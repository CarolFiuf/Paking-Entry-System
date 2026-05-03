"""
pipeline.py — Video Pipeline
DeepStream GPU pipeline (nếu có pyds) hoặc fallback GStreamer + OpenCV.

DeepStream chain (Phase 4+5):
  src0(plate) → flvdemux → h264parse → nvv4l2dec → queue ─┐
  src1(face)  → flvdemux → h264parse → nvv4l2dec →        │
                nvvidconv(flip-method) → queue ───────────┴→
                nvstreammux(batch=2) →
                nvinfer(plate, uid=1) →
                nvinfer(face_det,   uid=2, output-tensor-meta=1) →
                [Probe A: decode landmarks, attach user_meta to obj_meta] →
                nvtracker (nvDCF) →
                nvinfer(face_embed, uid=3, output-tensor-meta=1) →
                nvconv → caps RGBA → fakesink (Probe B reads results)

Probe B exports per source:
  - source 0: plate frame + plate detections (uid=1)
  - source 1: face frame + face data list (uid=2 obj + uid=3 embedding +
              tracker object_id + landmarks user_meta)

Fallback mode (no pyds):
  RTMP → GStreamer NVDEC → OpenCV → Python inference (StreamReader class).
"""

import cv2
import numpy as np
import logging
import time
from threading import Thread, Event, Lock
from queue import Queue, Empty

try:
    from face_meta_helpers import (
        decode_scrfd, extract_pgie_face_tensors,
        attach_landmarks_user_meta, read_landmarks_user_meta,
        backproject_landmarks_to_frame,
    )
    HAS_FACE_HELPERS = True
except Exception:
    HAS_FACE_HELPERS = False

log = logging.getLogger("pipeline")

_UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF

# ──────────────────────────────────────────────
# Thử import DeepStream Python bindings
# ──────────────────────────────────────────────
try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib
    import pyds
    HAS_DEEPSTREAM = True
    log.info("DeepStream SDK available")
except ImportError:
    HAS_DEEPSTREAM = False
    log.info("DeepStream not found — using GStreamer fallback")


# ──────────────────────────────────────────────
# DeepStream Pipeline (FIXED)
# ──────────────────────────────────────────────
class DeepStreamPipeline:
    """
    DeepStream pipeline cho 2 camera RTMP.

    Pipeline layout (explicit elements — không dùng uridecodebin):
      rtmpsrc0 → flvdemux → h264parse → nvv4l2decoder ─┐
                                                         ├→ nvstreammux → nvinfer → probe
      rtmpsrc1 → flvdemux → h264parse → nvv4l2decoder ─┘

    FIX: uridecodebin tạo dynamic pad, khi dùng với RTMP trong
    Gst.parse_launch thường không link được vào nvstreammux.
    Thay bằng explicit elements + pad-added signal cho flvdemux.
    """

    def __init__(self, plate_src: str, face_src: str, cfg: dict):
        Gst.init(None)

        self.cfg = cfg
        self._stop = Event()
        self._lock = Lock()

        ds_cfg = cfg.get("deepstream", {})
        self._face_chain = bool(ds_cfg.get("face_chain_enabled", False))
        self._face_align = bool(ds_cfg.get("face_align_enabled", False))
        self._net_w = 640    # SCRFD input size, hardcoded khớp face_det_config

        # Kết quả mới nhất từ Probe B
        self._plate_frame = None
        self._plate_detections = []
        self._face_frame = None
        self._face_data = []         # list[dict{bbox, conf, track_id, embedding, landmarks}]

        self._frame_seq = 0
        self._frame_event = Event()

        # Early-skip batch ngay trong probe, trước get_nvds_buf_surface
        self._skip_n = max(
            1, int(cfg.get("camera", {}).get("process_every_n", 1)))
        self._probe_counter = 0

        self._probe_count = 0
        self._probe_fps = 0.0
        self._probe_t0 = time.time()

        # Stats cho debug
        self._dbg_pre_last_log = time.time()
        self._dbg_pre_frames = 0
        self._dbg_pre_raw_dets = 0
        self._dbg_pre_obj_uid2 = 0
        self._dbg_pre_score_max = 0.0

        self._dbg_last_log = time.time()
        self._dbg_n_face_frames = 0
        self._dbg_n_face_all_obj = 0
        self._dbg_n_face_uid2 = 0
        self._dbg_n_face_dets = 0
        self._dbg_n_face_emb = 0
        self._dbg_n_face_lmk = 0

        self._build_pipeline(plate_src, face_src)

    def _build_pipeline(self, plate_src: str, face_src: str):
        """Xây dựng DeepStream pipeline bằng element API (không parse_launch)."""

        self._pipeline = Gst.Pipeline.new("parking-pipeline")
        ds_cfg = self.cfg["deepstream"]
        face_flip = int(self.cfg.get("camera", {}).get("face_rotate_nv", 0))

        # ── Streammux ──
        mux = self._make_element("nvstreammux", "mux")
        mux.set_property("batch-size", ds_cfg.get("batch_size", 2))
        mux.set_property("width", 1280)
        mux.set_property("height", 720)
        mux.set_property("batched-push-timeout", 40000)
        mux.set_property("live-source", 1)

        # ── Sources ──
        self._add_rtmp_source(plate_src, source_id=0, mux=mux,
                              name_prefix="plate", flip_method=0)
        self._add_rtmp_source(face_src,  source_id=1, mux=mux,
                              name_prefix="face",  flip_method=face_flip)

        # ── PGIE plate detector ──
        pgie_plate = self._make_element("nvinfer", "plate_det")
        pgie_plate.set_property("config-file-path", ds_cfg["plate_config"])

        chain_tail = pgie_plate
        self._pgie_face = None
        self._tracker = None
        self._sgie_face = None

        if self._face_chain and HAS_FACE_HELPERS:
            # ── PGIE face detector (SCRFD) ──
            pgie_face = self._make_element("nvinfer", "face_det")
            pgie_face.set_property("config-file-path",
                                    ds_cfg["face_det_config"])
            self._pgie_face = pgie_face

            # ── Tracker ──
            tracker = self._make_element("nvtracker", "face_tracker")
            tracker.set_property("ll-lib-file",
                                  "/opt/nvidia/deepstream/deepstream/lib/"
                                  "libnvds_nvmultiobjecttracker.so")
            tracker.set_property("ll-config-file",
                                  ds_cfg["tracker_config"])
            tracker.set_property("tracker-width",  640)
            tracker.set_property("tracker-height", 384)
            self._tracker = tracker

            # ── nvdspreprocess: ArcFace alignment via custom lib ──
            preproc = self._make_element("nvdspreprocess", "face_preprocess")
            preproc.set_property("config-file",
                                  ds_cfg["face_preprocess_config"])
            self._face_preproc = preproc

            # ── SGIE face embedding (ArcFace) — consumes preprocessed tensor ──
            sgie_face = self._make_element("nvinfer", "face_embed")
            sgie_face.set_property("config-file-path",
                                    ds_cfg["face_embed_config"])
            self._sgie_face = sgie_face

            # ── Probe A: trên src pad của pgie_face (TRƯỚC tracker) ──
            # Index-based attach (dets[i] → obj_meta[i]) is safe ở vị trí này
            # vì obj_meta_list đang ở parser NMS order; tracker downstream
            # mới shuffle. Nếu di chuyển probe xuống sau tracker, đổi sang
            # IoU-match.
            pgie_face_src = pgie_face.get_static_pad("src")
            pgie_face_src.add_probe(
                Gst.PadProbeType.BUFFER,
                self._probe_attach_landmarks, None)

            chain_tail = sgie_face
            log.info("Face chain enabled: PGIE_face → tracker → "
                     "nvdspreprocess(align, EGL map) → SGIE_embed")
        elif self._face_chain and not HAS_FACE_HELPERS:
            log.warning("face_chain_enabled=true nhưng face_meta_helpers "
                        "import fail — disable face chain")

        # ── Output convert + sink ──
        nvconv = self._make_element("nvvideoconvert", "nvconv_out")
        capsfilter = self._make_element("capsfilter", "caps_rgba")
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
        capsfilter.set_property("caps", caps)

        sink = self._make_element("fakesink", "sink")
        sink.set_property("sync", 0)
        sink.set_property("async", 0)

        # ── Link chain: mux → pgie_plate → [pgie_face → tracker →
        #   nvdspreprocess(align) → sgie] → nvconv → caps → sink ──
        prev = mux
        link_order = [pgie_plate]
        if self._pgie_face:
            link_order += [
                self._pgie_face, self._tracker,
                self._face_preproc, self._sgie_face,
            ]
        link_order += [nvconv, capsfilter, sink]
        for elem in link_order:
            if not prev.link(elem):
                log.error(f"Failed to link {prev.get_name()} → "
                          f"{elem.get_name()}")
                return
            prev = elem

        # ── Probe B: cuối pipeline (sink pad) ──
        sink_pad = sink.get_static_pad("sink")
        sink_pad.add_probe(
            Gst.PadProbeType.BUFFER, self._probe_callback, None)

        # ── Bus watch ──
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        log.info("DeepStream pipeline built")

    def _make_element(self, factory: str, name: str):
        """Tạo GStreamer element, add vào pipeline."""
        elem = Gst.ElementFactory.make(factory, name)
        if not elem:
            raise RuntimeError(
                f"Cannot create element: {factory} ({name}). "
                f"Plugin missing? Try: gst-inspect-1.0 {factory}")
        self._pipeline.add(elem)
        return elem

    def _add_rtmp_source(self, rtmp_url: str, source_id: int,
                         mux, name_prefix: str, flip_method: int = 0):
        """
        Thêm 1 RTMP source vào pipeline.

        Chain: rtmpsrc → flvdemux → (dynamic pad) → h264parse → nvv4l2decoder
               → [optional] nvvidconv(flip-method) → queue → mux.sink_N

        flip_method:
          0=none, 1=90CCW, 2=180, 3=90CW, 4=horiz, 6=vert
          Áp dụng per-source ingress để rotate face cam mà không demux/remux.
        """
        src = self._make_element("rtmpsrc", f"{name_prefix}_src")
        src.set_property("location", rtmp_url)
        src.set_property("timeout", 10)

        demux = self._make_element("flvdemux", f"{name_prefix}_demux")
        parse = self._make_element("h264parse", f"{name_prefix}_parse")
        decoder = self._make_element("nvv4l2decoder",
                                     f"{name_prefix}_decoder")

        queue = self._make_element("queue", f"{name_prefix}_queue")
        queue.set_property("max-size-buffers", 5)
        queue.set_property("leaky", 2)

        if not src.link(demux):
            log.error(f"Failed to link {name_prefix}_src → demux")

        if not parse.link(decoder):
            log.error(f"Failed to link {name_prefix}_parse → decoder")

        # Optional flip per source
        if flip_method != 0:
            flip = self._make_element("nvvideoconvert",
                                       f"{name_prefix}_flip")
            flip.set_property("flip-method", flip_method)
            if not decoder.link(flip):
                log.error(f"Failed to link {name_prefix}_decoder → flip")
            if not flip.link(queue):
                log.error(f"Failed to link {name_prefix}_flip → queue")
            log.info(f"[{name_prefix}] flip-method={flip_method}")
        else:
            if not decoder.link(queue):
                log.error(f"Failed to link {name_prefix}_decoder → queue")

        mux_sink = mux.get_request_pad(f"sink_{source_id}")
        queue_src = queue.get_static_pad("src")
        if mux_sink and queue_src:
            queue_src.link(mux_sink)
        else:
            log.error(f"Failed to get pads for mux.sink_{source_id}")

        demux.connect("pad-added", self._on_demux_pad_added,
                       parse, name_prefix)

    @staticmethod
    def _on_demux_pad_added(demux, pad, parse, name_prefix):
        """
        Callback khi flvdemux tạo pad mới.
        Chỉ link pad video (bỏ qua audio).
        """
        pad_name = pad.get_name()
        caps = pad.get_current_caps()
        struct_name = caps.get_structure(0).get_name() if caps else ""

        log.info(f"[{name_prefix}] flvdemux pad added: {pad_name} "
                 f"({struct_name})")

        # Chỉ link video, bỏ audio
        if pad_name.startswith("video") or "video" in struct_name:
            sink_pad = parse.get_static_pad("sink")
            if sink_pad and not sink_pad.is_linked():
                ret = pad.link(sink_pad)
                if ret == Gst.PadLinkReturn.OK:
                    log.info(f"[{name_prefix}] Linked video → h264parse")
                else:
                    log.error(f"[{name_prefix}] Failed to link video: {ret}")
        else:
            log.debug(f"[{name_prefix}] Ignoring non-video pad: {pad_name}")

    # ────────────────────────────────────────────────────────────────
    # Probe A: chạy trên src pad của PGIE_face, TRƯỚC nvtracker.
    # Đọc raw 9 tensors từ frame_user_meta (output-tensor-meta=1 của uid=2),
    # decode landmarks bằng face_meta_helpers.decode_scrfd, và attach
    # landmarks user_meta vào từng obj_meta theo NMS-sorted order.
    # ────────────────────────────────────────────────────────────────
    def _probe_attach_landmarks(self, pad, info, _):
        if not self._face_chain:
            return Gst.PadProbeReturn.OK
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        try:
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        except Exception:
            return Gst.PadProbeReturn.OK
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                fm = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            # Chỉ xử lý face cam
            if fm.source_id == 1:
                now = time.time()
                self._dbg_pre_frames += 1
                pre_uid2 = 0
                l_count = fm.obj_meta_list
                while l_count is not None:
                    try:
                        om_count = pyds.NvDsObjectMeta.cast(l_count.data)
                    except StopIteration:
                        break
                    if om_count.unique_component_id == 2:
                        pre_uid2 += 1
                    try:
                        l_count = l_count.next
                    except StopIteration:
                        break
                self._dbg_pre_obj_uid2 += pre_uid2

                try:
                    layers = extract_pgie_face_tensors(
                        fm, gie_uid=2, net_w=self._net_w)
                except Exception as e:
                    log.debug(f"extract_pgie_face_tensors err: {e}")
                    layers = {}

                if layers:
                    try:
                        dets = decode_scrfd(layers,
                                             self._net_w, self._net_w)
                    except Exception as e:
                        log.warning(f"decode_scrfd err: {e}")
                        dets = []

                    score_max = max(
                        float(np.asarray(v["score"]).max())
                        for v in layers.values()
                    ) if layers else 0.0
                    self._dbg_pre_score_max = max(
                        self._dbg_pre_score_max, score_max)
                    self._dbg_pre_raw_dets += len(dets)

                    # Backproject landmarks: 640x640 network → 1280x720 frame
                    # (streammux output coords). nvdspreprocess + libnvds_face_align
                    # warp source frame, nên landmarks phải ở frame coords.
                    for d in dets:
                        d["landmarks"] = backproject_landmarks_to_frame(
                            d["landmarks"], self._net_w, self._net_w,
                            1280, 720)

                    # Attach landmarks[i] → obj_meta[i] theo NMS order
                    l_obj = fm.obj_meta_list
                    idx = 0
                    while l_obj is not None and idx < len(dets):
                        try:
                            om = pyds.NvDsObjectMeta.cast(l_obj.data)
                        except StopIteration:
                            break
                        if om.unique_component_id == 2:
                            try:
                                attach_landmarks_user_meta(
                                    batch_meta, om,
                                    dets[idx]["landmarks"])
                            except Exception as e:
                                log.debug(f"attach landmarks err: {e}")
                            idx += 1
                        try:
                            l_obj = l_obj.next
                        except StopIteration:
                            break

                if now - self._dbg_pre_last_log >= 5.0:
                    log.info(
                        f"face_pgie_dbg: frames={self._dbg_pre_frames} "
                        f"raw_dets={self._dbg_pre_raw_dets} "
                        f"obj_uid2={self._dbg_pre_obj_uid2} "
                        f"score_max={self._dbg_pre_score_max:.3f}")
                    self._dbg_pre_last_log = now
                    self._dbg_pre_frames = 0
                    self._dbg_pre_raw_dets = 0
                    self._dbg_pre_obj_uid2 = 0
                    self._dbg_pre_score_max = 0.0

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    # ────────────────────────────────────────────────────────────────
    # Probe B: chạy ở fakesink. Tách frame + dữ liệu theo source_id +
    # obj_meta.unique_component_id.
    # ────────────────────────────────────────────────────────────────
    def _probe_callback(self, pad, info, user_data):
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        # FPS đo stream thật (mọi batch)
        self._probe_count += 1
        now = time.time()
        if now - self._probe_t0 >= 1.0:
            self._probe_fps = self._probe_count / (now - self._probe_t0)
            self._probe_count = 0
            self._probe_t0 = now

        # Early-skip
        self._probe_counter += 1
        if self._skip_n > 1 and self._probe_counter % self._skip_n != 0:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            source_id = frame_meta.source_id

            surface = pyds.get_nvds_buf_surface(hash(buf),
                                                frame_meta.batch_id)
            frame_rgba = np.array(surface, copy=True, order='C')
            frame = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)

            plate_dets = []
            face_data = []
            total_obj = 0
            uid2_post = 0

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                total_obj += 1
                uid = obj_meta.unique_component_id
                rect = obj_meta.rect_params
                bbox = (int(rect.left), int(rect.top),
                        int(rect.left + rect.width),
                        int(rect.top + rect.height))

                if source_id == 0 and uid == 1:
                    plate_dets.append({
                        "bbox": bbox,
                        "conf": float(obj_meta.confidence),
                    })
                elif source_id == 1 and uid == 2:
                    uid2_post += 1
                    fd = self._extract_face_meta(obj_meta, bbox)
                    face_data.append(fd)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            with self._lock:
                if source_id == 0:
                    self._plate_frame = frame
                    self._plate_detections = plate_dets
                    self._frame_seq += 1
                    self._frame_event.set()
                elif source_id == 1:
                    self._face_frame = frame
                    self._face_data = face_data

            # Debug stats
            if source_id == 1:
                self._dbg_n_face_frames += 1
                self._dbg_n_face_all_obj += total_obj
                self._dbg_n_face_uid2 += uid2_post
                self._dbg_n_face_dets += len(face_data)
                self._dbg_n_face_emb += sum(
                    1 for f in face_data if f.get("embedding") is not None)
                self._dbg_n_face_lmk += sum(
                    1 for f in face_data if f.get("landmarks") is not None)
                if now - self._dbg_last_log >= 5.0:
                    log.info(f"face_dbg: frames={self._dbg_n_face_frames} "
                             f"all_obj={self._dbg_n_face_all_obj} "
                             f"uid2={self._dbg_n_face_uid2} "
                             f"dets={self._dbg_n_face_dets} "
                             f"emb={self._dbg_n_face_emb} "
                             f"lmk={self._dbg_n_face_lmk}")
                    self._dbg_last_log = now
                    self._dbg_n_face_frames = 0
                    self._dbg_n_face_all_obj = 0
                    self._dbg_n_face_uid2 = 0
                    self._dbg_n_face_dets = 0
                    self._dbg_n_face_emb = 0
                    self._dbg_n_face_lmk = 0

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _extract_face_meta(self, obj_meta, bbox: tuple) -> dict:
        """Đọc embedding (SGIE uid=3 tensor meta) + landmarks user_meta."""
        import ctypes
        emb = None
        landmarks = None

        l_user = obj_meta.obj_user_meta_list
        while l_user is not None:
            try:
                u = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            mt = u.base_meta.meta_type
            if mt == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                try:
                    t = pyds.NvDsInferTensorMeta.cast(u.user_meta_data)
                    if t.unique_id == 3 and t.num_output_layers > 0:
                        ptr = pyds.get_ptr(t.out_buf_ptrs_host[0])
                        arr = np.ctypeslib.as_array(
                            ctypes.cast(ptr,
                                         ctypes.POINTER(ctypes.c_float)),
                            shape=(512,)
                        ).copy()
                        n = np.linalg.norm(arr)
                        emb = arr / n if n > 1e-6 else arr
                except Exception as e:
                    log.debug(f"emb read err: {e}")
            else:
                # Landmarks user_meta
                try:
                    lm = read_landmarks_user_meta(obj_meta) \
                        if HAS_FACE_HELPERS else None
                    if lm is not None and landmarks is None:
                        landmarks = lm
                except Exception:
                    pass

            try:
                l_user = l_user.next
            except StopIteration:
                break

        # pyds không expose UNTRACKED_OBJECT_ID; C header = 0xFFFFFFFFFFFFFFFF
        oid = obj_meta.object_id
        track_id = int(oid) if oid != _UNTRACKED_OBJECT_ID else None
        return {
            "bbox": bbox,
            "conf": float(obj_meta.confidence),
            "track_id": track_id,
            "embedding": emb,
            "landmarks": landmarks,
        }
    
    @property
    def stream_fps(self):
        return round(self._probe_fps, 1)
    
    def wait_new_frame(self, timeout=0.5) -> bool:
        """Block cho tới khi có frame mới từ probe."""
        self._frame_event.clear()
        return self._frame_event.wait(timeout=timeout)

    def _on_bus_message(self, bus, message):
        """Log GStreamer bus messages — rất quan trọng để debug."""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            src = message.src.get_name() if message.src else "?"
            log.error(f"GST ERROR [{src}]: {err.message}")
            log.error(f"  Debug: {debug}")
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            src = message.src.get_name() if message.src else "?"
            log.warning(f"GST WARN [{src}]: {warn.message}")
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self._pipeline:
                old, new, pending = message.parse_state_changed()
                log.info(f"Pipeline state: {old.value_nick} → "
                         f"{new.value_nick}")
        elif t == Gst.MessageType.STREAM_START:
            src = message.src.get_name() if message.src else "?"
            log.info(f"Stream started: {src}")
        elif t == Gst.MessageType.EOS:
            log.warning("End of stream")

    def start(self):
        """Start pipeline."""
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            log.error("Failed to set pipeline to PLAYING")
            # Log chi tiết
            ret2 = self._pipeline.get_state(5 * Gst.SECOND)
            log.error(f"Pipeline state: {ret2}")
        else:
            log.info(f"Pipeline set_state → PLAYING (ret={ret})")

        # GLib main loop trên thread riêng (cần cho bus messages)
        self._loop = GLib.MainLoop()
        self._loop_thread = Thread(target=self._loop.run, daemon=True)
        self._loop_thread.start()
        log.info("DeepStream pipeline started")

    def get_plate_data(self):
        """Lấy plate frame + detections mới nhất."""
        with self._lock:
            return self._plate_frame, self._plate_detections

    def get_face_frame(self):
        """Lấy face frame mới nhất."""
        with self._lock:
            return self._face_frame

    def get_face_data(self):
        """Lấy face frame + face_data list (mỗi face: bbox/conf/track_id/emb/lmk)."""
        with self._lock:
            return self._face_frame, list(self._face_data)

    def get_all(self):
        """Lấy plate + face frame + face data atomic."""
        with self._lock:
            return (self._plate_frame, self._plate_detections,
                    self._face_frame, list(self._face_data))

    def stop(self):
        self._stop.set()
        self._pipeline.set_state(Gst.State.NULL)
        if hasattr(self, "_loop"):
            self._loop.quit()
        log.info("DeepStream pipeline stopped")


# ──────────────────────────────────────────────
# GStreamer Fallback (StreamReader)
# ──────────────────────────────────────────────
class StreamReader:
    """Threaded stream reader — fallback khi không có DeepStream."""

    def __init__(self, source, name: str = "cam",
                 hw_decode: bool = True, reconnect_sec: float = 3.0):
        self.source = source
        self.name = name
        self.hw_decode = hw_decode
        self.reconnect_sec = reconnect_sec
        self._stop = Event()
        self._queue: Queue = Queue(maxsize=1)
        self._connected = False
        
        self._latest = None
        self._latest_lock = Lock()
        
        self.cap = None
        self.is_stream = isinstance(source, str) and \
            source.startswith(("rtmp://", "rtsp://", "http://"))

        self._connect()
        self._thread = Thread(target=self._reader, daemon=True,
                              name=f"reader-{name}")
        self._thread.start()
        
        self._read_count = 0
        self._read_fps = 0.0
        self._read_t0 = time.time()

    def _connect(self):
        if self.cap:
            self.cap.release()

        if self.hw_decode and isinstance(self.source, str):
            pipeline = self._build_gst(self.source)
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap or not self.cap.isOpened():
            if self.hw_decode:
                log.warning(f"[{self.name}] GStreamer failed, fallback")
            # self.cap = cv2.VideoCapture(str(self.source))
            self.cap = cv2.VideoCapture()
            self.cap.open(
                str(self.source),
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,  # 3s thay vì 30-60s
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000,
                ]
            )  

        self._connected = self.cap.isOpened()
        if self._connected:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info(f"[{self.name}] Connected: {w}x{h}")
        else:
            log.error(f"[{self.name}] Cannot open: {self.source}")

    def _build_gst(self, source: str) -> str:
        if source.startswith(("rtmp://", "rtsp://", "http://")):
            return (f'uridecodebin uri="{source}" ! '
                    "nvvidconv ! video/x-raw,format=BGRx ! "
                    "videoconvert ! video/x-raw,format=BGR ! "
                    "appsink drop=1 sync=0 max-buffers=1")
        return (f'filesrc location="{source}" ! '
                "decodebin ! nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=1")

    def _reader(self):
        while not self._stop.is_set():
            if not self._connected:
                if not self.is_stream:
                    self._queue.put(None)
                    break
                self._stop.wait(self.reconnect_sec)
                if self._stop.is_set():
                    break
                self._connect()
                continue

            ret, frame = self.cap.read()
            if not ret:
                if self.is_stream:
                    self._connected = False
                    continue
                self._queue.put(None)
                break
            
            self._read_count += 1
            now = time.time()
            if now - self._read_t0 >= 1.0:
                self._read_fps = self._read_count / (now - self._read_t0)
                self._read_count = 0
                self._read_t0 = now
                
            with self._latest_lock:
                self._latest = frame

            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put(frame)

    def read(self, timeout: float = 5.0):
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    @property
    def stream_fps(self):
        return round(self._read_fps, 1)

    @property
    def latest(self):
        """Frame mới nhất, không pop khỏi queue. Dùng cho web update thread."""
        with self._latest_lock:
            return self._latest
        
    @property
    def connected(self):
        return self._connected

    def release(self):
        self._stop.set()
        if self.cap:
            self.cap.release()
            
