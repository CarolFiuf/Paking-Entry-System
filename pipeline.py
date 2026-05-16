"""
pipeline.py — Video Pipeline
DeepStream GPU pipeline (nếu có pyds) hoặc fallback GStreamer + OpenCV.

DeepStream chain (split mode):
  plate_src → decode → nvstreammux(batch=1) → nvinfer(plate, uid=1)
            → nvconv RGBA → fakesink

  face_src  → decode → nvstreammux(batch=1) → nvinfer(face_det, uid=2)
            → nvconv RGBA → nvdsfaceembed(custom align + TensorRT ArcFace)
            → fakesink

Probe B exports:
  - plate pipeline: plate frame + plate detections (uid=1)
  - face pipeline: face frame + face data list (uid=2 obj +
                   embedding/landmarks/bbox từ nvdsfaceembed user meta)

Fallback mode (no pyds):
  RTMP → GStreamer NVDEC → OpenCV → Python inference (StreamReader class).
"""

import cv2
import numpy as np
import logging
import time
import ctypes
import os
from threading import Thread, Event, Lock
from queue import Queue, Empty

log = logging.getLogger("pipeline")

_UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF
_FACE_EMBED_META_DESC = "PARKING.FACE_EMBEDDING_META"
_FACE_EMBED_META_TYPE = None


class _FaceEmbeddingMeta(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("dims", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("embedding", ctypes.c_float * 512),
        ("landmarks", ctypes.c_float * 10),
        ("bbox", ctypes.c_float * 4),
        ("quality", ctypes.c_float),
    ]


def _get_face_embed_meta_type():
    global _FACE_EMBED_META_TYPE
    if _FACE_EMBED_META_TYPE is None:
        _FACE_EMBED_META_TYPE = pyds.nvds_get_user_meta_type(
            _FACE_EMBED_META_DESC)
    return _FACE_EMBED_META_TYPE

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
        self._register_local_plugins()

        self.cfg = cfg
        self._stop = Event()
        self._lock = Lock()

        ds_cfg = cfg.get("deepstream", {})
        self._plate_enabled = bool(ds_cfg.get("plate_enabled", True))
        self._face_enabled = bool(ds_cfg.get("face_enabled", True))
        if not self._plate_enabled and not self._face_enabled:
            raise RuntimeError("DeepStream needs at least one enabled source")

        self._plate_source_id = 0
        self._face_source_id = 0
        self._plate_gie_uid = 1
        self._face_gie_uid = 2

        self._net_w = 640    # SCRFD input size, hardcoded khớp face_det_config

        # Kết quả mới nhất từ Probe B
        self._plate_frame = None
        self._plate_detections = []
        self._face_frame = None
        self._face_data = []         # list[dict{bbox, conf, embedding, landmarks}]

        self._frame_seq = 0
        self._frame_event = Event()

        # Early-skip batch ngay trong probe, trước get_nvds_buf_surface
        self._skip_n = max(
            1, int(cfg.get("camera", {}).get("process_every_n", 1)))
        self._probe_counter = 0

        self._probe_count = 0
        self._probe_fps = 0.0
        self._probe_t0 = time.time()

        # Stats cho debug (Probe B)
        self._dbg_last_log = time.time()
        self._dbg_n_face_frames = 0
        self._dbg_n_face_all_obj = 0
        self._dbg_n_face_uid2 = 0
        self._dbg_n_face_dets = 0
        self._dbg_n_face_emb = 0
        self._dbg_n_face_lmk = 0
        self._dbg_emb_read_err = 0

        self._plate_pipeline = None
        self._face_pipeline = None
        self._pipelines = []
        self._pgie_face = None
        self._face_embedder = None

        self._build_split_pipelines(plate_src, face_src)

    @staticmethod
    def _register_local_plugins():
        """Expose repo-local GStreamer plugins without system install."""
        root = os.path.dirname(os.path.abspath(__file__))
        plugin_dir = os.path.join(root, "plugins", "gst-nvdsfaceembed")
        old = os.environ.get("GST_PLUGIN_PATH", "")
        paths = [p for p in old.split(os.pathsep) if p]
        if plugin_dir not in paths:
            os.environ["GST_PLUGIN_PATH"] = (
                plugin_dir if not old else plugin_dir + os.pathsep + old)
        try:
            Gst.Registry.get().scan_path(plugin_dir)
        except Exception as e:
            log.debug(f"Local plugin scan skipped: {e}")

    @staticmethod
    def _read_nvinfer_precluster_threshold(config_path: str,
                                           default: float = 0.65) -> float:
        """Keep nvdsfaceembed landmark decode threshold synced with PGIE."""
        try:
            path = os.path.abspath(config_path)
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.split("#", 1)[0].strip()
                    if not line.startswith("pre-cluster-threshold"):
                        continue
                    _, value = line.split("=", 1)
                    return float(value.strip())
        except Exception as e:
            log.debug(f"Could not read face detector threshold: {e}")
        return default

    @staticmethod
    def _configure_mux(mux, batch_size: int = 1):
        mux.set_property("batch-size", batch_size)
        mux.set_property("width", 1280)
        mux.set_property("height", 720)
        mux.set_property("batched-push-timeout", 40000)
        mux.set_property("live-source", 1)

    def _attach_bus(self, pipeline):
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

    def _link_many(self, elements: list) -> bool:
        for prev, elem in zip(elements, elements[1:]):
            if not prev.link(elem):
                log.error(f"Failed to link {prev.get_name()} → "
                          f"{elem.get_name()}")
                return False
        return True

    def _build_split_pipelines(self, plate_src: str, face_src: str):
        """Build independent batch=1 plate and face DeepStream pipelines."""
        if self._plate_enabled:
            self._build_plate_pipeline(plate_src)
        else:
            log.info("Plate pipeline disabled")

        if self._face_enabled:
            self._build_face_pipeline(face_src)
        else:
            log.info("Face pipeline disabled")

        log.info("DeepStream split pipelines built "
                 f"(plate={self._plate_enabled}, face={self._face_enabled}, "
                 "batch=1+1)")

    def _build_plate_pipeline(self, plate_src: str):
        ds_cfg = self.cfg["deepstream"]
        pipeline = Gst.Pipeline.new("parking-plate-pipeline")
        self._plate_pipeline = pipeline
        self._pipelines.append(pipeline)

        mux = self._make_element("nvstreammux", "plate_mux", pipeline)
        self._configure_mux(mux, 1)
        self._add_rtmp_source(plate_src, source_id=self._plate_source_id,
                              mux=mux, name_prefix="plate", flip_method=0,
                              pipeline=pipeline)

        pgie_plate = self._make_element("nvinfer", "plate_det", pipeline)
        pgie_plate.set_property("config-file-path", ds_cfg["plate_config"])

        nvconv = self._make_element("nvvideoconvert", "plate_nvconv_out",
                                    pipeline)
        capsfilter = self._make_element("capsfilter", "plate_caps_rgba",
                                        pipeline)
        capsfilter.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))

        sink = self._make_element("fakesink", "plate_sink", pipeline)
        sink.set_property("sync", 0)
        sink.set_property("async", 0)

        self._link_many([mux, pgie_plate, nvconv, capsfilter, sink])
        sink.get_static_pad("sink").add_probe(
            Gst.PadProbeType.BUFFER, self._probe_callback, "plate")
        self._attach_bus(pipeline)
        log.info("Plate pipeline: src → mux(batch=1) → plate_det → sink")

    def _build_face_pipeline(self, face_src: str):
        ds_cfg = self.cfg["deepstream"]
        face_flip = int(self.cfg.get("camera", {}).get("face_rotate_nv", 0))

        pipeline = Gst.Pipeline.new("parking-face-pipeline")
        self._face_pipeline = pipeline
        self._pipelines.append(pipeline)

        mux = self._make_element("nvstreammux", "face_mux", pipeline)
        self._configure_mux(mux, 1)
        self._add_rtmp_source(face_src, source_id=self._face_source_id,
                              mux=mux, name_prefix="face",
                              flip_method=face_flip, pipeline=pipeline)

        pgie_face = self._make_element("nvinfer", "face_det", pipeline)
        pgie_face.set_property("config-file-path", ds_cfg["face_det_config"])
        self._pgie_face = pgie_face

        nvconv = self._make_element("nvvideoconvert", "face_nvconv_out",
                                    pipeline)
        capsfilter = self._make_element("capsfilter", "face_caps_rgba",
                                        pipeline)
        capsfilter.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))

        embedder = self._make_face_embedder(pipeline, ds_cfg)
        self._face_embedder = embedder

        sink = self._make_element("fakesink", "face_sink", pipeline)
        sink.set_property("sync", 0)
        sink.set_property("async", 0)

        self._link_many([mux, pgie_face, nvconv, capsfilter, embedder, sink])

        sink.get_static_pad("sink").add_probe(
            Gst.PadProbeType.BUFFER, self._probe_callback, "face")
        self._attach_bus(pipeline)
        log.info("Face pipeline: src → mux(batch=1) → face_det → "
                 "nvdsfaceembed(aligned TRT ArcFace) → sink")

    def _make_face_embedder(self, pipeline, ds_cfg):
        embedder = self._make_element("nvdsfaceembed", "face_embed_aligned",
                                      pipeline)
        engine_path = ds_cfg.get(
            "face_embed_engine",
            "./models/face_embed_arcface_fp16.engine")
        embedder.set_property("engine-file", os.path.abspath(engine_path))
        embedder.set_property("gpu-id", 0)
        embedder.set_property("unique-id", 7)
        embedder.set_property("face-gie-id", self._face_gie_uid)
        embedder.set_property("source-id", self._face_source_id)
        embedder.set_property(
            "batch-size", int(ds_cfg.get("face_embed_batch_size", 16)))
        embedder.set_property(
            "align-on-gpu",
            bool(ds_cfg.get("face_embed_align_on_gpu", True)))
        embedder.set_property(
            "allow-cpu-fallback",
            bool(ds_cfg.get("face_embed_allow_cpu_fallback", True)))
        decode_thr = ds_cfg.get("face_embed_decode_conf_threshold")
        if decode_thr is None:
            decode_thr = self._read_nvinfer_precluster_threshold(
                ds_cfg["face_det_config"], 0.50)
        embedder.set_property("decode-conf-threshold", float(decode_thr))
        embedder.set_property("net-width", self._net_w)
        embedder.set_property("net-height", self._net_w)
        embedder.set_property("input-object-min-width", 32)
        embedder.set_property("input-object-min-height", 32)
        return embedder

    def _make_element(self, factory: str, name: str, pipeline=None):
        """Tạo GStreamer element, add vào pipeline."""
        elem = Gst.ElementFactory.make(factory, name)
        if not elem:
            raise RuntimeError(
                f"Cannot create element: {factory} ({name}). "
                f"Plugin missing? Try: gst-inspect-1.0 {factory}")
        if pipeline is None:
            raise RuntimeError(f"No target pipeline for element {name}")
        pipeline.add(elem)
        return elem

    def _add_rtmp_source(self, rtmp_url: str, source_id: int,
                         mux, name_prefix: str, flip_method: int = 0,
                         pipeline=None):
        """
        Thêm 1 RTMP source vào pipeline.

        Chain: rtmpsrc → flvdemux → (dynamic pad) → h264parse → nvv4l2decoder
               → [optional] nvvidconv(flip-method) → queue → mux.sink_N

        flip_method:
          0=none, 1=90CCW, 2=180, 3=90CW, 4=horiz, 6=vert
          Áp dụng per-source ingress để rotate face cam mà không demux/remux.
        """
        src = self._make_element("rtmpsrc", f"{name_prefix}_src", pipeline)
        src.set_property("location", rtmp_url)
        src.set_property("timeout", 10)

        demux = self._make_element("flvdemux", f"{name_prefix}_demux",
                                   pipeline)
        parse = self._make_element("h264parse", f"{name_prefix}_parse",
                                   pipeline)
        decoder = self._make_element("nvv4l2decoder",
                                     f"{name_prefix}_decoder", pipeline)

        queue = self._make_element("queue", f"{name_prefix}_queue", pipeline)
        queue.set_property("max-size-buffers", 5)
        queue.set_property("leaky", 2)

        if not src.link(demux):
            log.error(f"Failed to link {name_prefix}_src → demux")

        if not parse.link(decoder):
            log.error(f"Failed to link {name_prefix}_parse → decoder")

        # Optional flip per source
        if flip_method != 0:
            flip = self._make_element("nvvideoconvert",
                                       f"{name_prefix}_flip", pipeline)
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
    # Probe B: chạy ở fakesink. Tách frame + dữ liệu theo source_id +
    # obj_meta.unique_component_id.
    # ────────────────────────────────────────────────────────────────
    def _probe_callback(self, pad, info, user_data):
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK
        stream_kind = user_data

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
            is_plate_frame = (
                stream_kind == "plate" or
                (stream_kind is None and source_id == self._plate_source_id))
            is_face_frame = (
                stream_kind == "face" or
                (stream_kind is None and source_id == self._face_source_id))

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

                if is_plate_frame and uid == self._plate_gie_uid:
                    plate_dets.append({
                        "bbox": bbox,
                        "conf": float(obj_meta.confidence),
                    })
                elif is_face_frame and uid == self._face_gie_uid:
                    uid2_post += 1
                    face_data.append(
                        self._extract_face_meta(obj_meta, bbox, frame.shape))

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            with self._lock:
                if is_plate_frame:
                    self._plate_frame = frame
                    self._plate_detections = plate_dets
                    self._frame_seq += 1
                    self._frame_event.set()
                elif is_face_frame:
                    self._face_frame = frame
                    self._face_data = face_data
                    self._frame_seq += 1
                    self._frame_event.set()

            # Debug stats
            if is_face_frame:
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
                    self._dbg_emb_read_err = 0

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    @staticmethod
    def _clip_bbox(bbox: tuple, frame_shape) -> tuple:
        if bbox is None or frame_shape is None:
            return bbox
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _read_face_embed_meta(self, obj_meta):
        """
        Đọc PARKING.FACE_EMBEDDING_META do plugin gst-nvdsfaceembed attach
        vào obj_meta. Plugin đã L2-normalize embedding và cung cấp landmarks
        + bbox đã decode từ SCRFD tensor.
        """
        meta_type = _get_face_embed_meta_type()
        l_user = obj_meta.obj_user_meta_list
        while l_user is not None:
            try:
                u = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if u.base_meta.meta_type == meta_type:
                try:
                    ptr = pyds.get_ptr(u.user_meta_data)
                    meta = _FaceEmbeddingMeta.from_address(ptr)
                    if meta.version not in (1, 2) or meta.dims != 512:
                        return None, None, None
                    emb = np.ctypeslib.as_array(meta.embedding,
                                                shape=(512,)).copy()
                    lm = np.ctypeslib.as_array(meta.landmarks,
                                               shape=(10,)).copy()
                    face_bbox = None
                    if meta.version >= 2:
                        b = np.ctypeslib.as_array(meta.bbox,
                                                  shape=(4,)).copy()
                        if np.all(np.isfinite(b)) and b[2] > b[0] and b[3] > b[1]:
                            face_bbox = tuple(int(round(v)) for v in b)
                    return emb, lm.reshape(5, 2), face_bbox
                except Exception as e:
                    self._dbg_emb_read_err += 1
                    if self._dbg_emb_read_err <= 3:
                        log.warning(f"face embed meta read err: {e}")
                    return None, None, None

            try:
                l_user = l_user.next
            except StopIteration:
                break
        return None, None, None

    def _extract_face_meta(self, obj_meta, bbox: tuple,
                           frame_shape=None) -> dict:
        """
        Trích face data từ obj_meta: ưu tiên bbox + landmarks + embedding
        do plugin gst-nvdsfaceembed attach. Fallback bbox = PGIE rect_params.
        """
        emb, landmarks, face_bbox = self._read_face_embed_meta(obj_meta)

        oid = obj_meta.object_id
        track_id = int(oid) if oid != _UNTRACKED_OBJECT_ID else None
        final_bbox = bbox
        clipped_face_bbox = self._clip_bbox(face_bbox, frame_shape)
        if clipped_face_bbox is not None:
            final_bbox = clipped_face_bbox
        return {
            "bbox": final_bbox,
            "detector_bbox": bbox,
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
            if message.src in self._pipelines:
                old, new, pending = message.parse_state_changed()
                src = message.src.get_name() if message.src else "pipeline"
                log.info(f"{src} state: {old.value_nick} → "
                         f"{new.value_nick}")
        elif t == Gst.MessageType.STREAM_START:
            src = message.src.get_name() if message.src else "?"
            log.info(f"Stream started: {src}")
        elif t == Gst.MessageType.EOS:
            log.warning("End of stream")

    def start(self):
        """Start pipeline."""
        for pipeline in self._pipelines:
            ret = pipeline.set_state(Gst.State.PLAYING)
            name = pipeline.get_name()
            if ret == Gst.StateChangeReturn.FAILURE:
                log.error(f"{name}: failed to set PLAYING")
                ret2 = pipeline.get_state(5 * Gst.SECOND)
                log.error(f"{name}: state={ret2}")
            else:
                log.info(f"{name} set_state → PLAYING (ret={ret})")

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
        for pipeline in self._pipelines:
            pipeline.set_state(Gst.State.NULL)
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
            
