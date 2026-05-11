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
                nvconv → caps RGBA →
                nvdsfaceembed(custom align + TensorRT ArcFace) →
                fakesink (Probe B reads results)

Probe B exports per source:
  - source 0: plate frame + plate detections (uid=1)
  - source 1: face frame + face data list (uid=2 obj +
              embedding/landmarks when nvdsfaceembed succeeds)

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

# Landmarks/embeddings được decode + attach trong native DeepStream elements;
# Python probe chỉ đọc metadata cuối pipeline.
HAS_FACE_HELPERS = True

log = logging.getLogger("pipeline")

_UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF
_NVDS_ROI_META = 29
_FACE_EMBED_META_DESC = "PARKING.FACE_EMBEDDING_META"
_FACE_EMBED_META_TYPE = None


class _NvOSDColorParams(ctypes.Structure):
    _fields_ = [
        ("red", ctypes.c_double),
        ("green", ctypes.c_double),
        ("blue", ctypes.c_double),
        ("alpha", ctypes.c_double),
    ]


class _NvOSDRectParams(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_float),
        ("top", ctypes.c_float),
        ("width", ctypes.c_float),
        ("height", ctypes.c_float),
        ("border_width", ctypes.c_uint),
        ("border_color", _NvOSDColorParams),
        ("has_bg_color", ctypes.c_uint),
        ("reserved", ctypes.c_uint),
        ("bg_color", _NvOSDColorParams),
        ("has_color_info", ctypes.c_int),
        ("color_id", ctypes.c_int),
    ]


class _GList(ctypes.Structure):
    pass


_GList._fields_ = [
    ("data", ctypes.c_void_p),
    ("next", ctypes.POINTER(_GList)),
    ("prev", ctypes.POINTER(_GList)),
]


class _NvDsRoiMeta(ctypes.Structure):
    _fields_ = [
        ("roi", _NvOSDRectParams),
        ("roi_polygon", ctypes.c_uint * 16),
        ("converted_buffer", ctypes.c_void_p),
        ("frame_meta", ctypes.c_void_p),
        ("scale_ratio_x", ctypes.c_double),
        ("scale_ratio_y", ctypes.c_double),
        ("offset_left", ctypes.c_double),
        ("offset_top", ctypes.c_double),
        ("classifier_meta_list", ctypes.c_void_p),
        ("roi_user_meta_list", ctypes.POINTER(_GList)),
        ("object_meta", ctypes.c_void_p),
    ]


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
        self._face_source_id = 1
        self._plate_gie_uid = 1
        self._face_gie_uid = 2

        self._face_chain = bool(
            self._face_enabled and ds_cfg.get("face_chain_enabled", False))
        self._face_embed_backend = str(
            ds_cfg.get("face_embed_backend", "direct_sgie")).lower()
        self._face_tracker_enabled = bool(
            ds_cfg.get("face_tracker_enabled", False))
        self._face_align = bool(ds_cfg.get("face_align_enabled", False))
        self._read_face_embeddings = bool(
            ds_cfg.get("read_face_embeddings", True))
        self._net_w = 640    # SCRFD input size, hardcoded khớp face_det_config

        # Kết quả mới nhất từ Probe B
        self._plate_frame = None
        self._plate_detections = []
        self._face_frame = None
        self._face_data = []         # list[dict{bbox, conf, embedding, landmarks}]
        self._face_detector_cache = {}

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
        self._dbg_n_face_pre = 0
        self._dbg_n_face_dets = 0
        self._dbg_n_face_emb = 0
        self._dbg_n_face_lmk = 0
        self._dbg_n_roi_meta = 0
        self._dbg_n_roi_emb = 0
        self._dbg_n_roi_err = 0
        self._dbg_n_obj_roi_meta = 0
        self._dbg_n_obj_roi_emb = 0
        self._dbg_n_cls_meta = 0
        self._dbg_n_cls_labels = 0
        self._dbg_cls_uids = {}
        self._dbg_frame_user_types = {}
        self._dbg_frame_tensor_uids = {}
        self._dbg_obj_user_types = {}
        self._dbg_obj_tensor_uids = {}
        self._dbg_emb_read_err = 0
        self._dbg_emb_null_ptr = 0

        self._build_pipeline(plate_src, face_src)

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

    def _build_pipeline(self, plate_src: str, face_src: str):
        """Xây dựng DeepStream pipeline bằng element API (không parse_launch)."""

        self._pipeline = Gst.Pipeline.new("parking-pipeline")
        ds_cfg = self.cfg["deepstream"]
        face_flip = int(self.cfg.get("camera", {}).get("face_rotate_nv", 0))

        # ── Streammux ──
        mux = self._make_element("nvstreammux", "mux")
        highest_source_id = -1
        if self._plate_enabled:
            highest_source_id = max(highest_source_id, self._plate_source_id)
        if self._face_enabled:
            highest_source_id = max(highest_source_id, self._face_source_id)
        batch_size = max(int(ds_cfg.get("batch_size", 2)),
                         highest_source_id + 1)
        mux.set_property("batch-size", batch_size)
        mux.set_property("width", 1280)
        mux.set_property("height", 720)
        mux.set_property("batched-push-timeout", 40000)
        mux.set_property("live-source", 1)

        # ── Sources ──
        if self._plate_enabled:
            self._add_rtmp_source(plate_src, source_id=self._plate_source_id,
                                  mux=mux, name_prefix="plate",
                                  flip_method=0)
        else:
            log.info("Plate source disabled in DeepStream graph")

        if self._face_enabled:
            self._add_rtmp_source(face_src, source_id=self._face_source_id,
                                  mux=mux, name_prefix="face",
                                  flip_method=face_flip)
        else:
            log.info("Face source disabled in DeepStream graph")

        # ── PGIE plate detector ──
        pgie_plate = None
        if self._plate_enabled:
            pgie_plate = self._make_element("nvinfer", "plate_det")
            pgie_plate.set_property("config-file-path",
                                    ds_cfg["plate_config"])

        self._pgie_face = None
        self._tracker = None
        self._sgie_face = None
        self._face_embedder = None

        if self._face_chain and HAS_FACE_HELPERS:
            # ── PGIE face detector (SCRFD) ──
            pgie_face = self._make_element("nvinfer", "face_det")
            pgie_face.set_property("config-file-path",
                                    ds_cfg["face_det_config"])
            self._pgie_face = pgie_face

            if self._face_tracker_enabled:
                tracker = self._make_element("nvtracker", "face_tracker")
                tracker.set_property("ll-lib-file",
                                      "/opt/nvidia/deepstream/deepstream/lib/"
                                      "libnvds_nvmultiobjecttracker.so")
                tracker.set_property("ll-config-file",
                                      ds_cfg["tracker_config"])
                tracker.set_property("tracker-width",  640)
                tracker.set_property("tracker-height", 384)
                self._tracker = tracker

            if self._face_embed_backend in ("aligned_trt", "custom_trt"):
                self._face_preproc = None
                embedder = self._make_element(
                    "nvdsfaceembed", "face_embed_aligned")
                engine_path = ds_cfg.get(
                    "face_embed_engine",
                    "./models/face_embed_arcface_fp16.engine")
                embedder.set_property("engine-file",
                                      os.path.abspath(engine_path))
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
                embedder.set_property("decode-conf-threshold",
                                      float(decode_thr))
                embedder.set_property("net-width", self._net_w)
                embedder.set_property("net-height", self._net_w)
                embedder.set_property("input-object-min-width", 32)
                embedder.set_property("input-object-min-height", 32)
                self._face_embedder = embedder
                chain = "PGIE_face"
                if self._tracker:
                    chain += " → tracker"
                chain += " → nvdsfaceembed(aligned TensorRT ArcFace)"
                log.info(f"Face chain: {chain}")
            elif self._face_align:
                # ── nvdspreprocess: ArcFace alignment via custom lib ──
                preproc = self._make_element(
                    "nvdspreprocess", "face_preprocess")
                preproc.set_property("config-file",
                                      ds_cfg["face_preprocess_config"])
                self._face_preproc = preproc

                # ── SGIE face embedding (ArcFace) ──
                sgie_face = self._make_element("nvinfer", "face_embed")
                sgie_face.set_property("config-file-path",
                                        ds_cfg["face_embed_config"])
                self._sgie_face = sgie_face

                chain = "PGIE_face"
                if self._tracker:
                    chain += " → tracker"
                chain += " → nvdspreprocess(align) → SGIE_embed"
                log.info(f"Face chain: {chain}")
            else:
                self._face_preproc = None
                sgie_face = self._make_element("nvinfer", "face_embed")
                sgie_face.set_property("config-file-path",
                                        ds_cfg["face_embed_config"])
                self._sgie_face = sgie_face
                log.warning("face_align_enabled=false: SGIE_embed uses "
                            "nvinfer object crop/resize (no landmark align)")
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

        # ── Link chain:
        #   mux → [plate_det] → [face_det → optional tracker/preprocess/sgie]
        #       → nvconv → caps → sink
        prev = mux
        link_order = []
        if pgie_plate is not None:
            link_order.append(pgie_plate)
        if self._pgie_face:
            link_order.append(self._pgie_face)
            if self._tracker:
                link_order.append(self._tracker)
            if self._face_preproc and self._sgie_face:
                link_order += [self._face_preproc, self._sgie_face]
            elif self._sgie_face:
                link_order += [self._sgie_face]
        link_order += [nvconv, capsfilter]
        if self._face_embedder:
            link_order += [self._face_embedder]
        link_order += [sink]
        for elem in link_order:
            if not prev.link(elem):
                log.error(f"Failed to link {prev.get_name()} → "
                          f"{elem.get_name()}")
                return
            prev = elem

        # Only needed when a tracker is enabled; no-tracker mode already emits
        # current detector boxes at the sink probe.
        if self._pgie_face and self._tracker:
            face_det_src_pad = self._pgie_face.get_static_pad("src")
            if face_det_src_pad:
                face_det_src_pad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    self._probe_face_detector_boxes, None)

        # ── Probe B: cuối pipeline (sink pad) ──
        sink_pad = sink.get_static_pad("sink")
        sink_pad.add_probe(
            Gst.PadProbeType.BUFFER, self._probe_callback, None)

        # ── Bus watch ──
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        log.info("DeepStream pipeline built "
                 f"(plate={self._plate_enabled}, face={self._face_enabled}, "
                 f"batch={batch_size})")

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
    # Probe A: face detector output, trước tracker. Chỉ copy bbox/conf plain
    # Python để dashboard không phụ thuộc bbox post-tracker.
    # ────────────────────────────────────────────────────────────────
    def _probe_face_detector_boxes(self, pad, info, user_data):
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        updates = {}
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            if frame_meta.source_id == self._face_source_id:
                dets = []
                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    except StopIteration:
                        break

                    if obj_meta.unique_component_id == self._face_gie_uid:
                        rect = obj_meta.rect_params
                        bbox = (int(rect.left), int(rect.top),
                                int(rect.left + rect.width),
                                int(rect.top + rect.height))
                        dets.append({
                            "bbox": bbox,
                            "conf": float(obj_meta.confidence),
                            "track_id": None,
                            "embedding": None,
                            "landmarks": None,
                            "source": "detector",
                        })

                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break

                updates[(int(frame_meta.source_id),
                         int(frame_meta.frame_num))] = dets

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        if updates:
            with self._lock:
                self._face_detector_cache.update(updates)
                # Keep cache bounded for live streams.
                while len(self._face_detector_cache) > 60:
                    first_key = next(iter(self._face_detector_cache))
                    self._face_detector_cache.pop(first_key, None)

        return Gst.PadProbeReturn.OK

    def _consume_face_detector_boxes(self, source_id: int,
                                     frame_num: int) -> list:
        key = (int(source_id), int(frame_num))
        with self._lock:
            dets = self._face_detector_cache.pop(key, [])
        return list(dets)

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
            detector_face_data = (
                self._consume_face_detector_boxes(
                    source_id, frame_meta.frame_num)
                if source_id == self._face_source_id else []
            )
            roi_embeddings = (
                self._extract_face_roi_embeddings(frame_meta)
                if source_id == self._face_source_id else []
            )
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

                if (source_id == self._plate_source_id
                        and uid == self._plate_gie_uid):
                    plate_dets.append({
                        "bbox": bbox,
                        "conf": float(obj_meta.confidence),
                    })
                elif (source_id == self._face_source_id
                        and uid == self._face_gie_uid):
                    uid2_post += 1
                    fd = self._extract_face_meta(
                        obj_meta, bbox, roi_embeddings, frame.shape)
                    fd["source"] = "tracker" if self._tracker else "detector"
                    face_data.append(fd)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            with self._lock:
                if source_id == self._plate_source_id:
                    self._plate_frame = frame
                    self._plate_detections = plate_dets
                    self._frame_seq += 1
                    self._frame_event.set()
                elif source_id == self._face_source_id:
                    self._face_frame = frame
                    self._face_data = detector_face_data + face_data
                    self._frame_seq += 1
                    self._frame_event.set()

            # Debug stats
            if source_id == self._face_source_id:
                self._dbg_n_face_frames += 1
                self._dbg_n_face_all_obj += total_obj
                self._dbg_n_face_uid2 += uid2_post
                self._dbg_n_face_pre += len(detector_face_data)
                self._dbg_n_face_dets += len(detector_face_data) + len(face_data)
                self._dbg_n_face_emb += sum(
                    1 for f in face_data if f.get("embedding") is not None)
                self._dbg_n_face_lmk += sum(
                    1 for f in face_data if f.get("landmarks") is not None)
                if now - self._dbg_last_log >= 5.0:
                    log.info(f"face_dbg: frames={self._dbg_n_face_frames} "
                             f"all_obj={self._dbg_n_face_all_obj} "
                             f"uid2={self._dbg_n_face_uid2} "
                             f"pre={self._dbg_n_face_pre} "
                             f"dets={self._dbg_n_face_dets} "
                             f"emb={self._dbg_n_face_emb} "
                             f"lmk={self._dbg_n_face_lmk}")
                    self._dbg_last_log = now
                    self._dbg_n_face_frames = 0
                    self._dbg_n_face_all_obj = 0
                    self._dbg_n_face_uid2 = 0
                    self._dbg_n_face_pre = 0
                    self._dbg_n_face_dets = 0
                    self._dbg_n_face_emb = 0
                    self._dbg_n_face_lmk = 0
                    self._dbg_n_roi_meta = 0
                    self._dbg_n_roi_emb = 0
                    self._dbg_n_roi_err = 0
                    self._dbg_n_obj_roi_meta = 0
                    self._dbg_n_obj_roi_emb = 0
                    self._dbg_n_cls_meta = 0
                    self._dbg_n_cls_labels = 0
                    self._dbg_cls_uids = {}
                    self._dbg_frame_user_types = {}
                    self._dbg_frame_tensor_uids = {}
                    self._dbg_obj_user_types = {}
                    self._dbg_obj_tensor_uids = {}
                    self._dbg_emb_read_err = 0
                    self._dbg_emb_null_ptr = 0

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    @staticmethod
    def _bbox_iou(a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)

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

    def _read_embedding_tensor_meta(self, tensor_meta):
        if tensor_meta.unique_id != 3 or tensor_meta.num_output_layers <= 0:
            return None
        try:
            host_buf = tensor_meta.out_buf_ptrs_host[0]
        except TypeError:
            # Some pyds builds expose a single-output host pointer as a
            # PyCapsule instead of an indexable array.
            host_buf = tensor_meta.out_buf_ptrs_host
        ptr = pyds.get_ptr(host_buf)
        if not ptr:
            self._dbg_emb_null_ptr += 1
            return None
        arr = np.ctypeslib.as_array(
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)),
            shape=(512,)
        ).copy()
        n = np.linalg.norm(arr)
        return arr / n if n > 1e-6 else arr

    def _read_embedding_classifier_meta(self, obj_meta):
        l_cls = obj_meta.classifier_meta_list
        while l_cls is not None:
            try:
                cls = pyds.NvDsClassifierMeta.cast(l_cls.data)
            except StopIteration:
                break

            uid = int(cls.unique_component_id)
            self._dbg_cls_uids[uid] = self._dbg_cls_uids.get(uid, 0) + 1
            if cls.unique_component_id == 3:
                self._dbg_n_cls_meta += 1
                emb = np.zeros((512,), dtype=np.float32)
                seen = 0
                l_label = cls.label_info_list
                while l_label is not None:
                    try:
                        label = pyds.NvDsLabelInfo.cast(l_label.data)
                    except StopIteration:
                        break
                    idx = int(label.label_id)
                    if 0 <= idx < 512:
                        emb[idx] = float(label.result_prob)
                        seen += 1
                        self._dbg_n_cls_labels += 1
                    try:
                        l_label = l_label.next
                    except StopIteration:
                        break
                if seen == 512:
                    n = np.linalg.norm(emb)
                    return emb / n if n > 1e-6 else emb

            try:
                l_cls = l_cls.next
            except StopIteration:
                break
        return None

    def _read_custom_face_embedding_meta(self, obj_meta):
        meta_type = _get_face_embed_meta_type()
        l_user = obj_meta.obj_user_meta_list
        while l_user is not None:
            try:
                u = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if u.base_meta.meta_type == meta_type:
                try:
                    ptr = self._pyds_ptr(u.user_meta_data)
                    meta = _FaceEmbeddingMeta.from_address(ptr)
                    if meta.version not in (1, 2) or meta.dims != 512:
                        return None, None, None
                    emb = np.ctypeslib.as_array(meta.embedding,
                                                shape=(512,)).copy()
                    n = np.linalg.norm(emb)
                    if n > 1e-6:
                        emb = emb / n
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
                        log.warning(f"custom emb read err: {e}")
                    return None, None, None

            try:
                l_user = l_user.next
            except StopIteration:
                break
        return None, None, None

    @staticmethod
    def _pyds_ptr(value) -> int:
        try:
            return pyds.get_ptr(value)
        except Exception:
            return int(value)

    def _extract_face_roi_embeddings(self, frame_meta) -> list:
        """
        Read SGIE uid=3 embeddings attached by nvinfer as NVDS_ROI_META when
        input-tensor-from-meta=1. pyds does not expose NvDsRoiMeta, so this
        uses the DeepStream C struct layout from nvds_roi_meta.h.
        """
        if not self._read_face_embeddings:
            return []

        out = []
        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                u = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            try:
                mt = int(u.base_meta.meta_type)
            except Exception:
                mt = u.base_meta.meta_type
            self._dbg_frame_user_types[mt] = (
                self._dbg_frame_user_types.get(mt, 0) + 1)
            if u.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                try:
                    t = pyds.NvDsInferTensorMeta.cast(u.user_meta_data)
                    uid = int(t.unique_id)
                    self._dbg_frame_tensor_uids[uid] = (
                        self._dbg_frame_tensor_uids.get(uid, 0) + 1)
                except Exception:
                    pass

            if mt == _NVDS_ROI_META:
                self._dbg_n_roi_meta += 1
                try:
                    roi_ptr = self._pyds_ptr(u.user_meta_data)
                    bbox, emb = self._read_embedding_from_roi_ptr(roi_ptr)
                    if emb is not None:
                        self._dbg_n_roi_emb += 1
                        out.append({"bbox": bbox, "embedding": emb})
                except Exception as e:
                    self._dbg_n_roi_err += 1
                    log.debug(f"roi embedding read err: {e}")

            try:
                l_user = l_user.next
            except StopIteration:
                break

        return out

    def _read_embedding_from_roi_ptr(self, roi_ptr: int):
        roi = _NvDsRoiMeta.from_address(roi_ptr)
        r = roi.roi
        bbox = (int(r.left), int(r.top),
                int(r.left + r.width), int(r.top + r.height))

        # Critical safety guard: Gst-nvinfer's ROI meta release function
        # deletes roi.object_meta. In object-mode nvdspreprocess this pointer
        # is non-owning, so clear it after SGIE attaches ROI meta.
        roi.object_meta = None

        if not self._read_face_embeddings:
            return bbox, None

        emb = None
        l_roi = roi.roi_user_meta_list
        while bool(l_roi):
            ru = pyds.NvDsUserMeta.cast(l_roi.contents.data)
            if ru.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                t = pyds.NvDsInferTensorMeta.cast(ru.user_meta_data)
                emb = self._read_embedding_tensor_meta(t)
                if emb is not None:
                    break
            l_roi = l_roi.contents.next
        return bbox, emb

    def _extract_face_meta(self, obj_meta, bbox: tuple,
                           roi_embeddings: list = None,
                           frame_shape=None) -> dict:
        """Đọc embedding (SGIE uid=3 tensor meta)."""
        emb, landmarks, face_bbox = self._read_custom_face_embedding_meta(
            obj_meta)
        if emb is None:
            emb = self._read_embedding_classifier_meta(obj_meta)

        l_user = obj_meta.obj_user_meta_list
        while l_user is not None:
            try:
                u = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            mt = u.base_meta.meta_type
            try:
                mt_dbg = int(mt)
            except Exception:
                mt_dbg = mt
            self._dbg_obj_user_types[mt_dbg] = (
                self._dbg_obj_user_types.get(mt_dbg, 0) + 1)
            if mt == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                try:
                    t = pyds.NvDsInferTensorMeta.cast(u.user_meta_data)
                    uid = int(t.unique_id)
                    self._dbg_obj_tensor_uids[uid] = (
                        self._dbg_obj_tensor_uids.get(uid, 0) + 1)
                    emb = self._read_embedding_tensor_meta(t)
                except Exception as e:
                    self._dbg_emb_read_err += 1
                    if self._dbg_emb_read_err <= 3:
                        log.warning(f"emb read err: {e}")
                    else:
                        log.debug(f"emb read err: {e}")
            else:
                try:
                    mt_i = int(mt)
                except Exception:
                    mt_i = mt
                if mt_i == _NVDS_ROI_META:
                    self._dbg_n_obj_roi_meta += 1
                    try:
                        roi_ptr = self._pyds_ptr(u.user_meta_data)
                        _, roi_emb = self._read_embedding_from_roi_ptr(
                            roi_ptr)
                        if roi_emb is not None:
                            emb = roi_emb
                            self._dbg_n_obj_roi_emb += 1
                    except Exception as e:
                        self._dbg_n_roi_err += 1
                        log.debug(f"obj roi embedding read err: {e}")

            try:
                l_user = l_user.next
            except StopIteration:
                break

        if emb is None and roi_embeddings:
            best = None
            best_iou = 0.0
            for item in roi_embeddings:
                v = self._bbox_iou(bbox, item["bbox"])
                if v > best_iou:
                    best_iou = v
                    best = item
            if best is not None and best_iou >= 0.3:
                emb = best["embedding"]

        # pyds không expose UNTRACKED_OBJECT_ID; C header = 0xFFFFFFFFFFFFFFFF
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
            
