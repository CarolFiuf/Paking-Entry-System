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
from dataclasses import dataclass, field
from threading import Thread, Event, Lock
from queue import Queue, Empty

log = logging.getLogger("pipeline")


@dataclass
class _ProbeStats:
    """Counters cho probe debug. Reset sau mỗi lần log."""
    last_log: float = field(default_factory=time.time)
    face_frames: int = 0
    face_all_obj: int = 0
    face_uid2: int = 0
    face_dets: int = 0
    face_emb: int = 0
    emb_read_err: int = 0
    # Track ID stats: tổng số face có object_id != None, set track_id thấy
    # qua chu kỳ, đếm frame có ≥2 face (multi-person).
    face_tracked: int = 0
    face_untracked: int = 0
    multi_face_frames: int = 0
    track_ids: set = field(default_factory=set)
    track_hits: dict = field(default_factory=dict)   # track_id → frame count

    def reset(self):
        self.last_log = time.time()
        self.face_frames = 0
        self.face_all_obj = 0
        self.face_uid2 = 0
        self.face_dets = 0
        self.face_emb = 0
        self.emb_read_err = 0
        self.face_tracked = 0
        self.face_untracked = 0
        self.multi_face_frames = 0
        self.track_ids = set()
        self.track_hits = {}

    def log_summary(self):
        # Top 5 track theo số frame xuất hiện — phản ánh stability.
        top = sorted(self.track_hits.items(),
                     key=lambda kv: kv[1], reverse=True)[:5]
        top_str = ", ".join(f"#{t}:{n}" for t, n in top) or "—"
        log.info(f"face_probe: frames={self.face_frames} "
                 f"all_obj={self.face_all_obj} uid2={self.face_uid2} "
                 f"dets={self.face_dets} emb={self.face_emb} "
                 f"err={self.emb_read_err}")
        log.info(f"face_track: tracked={self.face_tracked} "
                 f"untracked={self.face_untracked} "
                 f"unique_ids={len(self.track_ids)} "
                 f"multi_face_frames={self.multi_face_frames} "
                 f"top={top_str}")

_FACE_EMBED_META_DESC = "PARKING.FACE_EMBEDDING_META"
_FACE_EMBED_META_TYPE = None

# nvtracker dùng UINT64_MAX cho object chưa được gán track_id.
_UNTRACKED_OBJECT_ID = (1 << 64) - 1


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

        # nvstreammux output size — bbox plugin sinh ra theo toạ độ này (surface
        # size sau mux), KHÔNG phải source camera size. Giữ ở 1 chỗ để khớp
        # _configure_mux + scale bbox về display frame.
        self._mux_w = 1280
        self._mux_h = 720

        # Display frames (downscale 640×360 BGR) — appsink push, lock riêng để
        # không cản probe meta. Web thread đọc, recognition KHÔNG đụng.
        self._display_w = 640
        self._display_h = 360
        self._display_fps = 10
        # Throttle theo stream-time (buf.pts, ns) thay vì wall-clock — file
        # replay/benchmark cũng cap đúng display_fps trên giây-video. Fallback
        # wall-clock khi buffer không có PTS.
        self._display_period_ns = int(Gst.SECOND / self._display_fps)
        self._disp_last_pass = {}
        self._face_display_frame = None
        self._plate_display_frame = None
        self._display_lock = Lock()

        # Meta state — probe push, main loop pull.
        # _plate_full_frame chỉ set khi probe thấy plate detection (cần cho OCR
        # crop). Khi không có plate, giữ None để tránh memcpy thừa.
        self._plate_detections = []
        self._plate_full_frame = None
        self._plate_source_size = (0, 0)
        self._face_data = []         # list[dict{bbox, conf, embedding, quality}]
        self._face_source_size = (0, 0)

        self._frame_seq = 0
        self._meta_event = Event()

        self._probe_count = 0
        self._probe_fps = 0.0
        self._probe_t0 = time.time()

        # Stats cho debug (Probe B). Bật/tắt qua deepstream.debug_probe.
        self._debug_probe = bool(
            cfg.get("deepstream", {}).get("debug_probe", False))
        self._stats = _ProbeStats()

        self._plate_pipeline = None
        self._face_pipeline = None
        self._pipelines = []
        self._pgie_face = None
        self._face_embedder = None
        self._face_tracker = None

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

    def _configure_mux(self, mux, batch_size: int = 1):
        mux.set_property("batch-size", batch_size)
        mux.set_property("width", self._mux_w)
        mux.set_property("height", self._mux_h)
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
        self._apply_nvinfer_interval(pgie_plate, ds_cfg.get(
            "plate_det_interval"), "plate")

        nvconv = self._make_element("nvvideoconvert", "plate_nvconv_out",
                                    pipeline)
        capsfilter = self._make_element("capsfilter", "plate_caps_rgba",
                                        pipeline)
        capsfilter.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))

        tee = self._make_element("tee", "plate_tee", pipeline)
        self._link_many([mux, pgie_plate, nvconv, capsfilter, tee])

        # Nhánh META: chỉ probe meta, plate full-res materialize on-demand
        # khi có plate detection (cho OCR crop).
        meta_queue = self._make_queue("plate_meta_queue", pipeline,
                                      max_buffers=2)
        meta_sink = self._make_element("fakesink", "plate_meta_sink", pipeline)
        meta_sink.set_property("sync", 0)
        meta_sink.set_property("async", 0)
        self._link_tee_branch(tee, [meta_queue, meta_sink])
        meta_sink.get_static_pad("sink").add_probe(
            Gst.PadProbeType.BUFFER, self._probe_callback, "plate")

        # Nhánh DISPLAY: downscale GPU → cap 10 fps → appsink BGR cho web.
        appsink = self._build_display_branch(
            pipeline, tee, name_prefix="plate")
        appsink.connect("new-sample", self._on_plate_appsink_sample)

        self._attach_bus(pipeline)
        log.info("Plate pipeline: src → mux → plate_det → tee "
                 "[meta_probe | display(640×360@10fps)]")

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
        self._apply_nvinfer_interval(pgie_face, ds_cfg.get(
            "face_det_interval"), "face")
        self._pgie_face = pgie_face

        nvconv = self._make_element("nvvideoconvert", "face_nvconv_out",
                                    pipeline)
        capsfilter = self._make_element("capsfilter", "face_caps_rgba",
                                        pipeline)
        capsfilter.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))

        embedder = self._make_face_embedder(pipeline, ds_cfg)
        self._face_embedder = embedder

        # nvtracker SAU embedder: parser SCRFD là stub, plugin tự tạo
        # obj_meta + embedding user_meta → tracker chỉ cần gán object_id
        # in-place lên obj_meta (user_meta được preserve). Trade-off: plugin
        # interval skip không có hiệu lực ở vị trí này (object_id chưa biết
        # lúc plugin chạy); muốn enable cần tách plugin thành facedet+embed.
        tracker = self._make_face_tracker(pipeline, ds_cfg)
        self._face_tracker = tracker

        tee = self._make_element("tee", "face_tee", pipeline)
        self._link_many([mux, pgie_face, nvconv, capsfilter,
                         embedder, tracker, tee])

        # Nhánh META: probe đọc obj_meta + embedding, KHÔNG materialize BGR.
        meta_queue = self._make_queue("face_meta_queue", pipeline,
                                      max_buffers=2)
        meta_sink = self._make_element("fakesink", "face_meta_sink", pipeline)
        meta_sink.set_property("sync", 0)
        meta_sink.set_property("async", 0)
        self._link_tee_branch(tee, [meta_queue, meta_sink])
        meta_sink.get_static_pad("sink").add_probe(
            Gst.PadProbeType.BUFFER, self._probe_callback, "face")

        # Nhánh DISPLAY: downscale GPU → cap 10 fps → appsink BGR cho web.
        appsink = self._build_display_branch(
            pipeline, tee, name_prefix="face")
        appsink.connect("new-sample", self._on_face_appsink_sample)

        self._attach_bus(pipeline)
        log.info("Face pipeline: src → mux → face_det → nvtracker → "
                 "nvdsfaceembed → tee [meta_probe | display(640×360@10fps)]")

    def _make_face_tracker(self, pipeline, ds_cfg):
        """nvtracker đặt SAU nvdsfaceembed — gán object_id lên obj_meta đã
        có sẵn (SCRFD parser stub nên obj_meta do plugin tạo). Tracker chỉ
        đọc bbox + pixel source để track, không đụng đến embedding user_meta.

        Mặc định NvDCF (visual feature) thay vì IOU vì face cam parking thường
        có driver+passenger sát nhau và face hay bị che tạm (cúi xuống mở
        khoá xe). NvDCF chống ID-switch tốt hơn ~5-10× với cost 1-2 ms/frame.
        """
        tracker = self._make_element("nvtracker", "face_tracker", pipeline)
        tracker.set_property(
            "ll-lib-file",
            "/opt/nvidia/deepstream/deepstream/lib/"
            "libnvds_nvmultiobjecttracker.so")
        tracker.set_property(
            "ll-config-file",
            os.path.abspath(ds_cfg.get(
                "face_tracker_config",
                "./configs/tracker_face_nvdcf.yml")))
        # NvDCF cần đủ pixel cho HOG; 960x544 ổn cho Orin Nano + 1-3 face.
        tracker.set_property(
            "tracker-width", int(ds_cfg.get("face_tracker_width", 960)))
        tracker.set_property(
            "tracker-height", int(ds_cfg.get("face_tracker_height", 544)))
        tracker.set_property("compute-hw", 1)            # GPU
        tracker.set_property("gpu-id", 0)
        tracker.set_property("display-tracking-id", 0)
        return tracker

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
        # Source of truth duy nhất: config.yaml.
        # Nếu thiếu, dùng plugin C default (0.30f).
        decode_thr = ds_cfg.get("face_embed_decode_conf_threshold")
        if decode_thr is not None:
            embedder.set_property("decode-conf-threshold", float(decode_thr))
        embedder.set_property("net-width", self._net_w)
        embedder.set_property("net-height", self._net_w)
        embedder.set_property("input-object-min-width", 32)
        embedder.set_property("input-object-min-height", 32)
        embedder.set_property("min-quality",
            float(ds_cfg.get("face_embed_min_quality", 0.0)))
        embedder.set_property("blur-threshold",
            float(self.cfg.get("face", {}).get("blur_threshold", 10.0)))
        # interval=N → mỗi track chỉ embed lại sau N frame kể từ lần embed
        # gần nhất. Yêu cầu nvtracker upstream gán object_id ổn định.
        embed_interval = ds_cfg.get("face_embed_interval")
        if embed_interval is not None:
            try:
                iv = int(embed_interval)
                if iv >= 0:
                    embedder.set_property("interval", iv)
                    log.info(f"nvdsfaceembed interval={iv} "
                             f"(embed 1/{iv+1} frame mỗi track)")
            except (TypeError, ValueError):
                log.warning(f"face_embed_interval không phải int: "
                            f"{embed_interval!r}")
        return embedder

    @staticmethod
    def _apply_nvinfer_interval(nvinfer_elem, interval_cfg, label: str):
        """Override nvinfer `interval` property nếu config khai báo.

        interval=N → chạy 1 frame, skip N frame. None hoặc thiếu → giữ giá trị
        trong .txt config (mặc định 0 = không skip). Tăng N để giảm GPU; đánh
        đổi: tracker lâu commit hơn N+1 lần và rủi ro miss mặt nếu subject
        chỉ xuất hiện trong frame skip.
        """
        if interval_cfg is None:
            return
        try:
            interval = int(interval_cfg)
        except (TypeError, ValueError):
            log.warning(f"{label}_det_interval không phải int: {interval_cfg!r}")
            return
        if interval < 0:
            return
        nvinfer_elem.set_property("interval", interval)
        log.info(f"{label}_det nvinfer interval={interval} "
                 f"(infer 1/{interval+1} frame)")

    def _make_queue(self, name: str, pipeline, max_buffers: int = 2,
                    leaky: int = 2):
        q = self._make_element("queue", name, pipeline)
        q.set_property("max-size-buffers", max_buffers)
        q.set_property("max-size-bytes", 0)
        q.set_property("max-size-time", 0)
        q.set_property("leaky", leaky)
        return q

    def _link_tee_branch(self, tee, elements: list) -> bool:
        """Request 1 src pad từ tee và link vào element đầu của chuỗi."""
        head = elements[0]
        tee_src = tee.get_request_pad("src_%u")
        head_sink = head.get_static_pad("sink")
        if not tee_src or not head_sink:
            log.error(f"Failed to get pads for tee → {head.get_name()}")
            return False
        if tee_src.link(head_sink) != Gst.PadLinkReturn.OK:
            log.error(f"Failed to link tee → {head.get_name()}")
            return False
        return self._link_many(elements)

    def _build_display_branch(self, pipeline, tee, name_prefix: str):
        """
        tee → queue → nvvideoconvert (NVMM RGBA → system mem BGRx 640×360)
              → caps(BGRx) → videoconvert → caps(BGR)
              → appsink(drop=1, max-buffers=1, emit-signals=1)

        Throttle: appsink drop=1 + Python callback @ 10 Hz tự bỏ frame thừa.
        """
        q = self._make_queue(f"{name_prefix}_disp_queue", pipeline,
                             max_buffers=2)
        nvconv = self._make_element("nvvideoconvert",
                                    f"{name_prefix}_disp_nvconv", pipeline)
        caps_bgrx = self._make_element("capsfilter",
                                       f"{name_prefix}_disp_caps_bgrx",
                                       pipeline)
        caps_bgrx.set_property(
            "caps", Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,"
                f"width={self._display_w},height={self._display_h}"))
        vconv = self._make_element("videoconvert",
                                   f"{name_prefix}_disp_videoconvert",
                                   pipeline)
        caps_bgr = self._make_element("capsfilter",
                                      f"{name_prefix}_disp_caps_bgr",
                                      pipeline)
        caps_bgr.set_property(
            "caps", Gst.Caps.from_string("video/x-raw,format=BGR"))
        appsink = self._make_element("appsink",
                                     f"{name_prefix}_disp_appsink", pipeline)
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", False)
        appsink.set_property("drop", True)
        appsink.set_property("max-buffers", 1)
        appsink.set_property("async", False)

        self._link_tee_branch(tee, [q, nvconv, caps_bgrx, vconv,
                                    caps_bgr, appsink])

        # Throttle ngay sau queue (trước nvvideoconvert) → nvvideoconvert
        # chỉ thấy ~display_fps buffer/giây, GPU downscale 1/3 work.
        q.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._display_throttle_probe,
            name_prefix)
        return appsink

    def _display_throttle_probe(self, pad, info, name_prefix):
        """Drop buffer trước nvvideoconvert nếu chưa đủ 1/display_fps stream-
        time kể từ buffer được pass cuối — giữ display branch ở ~display_fps,
        tiết kiệm GPU downscale work.

        Dùng buf.pts (nanosecond) thay vì wall-clock: file replay / benchmark
        cũng cap theo video time, không phụ thuộc decode speed. Fallback
        wall-clock khi PTS chưa hợp lệ (vd source không stamp time)."""
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        pts = buf.pts
        if pts == Gst.CLOCK_TIME_NONE:
            now_ns = int(time.monotonic() * Gst.SECOND)
        else:
            now_ns = int(pts)
        last = self._disp_last_pass.get(name_prefix, 0)
        if now_ns - last < self._display_period_ns:
            return Gst.PadProbeReturn.DROP
        self._disp_last_pass[name_prefix] = now_ns
        return Gst.PadProbeReturn.OK

    @staticmethod
    def _pull_bgr_from_appsink(appsink):
        """Helper: pull-sample → np.ndarray BGR copy (an toàn sau unmap)."""
        sample = appsink.emit("pull-sample")
        if sample is None:
            return None
        buf = sample.get_buffer()
        if buf is None:
            return None
        caps = sample.get_caps()
        s = caps.get_structure(0)
        ok, w = s.get_int("width")
        ok2, h = s.get_int("height")
        if not ok or not ok2:
            return None
        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return None
        try:
            arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
            if arr.size < h * w * 3:
                return None
            frame = arr[: h * w * 3].reshape(h, w, 3).copy()
        finally:
            buf.unmap(mapinfo)
        return frame

    def _on_face_appsink_sample(self, appsink):
        frame = self._pull_bgr_from_appsink(appsink)
        if frame is None:
            return Gst.FlowReturn.OK
        with self._display_lock:
            self._face_display_frame = frame
        return Gst.FlowReturn.OK

    def _on_plate_appsink_sample(self, appsink):
        frame = self._pull_bgr_from_appsink(appsink)
        if frame is None:
            return Gst.FlowReturn.OK
        with self._display_lock:
            self._plate_display_frame = frame
        return Gst.FlowReturn.OK

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

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            if stream_kind == "plate":
                self._handle_plate_meta(buf, frame_meta)
            elif stream_kind == "face":
                self._handle_face_meta(frame_meta, now)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _materialize_bgr(self, buf, batch_id):
        """RGBA NVMM → BGR np.array. CPU copy + cvtColor. Chỉ gọi cho plate
        khi có detection (cần frame full-res để OCR crop)."""
        surface = pyds.get_nvds_buf_surface(hash(buf), batch_id)
        return cv2.cvtColor(np.array(surface, copy=True, order='C'),
                            cv2.COLOR_RGBA2BGR)

    def _frame_size_from_meta(self, frame_meta):
        """
        Bbox trong obj_meta theo toạ độ mux output (surface size sau
        nvstreammux), KHÔNG phải source camera. Plugin nvdsfaceembed lấy
        frame_w/h từ surface — giá trị này luôn là mux size. Dùng
        source_frame_width/height của frame_meta sẽ sai khi camera native
        khác mux config (vd camera 1080×608 nhưng mux 1280×720).
        """
        return (self._mux_w, self._mux_h)

    def _collect_plate_dets(self, frame_meta):
        dets = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if obj_meta.unique_component_id == self._plate_gie_uid:
                rect = obj_meta.rect_params
                bbox = (int(rect.left), int(rect.top),
                        int(rect.left + rect.width),
                        int(rect.top + rect.height))
                dets.append({"bbox": bbox,
                             "conf": float(obj_meta.confidence)})
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        return dets

    def _collect_face_data(self, frame_meta, frame_size):
        face_data = []
        total_obj = 0
        uid_match = 0
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            total_obj += 1
            if obj_meta.unique_component_id == self._face_gie_uid:
                uid_match += 1
                rect = obj_meta.rect_params
                bbox = (int(rect.left), int(rect.top),
                        int(rect.left + rect.width),
                        int(rect.top + rect.height))
                face_data.append(
                    self._extract_face_meta(obj_meta, bbox, frame_size))
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        return face_data, total_obj, uid_match

    def _handle_plate_meta(self, buf, frame_meta):
        plate_dets = self._collect_plate_dets(frame_meta)
        # Chỉ materialize BGR khi có ít nhất 1 plate — OCR cần frame full-res.
        # Còn lại để None để tránh ~5-10 ms cvtColor mỗi frame idle.
        full_frame = None
        if plate_dets:
            full_frame = self._materialize_bgr(buf, frame_meta.batch_id)
        size = self._frame_size_from_meta(frame_meta)
        with self._lock:
            self._plate_detections = plate_dets
            self._plate_full_frame = full_frame
            self._plate_source_size = size
            self._frame_seq += 1
            self._meta_event.set()

    def _handle_face_meta(self, frame_meta, now):
        size = self._frame_size_from_meta(frame_meta)
        face_data, total_obj, uid_match = self._collect_face_data(
            frame_meta, size)
        with self._lock:
            self._face_data = face_data
            self._face_source_size = size
            self._frame_seq += 1
            self._meta_event.set()

        if self._debug_probe:
            s = self._stats
            s.face_frames += 1
            s.face_all_obj += total_obj
            s.face_uid2 += uid_match
            s.face_dets += len(face_data)
            s.face_emb += sum(
                1 for f in face_data if f.get("embedding") is not None)
            tids_this_frame = [f.get("object_id") for f in face_data]
            tracked_now = [t for t in tids_this_frame if t is not None]
            s.face_tracked += len(tracked_now)
            s.face_untracked += len(tids_this_frame) - len(tracked_now)
            for t in tracked_now:
                s.track_ids.add(t)
                s.track_hits[t] = s.track_hits.get(t, 0) + 1
            if len(face_data) >= 2:
                s.multi_face_frames += 1
            if now - s.last_log >= 5.0:
                s.log_summary()
                s.reset()

    @staticmethod
    def _clip_bbox(bbox: tuple, frame_size) -> tuple:
        """frame_size: (width, height) tuple."""
        if bbox is None or frame_size is None:
            return bbox
        w, h = frame_size
        if w <= 0 or h <= 0:
            return bbox
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
        vào obj_meta. Plugin đã L2-normalize embedding, decode bbox từ
        landmarks, và (v2+) tính quality score trên aligned tensor.

        Returns: (embedding | None, face_bbox | None, quality | None)
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
                    face_bbox = None
                    quality = None
                    if meta.version >= 2:
                        b = np.ctypeslib.as_array(meta.bbox,
                                                  shape=(4,)).copy()
                        if np.all(np.isfinite(b)) and b[2] > b[0] and b[3] > b[1]:
                            face_bbox = tuple(int(round(v)) for v in b)
                        q = float(meta.quality)
                        # Plugin sets 0.0 nếu chưa compute (back-compat).
                        if np.isfinite(q) and q > 0.0:
                            quality = q
                    return emb, face_bbox, quality
                except Exception as e:
                    self._stats.emb_read_err += 1
                    if self._stats.emb_read_err <= 3:
                        log.warning(f"face embed meta read err: {e}")
                    return None, None, None

            try:
                l_user = l_user.next
            except StopIteration:
                break
        return None, None, None

    def _extract_face_meta(self, obj_meta, bbox: tuple,
                           frame_size=None) -> dict:
        """
        Trích face data từ obj_meta. Plugin attach embedding + (v2) bbox
        decode từ landmarks + quality. PGIE rect_params làm fallback bbox.
        frame_size: (width, height) — dùng để clip bbox về biên frame.
        """
        emb, face_bbox, quality = self._read_face_embed_meta(obj_meta)

        final_bbox = bbox
        clipped_face_bbox = self._clip_bbox(face_bbox, frame_size)
        if clipped_face_bbox is not None:
            final_bbox = clipped_face_bbox

        # nvtracker gán object_id; UNTRACKED khi chưa kịp commit track.
        tid = int(obj_meta.object_id)
        if tid == _UNTRACKED_OBJECT_ID:
            tid = None

        return {
            "bbox": final_bbox,
            "conf": float(obj_meta.confidence),
            "embedding": emb,
            "quality": quality,
            "object_id": tid,
        }
    
    @property
    def stream_fps(self):
        return round(self._probe_fps, 1)
    
    def wait_new_frame(self, timeout=0.5) -> bool:
        """Block cho tới khi probe đẩy meta mới."""
        self._meta_event.clear()
        return self._meta_event.wait(timeout=timeout)

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

    def get_all(self):
        """
        Snapshot cho main loop + web thread.

        Trả dict:
          plate_full: BGR full-res, chỉ set khi có plate detection (else None)
          plate_dets: list detection của plate
          plate_source_size: (w, h)
          plate_display: BGR 640×360 (do appsink push), có thể None khi pipeline
                         vừa khởi động hoặc plate disabled
          face_data: list face dict {bbox, conf, embedding, quality}
          face_source_size: (w, h)
          face_display: BGR 640×360 (do appsink push), có thể None
        """
        with self._lock:
            plate_full = self._plate_full_frame
            plate_dets = list(self._plate_detections)
            plate_src = self._plate_source_size
            face_data = list(self._face_data)
            face_src = self._face_source_size
        with self._display_lock:
            plate_disp = self._plate_display_frame
            face_disp = self._face_display_frame
        return {
            "plate_full": plate_full,
            "plate_dets": plate_dets,
            "plate_source_size": plate_src,
            "plate_display": plate_disp,
            "face_data": face_data,
            "face_source_size": face_src,
            "face_display": face_disp,
        }

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
            
