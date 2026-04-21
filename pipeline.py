"""
pipeline.py — Video Pipeline
DeepStream GPU pipeline (nếu có pyds) hoặc fallback GStreamer + OpenCV.

DeepStream mode:
  RTMP → rtmpsrc → flvdemux → h264parse → nvv4l2decoder → nvstreammux
       → nvinfer (plate YOLO) → probe callback
  - Zero-copy decode→detect trên GPU
  - Plate detection qua nvinfer, kết quả qua NvDsObjectMeta
  - Face camera: decode chung pipeline, extract frame cho insightface

Fallback mode:
  RTMP → GStreamer NVDEC → OpenCV → Python inference

FIX vs v3:
  - Thay uridecodebin (dynamic pad, hay lỗi RTMP) bằng explicit elements
  - Thêm bus watch để log lỗi GStreamer
  - Thêm retry/reconnect khi source mất
"""

import cv2
import numpy as np
import logging
import time
from threading import Thread, Event, Lock
from queue import Queue, Empty

log = logging.getLogger("pipeline")

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

        # Kết quả mới nhất từ probe
        self._plate_frame = None
        self._plate_detections = []
        self._face_frame = None
        
        self._frame_seq = 0          # ← thêm counter
        self._frame_event = Event()  # ← thêm event

        self._probe_count = 0
        self._probe_fps = 0.0
        self._probe_t0 = time.time()

        self._build_pipeline(plate_src, face_src)

    def _build_pipeline(self, plate_src: str, face_src: str):
        """Xây dựng DeepStream pipeline bằng element API (không parse_launch)."""

        self._pipeline = Gst.Pipeline.new("parking-pipeline")

        # ── Streammux ──
        mux = self._make_element("nvstreammux", "mux")
        mux.set_property("batch-size", 2)
        mux.set_property("width", 1280)
        mux.set_property("height", 720)
        mux.set_property("batched-push-timeout", 40000)
        mux.set_property("live-source", 1)

        # ── Source 0: plate camera ──
        self._add_rtmp_source(plate_src, source_id=0, mux=mux,
                              name_prefix="plate")

        # ── Source 1: face camera ──
        self._add_rtmp_source(face_src, source_id=1, mux=mux,
                              name_prefix="face")

        # ── nvinfer (plate detection) ──
        nvinfer = self._make_element("nvinfer", "plate_det")
        nvinfer.set_property("config-file-path",
                             self.cfg["deepstream"]["plate_config"])

        # ── nvvideoconvert → RGBA (cho pyds.get_nvds_buf_surface) ──
        nvconv = self._make_element("nvvideoconvert", "nvconv_out")

        # Caps filter: RGBA format
        capsfilter = self._make_element("capsfilter", "caps_rgba")
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
        capsfilter.set_property("caps", caps)

        # ── Fakesink ──
        sink = self._make_element("fakesink", "sink")
        sink.set_property("sync", 0)
        sink.set_property("async", 0)

        # ── Link: mux → nvinfer → nvconv → caps → sink ──
        if not mux.link(nvinfer):
            log.error("Failed to link mux → nvinfer")
        if not nvinfer.link(nvconv):
            log.error("Failed to link nvinfer → nvconv")
        if not nvconv.link(capsfilter):
            log.error("Failed to link nvconv → capsfilter")
        if not capsfilter.link(sink):
            log.error("Failed to link capsfilter → sink")

        # ── Probe trước sink ──
        sink_pad = sink.get_static_pad("sink")
        sink_pad.add_probe(
            Gst.PadProbeType.BUFFER, self._probe_callback, None)

        # ── Bus watch cho error logging ──
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        log.info("DeepStream pipeline built (explicit elements)")

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
                         mux, name_prefix: str):
        """
        Thêm 1 RTMP source vào pipeline.

        Chain: rtmpsrc → flvdemux → (dynamic pad) → h264parse
               → nvv4l2decoder → queue → mux.sink_N

        flvdemux có dynamic pad nên cần connect signal "pad-added".
        """
        # rtmpsrc
        src = self._make_element("rtmpsrc", f"{name_prefix}_src")
        src.set_property("location", rtmp_url)
        # Timeout để không treo mãi khi RTMP chưa sẵn sàng
        src.set_property("timeout", 10)

        # flvdemux — dynamic pad
        demux = self._make_element("flvdemux", f"{name_prefix}_demux")

        # h264parse
        parse = self._make_element("h264parse", f"{name_prefix}_parse")

        # nvv4l2decoder — hardware decode trên Jetson
        decoder = self._make_element("nvv4l2decoder",
                                     f"{name_prefix}_decoder")

        # queue — buffer giữa decoder và mux
        queue = self._make_element("queue", f"{name_prefix}_queue")
        queue.set_property("max-size-buffers", 5)
        queue.set_property("leaky", 2)  # downstream = drop old

        # Link static elements: rtmpsrc → flvdemux
        if not src.link(demux):
            log.error(f"Failed to link {name_prefix}_src → demux")

        # Link static: h264parse → decoder → queue
        if not parse.link(decoder):
            log.error(f"Failed to link {name_prefix}_parse → decoder")
        if not decoder.link(queue):
            log.error(f"Failed to link {name_prefix}_decoder → queue")

        # Link queue → mux.sink_N (request pad)
        mux_sink = mux.get_request_pad(f"sink_{source_id}")
        queue_src = queue.get_static_pad("src")
        if mux_sink and queue_src:
            queue_src.link(mux_sink)
        else:
            log.error(f"Failed to get pads for mux.sink_{source_id}")

        # flvdemux dynamic pad → h264parse
        # Khi RTMP stream bắt đầu, flvdemux tạo pad "video"
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

    def _probe_callback(self, pad, info, user_data):
        """
        Probe callback — chạy trên GStreamer thread.
        Đọc NvDsBatchMeta → tách frame + detection theo source_id.
        """
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            source_id = frame_meta.source_id

            # Extract frame as numpy (RGBA). cvtColor làm sau, ngoài lock.
            surface = pyds.get_nvds_buf_surface(hash(buf),
                                                frame_meta.batch_id)
            frame_rgba = np.array(surface, copy=True, order='C')

            # Extract detections (source 0 = plate cam có nvinfer)
            detections = []
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    rect = obj_meta.rect_params
                    detections.append({
                        "bbox": (int(rect.left), int(rect.top),
                                 int(rect.left + rect.width),
                                 int(rect.top + rect.height)),
                        "conf": obj_meta.confidence
                    })
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # cvtColor ngoài lock — giảm thời gian block GStreamer thread
            frame = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)

            with self._lock:
                if source_id == 0:
                    self._plate_frame = frame
                    self._plate_detections = detections
                    self._frame_seq += 1
                    self._frame_event.set()
                elif source_id == 1:
                    self._face_frame = frame

            try:
                l_frame = l_frame.next
            except StopIteration:
                break
            
        self._probe_count += 1
        now = time.time()
        if now - self._probe_t0 >= 1.0:
            self._probe_fps = self._probe_count / (now - self._probe_t0)
            self._probe_count = 0
            self._probe_t0 = now

        return Gst.PadProbeReturn.OK
    
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

    def get_all(self):
        """Lấy plate + face frame atomic — tránh mismatch giữa 2 lock."""
        with self._lock:
            return self._plate_frame, self._plate_detections, self._face_frame

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
            