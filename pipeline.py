"""
pipeline.py — Video Pipeline
DeepStream GPU pipeline (nếu có pyds) hoặc fallback GStreamer + OpenCV.

DeepStream mode:
  RTMP → nvstreammux → nvinfer (plate YOLO) → probe callback
  - Zero-copy decode→detect trên GPU
  - Plate detection qua nvinfer, kết quả qua NvDsObjectMeta
  - Face camera: decode chung pipeline, extract frame cho insightface

Fallback mode:
  RTMP → GStreamer NVDEC → OpenCV → Python inference
  - Giống bản trước, dùng StreamReader
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
# DeepStream Pipeline
# ──────────────────────────────────────────────
class DeepStreamPipeline:
    """
    DeepStream pipeline cho 2 camera RTMP.

    Pipeline layout:
      source0 (plate RTMP) ─┐
                             ├→ nvstreammux → nvinfer (plate YOLOv8) → probe
      source1 (face RTMP)  ─┘

    Probe callback tách kết quả theo source_id:
      - source 0: plate detections (bbox + conf)
      - source 1: raw frame cho insightface (detect+embed ở Python)

    Lý do không dùng nvinfer cho face:
      InsightFace FaceAnalysis.get() = detect + align + embed trong 1 call.
      Tách detect ra nvinfer → phải tự align + chạy ArcFace riêng → phức tạp hơn.
      Face camera chỉ cần frame, insightface đã đủ nhanh (~10ms).
    """

    def __init__(self, plate_src: str, face_src: str, cfg: dict):
        Gst.init(None)

        self.cfg = cfg
        self._stop = Event()
        self._lock = Lock()

        # Kết quả mới nhất từ probe
        self._plate_frame = None
        self._plate_detections = []  # [{bbox, conf}, ...]
        self._face_frame = None

        self._build_pipeline(plate_src, face_src)

    def _build_pipeline(self, plate_src: str, face_src: str):
        """Xây dựng DeepStream pipeline bằng Gst.parse_launch."""
        plate_engine = self.cfg["plate_detector"]["engine"]

        # Pipeline string
        pipe_str = (
            # Source 0: plate camera
            f'uridecodebin uri="{plate_src}" name=src0 ! '
            "queue ! nvvideoconvert ! "
            'video/x-raw(memory:NVMM),format=NV12 ! '
            "mux.sink_0 "

            # Source 1: face camera
            f'uridecodebin uri="{face_src}" name=src1 ! '
            "queue ! nvvideoconvert ! "
            'video/x-raw(memory:NVMM),format=NV12 ! '
            "mux.sink_1 "

            # Streammux: batch 2 sources
            "nvstreammux name=mux batch-size=2 "
            "width=1280 height=720 "
            "batched-push-timeout=40000 "
            "live-source=1 ! "

            # Plate detection (nvinfer)
            "nvinfer name=plate_det "
            f'config-file-path="{self.cfg["deepstream"]["plate_config"]}" ! '
            
            # Convert to rgba
            "nvvideoconvert ! "
            'video/x-raw(memory:NVMM),format=RGBA ! '

            # Dùng fakesink, lấy data qua probe
            "fakesink name=sink sync=0"
        )

        self._pipeline = Gst.parse_launch(pipe_str)

        # Attach probe trước fakesink
        sink = self._pipeline.get_by_name("sink")
        sink_pad = sink.get_static_pad("sink")
        sink_pad.add_probe(
            Gst.PadProbeType.BUFFER, self._probe_callback, None)

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

            # Extract frame as numpy
            # frame = pyds.get_nvds_buf_surface(hash(buf),
            #                                    frame_meta.batch_id)
            surface = pyds.get_nvds_buf_surface(hash(buf),
                                               frame_meta.batch_id)
            frame = np.array(surface, copy=True, order='C')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            # Extract detections (chỉ source 0 = plate cam có nvinfer)
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

            with self._lock:
                if source_id == 0:
                    self._plate_frame = frame.copy()
                    self._plate_detections = detections
                    log.info(f"[probe] plate frame {frame.shape}, "
                             f"{len(detections)} dets")
                elif source_id == 1:
                    self._face_frame = frame.copy()
                    log.info(f"[probe] face frame {frame.shape}")

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def start(self):
        """Start pipeline."""
        self._pipeline.set_state(Gst.State.PLAYING)
        # GLib main loop trên thread riêng
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

    def stop(self):
        self._stop.set()
        self._pipeline.set_state(Gst.State.NULL)
        if hasattr(self, "_loop"):
            self._loop.quit()
        log.info("DeepStream pipeline stopped")


# ──────────────────────────────────────────────
# GStreamer Fallback (StreamReader từ bản trước)
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
        self.cap = None
        self.is_stream = isinstance(source, str) and \
            source.startswith(("rtmp://", "rtsp://", "http://"))

        self._connect()
        self._thread = Thread(target=self._reader, daemon=True,
                              name=f"reader-{name}")
        self._thread.start()

    def _connect(self):
        if self.cap:
            self.cap.release()

        if self.hw_decode and isinstance(self.source, str):
            pipeline = self._build_gst(self.source)
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap or not self.cap.isOpened():
            if self.hw_decode:
                log.warning(f"[{self.name}] GStreamer failed, fallback")
            self.cap = cv2.VideoCapture(str(self.source))

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
    def connected(self):
        return self._connected

    def release(self):
        self._stop.set()
        if self.cap:
            self.cap.release()
