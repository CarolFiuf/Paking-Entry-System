import cv2
import numpy as np
import yaml
import time
import re
import signal
import logging
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

from engine import PlateDetector, PlateOCRYolo, PlateOCR, FaceEngine
from database import ParkingDB
from pipeline import HAS_DEEPSTREAM, DeepStreamPipeline, StreamReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
class PlateValidator:
    """
    Validate + normalize biển số VN.

    Formats biển xe máy VN:
      Mới: 99B1-257.39  → {2số}{chữ}{1số}-{3số}.{2số}   (có dot)
      Cũ:  29B-12345    → {2số}{chữ}-{5số}               (không dot, không series)
      Mới: 51G1-23456   → {2số}{chữ}{1số}-{5số}          (không dot, có series)
      Mới: 30AB-12345   → {2số}{2chữ}-{5số}              (không dot, 2 chữ)
      NG:  80-123-NG-001 → {2số}-{3số}-{NN|NG|QT|CV}-{3số}     (biển người nước ngoài)
    """
    def __init__(self, regex_str: str = None):
        self._fix = str.maketrans("OI", "01")

    def __call__(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip().upper().translate(self._fix)
        # Giữ dot, strip phần còn lại
        clean = re.sub(r"[^A-Z0-9.]", "", t)

        if len(clean) < 7 or len(clean) > 13:
            return ""

        # ── Có dot → thử nhiều format (biển nước ngoài luôn không dot → skip) ──
        if "." in clean:
            # Format mới có series: XXYN-NNN.NN  (vd: 99B1-257.39)
            m = re.match(
                r"^(\d{2})([A-Z])(\d{1})(\d{3})\.(\d{2})$",
                clean)
            if m:
                return (f"{m.group(1)}{m.group(2)}{m.group(3)}"
                        f"-{m.group(4)}.{m.group(5)}")

            # Format cũ: XXY-NNN.NN  (1 chữ, không series)
            m = re.match(
                r"^(\d{2})([A-Z])(\d{3})\.(\d{2})$",
                clean)
            if m:
                return (f"{m.group(1)}{m.group(2)}"
                        f"-{m.group(3)}.{m.group(4)}")

            # Format 2 chữ: XXYY-NNN.NN
            m = re.match(
                r"^(\d{2})([A-Z]{2})(\d{3})\.(\d{2})$",
                clean)
            if m:
                return (f"{m.group(1)}{m.group(2)}"
                        f"-{m.group(3)}.{m.group(4)}")

            return ""

        # ── Không dot → thử nhiều format ──
        nodot = clean

        # Format nước ngoài: XX-NNN-NN/NG-NNN
        m = re.match(r"^(\d{2})(\d{3})(NN|NG|QT|CV)(\d{3})$", nodot)
        if m:
            n4 = int(m.group(4))
            if 1 <= n4 <= 999:
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}-{m.group(4)}"
            return ""

        # Format cũ: XXY-NNNNN (không series, 5 số)
        m = re.match(r"^(\d{2})([A-Z])(\d{4,5})$", nodot)
        if m:
            return f"{m.group(1)}{m.group(2)}-{m.group(3)}"

        # Format mới: XXYN-NNNNN (1 series + 4-5 số)
        m = re.match(r"^(\d{2})([A-Z])(\d)(\d{4,5})$", nodot)
        if m:
            return f"{m.group(1)}{m.group(2)}{m.group(3)}-{m.group(4)}"

        # Format mới: XXYY-NNNNN (2 chữ cái, 5 số)
        m = re.match(r"^(\d{2})([A-Z]{2})(\d{4,5})$", nodot)
        if m:
            return f"{m.group(1)}{m.group(2)}-{m.group(3)}"

        return ""


class PlateVoter:
    def __init__(self, n=5, min_votes=3):
        self.n, self.min_votes = n, min_votes
        self._buf = []

    def vote(self, text):
        if not text:
            return ""
        self._buf.append(text)
        if len(self._buf) > self.n:
            self._buf.pop(0)
        if len(self._buf) < self.min_votes:
            return ""
        best, count = Counter(self._buf).most_common(1)[0]
        return best if count >= len(self._buf) * 0.5 else ""

    def clear(self):
        self._buf.clear()


class IdentityTracker:
    """
    Session-level face embedding aggregator. Identity continuity qua frame
    do nvtracker (NvDCF) cung cấp qua `object_id`; class này chỉ gom embedding
    theo track_id trong suốt 1 session (từ khi face xuất hiện đến khi plate
    vote stable).

    Vai trò:
      - Best-of-N: giữ embedding quality cao nhất mỗi track.
      - min_hits gate: bỏ qua track ephemeral (1-2 frame rồi mất).
      - max_slots cap: tối đa N identity/xe (xe máy chở 2 + dự phòng).
      - Snapshot atomic khi commit: drain slot đã committed → list embedding.

    update_batch(faces):
      - Mỗi face có `object_id` từ nvtracker; lookup slot theo ID.
      - Nếu có embedding mới và quality cao hơn → replace best (best-of-N).
      - Track không xuất hiện > stale_frames → dọn slot.
    """

    def __init__(self, max_slots=3, min_seed_quality=0.3, min_hits=2,
                 stale_frames=60):
        self.max_slots = max_slots
        self.min_seed_quality = min_seed_quality
        self.min_hits = min_hits
        self.stale_frames = stale_frames
        self._slots = {}        # track_id → dict{embedding, quality, ...}
        self._frame_idx = 0
        # Stamp của face frame thật cuối cùng đã ingest. Dùng để bỏ qua
        # snapshot stale khi main loop wake từ camera khác (xem process_entry).
        self._last_face_stamp = None

    def update_batch(self, faces, frame_stamp=None):
        """
        faces: list face dict; mỗi dict có thể chứa `frame_stamp` (DS path).
        frame_stamp: stamp ở mức batch (caller pass thẳng cho trường hợp
            face_data=[] — không có face dict để đính). Nếu None, suy ra từ
            face dict đầu tiên có stamp.

        Hành vi: nếu stamp đã thấy ở lần update_batch trước → return không
        side-effect. Vẫn tick `_frame_idx` chỉ khi đây là frame face thật mới,
        nên stale_frames đếm theo frame face chứ không theo wake-up.
        """
        if frame_stamp is None and faces:
            for f in faces:
                s = f.get("frame_stamp")
                if s is not None:
                    frame_stamp = s
                    break
        if frame_stamp is not None and frame_stamp == self._last_face_stamp:
            return
        if frame_stamp is not None:
            self._last_face_stamp = frame_stamp

        self._frame_idx += 1
        for f in faces:
            tid = f.get("object_id")
            if tid is None:
                # nvtracker chưa kịp commit track — skip frame, đợi frame sau.
                continue
            slot = self._slots.get(tid)
            if slot is None:
                # Seed slot mới: cần embedding hợp lệ + quality đạt ngưỡng + còn chỗ.
                if len(self._slots) >= self.max_slots:
                    continue
                if f.get("embedding") is None or f.get("quality") is None:
                    continue
                if f["quality"] < self.min_seed_quality:
                    continue
                self._slots[tid] = {
                    "embedding": np.asarray(f["embedding"], dtype=np.float32),
                    "quality":   float(f["quality"]),
                    "conf":      float(f["conf"]),
                    "bbox":      f.get("bbox"),
                    "hits":      1,
                    "last_seen": self._frame_idx,
                }
            else:
                # Track đã có slot — count hits luôn, kể cả frame plugin skip.
                slot["hits"] += 1
                slot["last_seen"] = self._frame_idx
                slot["bbox"] = f.get("bbox") or slot["bbox"]
                # Best-of-N: chỉ cập nhật embedding khi có cái mới quality cao hơn.
                if (f.get("embedding") is not None
                        and f.get("quality") is not None
                        and f["quality"] > slot["quality"]):
                    slot["embedding"] = np.asarray(
                        f["embedding"], dtype=np.float32)
                    slot["quality"]   = float(f["quality"])
                    slot["conf"]      = float(f["conf"])

        # Dọn track không còn xuất hiện quá lâu — tránh slot bám mãi.
        if self.stale_frames > 0 and self._slots:
            stale = [tid for tid, s in self._slots.items()
                     if self._frame_idx - s["last_seen"] > self.stale_frames]
            for tid in stale:
                del self._slots[tid]

    def committed(self):
        """Slot đủ điều kiện commit (hits ≥ min_hits)."""
        return [s for s in self._slots.values() if s["hits"] >= self.min_hits]

    def best_committed(self):
        """Slot quality cao nhất trong committed, dùng cho display + exit query."""
        c = self.committed()
        return max(c, key=lambda s: s["quality"]) if c else None

    @property
    def ready(self):
        return any(s["hits"] >= self.min_hits for s in self._slots.values())

    @property
    def slots(self):
        return list(self._slots.values())

    def clear(self):
        self._slots.clear()


# ──────────────────────────────────────────────
# Parking System
# ──────────────────────────────────────────────
class ParkingSystem:

    def __init__(self, cfg_path: str = "config.yaml"):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        # ── Models ──
        log.info("Loading models...")
        t0 = time.time()

        self.use_deepstream = (self.cfg["deepstream"]["enabled"]
                               and HAS_DEEPSTREAM)
        ds_cfg = self.cfg.get("deepstream", {})
        self._ds_plate_enabled = bool(
            self.use_deepstream and ds_cfg.get("plate_enabled", True))
        self._ds_face_enabled = bool(
            self.use_deepstream and ds_cfg.get("face_enabled", True))
        if not self.use_deepstream:
            pcfg = self.cfg["plate_detector"]
            self.plate_det = PlateDetector(
                pcfg["model"], pcfg["imgsz"], pcfg["conf"], pcfg["device"])
        else:
            self.plate_det = None
            if self._ds_plate_enabled:
                log.info("PlateDetector skipped (DeepStream plate nvinfer)")
            else:
                log.info("PlateDetector skipped (DeepStream plate disabled)")

        # ★ v5: OCR backend selection
        ocr_cfg = self.cfg["plate_ocr"]
        ocr_backend = ocr_cfg.get("backend", "yolo")
        self.plate_ocr = None

        if self.use_deepstream and not self._ds_plate_enabled:
            ocr_backend = "disabled"
            log.info("Plate OCR skipped (DeepStream plate disabled)")
        elif self._ds_plate_enabled:
            # SGIE plate_ocr (DS) đã chạy YOLO char-det trên GPU; combine probe
            # gom char → string trong pipeline.py. Python OCR không cần load.
            ocr_backend = "deepstream-sgie"
            log.info("Plate OCR via DeepStream SGIE "
                     "(Python OCR model not loaded)")
        elif ocr_backend == "yolo":
            self.plate_ocr = PlateOCRYolo(
                model_path=ocr_cfg["model"],
                imgsz=ocr_cfg.get("imgsz", 320),
                conf=ocr_cfg.get("conf", 0.3),
                device=ocr_cfg.get("device", 0))
        else:
            # Fallback PaddleOCR
            log.info("Using PaddleOCR fallback (backend='paddle')")
            self.plate_ocr = PlateOCR(
                ocr_cfg.get("lang", "en"),
                ocr_cfg.get("use_gpu", True))

        self.face_eng = None
        if self._ds_face_enabled:
            log.info("FaceEngine skipped (DeepStream face chain)")
        else:
            self._load_face_engine()

        log.info(f"Models loaded in {time.time()-t0:.1f}s "
                 f"(DeepStream={'ON' if self.use_deepstream else 'OFF'}, "
                 f"OCR={ocr_backend})")

        # ── Database ──
        dcfg = self.cfg["database"]
        self.db = ParkingDB(
            host=dcfg["host"], port=dcfg["port"],
            dbname=dcfg["dbname"], user=dcfg["user"],
            password=dcfg["password"], max_cap=dcfg["max_capacity"])

        # ── Recognition helpers ──
        rcfg = self.cfg["recognition"]
        fcfg_full = self.cfg.get("face", {})
        self.validator = PlateValidator()
        self.plate_voter = PlateVoter(rcfg["plate_vote_frames"])
        self.face_thr = rcfg["face_threshold"]
        self.blur_thr = fcfg_full.get("blur_threshold", 35.0)
        self._face_min_quality = float(fcfg_full.get("min_quality", 0.3))
        self.tracker = IdentityTracker(
            max_slots=rcfg.get("max_identities", 3),
            min_seed_quality=rcfg.get("min_seed_quality",
                                      self._face_min_quality),
            min_hits=rcfg.get("identity_min_hits", 2),
            stale_frames=rcfg.get("identity_stale_frames", 60),
        )

        # Fallback path không có nvtracker → tự gán object_id qua IoU.
        self._fallback_prev_tracks = []   # [(bbox, track_id), ...]
        self._fallback_next_id = 0
        # Stamp tăng mỗi lần _run_face chạy → đính vào face dict cho symmetric
        # với DS path. Fallback không có vấn đề stale-snapshot nhưng giữ field
        # để IdentityTracker dedup hoạt động đồng nhất.
        self._fallback_face_stamp = 0

        # ── Web state (shared reference với web.py) ──
        self.state = {
            "mode": "entry", "stream_fps":0, "fps": 0,
            "plate_cam_ok": False, "face_cam_ok": False,
            "deepstream": self.use_deepstream,
        }
        
        # Lưu result gần nhất để annotate frame
        self._last_result = {"ok": False}

        # ── Thread pool cho parallel OCR + face inference ──
        # Lazy: chỉ tạo khi fallback path cần (DS mode không dùng).
        self._executor = None

        self._cached_stats = self.db.stats()
        self._face_rotate = self.cfg.get("camera", {}).get("face_rotate", -1)
        self._rot_map = {
            1: cv2.ROTATE_90_CLOCKWISE,
            2: cv2.ROTATE_180,
            3: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }

        self.running = False

    def _get_executor(self):
        """Lazy ThreadPoolExecutor: tạo lần đầu fallback path gọi tới."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="infer")
        return self._executor

    def _load_face_engine(self):
        """Lazy-load InsightFace only for fallback / non-DS face path."""
        if self.face_eng is None:
            fcfg = self.cfg["face"]
            self.face_eng = FaceEngine(
                fcfg["model_pack"], tuple(fcfg["det_size"]))
        return self.face_eng

    # ──────────────────────────────────────────────
    # Face helpers (DS path đẩy face_data từ pipeline)
    # ──────────────────────────────────────────────
    def _ensure_quality(self, face_data, frame_face):
        """
        Đảm bảo mỗi face có field `quality`. Plugin nvdsfaceembed (v2+) đã set
        sẵn; fallback dùng CPU FaceEngine.quality khi plugin chưa fill (vd
        plugin chưa rebuild với A1) hoặc khi chạy InsightFace path.
        """
        if not face_data:
            return face_data
        for f in face_data:
            if f.get("embedding") is None:
                f["quality"] = None
                continue
            if f.get("quality") is not None:
                continue
            if frame_face is None:
                f["quality"] = None
                continue
            _, q = FaceEngine.quality(frame_face, f["bbox"], self.blur_thr)
            f["quality"] = q
        return face_data

    def _pick_display_face(self, face_data):
        """Face để vẽ bbox dashboard: ưu tiên có embedding, max conf."""
        if not face_data:
            return None
        with_emb = [f for f in face_data if f.get("embedding") is not None]
        pool = with_emb if with_emb else face_data
        return max(pool, key=lambda f: float(f.get("conf", 0.0)))

    # ── Parallel inference helpers ──
    def _run_ocr(self, crop):
        """OCR + validate. Thread-safe (GPU releases GIL)."""
        t0 = time.time()
        raw_text, ocr_conf = self.plate_ocr(crop)
        plate = self.validator(raw_text)
        dt = (time.time() - t0) * 1000
        return raw_text, ocr_conf, plate, dt

    def _assign_fallback_track_id(self, bbox):
        """
        Gán object_id tổng hợp cho fallback path (không có nvtracker).
        Match với prev frame bằng IoU > 0.5 → reuse ID; khác → ID mới.
        Đủ cho test/dev mode; production luôn dùng DS path + NvDCF.
        """
        x1, y1, x2, y2 = bbox
        best_id, best_iou = None, 0.0
        for prev_bbox, prev_id in self._fallback_prev_tracks:
            ax1, ay1, ax2, ay2 = prev_bbox
            ix1 = max(x1, ax1); iy1 = max(y1, ay1)
            ix2 = min(x2, ax2); iy2 = min(y2, ay2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = max(0, x2 - x1) * max(0, y2 - y1)
            area_b = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            iou = inter / (area_a + area_b - inter + 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_id = prev_id
        if best_id is not None and best_iou >= 0.5:
            return best_id
        self._fallback_next_id += 1
        return self._fallback_next_id

    def _run_face(self, frame_face):
        """
        Fallback non-DS face inference: chạy InsightFace, attach quality +
        synthetic object_id (IoU matching vs frame trước) cho mỗi face.
        Returns: (face_data: list[dict], dt_ms)
        """
        t0 = time.time()
        faces = self._load_face_engine()(frame_face)
        self._fallback_face_stamp += 1
        stamp = self._fallback_face_stamp
        out = []
        new_prev = []
        for f in faces:
            _, q = FaceEngine.quality(frame_face, f["bbox"], self.blur_thr)
            tid = self._assign_fallback_track_id(f["bbox"])
            new_prev.append((f["bbox"], tid))
            out.append({
                "bbox": f["bbox"],
                "conf": f["conf"],
                "embedding": f["embedding"],
                "quality": q,
                "object_id": tid,
                "frame_stamp": stamp,
            })
        self._fallback_prev_tracks = new_prev
        dt = (time.time() - t0) * 1000
        return out, dt

    # ──────────────────────────────────────────────
    # Plate crop + display helpers
    # ──────────────────────────────────────────────
    @staticmethod
    def _crop_plate(frame_plate, bbox, source_size=None):
        """Crop plate thumbnail. Nếu source_size khác kích thước frame
        (DS path: bbox ở toạ độ mux 1280×720, frame là display 640×360),
        scale bbox về toạ độ frame trước khi crop."""
        h, w = frame_plate.shape[:2]
        x1, y1, x2, y2 = bbox
        if source_size and source_size[0] > 0 and source_size[1] > 0:
            src_w, src_h = source_size
            if (src_w, src_h) != (w, h):
                sx = w / src_w
                sy = h / src_h
                x1, y1, x2, y2 = (int(x1 * sx), int(y1 * sy),
                                  int(x2 * sx), int(y2 * sy))
        box_w, box_h = x2 - x1, y2 - y1
        mx = max(10, int(box_w * 0.08))
        my = max(8, int(box_h * 0.12))
        return frame_plate[max(0, y1 - my):min(h, y2 + my),
                           max(0, x1 - mx):min(w, x2 + mx)]

    def _fill_display(self, result, frame_face, face_data,
                      face_source_size=None):
        """
        Set face_bbox/face_conf/face_crop cho dashboard event payload.

        bbox luôn ở toạ độ source (full-res). Nếu frame_face là display frame
        đã downscale (DS mode), face_source_size khác kích thước frame_face
        nên cần scale bbox về toạ độ frame_face trước khi cắt.
        """
        best_f = self._pick_display_face(face_data)
        if not best_f:
            return
        # Tất cả face để vẽ trên dashboard (bbox + conf + object_id để debug).
        result["face_bboxes"] = [
            {"bbox": f["bbox"],
             "conf": float(f.get("conf", 0.0)),
             "object_id": f.get("object_id")}
            for f in face_data if f.get("bbox") is not None
        ]
        # Face "chính" — dùng cho crop + label MATCH + commit logic.
        result["face_bbox"] = best_f["bbox"]
        result["face_conf"] = float(best_f.get("conf", 0.0))
        if frame_face is None:
            return
        bx1, by1, bx2, by2 = best_f["bbox"]
        fh, fw = frame_face.shape[:2]
        if face_source_size and face_source_size[0] > 0:
            src_w, src_h = face_source_size
            sx = fw / src_w
            sy = fh / src_h
            bx1, by1, bx2, by2 = (int(bx1 * sx), int(by1 * sy),
                                  int(bx2 * sx), int(by2 * sy))
        fc = frame_face[max(0, by1):min(fh, by2),
                        max(0, bx1):min(fw, bx2)]
        if fc.size > 0:
            result["face_crop"] = fc.copy()

    # ──────────────────────────────────────────────
    # ENTRY / EXIT — unified tracker-based logic
    # Cả DS path (face_data từ pipeline) và fallback (face_data=None) đều
    # đẩy vào IdentityTracker. DB nhận list embedding/quality từ committed
    # slots khi plate vote stable.
    # ──────────────────────────────────────────────
    def process_entry(self, frame_plate, plate_dets,
                      frame_face, face_data=None,
                      face_source_size=None,
                      plate_source_size=None,
                      face_stamp=None,
                      plate_fresh=True, face_fresh=True) -> dict:
        result = {"ok": False, "plate": "", "face_conf": 0,
                  "plate_bbox": None, "face_bbox": None}

        # Plate detect fallback (non-DS).
        if plate_dets is None and self.plate_det:
            plate_dets = self.plate_det(frame_plate) if frame_plate is not None \
                else []

        # Face inference fallback (non-DS path).
        dt_face = 0.0
        if face_data is None and frame_face is not None:
            t_parallel = time.time()
            fut_ocr = None
            if plate_dets and frame_plate is not None:
                best_p = max(plate_dets, key=lambda p: p["conf"])
                crop = self._crop_plate(frame_plate, best_p["bbox"],
                                        source_size=plate_source_size)
                fut_ocr = self._get_executor().submit(self._run_ocr, crop) \
                    if crop.size > 0 else None
            fut_face = self._get_executor().submit(self._run_face, frame_face)
            face_data, dt_face = fut_face.result()
            if fut_ocr is not None:
                ocr_result = fut_ocr.result()
            else:
                ocr_result = None
            dt_parallel = (time.time() - t_parallel) * 1000
            self.state["timing"] = {
                "face_ms": round(dt_face, 1),
                "parallel_ms": round(dt_parallel, 1),
            }
        else:
            ocr_result = None
            self.state["timing"] = {}

        # Quality fill + push vào tracker CHỈ khi face cam có frame mới.
        # Snapshot stale (plate-only wake) → giữ nguyên state tracker, dùng
        # face_data cũ cho display để không flicker bbox.
        face_data = self._ensure_quality(face_data, frame_face)
        if face_fresh:
            self.tracker.update_batch(face_data or [], frame_stamp=face_stamp)
        if face_data:
            self._fill_display(result, frame_face, face_data,
                               face_source_size=face_source_size)

        # Plate display (bbox) luôn fill nếu có detection — dù plate stale,
        # vẫn vẽ bbox cũ để không flicker giữa các plate wake.
        best_p = max(plate_dets, key=lambda p: p["conf"]) if plate_dets else None
        if best_p is not None:
            result["plate_bbox"] = best_p["bbox"]

        # Vote / OCR / commit CHỈ khi plate fresh + có detection. Stale wake
        # (face-only) không được vote lại vì sẽ inflate plate_voter buffer.
        if not plate_fresh or best_p is None:
            return result

        # Crop chỉ cần cho display emit (plate_crop trong WebSocket payload).
        # DS path: frame_plate là display 640×360 — _crop_plate scale bbox từ
        # toạ độ mux về display qua plate_source_size.
        crop = None
        if frame_plate is not None:
            crop = self._crop_plate(frame_plate, best_p["bbox"],
                                    source_size=plate_source_size)
            if crop.size > 0:
                result["plate_crop"] = crop.copy()

        # OCR text:
        #   DS path  → best_p["text"] do combine probe gắn từ SGIE.
        #   Fallback → ocr_result đã chạy parallel ở trên, hoặc serial _run_ocr.
        if "text" in best_p:
            raw_text = best_p["text"]
            ocr_conf = float(best_p.get("text_conf", 0.0))
            plate = self.validator(raw_text) if raw_text else ""
            dt_ocr = 0.0
        elif ocr_result is not None:
            raw_text, ocr_conf, plate, dt_ocr = ocr_result
        else:
            if crop is None or crop.size == 0:
                return result
            raw_text, ocr_conf, plate, dt_ocr = self._run_ocr(crop)
        self.state["timing"]["ocr_ms"] = round(dt_ocr, 1)
        if raw_text:
            log.debug(f"ENTRY OCR: raw='{raw_text}' conf={ocr_conf:.2f} "
                      f"→ '{plate}' ({dt_ocr:.0f}ms)")

        # Vote.
        stable = self.plate_voter.vote(plate)
        if not stable:
            result["plate"] = plate
            return result
        result["plate"] = stable
        log.info(f"ENTRY: plate voted → '{stable}'")

        # Commit: cần có ít nhất 1 committed slot.
        slots = self.tracker.committed()
        if not slots:
            log.debug(f"ENTRY: plate '{stable}' stable nhưng tracker chưa "
                      f"có slot đủ hits")
            return result

        embeddings = [s["embedding"] for s in slots]
        qualities = [s["quality"] for s in slots]
        best_q = max(qualities)
        code = self.db.entry(
            stable, embeddings, ocr_conf, best_q,
            qualities=qualities,
        )
        if code > 0:
            result["ok"] = True
            self.plate_voter.clear()
            self.tracker.clear()
            log.info(f"✅ ENTRY OK: {stable} (id={code}, "
                     f"ids={len(slots)})")
            self._emit("entry", {"plate": stable}, result)
        elif code == -1:
            log.warning("❌ BÃI ĐẦY")
        elif code == -2:
            log.warning(f"❌ TRÙNG BIỂN SỐ: {stable}")
            self.plate_voter.clear()
            self.tracker.clear()
        return result

    def process_exit(self, frame_face, frame_plate,
                     plate_dets=None, face_data=None,
                     face_source_size=None,
                     plate_source_size=None,
                     face_stamp=None,
                     plate_fresh=True, face_fresh=True) -> dict:
        result = {"ok": False, "plate": "", "sim": 0.0,
                  "face_bbox": None, "plate_bbox": None}

        if plate_dets is None and self.plate_det:
            plate_dets = self.plate_det(frame_plate) if frame_plate is not None \
                else []

        dt_face = 0.0
        if face_data is None and frame_face is not None:
            face_data, dt_face = self._run_face(frame_face)
            self.state["timing"] = {"face_ms": round(dt_face, 1)}
        else:
            self.state["timing"] = {}

        face_data = self._ensure_quality(face_data, frame_face)
        # Tracker update chỉ khi face fresh — xem ghi chú trong process_entry.
        if face_fresh:
            self.tracker.update_batch(face_data or [], frame_stamp=face_stamp)
        if face_data:
            self._fill_display(result, frame_face, face_data,
                               face_source_size=face_source_size)

        # Plate display luôn fill cho continuity.
        best_p = max(plate_dets, key=lambda p: p["conf"]) if plate_dets else None
        if best_p is not None:
            result["plate_bbox"] = best_p["bbox"]

        if not plate_fresh or best_p is None:
            return result

        # Crop chỉ cho display payload; DS path có text sẵn nên crop optional.
        crop = None
        if frame_plate is not None:
            crop = self._crop_plate(frame_plate, best_p["bbox"],
                                    source_size=plate_source_size)
            if crop.size > 0:
                result["plate_crop"] = crop.copy()

        if "text" in best_p:
            raw_text = best_p["text"]
            plate = self.validator(raw_text) if raw_text else ""
            dt_ocr = 0.0
        else:
            if crop is None or crop.size == 0:
                return result
            raw_text, _, plate, dt_ocr = self._run_ocr(crop)
        self.state["timing"]["ocr_ms"] = round(dt_ocr, 1)

        exit_plate = self.plate_voter.vote(plate)
        if not exit_plate:
            return result

        # Verify face: query DB cho slot quality cao nhất (slots khác sẽ vẫn
        # match qua MIN(distance) trong active_faces vì store nhiều embedding).
        best_slot = self.tracker.best_committed()
        if best_slot is None:
            log.debug(f"EXIT: plate '{exit_plate}' stable nhưng tracker "
                      f"chưa có committed slot")
            return result

        match = self.db.match_exit_by_plate(
            exit_plate, best_slot["embedding"], threshold=self.face_thr)
        if not match:
            log.debug(f"EXIT: {exit_plate} — no embedding match "
                      f"(thr={self.face_thr})")
            return result

        sim = match["sim"]
        ok = self.db.exit(match["active_id"], sim)
        if not ok:
            return result

        result["ok"] = True
        result["plate"] = match["plate"]
        result["sim"] = sim
        self.plate_voter.clear()
        self.tracker.clear()
        log.info(f"✅ EXIT: {match['plate']} (sim={sim:.3f}, "
                 f"embeddings={match['n_embeddings']})")
        self._emit("exit",
                   {"plate": match["plate"], "sim": sim}, result)
        return result

    def _emit(self, event_type: str, data: dict, result: dict = None):
        """Push event ra web dashboard."""
        import base64
        try:
            if result:
                for key in ("plate_crop", "face_crop"):
                    img = result.get(key)
                    if img is not None:
                        _, jpg = cv2.imencode(".jpg", img,
                                              [cv2.IMWRITE_JPEG_QUALITY, 85])
                        data[key] = base64.b64encode(jpg).decode("ascii")
            from web import notify_sync
            notify_sync({"type": event_type, "data": data})
        except Exception as e:
            log.debug(f"Web notify failed: {e}")
        
    def _apply_rotation(self, frame):
        """Apply cached rotation. Dùng cho web thread — không auto-detect."""
        rot = self._rot_map.get(self._face_rotate)
        return cv2.rotate(frame, rot) if rot else frame

    def _rotate_face(self, frame):
        """Rotate face frame. Auto-detect orientation nếu chưa biết."""
        if self._face_rotate >= 0:
            return self._apply_rotation(frame)

        # Auto-detect (chạy 1 lần duy nhất khi _face_rotate == -1)
        for code, rot, name in [
            (0, None, "no rotation"),
            (1, cv2.ROTATE_90_CLOCKWISE, "90° CW"),
            (3, cv2.ROTATE_90_COUNTERCLOCKWISE, "90° CCW"),
            (2, cv2.ROTATE_180, "180°"),
        ]:
            test = frame if rot is None else cv2.rotate(frame, rot)
            if self._load_face_engine()(test):
                self._face_rotate = code
                log.info(f"Face rotation auto-detected: {name} (code={code})")
                return test
        return frame

    # ── ANNOTATE FRAMES CHO WEB ──
    # Annotation kích thước hiển thị thực = display_frame_w × browser_scale.
    # Để không phụ thuộc browser stretch, scale các hằng số (border, font,
    # padding) theo width frame so với 1280px chuẩn cũ.
    _ANNOT_REF_W = 1280

    @classmethod
    def _annot_style(cls, frame_w):
        s = max(0.5, frame_w / cls._ANNOT_REF_W)
        return {
            "border": max(1, int(round(2 * s))),
            "pad_h":  max(14, int(round(28 * s))),
            "char_w_face":  max(7, int(round(12 * s))),
            "char_w_plate": max(10, int(round(16 * s))),
            "font_scale_face":  0.6 * s,
            "font_scale_plate": 0.7 * s,
            "text_thick": max(1, int(round(2 * s))),
        }

    @staticmethod
    def _scale_bbox(bbox, frame_shape, source_size):
        """bbox đang ở source coord → scale về kích thước frame."""
        x1, y1, x2, y2 = bbox
        if not source_size or source_size[0] <= 0 or source_size[1] <= 0:
            return int(x1), int(y1), int(x2), int(y2)
        src_w, src_h = source_size
        fh, fw = frame_shape[:2]
        if (src_w, src_h) == (fw, fh):
            return int(x1), int(y1), int(x2), int(y2)
        sx = fw / src_w
        sy = fh / src_h
        return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)

    def _annotate_plate(self, frame, result, source_size=None):
        if not result.get("plate_bbox"):
            return frame
        vis = frame.copy()
        st = self._annot_style(vis.shape[1])
        x1, y1, x2, y2 = self._scale_bbox(result["plate_bbox"],
                                          vis.shape, source_size)
        color = (0, 255, 0) if result.get("ok") else (0, 255, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, st["border"])
        plate = result.get("plate", "")
        if plate:
            cv2.rectangle(vis, (x1, y1 - st["pad_h"]),
                          (x1 + len(plate) * st["char_w_plate"], y1),
                          (0, 0, 0), -1)
            cv2.putText(vis, plate, (x1 + 4, y1 - int(st["pad_h"] * 0.3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        st["font_scale_plate"], color, st["text_thick"])
        return vis

    def _annotate_face(self, frame, result, source_size=None):
        faces = result.get("face_bboxes")
        primary = result.get("face_bbox")
        if not faces and not primary:
            return frame
        vis = frame.copy()
        st = self._annot_style(vis.shape[1])

        # Vẽ tất cả face — bbox phụ màu vàng mỏng để dễ phân biệt với primary.
        if faces:
            for f in faces:
                bbox = f.get("bbox")
                if bbox is None or bbox == primary:
                    continue
                x1, y1, x2, y2 = self._scale_bbox(bbox, vis.shape, source_size)
                cv2.rectangle(vis, (x1, y1), (x2, y2),
                              (0, 200, 200), max(1, st["border"] - 1))
                tid = f.get("object_id")
                if tid is not None:
                    cv2.putText(vis, f"#{tid}",
                                (x1 + 2, y1 + st["pad_h"] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                st["font_scale_face"] * 0.8,
                                (0, 200, 200), 1)

        # Face primary (best) — vẽ đậm hơn + label conf/MATCH.
        if primary:
            x1, y1, x2, y2 = self._scale_bbox(primary, vis.shape, source_size)
            color = (0, 255, 0) if result.get("ok") else (0, 255, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, st["border"])
            label = ""
            if result.get("ok") and result.get("sim"):
                label = f"MATCH {result['sim']:.2f}"
            elif result.get("face_conf"):
                label = f"face {result['face_conf']:.2f}"
            if label:
                cv2.rectangle(vis, (x1, y1 - st["pad_h"]),
                              (x1 + len(label) * st["char_w_face"], y1),
                              (0, 0, 0), -1)
                cv2.putText(vis, label,
                            (x1 + 4, y1 - int(st["pad_h"] * 0.3)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            st["font_scale_face"], color, st["text_thick"])
        return vis
    
    def _web_update_loop(self, cam_plate, cam_face, interval: float = 0.1):
        from web import update_frame
        log.info("Web update thread started (fallback mode)")
        while self.running:
            t0 = time.time()
            fp = cam_plate.latest
            ff = cam_face.latest
            try:
                if fp is not None:
                    update_frame("plate",
                                 self._annotate_plate(fp, self._last_result))
                if ff is not None:
                    ff_rot = self._apply_rotation(ff)
                    update_frame("face",
                                 self._annotate_face(ff_rot, self._last_result))
            except Exception:
                pass
            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

    def _web_update_loop_ds(self, ds, interval: float = 0.1):
        from web import update_frame
        log.info("Web update thread started (DeepStream mode)")
        while self.running:
            t0 = time.time()
            snap = ds.get_all()
            plate_disp = snap["plate_display"]
            face_disp = snap["face_display"]
            plate_src = snap["plate_source_size"]
            face_src = snap["face_source_size"]
            try:
                if plate_disp is not None:
                    update_frame("plate",
                                 self._annotate_plate(plate_disp,
                                                      self._last_result,
                                                      source_size=plate_src))
                if face_disp is not None:
                    update_frame("face",
                                 self._annotate_face(face_disp,
                                                     self._last_result,
                                                     source_size=face_src))
            except Exception:
                pass
            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

    # ── RUN (DEEPSTREAM MODE) ──
    def _run_deepstream(self, mode: str, show: bool):
        ccfg = self.cfg["camera"]
        ds = DeepStreamPipeline(ccfg["plate"], ccfg["face"], self.cfg)
        ds.start()
        
        web_thread = Thread(
            target=self._web_update_loop_ds,
            args=(ds,),
            daemon=True,
            name="web-update-ds"
        )
        web_thread.start()

        frame_idx = 0
        fps, t_fps, n_fps = 0.0, time.time(), 0
        cooldown_until = 0.0

        log.info(f"DeepStream mode={mode}")

        try:
            while self.running:
                plate_fresh, face_fresh = ds.wait_new_frame(timeout=0.5)
                if not (plate_fresh or face_fresh):
                    continue

                snap = ds.get_all()
                plate_disp = snap["plate_display"]
                plate_dets = snap["plate_dets"]
                face_data = snap["face_data"]
                face_disp = snap["face_display"]
                face_src_sz = snap["face_source_size"]
                plate_src_sz = snap["plate_source_size"]
                face_stamp = snap["face_stamp"]

                # "Cam ok" = pipeline đã probe ít nhất 1 frame (source_size>0).
                plate_seen = plate_src_sz[0] > 0
                face_seen = face_src_sz[0] > 0
                self.state["plate_cam_ok"] = plate_seen
                self.state["face_cam_ok"] = face_seen
                frame_idx += 1

                plate_ready = (not self._ds_plate_enabled) or plate_seen
                face_ready = (not self._ds_face_enabled) or face_seen

                if not plate_ready or not face_ready:
                    time.sleep(0.01)
                    if frame_idx % 300 == 0:
                        plate_state = ("DISABLED" if not self._ds_plate_enabled
                                       else ("OK" if plate_seen else "NONE"))
                        face_state = ("DISABLED" if not self._ds_face_enabled
                                      else ("OK" if face_seen else "NONE"))
                        log.warning(f"Waiting frames... "
                                    f"plate={plate_state} face={face_state}")
                    continue

                # Single-camera debug mode: keep the pipeline and dashboard
                # alive, but do not call entry/exit logic that requires both
                # plate and face frames.
                if not (self._ds_plate_enabled and self._ds_face_enabled):
                    # Reset trước, tránh giữ bbox cũ khi frame mới không có mặt.
                    self._last_result = {"ok": False}
                    if self._ds_face_enabled and face_data:
                        best = self._pick_display_face(face_data)
                        self._last_result = {
                            "ok": False,
                            "face_bbox": best.get("bbox") if best else None,
                            "face_conf": float(best.get("conf", 0.0))
                                if best else 0.0,
                            "face_bboxes": [
                                {"bbox": f["bbox"],
                                 "conf": float(f.get("conf", 0.0)),
                                 "object_id": f.get("object_id")}
                                for f in face_data
                                if f.get("bbox") is not None
                            ],
                        }

                    n_fps += 1
                    now = time.time()
                    elapsed = now - t_fps
                    if elapsed >= 1.0:
                        self.state["fps"] = round(n_fps / elapsed, 1)
                        self.state["stream_fps"] = ds.stream_fps
                        self._cached_stats = self.db.stats()
                        n_fps, t_fps = 0, now
                    continue

                # DeepStream face frames are already rotated at source ingress
                # via camera.face_rotate_nv. Do not rotate again here.

                t0 = time.time()

                if t0 < cooldown_until:
                    result = {"ok": False}
                elif mode == "entry":
                    result = self.process_entry(
                        plate_disp, plate_dets, face_disp,
                        face_data=face_data if self._ds_face_enabled else None,
                        face_source_size=face_src_sz,
                        plate_source_size=plate_src_sz,
                        face_stamp=face_stamp if self._ds_face_enabled else None,
                        plate_fresh=plate_fresh and self._ds_plate_enabled,
                        face_fresh=face_fresh and self._ds_face_enabled)
                else:
                    result = self.process_exit(
                        face_disp, plate_disp, plate_dets,
                        face_data=face_data if self._ds_face_enabled else None,
                        face_source_size=face_src_sz,
                        plate_source_size=plate_src_sz,
                        face_stamp=face_stamp if self._ds_face_enabled else None,
                        plate_fresh=plate_fresh and self._ds_plate_enabled,
                        face_fresh=face_fresh and self._ds_face_enabled)

                self._last_result = result

                if result.get("ok"):
                    cooldown_until = t0 + 0.5

                n_fps += 1
                now = time.time()
                elapsed = now - t_fps
                if elapsed >= 1.0:
                    self.state["fps"] = round(n_fps / elapsed, 1)
                    self.state["stream_fps"] = ds.stream_fps
                    self._cached_stats = self.db.stats()
                    n_fps, t_fps = 0, now

                if show:
                    if plate_disp is not None and face_disp is not None:
                        self._show_dual(plate_disp, face_disp, result, mode)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("m"):
                        mode = "exit" if mode == "entry" else "entry"
                        self.state["mode"] = mode
                        self.plate_voter.clear()
                        self.tracker.clear()
                        log.info(f"Mode → {mode.upper()}")

        except KeyboardInterrupt:
            pass
        finally:
            ds.stop()

    # ── RUN (FALLBACK MODE) ──
    def _run_fallback(self, mode: str, show: bool):
        ccfg = self.cfg["camera"]
        hw = ccfg["hw_decode"]
        reconn = ccfg.get("reconnect_sec", 3)

        cam_plate = StreamReader(ccfg["plate"], name="plate",
                                 hw_decode=hw, reconnect_sec=reconn)
        cam_face = StreamReader(ccfg["face"], name="face",
                                hw_decode=hw, reconnect_sec=reconn)

        web_thread = Thread(
            target=self._web_update_loop,
            args=(cam_plate, cam_face),
            daemon=True,
            name="web-update"
        )
        web_thread.start()
        
        skip_n = ccfg["process_every_n"]
        frame_idx = 0
        fps, t_fps, n_fps = 0.0, time.time(), 0
        cooldown_until = 0.0

        log.info(f"Fallback mode={mode}")

        try:
            while self.running:
                fp = cam_plate.read(timeout=5.0)
                ff = cam_face.read(timeout=5.0)

                self.state["plate_cam_ok"] = cam_plate.connected
                self.state["face_cam_ok"] = cam_face.connected

                if fp is None or ff is None:
                    if cam_plate.is_stream or cam_face.is_stream:
                        time.sleep(0.1)
                        continue
                    break

                frame_idx += 1

                if skip_n > 1 and frame_idx % skip_n != 0:
                    continue

                ff = self._rotate_face(ff)

                t0 = time.time()

                if t0 < cooldown_until:
                    result = {"ok": False}
                elif mode == "entry":
                    result = self.process_entry(fp, None, ff)
                else:
                    result = self.process_exit(ff, fp)

                self._last_result = result

                if result.get("ok"):
                    cooldown_until = t0 + 0.5

                n_fps += 1
                now = time.time()
                elapsed = now - t_fps
                if elapsed >= 1.0:
                    self.state["fps"] = round(n_fps / elapsed, 1)
                    self.state["stream_fps"] = round(
                        (cam_plate.stream_fps + cam_face.stream_fps) / 2, 1)
                    self._cached_stats = self.db.stats()
                    n_fps, t_fps = 0, now
                    
                if show:
                    self._show_dual(fp, ff, result, mode)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("m"):
                        mode = "exit" if mode == "entry" else "entry"
                        self.state["mode"] = mode
                        self.plate_voter.clear()
                        self.tracker.clear()
                        log.info(f"Mode → {mode.upper()}")

        except KeyboardInterrupt:
            pass
        finally:
            cam_plate.release()
            cam_face.release()

    def _show_dual(self, fp, ff, result, mode):
        stats = self._cached_stats
        fps = self.state.get("fps", 0.0)
        h = 360
        p = cv2.resize(fp, (int(fp.shape[1]*h/fp.shape[0]), h))
        f = cv2.resize(ff, (int(ff.shape[1]*h/ff.shape[0]), h))

        if result.get("plate_bbox"):
            sx = p.shape[1] / fp.shape[1]
            sy = p.shape[0] / fp.shape[0]
            x1, y1, x2, y2 = result["plate_bbox"]
            cv2.rectangle(p, (int(x1*sx), int(y1*sy)),
                          (int(x2*sx), int(y2*sy)), (0, 255, 0), 2)
        if result.get("face_bbox"):
            sx = f.shape[1] / ff.shape[1]
            sy = f.shape[0] / ff.shape[0]
            x1, y1, x2, y2 = result["face_bbox"]
            cv2.rectangle(f, (int(x1*sx), int(y1*sy)),
                          (int(x2*sx), int(y2*sy)), (0, 255, 255), 2)

        color = (0, 255, 0) if result.get("ok") else (100, 100, 100)
        for img, label in [(p, "PLATE"), (f, "FACE")]:
            w = img.shape[1]
            cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
            info = (f"{label}|{mode.upper()} FPS:{fps:.0f} "
                    f"Lot:{stats['current']}/{stats['capacity']}")
            cv2.putText(img, info, (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        vis = np.hstack([p, f])
        disp_w = self.cfg["display"]["window_width"]
        if vis.shape[1] > disp_w:
            s = disp_w / vis.shape[1]
            vis = cv2.resize(vis, None, fx=s, fy=s)
        cv2.imshow("Parking", vis)

    # ── PUBLIC RUN ──
    def run(self, mode: str = "entry", show: bool = True):
        self.running = True
        self.state["mode"] = mode
        log.info(f"Starting (deepstream={self.use_deepstream})")
        log.info(f"DB: {self.db.stats()}")

        try:
            if self.use_deepstream:
                self._run_deepstream(mode, show)
            else:
                self._run_fallback(mode, show)
        finally:
            self.running = False
            if self._executor is not None:
                self._executor.shutdown(wait=False)
            self.db.close()
            cv2.destroyAllWindows()
            log.info(f"Done. {self.db.stats()}")


# ──────────────────────────────────────────────
# Web server launcher
# ──────────────────────────────────────────────
def start_web(cfg: dict, db, state: dict):
    import uvicorn
    from web import app, init
    init(db, state)
    host = cfg["web"]["host"]
    port = cfg["web"]["port"]
    log.info(f"Web dashboard: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port,
                log_level="warning", access_log=False)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Smart Parking System")
    parser.add_argument("--config", default="config.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--entry", action="store_true")
    group.add_argument("--exit", action="store_true")
    group.add_argument("--benchmark", metavar="VIDEO")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--debug", action="store_true",
                        help="Bật debug logging chi tiết")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    system = ParkingSystem(args.config)

    def sig_handler(s, f):
        system.running = False
    signal.signal(signal.SIGINT, sig_handler)

    # Start web dashboard
    if not args.no_web and system.cfg["web"]["enabled"]:
        web_thread = Thread(
            target=start_web,
            args=(system.cfg, system.db, system.state),
            daemon=True)
        web_thread.start()
        time.sleep(1)

    if args.benchmark:
        reader = StreamReader(args.benchmark, name="bench",
                              hw_decode=system.cfg["camera"]["hw_decode"])
        times = {"plate_det": [], "ocr": [], "face": [],
                 "db": [], "total": []}
        count = 0

        while count < args.frames:
            frame = reader.read()
            if frame is None:
                break
            tt = time.time()

            t0 = time.time()
            plates = system.plate_det(frame) if system.plate_det else []
            times["plate_det"].append(time.time() - t0)

            if plates:
                best = max(plates, key=lambda p: p["conf"])
                x1, y1, x2, y2 = best["bbox"]
                t0 = time.time()
                system.plate_ocr(frame[y1:y2, x1:x2])
                times["ocr"].append(time.time() - t0)

            t0 = time.time()
            system.face_eng(frame)
            times["face"].append(time.time() - t0)

            t0 = time.time()
            system.db.find_by_plate("00A00000")
            times["db"].append(time.time() - t0)

            times["total"].append(time.time() - tt)
            count += 1

        reader.release()
        print(f"\n{'='*55}\n  BENCHMARK ({count} frames)\n{'='*55}")
        for k, v in times.items():
            if v:
                a = np.array(v) * 1000
                print(f"  {k:10s}  avg={a.mean():6.1f}ms  "
                      f"p95={np.percentile(a,95):6.1f}ms")
        if times["total"]:
            print(f"\n  FPS: {1000/(np.mean(times['total'])*1000):.1f}")
        print(f"{'='*55}\n")

    elif args.entry:
        system.run(mode="entry", show=not args.no_show)
    elif args.exit:
        system.run(mode="exit", show=not args.no_show)


if __name__ == "__main__":
    main()
