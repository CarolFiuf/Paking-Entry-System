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
from pipeline import (HAS_DEEPSTREAM, HAS_FACE_HELPERS,
                      DeepStreamPipeline, StreamReader)

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


class EmbeddingAvg:
    """
    Weighted average by quality score.
    Frame chất lượng cao (ít blur, sáng vừa, face to) được weight cao hơn.
    """
 
    def __init__(self, n=3):
        self.n = n
        self._buf = []  # list of (embedding, quality)
        self._latest = None

    def update(self, emb, quality=1.0):
        self._buf.append((emb, max(quality, 0.01)))  # avoid zero weight
        if len(self._buf) > self.n:
            self._buf.pop(0)

        # Weighted average
        weights = np.array([q for _, q in self._buf])
        weights = weights / weights.sum()
        avg = sum(w * e for (e, _), w in zip(self._buf, weights))
        n = np.linalg.norm(avg)
        self._latest = avg / n if n > 0 else avg
        return self._latest

    @property
    def ready(self):
        return self._latest is not None

    @property
    def _latest_quality(self):
        """Max quality trong buf — đại diện cho 'best frame seen' của track."""
        return max((q for _, q in self._buf), default=0.0)

    def clear(self):
        self._buf.clear()
        self._latest = None


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
        self._ds_face_chain = bool(
            self.use_deepstream
            and self.cfg.get("deepstream", {}).get("face_chain_enabled", False)
            and HAS_FACE_HELPERS
        )
        if not self.use_deepstream:
            pcfg = self.cfg["plate_detector"]
            self.plate_det = PlateDetector(
                pcfg["model"], pcfg["imgsz"], pcfg["conf"], pcfg["device"])
        else:
            self.plate_det = None
            log.info("PlateDetector skipped (DeepStream nvinfer)")

        # ★ v5: OCR backend selection
        ocr_cfg = self.cfg["plate_ocr"]
        ocr_backend = ocr_cfg.get("backend", "yolo")

        if ocr_backend == "yolo":
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
        if self._ds_face_chain:
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
        self.face_avg = EmbeddingAvg(rcfg["face_avg_frames"])  # fallback path
        self.face_thr = rcfg["face_threshold"]
        self.blur_thr = fcfg_full.get("blur_threshold", 35.0)

        # Phase 8: per-track state cho DeepStream path.
        # face_tracks[tid] = {"avg": EmbeddingAvg, "last_update": int, "n_frames": int}
        self.face_tracks: dict = {}
        self.cur_frame_id = 0
        self._face_avg_frames = rcfg["face_avg_frames"]
        self._multi_emb = bool(fcfg_full.get("multi_embedding", True))
        self._k_max = int(fcfg_full.get("k_max", 5))
        self._min_track_frames = int(fcfg_full.get("min_track_frames", 5))
        self._track_idle_drop = int(
            fcfg_full.get("track_idle_drop_frames", 30))
        self._plate_face_max_gap = int(
            fcfg_full.get("plate_face_max_gap_frames", 30))
        self._face_min_quality = float(fcfg_full.get("min_quality", 0.4))
        self._dedup_cos = float(fcfg_full.get("dedup_cos", 0.92))

        # ── Web state (shared reference với web.py) ──
        self.state = {
            "mode": "entry", "stream_fps":0, "fps": 0,
            "plate_cam_ok": False, "face_cam_ok": False,
            "deepstream": self.use_deepstream,
        }
        
        # Lưu result gần nhất để annotate frame
        self._last_result = {"ok": False}

        # ── Thread pool cho parallel OCR + face inference ──
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="infer")

        self._cached_stats = self.db.stats()
        self._face_rotate = self.cfg.get("camera", {}).get("face_rotate", -1)
        self._rot_map = {
            1: cv2.ROTATE_90_CLOCKWISE,
            2: cv2.ROTATE_180,
            3: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }

        self.running = False

    def _load_face_engine(self):
        """Lazy-load InsightFace only for fallback / non-DS face path."""
        if self.face_eng is None:
            fcfg = self.cfg["face"]
            self.face_eng = FaceEngine(
                fcfg["model_pack"], tuple(fcfg["det_size"]))
        return self.face_eng

    # ──────────────────────────────────────────────
    # Phase 8 — Per-track helpers (DeepStream path)
    # ──────────────────────────────────────────────
    def _expire_tracks(self):
        """Invariant 1 (idle expiry): drop tracks không update > N frames."""
        if not self.face_tracks:
            return
        cutoff = self.cur_frame_id - self._track_idle_drop
        expired = [tid for tid, t in self.face_tracks.items()
                   if t["last_update"] < cutoff]
        for tid in expired:
            del self.face_tracks[tid]

    def _reset_tracks(self):
        """Invariant 1 (hard reset): gọi sau entry/exit success + mode switch."""
        if self.face_tracks:
            self.face_tracks.clear()

    def _has_fresh_track(self) -> bool:
        """
        Invariant 2: tại commit time, phải có ít nhất 1 track:
          - n_frames >= min_track_frames (track "chín")
          - last_update gần đây (≤ plate_face_max_gap_frames)
        """
        cutoff = self.cur_frame_id - self._plate_face_max_gap
        for t in self.face_tracks.values():
            if (t["n_frames"] >= self._min_track_frames
                    and t["last_update"] >= cutoff
                    and t["avg"].ready):
                return True
        return False

    def _update_face_tracks(self, face_data, frame_face):
        """
        Tích lũy embedding per-track từ DeepStream face_data.

        face_data: list[{bbox, conf, track_id, embedding, landmarks}]
        Bỏ qua face không có track_id hoặc embedding (untracked / SGIE miss).
        """
        if not face_data:
            return
        for f in face_data:
            tid = f.get("track_id")
            emb = f.get("embedding")
            bbox = f.get("bbox")
            if tid is None or emb is None or bbox is None:
                continue

            # Per-frame quality gate (CPU; sẽ thay bằng landmark geometry sau MVP)
            ok, q = FaceEngine.quality(frame_face, bbox, self.blur_thr)
            if not ok or q is None or q < self._face_min_quality:
                continue

            t = self.face_tracks.get(tid)
            if t is None:
                t = {
                    "avg": EmbeddingAvg(self._face_avg_frames),
                    "last_update": self.cur_frame_id,
                    "n_frames": 0,
                }
                self.face_tracks[tid] = t

            t["avg"].update(emb, q)
            t["last_update"] = self.cur_frame_id
            t["n_frames"] += 1

    def _build_bank(self) -> list:
        """
        Build embedding bank cho entry: quality-sorted, dedup, capped to k_max.

        Lọc cùng condition với Invariant 2 (mature + fresh + ready). Không lọc
        freshness ở đây sẽ cho phép track vừa-chưa-expire (idle 25/30 frames)
        lọt vào bank → embedding của xe cũ bị gắn cho xe mới.

        Returns: list[(emb: ndarray, quality: float, track_id: int)]
        """
        cutoff = self.cur_frame_id - self._plate_face_max_gap
        candidates = [
            (tid, t["avg"]._latest, t["avg"]._latest_quality, t["n_frames"])
            for tid, t in self.face_tracks.items()
            if (t["n_frames"] >= self._min_track_frames
                and t["last_update"] >= cutoff
                and t["avg"].ready)
        ]
        candidates.sort(key=lambda x: x[2], reverse=True)
        bank = []
        for tid, emb, q, _ in candidates:
            if any(float(np.dot(emb, b[0])) > self._dedup_cos for b in bank):
                continue
            bank.append((emb, q, tid))
            if len(bank) >= self._k_max:
                break
        return bank

    def _best_track_emb(self):
        """
        Pick highest-quality MATURE + FRESH track cho exit candidate.

        Phải khớp condition với _has_fresh_track / _build_bank, nếu không gate
        có thể pass trên track B (fresh) nhưng picker chọn track A (stale,
        higher quality) → match wrong car.
        """
        cutoff = self.cur_frame_id - self._plate_face_max_gap
        eligible = [
            t for t in self.face_tracks.values()
            if (t["n_frames"] >= self._min_track_frames
                and t["last_update"] >= cutoff
                and t["avg"].ready)
        ]
        if not eligible:
            return None, 0.0
        best = max(eligible, key=lambda t: t["avg"]._latest_quality)
        return best["avg"]._latest, float(best["avg"]._latest_quality)

    # ── Parallel inference helpers ──
    def _run_ocr(self, crop):
        """OCR + validate. Thread-safe (GPU releases GIL)."""
        t0 = time.time()
        raw_text, ocr_conf = self.plate_ocr(crop)
        plate = self.validator(raw_text)
        dt = (time.time() - t0) * 1000
        return raw_text, ocr_conf, plate, dt

    def _run_face(self, frame_face):
        """Face detect + quality. Returns (best_face, quality, face_crop, dt_ms) or None."""
        t0 = time.time()
        faces = self._load_face_engine()(frame_face)
        if not faces:
            return None
        best_f = max(faces, key=lambda f: f["conf"])
        fx1, fy1, fx2, fy2 = best_f["bbox"]
        fh, fw = frame_face.shape[:2]
        face_crop = frame_face[max(0, fy1):min(fh, fy2),
                               max(0, fx1):min(fw, fx2)]
        fc = face_crop.copy() if face_crop.size > 0 else None
        _, quality = FaceEngine.quality(frame_face, best_f["bbox"], self.blur_thr)
        dt = (time.time() - t0) * 1000
        return best_f, quality, fc, dt

    # ── ENTRY ──
    def process_entry(self, frame_plate, plate_dets,
                      frame_face, face_data=None) -> dict:
        # Phase 8: DeepStream path (face_data từ pipeline). Fallback (face_data=None)
        # giữ nguyên code path cũ với face_avg singular.
        if face_data is not None and self._multi_emb:
            return self._process_entry_ds(
                frame_plate, plate_dets, frame_face, face_data)

        result = {"ok": False, "plate": "", "face_conf": 0,
                  "plate_bbox": None, "face_bbox": None}

        # 1) Plate detection
        if plate_dets is None:
            plate_dets = self.plate_det(frame_plate) if self.plate_det else []

        if not plate_dets:
            log.debug("ENTRY: no plate detected")
            return result
        best_p = max(plate_dets, key=lambda p: p["conf"])
        x1, y1, x2, y2 = best_p["bbox"]
        result["plate_bbox"] = best_p["bbox"]

        log.debug(f"ENTRY: plate det bbox=({x1},{y1},{x2},{y2}) "
                  f"conf={best_p['conf']:.2f}")

        # 2) Crop
        h, w = frame_plate.shape[:2]
        box_w, box_h = x2 - x1, y2 - y1
        mx = max(10, int(box_w * 0.08))
        my = max(8, int(box_h * 0.12))
        crop = frame_plate[max(0, y1 - my):min(h, y2 + my),
                           max(0, x1 - mx):min(w, x2 + mx)]

        log.debug(f"ENTRY CROP: frame={w}x{h} bbox=({x1},{y1},{x2},{y2}) "
                  f"margin=({mx},{my}) "
                  f"crop={crop.shape if crop.size > 0 else 'EMPTY'}")

        if crop.size > 0:
            result["plate_crop"] = crop.copy()

        # 3) OCR + Face song song (GPU releases GIL → true overlap)
        t_parallel = time.time()
        fut_ocr = self._executor.submit(self._run_ocr, crop)
        fut_face = self._executor.submit(self._run_face, frame_face)

        raw_text, ocr_conf, plate, dt_ocr = fut_ocr.result()
        face_data = fut_face.result()
        dt_parallel = (time.time() - t_parallel) * 1000

        dt_face = 0.0
        log.debug(f"ENTRY OCR: raw='{raw_text}' conf={ocr_conf:.2f} "
                  f"→ valid='{plate}'")

        # 4) Process face result — tích lũy embedding sớm
        if face_data:
            best_f, quality, face_crop, dt_face = face_data
            result["face_bbox"] = best_f["bbox"]
            result["face_conf"] = best_f["conf"]
            if face_crop is not None:
                result["face_crop"] = face_crop
            if quality is not None:
                self.face_avg.update(best_f["embedding"], quality)

        # Timing: nếu parallel hoạt động → total ≈ max(ocr, face)
        saved = dt_ocr + dt_face - dt_parallel
        log.debug(f"⏱ ENTRY ocr={dt_ocr:.0f}ms face={dt_face:.0f}ms "
                  f"parallel={dt_parallel:.0f}ms "
                  f"(saved {saved:.0f}ms)")
        self.state["timing"] = {
            "ocr_ms": round(dt_ocr, 1),
            "face_ms": round(dt_face, 1),
            "parallel_ms": round(dt_parallel, 1),
            "saved_ms": round(saved, 1),
        }

        # 5) Vote
        stable = self.plate_voter.vote(plate)
        if not stable:
            result["plate"] = plate
            log.debug(f"ENTRY: voting... buf={self.plate_voter._buf}")
            return result
        result["plate"] = stable
        log.info(f"ENTRY: plate voted → '{stable}'")

        # 6) Register — dùng embedding đã tích lũy từ các frame trước
        if not self.face_avg.ready:
            log.debug("ENTRY: plate ready nhưng chưa có face embedding")
            return result

        face_conf = result.get("face_conf", 0)
        code = self.db.entry(stable, self.face_avg._latest,
                             ocr_conf, face_conf)
        if code > 0:
            result["ok"] = True
            self.plate_voter.clear()
            self.face_avg.clear()
            log.info(f"✅ ENTRY OK: {stable} (id={code})")
            self._emit("entry", {"plate": stable}, result)
        elif code == -1:
            log.warning("❌ BÃI ĐẦY")
        elif code == -2:
            log.warning(f"❌ TRÙNG BIỂN SỐ: {stable}")

        return result

    # ── EXIT (plate-first → face verify) ──
    def process_exit(self, frame_face, frame_plate,
                     plate_dets=None, face_data=None) -> dict:
        # Phase 8: DeepStream path. Fallback giữ nguyên.
        if face_data is not None and self._multi_emb:
            return self._process_exit_ds(
                frame_face, frame_plate, plate_dets, face_data)

        result = {"ok": False, "plate": "", "sim": 0.0,
                  "face_bbox": None, "plate_bbox": None}

        # ── Bước 1: Plate detection + crop ──
        if plate_dets is None and self.plate_det:
            plate_dets = self.plate_det(frame_plate)
        if not plate_dets:
            return result

        best_p = max(plate_dets, key=lambda p: p["conf"])
        result["plate_bbox"] = best_p["bbox"]
        x1, y1, x2, y2 = best_p["bbox"]
        h, w = frame_plate.shape[:2]
        box_w, box_h = x2 - x1, y2 - y1
        mx = max(10, int(box_w * 0.08))
        my = max(8, int(box_h * 0.12))
        crop = frame_plate[max(0, y1 - my):min(h, y2 + my),
                           max(0, x1 - mx):min(w, x2 + mx)]
        if crop.size == 0:
            return result
        result["plate_crop"] = crop.copy()

        # ── Bước 2: OCR + Face song song ──
        t_parallel = time.time()
        fut_ocr = self._executor.submit(self._run_ocr, crop)
        fut_face = self._executor.submit(self._run_face, frame_face)

        raw_text, _, plate, dt_ocr = fut_ocr.result()
        face_data = fut_face.result()
        dt_parallel = (time.time() - t_parallel) * 1000

        dt_face = 0.0
        # Process face — tích lũy embedding sớm
        if face_data:
            best_f, quality, face_crop, dt_face = face_data
            result["face_bbox"] = best_f["bbox"]
            if face_crop is not None:
                result["face_crop"] = face_crop
            if quality is not None:
                self.face_avg.update(best_f["embedding"], quality)

        saved = dt_ocr + dt_face - dt_parallel
        log.debug(f"⏱ EXIT ocr={dt_ocr:.0f}ms face={dt_face:.0f}ms "
                  f"parallel={dt_parallel:.0f}ms "
                  f"(saved {saved:.0f}ms)")
        self.state["timing"] = {
            "ocr_ms": round(dt_ocr, 1),
            "face_ms": round(dt_face, 1),
            "parallel_ms": round(dt_parallel, 1),
            "saved_ms": round(saved, 1),
        }

        # ── Bước 3: Vote + tìm record ──
        exit_plate = self.plate_voter.vote(plate)
        if not exit_plate:
            return result

        record = self.db.find_by_plate(exit_plate)
        if not record:
            log.debug(f"EXIT: biển {exit_plate} không tìm thấy trong DB")
            return result

        # ── Bước 4: Verify face embedding ──
        if not self.face_avg.ready:
            return result

        emb = self.face_avg._latest
        sim = float(np.dot(emb, record["embedding"])
                    / (np.linalg.norm(emb) * np.linalg.norm(record["embedding"])
                       + 1e-8))
        if sim < self.face_thr:
            log.debug(f"EXIT: face không khớp cho {exit_plate} "
                      f"(sim={sim:.3f} < thr={self.face_thr})")
            return result

        # ── Bước 5: Xác nhận xe ra ──
        self.db.exit(record["id"], sim)
        result["ok"] = True
        result["plate"] = record["plate"]
        result["sim"] = sim
        self.plate_voter.clear()
        self.face_avg.clear()
        log.info(f"✅ EXIT: {record['plate']} (sim={sim:.3f})")
        self._emit("exit", {"plate": record["plate"],
                            "sim": sim}, result)
        return result

    # ──────────────────────────────────────────────
    # Phase 8 — DeepStream paths (per-track logic)
    # ──────────────────────────────────────────────
    def _process_entry_ds(self, frame_plate, plate_dets,
                          frame_face, face_data) -> dict:
        result = {"ok": False, "plate": "", "face_conf": 0,
                  "plate_bbox": None, "face_bbox": None}

        # Invariant 1: expire stale tracks every frame.
        self._expire_tracks()

        # Update face_tracks từ DeepStream face_data (per-track).
        # Vẫn làm dù chưa có plate — giúp track "chín" sớm hơn.
        if face_data:
            self._update_face_tracks(face_data, frame_face)
            # Best face cho display + web event (highest conf trong frame này).
            valid = [f for f in face_data
                     if f.get("bbox") is not None]
            if valid:
                best_f = max(valid, key=lambda f: float(f.get("conf", 0.0)))
                bx1, by1, bx2, by2 = best_f["bbox"]
                result["face_bbox"] = best_f["bbox"]
                result["face_conf"] = float(best_f.get("conf", 0.0))
                # Crop face từ frame_face cho web event payload.
                fh, fw = frame_face.shape[:2]
                fc = frame_face[max(0, by1):min(fh, by2),
                                max(0, bx1):min(fw, bx2)]
                if fc.size > 0:
                    result["face_crop"] = fc.copy()

        if not plate_dets:
            return result
        best_p = max(plate_dets, key=lambda p: p["conf"])
        x1, y1, x2, y2 = best_p["bbox"]
        result["plate_bbox"] = best_p["bbox"]

        # Crop plate cho OCR.
        h, w = frame_plate.shape[:2]
        box_w, box_h = x2 - x1, y2 - y1
        mx = max(10, int(box_w * 0.08))
        my = max(8, int(box_h * 0.12))
        crop = frame_plate[max(0, y1 - my):min(h, y2 + my),
                           max(0, x1 - mx):min(w, x2 + mx)]
        if crop.size == 0:
            return result
        result["plate_crop"] = crop.copy()

        # OCR (face inference đã chạy trên GPU qua DeepStream → no parallel needed).
        raw_text, ocr_conf, plate, dt_ocr = self._run_ocr(crop)
        self.state["timing"] = {"ocr_ms": round(dt_ocr, 1)}
        log.debug(f"ENTRY OCR: raw='{raw_text}' conf={ocr_conf:.2f} "
                  f"→ valid='{plate}' ({dt_ocr:.0f}ms)")

        # Vote.
        stable = self.plate_voter.vote(plate)
        if not stable:
            result["plate"] = plate
            return result
        result["plate"] = stable
        log.info(f"ENTRY: plate voted → '{stable}'")

        # Invariant 2: fresh face required at commit.
        if not self._has_fresh_track():
            log.warning(
                f"ENTRY: plate '{stable}' stable nhưng không có fresh face "
                f"track (gap > {self._plate_face_max_gap} frames) — reject")
            self.plate_voter.clear()
            return result

        # Build embedding bank.
        bank = self._build_bank()
        if not bank:
            log.debug("ENTRY: bank rỗng — chưa track nào đủ chín")
            return result

        embs = [b[0] for b in bank]
        quals = [float(b[1]) for b in bank]
        tids = [int(b[2]) for b in bank]
        face_conf_for_db = quals[0]

        code = self.db.entry(
            stable, embs, ocr_conf, face_conf_for_db,
            qualities=quals, track_ids=tids,
        )
        if code > 0:
            result["ok"] = True
            self.plate_voter.clear()
            self._reset_tracks()
            log.info(f"✅ ENTRY OK: {stable} (id={code}, "
                     f"bank={len(embs)} embeddings)")
            self._emit("entry", {"plate": stable}, result)
        elif code == -1:
            log.warning("❌ BÃI ĐẦY")
        elif code == -2:
            log.warning(f"❌ TRÙNG BIỂN SỐ: {stable}")

        return result

    def _process_exit_ds(self, frame_face, frame_plate,
                         plate_dets, face_data) -> dict:
        result = {"ok": False, "plate": "", "sim": 0.0,
                  "face_bbox": None, "plate_bbox": None}

        # Invariant 1: expire + update.
        self._expire_tracks()
        if face_data:
            self._update_face_tracks(face_data, frame_face)
            valid = [f for f in face_data if f.get("bbox") is not None]
            if valid:
                best_f = max(valid, key=lambda f: float(f.get("conf", 0.0)))
                bx1, by1, bx2, by2 = best_f["bbox"]
                result["face_bbox"] = best_f["bbox"]
                fh, fw = frame_face.shape[:2]
                fc = frame_face[max(0, by1):min(fh, by2),
                                max(0, bx1):min(fw, bx2)]
                if fc.size > 0:
                    result["face_crop"] = fc.copy()

        if not plate_dets:
            return result
        best_p = max(plate_dets, key=lambda p: p["conf"])
        result["plate_bbox"] = best_p["bbox"]
        x1, y1, x2, y2 = best_p["bbox"]
        h, w = frame_plate.shape[:2]
        box_w, box_h = x2 - x1, y2 - y1
        mx = max(10, int(box_w * 0.08))
        my = max(8, int(box_h * 0.12))
        crop = frame_plate[max(0, y1 - my):min(h, y2 + my),
                           max(0, x1 - mx):min(w, x2 + mx)]
        if crop.size == 0:
            return result
        result["plate_crop"] = crop.copy()

        raw_text, _, plate, dt_ocr = self._run_ocr(crop)
        self.state["timing"] = {"ocr_ms": round(dt_ocr, 1)}

        exit_plate = self.plate_voter.vote(plate)
        if not exit_plate:
            return result

        # Invariant 2: fresh face required at commit.
        if not self._has_fresh_track():
            log.debug(
                f"EXIT: plate '{exit_plate}' stable nhưng không có fresh face")
            return result

        # Pick best track as candidate.
        candidate_emb, cand_q = self._best_track_emb()
        if candidate_emb is None:
            return result

        match = self.db.match_exit_by_plate(
            exit_plate, candidate_emb, threshold=self.face_thr)
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
        self._reset_tracks()
        log.info(f"✅ EXIT: {match['plate']} (sim={sim:.3f}, "
                 f"bank={match['n_embeddings']})")
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
    def _annotate_plate(self, frame, result):
        if not result.get("plate_bbox"):
            return frame
        vis = frame.copy()
        x1, y1, x2, y2 = result["plate_bbox"]
        color = (0, 255, 0) if result.get("ok") else (0, 255, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        plate = result.get("plate", "")
        if plate:
            cv2.rectangle(vis, (x1, y1-28), (x1+len(plate)*16, y1),
                          (0, 0, 0), -1)
            cv2.putText(vis, plate, (x1+4, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return vis

    def _annotate_face(self, frame, result):
        if not result.get("face_bbox"):
            return frame
        vis = frame.copy()
        x1, y1, x2, y2 = result["face_bbox"]
        color = (0, 255, 0) if result.get("ok") else (0, 255, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = ""
        if result.get("ok") and result.get("sim"):
            label = f"MATCH {result['sim']:.2f}"
        elif result.get("face_conf"):
            label = f"face {result['face_conf']:.2f}"
        if label:
            cv2.rectangle(vis, (x1, y1-28), (x1+len(label)*12, y1),
                          (0, 0, 0), -1)
            cv2.putText(vis, label, (x1+4, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return vis
    
    def _web_update_loop(self, cam_plate, cam_face, interval: float = 0.1):
        from web import update_frame
        log.info("Web update thread started (fallback mode)")
        while self.running:
            t0 = time.time()
            fp = cam_plate.latest
            ff = cam_face.latest
            if fp is not None and ff is not None:
                ff_rot = self._apply_rotation(ff)
                try:
                    update_frame("plate",
                                 self._annotate_plate(fp, self._last_result))
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
            fp, _, ff, _ = ds.get_all()
            if fp is not None and ff is not None:
                try:
                    update_frame("plate",
                                 self._annotate_plate(fp, self._last_result))
                    update_frame("face",
                                 self._annotate_face(ff, self._last_result))
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
                if not ds.wait_new_frame(timeout=0.5):
                    continue

                fp, plate_dets, ff, face_data = ds.get_all()

                if fp is None or ff is None:
                    time.sleep(0.01)
                    frame_idx += 1
                    if frame_idx % 300 == 0:
                        log.warning(f"Waiting frames... "
                                    f"plate={'OK' if fp is not None else 'NONE'} "
                                    f"face={'OK' if ff is not None else 'NONE'}")
                    continue

                frame_idx += 1
                self.cur_frame_id += 1

                self.state["plate_cam_ok"] = True
                self.state["face_cam_ok"] = True

                # DeepStream face frames are already rotated at source ingress
                # via camera.face_rotate_nv. Do not rotate again here.

                t0 = time.time()

                if t0 < cooldown_until:
                    result = {"ok": False}
                elif mode == "entry":
                    result = self.process_entry(
                        fp, plate_dets, ff,
                        face_data=face_data if self._ds_face_chain else None)
                else:
                    result = self.process_exit(
                        ff, fp, plate_dets,
                        face_data=face_data if self._ds_face_chain else None)

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
                    self._show_dual(fp, ff, result, mode)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("m"):
                        mode = "exit" if mode == "entry" else "entry"
                        self.state["mode"] = mode
                        self.plate_voter.clear()
                        self.face_avg.clear()
                        self._reset_tracks()
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
                self.cur_frame_id += 1

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
                        self.face_avg.clear()
                        self._reset_tracks()
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
