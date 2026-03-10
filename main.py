#!/usr/bin/env python3
"""
Smart Parking System
DeepStream pipeline (hoặc GStreamer fallback) + FastAPI web dashboard.

Usage:
  python main.py --entry            # Entry mode
  python main.py --exit             # Exit mode
  python main.py --benchmark vid.mp4
  python main.py --entry --no-web   # Không chạy web dashboard
"""

import cv2
import numpy as np
import yaml
import time
import re
import signal
import logging
import argparse
from collections import Counter
from threading import Thread

from engine import PlateDetector, PlateOCR, FaceEngine
from database import ParkingDB, DIM
from pipeline import HAS_DEEPSTREAM, DeepStreamPipeline, StreamReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ──────────────────────────────────────────────
# Helpers (compact — không thay đổi logic)
# ──────────────────────────────────────────────
class PlateValidator:
    def __init__(self, regex_str: str):
        self.regex = re.compile(regex_str)
        self._fix = str.maketrans("OI", "01")

    def __call__(self, text: str) -> str:
        t = text.strip().upper().translate(self._fix)
        t = t.replace(" ", "").replace("_", "-")
        return t if self.regex.match(t) else ""


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
    def __init__(self, n=3):
        self.n = n
        self._buf = []

    def update(self, emb):
        self._buf.append(emb)
        if len(self._buf) > self.n:
            self._buf.pop(0)
        avg = np.mean(self._buf, axis=0)
        n = np.linalg.norm(avg)
        return avg / n if n > 0 else avg

    def clear(self):
        self._buf.clear()


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

        # PlateDetector chỉ cần cho fallback mode
        self.use_deepstream = (self.cfg["deepstream"]["enabled"]
                               and HAS_DEEPSTREAM)
        if not self.use_deepstream:
            pcfg = self.cfg["plate_detector"]
            self.plate_det = PlateDetector(
                pcfg["model"], pcfg["imgsz"], pcfg["conf"], pcfg["device"])
        else:
            self.plate_det = None
            log.info("PlateDetector skipped (DeepStream nvinfer)")

        self.plate_ocr = PlateOCR(
            self.cfg["plate_ocr"]["lang"], self.cfg["plate_ocr"]["use_gpu"])

        fcfg = self.cfg["face"]
        self.face_eng = FaceEngine(
            fcfg["model_pack"], tuple(fcfg["det_size"]))

        log.info(f"Models loaded in {time.time()-t0:.1f}s "
                 f"(DeepStream={'ON' if self.use_deepstream else 'OFF'})")

        # ── Database ──
        dcfg = self.cfg["database"]
        self.db = ParkingDB(
            host=dcfg["host"], port=dcfg["port"],
            dbname=dcfg["dbname"], user=dcfg["user"],
            password=dcfg["password"], max_cap=dcfg["max_capacity"])

        # ── Recognition helpers ──
        rcfg = self.cfg["recognition"]
        self.validator = PlateValidator(rcfg["plate_regex"])
        self.plate_voter = PlateVoter(rcfg["plate_vote_frames"])
        self.face_avg = EmbeddingAvg(rcfg["face_avg_frames"])
        self.face_thr = rcfg["face_threshold"]

        # ── Web state (shared with web.py) ──
        self.state = {
            "mode": "entry", "fps": 0,
            "plate_cam_ok": False, "face_cam_ok": False,
            "deepstream": self.use_deepstream,
        }
        self.running = False

    # ── ENTRY ──
    def process_entry(self, frame_plate, plate_dets,
                      frame_face) -> dict:
        """
        Entry flow.
        plate_dets: từ DeepStream nvinfer hoặc self.plate_det()
        """
        result = {"ok": False, "plate": "", "face_conf": 0,
                  "plate_bbox": None, "face_bbox": None}

        # 1) Plate detection
        if plate_dets is None:
            # Fallback mode: detect bằng Python
            plate_dets = self.plate_det(frame_plate) if self.plate_det else []

        if not plate_dets:
            return result
        best_p = max(plate_dets, key=lambda p: p["conf"])
        x1, y1, x2, y2 = best_p["bbox"]
        result["plate_bbox"] = best_p["bbox"]

        # 2) Crop + OCR
        h, w = frame_plate.shape[:2]
        m = 5
        crop = frame_plate[max(0, y1-m):min(h, y2+m),
                           max(0, x1-m):min(w, x2+m)]
        raw_text, ocr_conf = self.plate_ocr(crop)
        plate = self.validator(raw_text)

        # 3) Vote
        stable = self.plate_voter.vote(plate)
        if not stable:
            result["plate"] = plate
            return result
        result["plate"] = stable

        # 4) Face (luôn dùng insightface — detect+align+embed 1 call)
        faces = self.face_eng(frame_face)
        if not faces:
            return result
        best_f = max(faces, key=lambda f: f["conf"])
        result["face_bbox"] = best_f["bbox"]
        result["face_conf"] = best_f["conf"]

        if not FaceEngine.quality_ok(frame_face, best_f["bbox"]):
            return result

        emb = self.face_avg.update(best_f["embedding"])

        # 5) Register
        code = self.db.entry(stable, emb, ocr_conf, best_f["conf"])
        if code > 0:
            result["ok"] = True
            self.plate_voter.clear()
            self.face_avg.clear()
            log.info(f"✅ ENTRY: {stable}")
            self._emit("entry", {"plate": stable})
        elif code == -1:
            log.warning("BÃI ĐẦY")
        elif code == -2:
            log.warning(f"TRÙNG: {stable}")

        return result

    # ── EXIT ──
    def process_exit(self, frame_face, frame_plate,
                     plate_dets=None) -> dict:
        result = {"ok": False, "plate": "", "sim": 0.0,
                  "face_bbox": None, "plate_bbox": None}

        faces = self.face_eng(frame_face)
        if not faces:
            return result
        best_f = max(faces, key=lambda f: f["conf"])
        result["face_bbox"] = best_f["bbox"]

        if not FaceEngine.quality_ok(frame_face, best_f["bbox"]):
            return result

        emb = self.face_avg.update(best_f["embedding"])
        match = self.db.match_exit(emb, self.face_thr)
        if not match:
            return result

        # Verify plate
        if plate_dets is None and self.plate_det:
            plate_dets = self.plate_det(frame_plate)
        if plate_dets:
            best_p = max(plate_dets, key=lambda p: p["conf"])
            result["plate_bbox"] = best_p["bbox"]
            x1, y1, x2, y2 = best_p["bbox"]
            h, w = frame_plate.shape[:2]
            crop = frame_plate[max(0, y1-5):min(h, y2+5),
                               max(0, x1-5):min(w, x2+5)]
            raw_text, _ = self.plate_ocr(crop)
            exit_plate = self.validator(raw_text)
            if exit_plate and exit_plate != match["plate"]:
                log.warning(
                    f"Face→{match['plate']} but camera→{exit_plate}")

        self.db.exit(match["id"], match["sim"])
        result["ok"] = True
        result["plate"] = match["plate"]
        result["sim"] = match["sim"]
        self.face_avg.clear()
        log.info(f"✅ EXIT: {match['plate']} (sim={match['sim']:.3f})")
        self._emit("exit", {"plate": match["plate"],
                            "sim": match["sim"]})
        return result

    def _emit(self, event_type: str, data: dict):
        """Push event ra web dashboard."""
        try:
            from web import notify_sync
            notify_sync({"type": event_type, "data": data})
        except Exception:
            pass

    # ── RUN (DEEPSTREAM MODE) ──
    def _run_deepstream(self, mode: str, show: bool):
        """Main loop với DeepStream pipeline."""
        ccfg = self.cfg["camera"]
        ds = DeepStreamPipeline(ccfg["plate"], ccfg["face"], self.cfg)
        ds.start()

        skip_n = ccfg["process_every_n"]
        frame_idx = 0
        fps, t_fps, n_fps = 0.0, time.time(), 0
        cooldown_until = 0.0

        log.info(f"DeepStream mode={mode}")

        try:
            while self.running:
                # Lấy data từ DeepStream probe
                fp, plate_dets = ds.get_plate_data()
                ff = ds.get_face_frame()

                if fp is None or ff is None:
                    time.sleep(0.01)
                    if frame_idx % 300 == 0:
                        log.warning(f"Waiting frames... plate={fp is not None} face={ff is not None}")
                    frame_idx += 1
                    continue

                frame_idx += 1
                if skip_n > 1 and frame_idx % skip_n != 0:
                    continue

                t0 = time.time()
                self.state["plate_cam_ok"] = True
                self.state["face_cam_ok"] = True
                # log.info(f"Frame {frame_idx}: plate={fp.shape} face={ff.shape} "
                #          f"dets={len(plate_dets)}")

                if t0 < cooldown_until:
                    result = {"ok": False}
                elif mode == "entry":
                    result = self.process_entry(fp, plate_dets, ff)
                else:
                    result = self.process_exit(ff, fp, plate_dets)

                if result.get("ok"):
                        # Feed frames cho web MJPEG stream
                    cooldown_until = t0 + 2.0

                n_fps += 1
                if time.time() - t_fps >= 1.0:
                    fps = n_fps / (time.time() - t_fps)
                    n_fps, t_fps = 0, time.time()
                    self.state["fps"] = round(fps, 1)
                    
                # Feed frames cho web stream (MỌI frame, không chỉ khi match)
                # Feed frames cho web stream — có vẽ kết quả detect
                # Feed frames cho web stream (MỌI frame, không chỉ khi match)
                try:
                    from web import update_frame
                    update_frame("plate", fp)
                    update_frame("face", ff)
                except Exception:
                    pass

                if show:
                    self._show_dual(fp, ff, result, mode, fps)

                # Toggle mode
                if show:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("m"):
                        mode = "exit" if mode == "entry" else "entry"
                        self.state["mode"] = mode
                        self.plate_voter.clear()
                        self.face_avg.clear()
                        log.info(f"Mode → {mode.upper()}")

        except KeyboardInterrupt:
            pass
        finally:
            ds.stop()

    # ── RUN (FALLBACK MODE) ──
    def _run_fallback(self, mode: str, show: bool):
        """Main loop với GStreamer StreamReader + Python inference."""
        ccfg = self.cfg["camera"]
        hw = ccfg["hw_decode"]
        reconn = ccfg.get("reconnect_sec", 3)

        cam_plate = StreamReader(ccfg["plate"], name="plate",
                                 hw_decode=hw, reconnect_sec=reconn)
        cam_face = StreamReader(ccfg["face"], name="face",
                                hw_decode=hw, reconnect_sec=reconn)

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

                t0 = time.time()

                if t0 < cooldown_until:
                    result = {"ok": False}
                elif mode == "entry":
                    result = self.process_entry(fp, None, ff)
                else:
                    result = self.process_exit(ff, fp)

                if result.get("ok"):
                    # Feed frames cho web MJPEG stream
                    cooldown_until = t0 + 2.0

                n_fps += 1
                if time.time() - t_fps >= 1.0:
                    fps = n_fps / (time.time() - t_fps)
                    n_fps, t_fps = 0, time.time()
                    self.state["fps"] = round(fps, 1)
                    
                # Feed frames cho web stream (MỌI frame, không chỉ khi match)
                # Feed frames cho web stream — có vẽ kết quả detect
                # Feed frames cho web stream (MỌI frame, không chỉ khi match)
                try:
                    from web import update_frame
                    update_frame("plate", fp)
                    update_frame("face", ff)
                except Exception:
                    pass

                if show:
                    self._show_dual(fp, ff, result, mode, fps)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("m"):
                        mode = "exit" if mode == "entry" else "entry"
                        self.state["mode"] = mode
                        self.plate_voter.clear()
                        self.face_avg.clear()
                        log.info(f"Mode → {mode.upper()}")

        except KeyboardInterrupt:
            pass
        finally:
            cam_plate.release()
            cam_face.release()
            
    def _annotate_plate(self, frame, result):
        """Vẽ plate bbox + text lên frame cho web stream."""
        vis = frame.copy()
        if result.get("plate_bbox"):
            x1, y1, x2, y2 = result["plate_bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            plate = result.get("plate", "")
            if plate:
                # Nền đen cho text dễ đọc
                cv2.rectangle(vis, (x1, y1-28), (x1+len(plate)*16, y1),
                              (0, 0, 0), -1)
                cv2.putText(vis, plate, (x1+4, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return vis

    def _annotate_face(self, frame, result):
        """Vẽ face bbox + similarity lên frame cho web stream."""
        vis = frame.copy()
        if result.get("face_bbox"):
            x1, y1, x2, y2 = result["face_bbox"]
            # Xanh lá khi match, vàng khi đang scan
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

    def _show_dual(self, fp, ff, result, mode, fps):
        """Hiển thị 2 cam cạnh nhau (cho --show mode)."""
        stats = self.db.stats()
        h = 360
        p = cv2.resize(fp, (int(fp.shape[1]*h/fp.shape[0]), h))
        f = cv2.resize(ff, (int(ff.shape[1]*h/ff.shape[0]), h))

        # Draw bboxes
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

        # Status bars
        color = (0, 255, 0) if result.get("ok") else (100, 100, 100)
        for img, label in [(p, "PLATE"), (f, "FACE")]:
            w = img.shape[1]
            cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
            info = f"{label}|{mode.upper()} FPS:{fps:.0f} " \
                   f"Lot:{stats['current']}/{stats['capacity']}"
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
            self.db.close()
            cv2.destroyAllWindows()
            log.info(f"Done. {self.db.stats()}")


# ──────────────────────────────────────────────
# Web server launcher
# ──────────────────────────────────────────────
def start_web(cfg: dict, db, state: dict):
    """Chạy FastAPI trên thread riêng."""
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
    args = parser.parse_args()

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
        time.sleep(1)  # Đợi server ready

    if args.benchmark:
        # Benchmark dùng fallback mode
        reader = StreamReader(args.benchmark, name="bench",
                              hw_decode=system.cfg["camera"]["hw_decode"])
        times = {"plate_det": [], "ocr": [], "face": [],
                 "db": [], "total": []}
        dummy_emb = np.random.randn(DIM).astype(np.float32)
        dummy_emb /= np.linalg.norm(dummy_emb)
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
            system.db.match_exit(dummy_emb, system.face_thr)
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
