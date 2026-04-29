import argparse
import logging
import signal
import time
from datetime import datetime
from pathlib import Path
from threading import Thread

import cv2

from main import ParkingSystem, start_web
from pipeline import DeepStreamPipeline, StreamReader


log = logging.getLogger("main_auto_cycle")


def build_profile_prefix(arg_value: str | None):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if arg_value is None:
        return None
    if arg_value == "":
        return str(Path("runs") / f"auto_cycle_{stamp}")

    p = Path(arg_value)
    if p.exists() and p.is_dir():
        return str(p / f"auto_cycle_{stamp}")
    if arg_value.endswith(("/", "\\")):
        return str(p / f"auto_cycle_{stamp}")
    if p.suffix.lower() == ".csv":
        p = p.with_suffix("")
    return str(p)


class AutoCycleParkingSystem(ParkingSystem):
    """
    Chạy một tiến trình duy nhất theo chu kỳ entry -> exit -> entry.
    Dùng cho thử nghiệm khi chỉ có một người/xe và không chạy riêng 2 mode.
    """

    def __init__(self, cfg_path: str = "config.yaml",
                 max_cycles: int = 0, switch_delay: float = 0.5,
                 start_mode: str = "entry"):
        super().__init__(cfg_path)
        self.max_cycles = max(0, int(max_cycles))
        self.switch_delay = max(0.0, float(switch_delay))
        if start_mode not in {"entry", "exit"}:
            raise ValueError("start_mode must be 'entry' or 'exit'")
        self.start_mode = start_mode
        self._completed_cycles = 0
        self._stop_reason = "not started"

    def _switch_after_success(self, mode: str, result: dict):
        if not result.get("ok"):
            return mode, False

        if mode == "entry":
            next_mode = "exit"
            log.info("AUTO: ENTRY OK -> switch to EXIT")
        else:
            self._completed_cycles += 1
            next_mode = "entry"
            log.info(
                f"AUTO: EXIT OK -> switch to ENTRY "
                f"(cycles={self._completed_cycles})")

        self.state["mode"] = next_mode
        self.plate_voter.clear()
        self.face_avg.clear()
        if self.max_cycles and self._completed_cycles >= self.max_cycles:
            self._stop_reason = f"completed {self._completed_cycles} cycles"
            log.info(f"AUTO: reached {self.max_cycles} cycles -> stop")
            self.running = False
        return next_mode, True

    def _run_deepstream_auto(self, show: bool):
        ccfg = self.cfg["camera"]
        ds = DeepStreamPipeline(ccfg["plate"], ccfg["face"], self.cfg)
        ds.start()

        web_thread = Thread(
            target=self._web_update_loop_ds,
            args=(ds,),
            daemon=True,
            name="web-update-ds")
        web_thread.start()

        mode = self.start_mode
        self.state["mode"] = mode
        frame_idx = 0
        t_fps, n_fps = time.time(), 0
        cooldown_until = 0.0

        log.info(f"Auto-cycle DeepStream mode started (start={mode})")

        try:
            while self.running:
                if not ds.wait_new_frame(timeout=0.5):
                    continue

                fp, plate_dets, ff = ds.get_all()
                if fp is None or ff is None:
                    time.sleep(0.01)
                    frame_idx += 1
                    if frame_idx % 300 == 0:
                        log.warning(
                            "Waiting frames... "
                            f"plate={'OK' if fp is not None else 'NONE'} "
                            f"face={'OK' if ff is not None else 'NONE'}")
                    continue

                frame_idx += 1
                self.state["plate_cam_ok"] = True
                self.state["face_cam_ok"] = True
                ff = self._rotate_face(ff)

                probe_to_app_ms = None
                if ds.last_plate_frame_ts > 0:
                    probe_to_app_ms = (
                        time.time() - ds.last_plate_frame_ts) * 1000

                t0 = time.time()
                processed = False
                if t0 < cooldown_until:
                    result = {"ok": False}
                elif mode == "entry":
                    processed = True
                    result = self.process_entry(fp, plate_dets, ff)
                else:
                    processed = True
                    result = self.process_exit(ff, fp, plate_dets)

                if processed:
                    loop_ms = (time.time() - t0) * 1000
                    self._record_runtime_sample(
                        mode, result, loop_ms,
                        {"probe_to_app_ms": probe_to_app_ms})

                mode, switched = self._switch_after_success(mode, result)
                if switched:
                    cooldown_until = time.time() + self.switch_delay

                self._last_result = result
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
                        self._stop_reason = "user pressed q"
                        break

        except KeyboardInterrupt:
            self._stop_reason = "keyboard interrupt"
            pass
        finally:
            ds.stop()

    def _run_fallback_auto(self, show: bool):
        ccfg = self.cfg["camera"]
        cam_plate = StreamReader(
            ccfg["plate"], name="plate",
            hw_decode=ccfg["hw_decode"],
            reconnect_sec=ccfg.get("reconnect_sec", 3))
        cam_face = StreamReader(
            ccfg["face"], name="face",
            hw_decode=ccfg["hw_decode"],
            reconnect_sec=ccfg.get("reconnect_sec", 3))

        web_thread = Thread(
            target=self._web_update_loop,
            args=(cam_plate, cam_face),
            daemon=True,
            name="web-update")
        web_thread.start()

        mode = self.start_mode
        self.state["mode"] = mode
        skip_n = self.cfg["camera"]["process_every_n"]
        frame_idx = 0
        t_fps, n_fps = time.time(), 0
        cooldown_until = 0.0

        log.info(f"Auto-cycle fallback mode started (start={mode})")

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
                    self._stop_reason = "input stream ended"
                    break

                frame_idx += 1
                if skip_n > 1 and frame_idx % skip_n != 0:
                    continue

                ff = self._rotate_face(ff)

                t0 = time.time()
                processed = False
                if t0 < cooldown_until:
                    result = {"ok": False}
                elif mode == "entry":
                    processed = True
                    result = self.process_entry(fp, None, ff)
                else:
                    processed = True
                    result = self.process_exit(ff, fp)

                if processed:
                    loop_ms = (time.time() - t0) * 1000
                    self._record_runtime_sample(mode, result, loop_ms)

                mode, switched = self._switch_after_success(mode, result)
                if switched:
                    cooldown_until = time.time() + self.switch_delay

                self._last_result = result
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
                        self._stop_reason = "user pressed q"
                        break

        except KeyboardInterrupt:
            self._stop_reason = "keyboard interrupt"
            pass
        finally:
            cam_plate.release()
            cam_face.release()

    def run_auto(self, show: bool = True):
        self.running = True
        self._stop_reason = "running"
        self.state["mode"] = self.start_mode
        log.info(f"Starting auto-cycle (deepstream={self.use_deepstream})")
        log.info(f"DB: {self.db.stats()}")

        try:
            if self.use_deepstream:
                self._run_deepstream_auto(show)
            else:
                self._run_fallback_auto(show)
        except Exception:
            self._stop_reason = "unhandled exception"
            log.exception("Auto-cycle stopped by an unhandled error")
            raise
        finally:
            self.running = False
            log.info(f"Stop reason: {self._stop_reason}")
            self._log_runtime_profile()
            self._export_runtime_profile()
            self._executor.shutdown(wait=False)
            self.db.close()
            cv2.destroyAllWindows()
            log.info(f"Done. {self.db.stats()}")


def main():
    parser = argparse.ArgumentParser(
        description="Smart Parking auto-cycle: entry -> exit -> entry")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Số chu kỳ entry+exit cần chạy; 0 = chạy mãi.")
    parser.add_argument("--start-mode", choices=["entry", "exit"],
                        default="entry",
                        help="Mode khởi đầu. Dùng exit nếu xe đang còn active.")
    parser.add_argument("--switch-delay", type=float, default=0.5,
                        help="Thời gian nghỉ sau khi đổi mode.")
    parser.add_argument(
        "--profile-csv",
        nargs="?",
        const="",
        default=None,
        help=("Export runtime profiler CSV. "
              "Use no value for auto path under runs/."))
    parser.add_argument("--debug", action="store_true",
                        help="Bật debug logging chi tiết")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    system = AutoCycleParkingSystem(
        args.config,
        max_cycles=args.cycles,
        switch_delay=args.switch_delay,
        start_mode=args.start_mode)
    system.profile_csv_prefix = build_profile_prefix(args.profile_csv)

    def sig_handler(s, f):
        system._stop_reason = "signal interrupt"
        system.running = False
    signal.signal(signal.SIGINT, sig_handler)

    if not args.no_web and system.cfg["web"]["enabled"]:
        web_thread = Thread(
            target=start_web,
            args=(system.cfg, system.db, system.state),
            daemon=True)
        web_thread.start()
        time.sleep(1)

    system.run_auto(show=not args.no_show)


if __name__ == "__main__":
    main()
