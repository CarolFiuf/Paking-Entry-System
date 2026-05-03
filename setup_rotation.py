"""
setup_rotation.py — auto-detect face camera rotation (run once at setup).

Connects to face cam, grabs frames, tries 4 orientations with FaceEngine.
Writes the detected rotation back to config.yaml.

  - cv2 code: 0=no, 1=90CW, 2=180, 3=90CCW
  - nvvidconv flip-method: 0=identity, 1=90CCW, 2=180, 3=90CW
    (yes, cv2 and nvvidconv use INVERTED 90° codes)

Usage:
  python setup_rotation.py [--config config.yaml] [--frames 5] [--write]
"""

import argparse
import logging
import sys
import time

import cv2
import yaml

from engine import FaceEngine
from pipeline import StreamReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("setup_rotation")

# cv2 → nvvidconv mapping
CV2_TO_NV = {0: 0, 1: 3, 2: 2, 3: 1}

ROTATIONS = [
    (0, None,                            "no rotation"),
    (1, cv2.ROTATE_90_CLOCKWISE,         "90 CW"),
    (3, cv2.ROTATE_90_COUNTERCLOCKWISE,  "90 CCW"),
    (2, cv2.ROTATE_180,                  "180"),
]


def detect_rotation(face_eng: FaceEngine, frame, max_attempts: int = 1) -> int:
    """Return cv2 rotation code (0-3) or -1 if no orientation detects faces."""
    for code, rot, name in ROTATIONS:
        test = frame if rot is None else cv2.rotate(frame, rot)
        faces = face_eng(test)
        if faces:
            log.info(f"  ✅ {name} (cv2={code}): "
                     f"{len(faces)} face(s) detected")
            return code
        else:
            log.debug(f"  ✗ {name} (cv2={code}): no face")
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--frames", type=int, default=5,
                    help="number of frames to test (vote majority)")
    ap.add_argument("--write", action="store_true",
                    help="update config.yaml with detected rotation")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    face_url = cfg["camera"]["face"]
    log.info(f"Connecting to face cam: {face_url}")

    face_eng = FaceEngine(cfg.get("face", {}))
    reader = StreamReader(face_url, name="face_setup",
                          hw_decode=cfg["camera"].get("hw_decode", True))

    # Wait for first frame
    deadline = time.time() + 15
    frame = None
    while time.time() < deadline:
        frame = reader.read(timeout=2.0)
        if frame is not None:
            break
    if frame is None:
        log.error("Cannot read frame from face cam after 15s")
        reader.release()
        sys.exit(2)

    log.info(f"Got first frame: {frame.shape}, sampling {args.frames} more...")

    # Vote across N frames
    votes = []
    for i in range(args.frames):
        f = reader.read(timeout=2.0)
        if f is None:
            continue
        log.info(f"Frame {i + 1}/{args.frames}:")
        code = detect_rotation(face_eng, f)
        if code >= 0:
            votes.append(code)

    reader.release()

    if not votes:
        log.error("No orientation produced face detections — check camera "
                  "angle, lighting, and face_eng config")
        sys.exit(3)

    # Majority vote
    from collections import Counter
    cv2_code, count = Counter(votes).most_common(1)[0]
    nv_code = CV2_TO_NV[cv2_code]
    name = next(n for c, _, n in ROTATIONS if c == cv2_code)

    log.info(f"\n=== Detected ===\n"
             f"cv2 code:        {cv2_code}  ({name})\n"
             f"nvvidconv code:  {nv_code}\n"
             f"votes:           {count}/{len(votes)}")

    if args.write:
        cfg.setdefault("camera", {})["face_rotate"] = cv2_code
        cfg["camera"]["face_rotate_nv"] = nv_code
        with open(args.config, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        log.info(f"Wrote camera.face_rotate={cv2_code}, "
                 f"camera.face_rotate_nv={nv_code} → {args.config}")
    else:
        log.info("Run with --write to update config.yaml")


if __name__ == "__main__":
    main()
