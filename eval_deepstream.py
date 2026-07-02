#!/usr/bin/env python3
"""
Đánh giá các component DeepStream THẬT (TensorRT FP16 + plugin nvdsscrfddec /
nvdsfaceembed + OCR SGIE + NvDCF), thay vì đường fallback engine.py mà
component_benchmark.ipynb đang đo.

Cách hoạt động: đưa một danh sách ảnh có ground-truth qua đúng graph DeepStream
(không qua RTMP), probe ở fakesink dump metadata per-frame ra JSONL, rồi tính
metric so sánh được với notebook.

    python eval_deepstream.py face  --limit 400      # LFW: det rate, sim, top-1
    python eval_deepstream.py plate --limit 300      # plate det IoU + OCR acc

Frame↔file mapping: ảnh được copy/re-encode về <seqdir>/%06d.jpg theo đúng thứ
tự manifest; multifilesrc phát tuần tự nên frame thứ N = manifest[N].
"""
import os
import sys
import glob
import json
import time
import random
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402

# Tái dùng meta-struct + hàm OCR module-level từ pipeline.py (KHÔNG kéo theo
# engine.py / insightface). Đây là đúng struct mà plugin C++ ghi ra.
from pipeline import (  # noqa: E402
    DeepStreamPipeline,
    _get_face_embed_meta_type,
    _get_face_landmarks_meta_type,
    _FaceEmbeddingMeta,
    _FaceLandmarksMeta,
    _FACE_EMBED_META_VERSION,
    _FACE_EMB_FLAG_VALID,
    _FACE_LANDMARKS_META_VERSION,
    _FACE_LM_FLAG_VALID,
    _OCR_CLASS_CHARS,
    _MAX_PLATE_CHARS,
    _sort_chars_for_plate,
    _enforce_plate_format,
    _UNTRACKED_OBJECT_ID,
)
import pyds  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get(
    "EVAL_SCRATCH", "/tmp/claude-1000/-home-somethink-parking-system/"
    "72d218ef-276b-45f3-90cb-a3373fcdd2fc/scratchpad"))
SCRATCH.mkdir(parents=True, exist_ok=True)

MUX_W, MUX_H = 1280, 720
FACE_GIE_UID = 2
PLATE_GIE_UID = 1
OCR_SGIE_UID = 3


# ───────────────────────── seq dir / manifest ──────────────────────────
def stage_images(paths, seqdir):
    """Re-encode danh sách ảnh về <seqdir>/%06d.jpg theo thứ tự. Trả manifest
    = list path gốc (index = frame index). Re-encode (không symlink) để chuẩn
    hoá format + đảm bảo jpegparse đọc được mọi ảnh."""
    seqdir = Path(seqdir)
    if seqdir.exists():
        for f in seqdir.glob("*.jpg"):
            f.unlink()
    seqdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        idx = len(manifest)
        cv2.imwrite(str(seqdir / f"{idx:06d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 98])
        manifest.append(str(p))
    return manifest


# ───────────────────────── graph builders ──────────────────────────────
def _mk(factory, name, pipeline):
    e = Gst.ElementFactory.make(factory, name)
    if not e:
        raise RuntimeError(f"Không tạo được element {factory} ({name})")
    pipeline.add(e)
    return e


def _add_image_source(pipeline, seqdir, mux, n_images):
    """multifilesrc(%06d.jpg) → jpegparse → jpegdec → nvvideoconvert(NVMM) →
    mux.sink_0. jpegdec (CPU) + nvvideoconvert upload — detector vẫn nhận NVMM
    NV12 ở mux size như production."""
    src = _mk("multifilesrc", "imgsrc", pipeline)
    src.set_property("location", str(Path(seqdir) / "%06d.jpg"))
    src.set_property("index", 0)
    src.set_property("stop-index", n_images - 1)
    src.set_property("loop", False)
    src.set_property("caps", Gst.Caps.from_string(
        "image/jpeg,framerate=30/1"))

    parse = _mk("jpegparse", "imgparse", pipeline)
    dec = _mk("jpegdec", "imgdec", pipeline)
    conv = _mk("nvvideoconvert", "img_nvconv", pipeline)
    caps = _mk("capsfilter", "img_caps", pipeline)
    caps.set_property("caps", Gst.Caps.from_string(
        "video/x-raw(memory:NVMM),format=NV12"))

    for a, b in [(src, parse), (parse, dec), (dec, conv), (conv, caps)]:
        if not a.link(b):
            raise RuntimeError(f"link {a.get_name()}→{b.get_name()} fail")
    sink = mux.get_request_pad("sink_0")
    if not caps.get_static_pad("src").link(sink) == Gst.PadLinkReturn.OK:
        raise RuntimeError("link img_caps→mux.sink_0 fail")


def _make_mux(pipeline):
    mux = _mk("nvstreammux", "mux", pipeline)
    mux.set_property("batch-size", 1)
    mux.set_property("width", MUX_W)
    mux.set_property("height", MUX_H)
    mux.set_property("batched-push-timeout", 40000)
    mux.set_property("live-source", 0)
    return mux


def build_face_pipeline(seqdir, n_images, cfg):
    ds = cfg["deepstream"]
    pipeline = Gst.Pipeline.new("eval-face")
    mux = _make_mux(pipeline)
    _add_image_source(pipeline, seqdir, mux, n_images)

    pgie = _mk("nvinfer", "face_det", pipeline)
    pgie.set_property("config-file-path", os.path.abspath(ds["face_det_config"]))

    dec = _mk("nvdsscrfddec", "face_scrfd_dec", pipeline)
    dec.set_property("face-gie-id", FACE_GIE_UID)
    dec.set_property("source-id", 0)
    dec.set_property("net-width", 640)
    dec.set_property("net-height", 640)
    thr = ds.get("face_embed_decode_conf_threshold")
    if thr is not None:
        dec.set_property("decode-conf-threshold", float(thr))
    dec.set_property("input-object-min-width", 32)
    dec.set_property("input-object-min-height", 32)

    conv = _mk("nvvideoconvert", "face_nvconv", pipeline)
    caps = _mk("capsfilter", "face_caps", pipeline)
    caps.set_property("caps", Gst.Caps.from_string(
        "video/x-raw(memory:NVMM),format=RGBA"))

    tracker = _mk("nvtracker", "face_tracker", pipeline)
    tracker.set_property(
        "ll-lib-file",
        "/opt/nvidia/deepstream/deepstream/lib/"
        "libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", os.path.abspath(
        ds.get("face_tracker_config", "./configs/tracker_face_nvdcf.yml")))
    tracker.set_property("tracker-width", int(ds.get("face_tracker_width", 960)))
    tracker.set_property("tracker-height", int(ds.get("face_tracker_height", 544)))
    tracker.set_property("compute-hw", 1)
    tracker.set_property("gpu-id", 0)
    tracker.set_property("display-tracking-id", 0)

    emb = _mk("nvdsfaceembed", "face_embed", pipeline)
    emb.set_property("engine-file", os.path.abspath(
        ds.get("face_embed_engine", "./models/face_embed_arcface_fp16.engine")))
    emb.set_property("gpu-id", 0)
    emb.set_property("unique-id", 7)
    emb.set_property("face-gie-id", FACE_GIE_UID)
    emb.set_property("source-id", 0)
    emb.set_property("batch-size", int(ds.get("face_embed_batch_size", 8)))
    emb.set_property("align-on-gpu", bool(ds.get("face_embed_align_on_gpu", True)))
    emb.set_property("allow-cpu-fallback",
                     bool(ds.get("face_embed_allow_cpu_fallback", True)))
    # Eval: KHÔNG gate quality, KHÔNG skip interval — embed mọi ảnh để đo full.
    emb.set_property("min-quality", 0.0)
    emb.set_property("blur-threshold",
                     float(cfg.get("face", {}).get("blur_threshold", 10.0)))
    emb.set_property("interval", 0)

    sink = _mk("fakesink", "sink", pipeline)
    sink.set_property("sync", False)
    sink.set_property("async", False)

    for a, b in [(mux, pgie), (pgie, dec), (dec, conv), (conv, caps),
                 (caps, tracker), (tracker, emb), (emb, sink)]:
        if not a.link(b):
            raise RuntimeError(f"link {a.get_name()}→{b.get_name()} fail")
    return pipeline, sink


def build_ocr_pipeline(seqdir, n_images, cfg):
    """OCR-only graph để đo riêng component OCR trên crop biển số có GT text.
    Chạy chính engine plate_ocr như PGIE (process-mode=1) trực tiếp trên crop —
    decouple khỏi lỗi detect, đúng cách notebook fallback đo (PlateOCRYolo trên
    crop). gie-unique-id giữ =3 nên combine probe (OCR_SGIE_UID) khớp."""
    ds = cfg["deepstream"]
    pipeline = Gst.Pipeline.new("eval-ocr")
    mux = _make_mux(pipeline)
    _add_image_source(pipeline, seqdir, mux, n_images)

    ocr = _mk("nvinfer", "plate_ocr_pgie", pipeline)
    ocr.set_property("config-file-path", os.path.abspath(ds["plate_ocr_config"]))
    ocr.set_property("process-mode", 1)   # override SGIE→PGIE

    sink = _mk("fakesink", "sink", pipeline)
    sink.set_property("sync", False)
    sink.set_property("async", False)

    for a, b in [(mux, ocr), (ocr, sink)]:
        if not a.link(b):
            raise RuntimeError(f"link {a.get_name()}→{b.get_name()} fail")
    return pipeline, sink


def build_plate_pipeline(seqdir, n_images, cfg):
    ds = cfg["deepstream"]
    pipeline = Gst.Pipeline.new("eval-plate")
    mux = _make_mux(pipeline)
    _add_image_source(pipeline, seqdir, mux, n_images)

    pgie = _mk("nvinfer", "plate_det", pipeline)
    pgie.set_property("config-file-path", os.path.abspath(ds["plate_config"]))

    sgie = _mk("nvinfer", "plate_ocr_sgie", pipeline)
    sgie.set_property("config-file-path", os.path.abspath(ds["plate_ocr_config"]))

    conv = _mk("nvvideoconvert", "plate_nvconv", pipeline)
    caps = _mk("capsfilter", "plate_caps", pipeline)
    caps.set_property("caps", Gst.Caps.from_string(
        "video/x-raw(memory:NVMM),format=RGBA"))

    sink = _mk("fakesink", "sink", pipeline)
    sink.set_property("sync", False)
    sink.set_property("async", False)

    for a, b in [(mux, pgie), (pgie, sgie), (sgie, conv), (conv, caps),
                 (caps, sink)]:
        if not a.link(b):
            raise RuntimeError(f"link {a.get_name()}→{b.get_name()} fail")
    return pipeline, sgie, sink


# ───────────────────────── probes (dump) ───────────────────────────────
class Dumper:
    """Đếm frame theo thứ tự buffer (= manifest index) và lưu records."""
    def __init__(self):
        self.frame_no = 0
        self.records = {}

    # ---- FACE ----
    def face_probe(self, pad, info, _):
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK
        bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = bm.frame_meta_list
        while l_frame is not None:
            fm = pyds.NvDsFrameMeta.cast(l_frame.data)
            faces = self._read_faces(fm)
            self.records[self.frame_no] = faces
            self.frame_no += 1
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    def _read_faces(self, fm):
        emb_t = _get_face_embed_meta_type()
        lm_t = _get_face_landmarks_meta_type()
        out = []
        l_obj = fm.obj_meta_list
        while l_obj is not None:
            om = pyds.NvDsObjectMeta.cast(l_obj.data)
            if om.unique_component_id == FACE_GIE_UID:
                r = om.rect_params
                emb, q, det_conf = None, None, None
                lu = om.obj_user_meta_list
                while lu is not None:
                    u = pyds.NvDsUserMeta.cast(lu.data)
                    mt = u.base_meta.meta_type
                    if mt == emb_t:
                        m = _FaceEmbeddingMeta.from_address(
                            pyds.get_ptr(u.user_meta_data))
                        if (m.version == _FACE_EMBED_META_VERSION
                                and m.dims == 512
                                and (m.flags & _FACE_EMB_FLAG_VALID)):
                            emb = np.ctypeslib.as_array(
                                m.embedding, shape=(512,)).astype(np.float32)
                            emb = emb.copy()
                            q = float(m.quality)
                    elif mt == lm_t:
                        lm = _FaceLandmarksMeta.from_address(
                            pyds.get_ptr(u.user_meta_data))
                        if (lm.version == _FACE_LANDMARKS_META_VERSION
                                and (lm.flags & _FACE_LM_FLAG_VALID)):
                            det_conf = float(lm.det_conf)
                    try:
                        lu = lu.next
                    except StopIteration:
                        break
                tid = int(om.object_id)
                out.append({
                    "bbox": [int(r.left), int(r.top),
                             int(r.left + r.width), int(r.top + r.height)],
                    "conf": float(om.confidence),
                    "det_conf": det_conf,
                    "quality": q,
                    "object_id": None if tid == _UNTRACKED_OBJECT_ID else tid,
                    "embedding": emb.tolist() if emb is not None else None,
                })
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        return out

    # ---- PLATE ----
    def plate_probe(self, pad, info, _):
        """Chạy ở src của SGIE: vừa gom det vừa combine char→text (port từ
        DeepStreamPipeline._combine_frame_chars)."""
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK
        bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = bm.frame_meta_list
        while l_frame is not None:
            fm = pyds.NvDsFrameMeta.cast(l_frame.data)
            dets, chars = [], []
            l_obj = fm.obj_meta_list
            while l_obj is not None:
                om = pyds.NvDsObjectMeta.cast(l_obj.data)
                r = om.rect_params
                if om.unique_component_id == PLATE_GIE_UID:
                    dets.append({
                        "bbox": [int(r.left), int(r.top),
                                 int(r.left + r.width), int(r.top + r.height)],
                        "conf": float(om.confidence)})
                elif om.unique_component_id == OCR_SGIE_UID:
                    cid = int(om.class_id)
                    if 0 <= cid < len(_OCR_CLASS_CHARS):
                        chars.append({
                            "char": _OCR_CLASS_CHARS[cid],
                            "conf": float(om.confidence),
                            "cx": r.left + r.width * 0.5,
                            "cy": r.top + r.height * 0.5,
                            "h": r.height})
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            text = self._combine(chars)
            self.records[self.frame_no] = {"dets": dets, "text": text}
            self.frame_no += 1
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    @staticmethod
    def _combine(chars):
        if not chars:
            return ""
        if len(chars) > _MAX_PLATE_CHARS:
            chars = sorted(chars, key=lambda c: c["conf"],
                           reverse=True)[:_MAX_PLATE_CHARS]
        ordered = _sort_chars_for_plate(chars)
        return _enforce_plate_format("".join(c["char"] for c in ordered))


# ───────────────────────── run loop ────────────────────────────────────
def run_pipeline(pipeline, expected_frames, label):
    loop = GLib.MainLoop()
    state = {"err": None}

    def on_msg(bus, msg):
        t = msg.type
        if t == Gst.MessageType.EOS:
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            state["err"] = f"{err.message} | {dbg}"
            loop.quit()
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_msg)

    pipeline.set_state(Gst.State.PLAYING)
    # Safety timeout: ~0.1s/ảnh + 30s headroom.
    GLib.timeout_add_seconds(int(expected_frames * 0.2) + 30, loop.quit)
    t0 = time.time()
    loop.run()
    pipeline.set_state(Gst.State.NULL)
    dt = time.time() - t0
    if state["err"]:
        print(f"  [GST ERROR] {state['err']}")
    print(f"  {label}: chạy {dt:.1f}s")
    return state["err"]


# ───────────────────────── FACE eval ───────────────────────────────────
def eval_face(cfg, limit, n_pairs):
    lfw_root = ROOT / "benchmark_data" / "lfw_funneled"
    all_imgs = sorted(glob.glob(str(lfw_root / "*" / "*.jpg")))
    person_imgs = defaultdict(list)
    for p in all_imgs:
        person_imgs[Path(p).parent.name].append(p)
    multi = {k: v for k, v in person_imgs.items() if len(v) >= 2}

    rng = random.Random(2026)
    # Tập det-rate: sample ngẫu nhiên `limit` ảnh.
    det_sample = rng.sample(all_imgs, min(limit, len(all_imgs)))
    # Tập pairs: same (2 ảnh cùng người) + diff (2 người khác nhau).
    same_pairs, diff_pairs = [], []
    multi_names = list(multi.keys())
    for name in rng.sample(multi_names, min(n_pairs, len(multi_names))):
        a, b = rng.sample(multi[name], 2)
        same_pairs.append((a, b))
    all_names = list(person_imgs.keys())
    while len(diff_pairs) < n_pairs:
        n1, n2 = rng.sample(all_names, 2)
        diff_pairs.append((person_imgs[n1][0], person_imgs[n2][0]))

    # Gom toàn bộ ảnh cần chạy qua DeepStream (dedup) → 1 lần dump.
    needed = list(dict.fromkeys(
        det_sample
        + [p for pr in same_pairs for p in pr]
        + [p for pr in diff_pairs for p in pr]))
    seqdir = SCRATCH / "seq_face"
    print(f"Staging {len(needed)} ảnh → {seqdir} ...")
    manifest = stage_images(needed, seqdir)
    idx_of = {p: i for i, p in enumerate(manifest)}

    pipeline, sink = build_face_pipeline(str(seqdir), len(manifest), cfg)
    dumper = Dumper()
    sink.get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER, dumper.face_probe, None)
    print(f"Chạy DeepStream face graph trên {len(manifest)} ảnh ...")
    run_pipeline(pipeline, len(manifest), "face")
    print(f"  frames probed = {dumper.frame_no} / {len(manifest)}")

    dump_path = SCRATCH / "face_dump.json"
    dump_path.write_text(json.dumps(
        {"manifest": manifest, "records": dumper.records}))
    print(f"  dump → {dump_path}")

    def best_face(idx):
        fs = dumper.records.get(idx, [])
        fs = [f for f in fs if f["embedding"] is not None]
        if not fs:
            return None
        f = max(fs, key=lambda x: x["conf"])
        e = np.asarray(f["embedding"], dtype=np.float32)
        n = np.linalg.norm(e)
        return e / n if n > 0 else None

    # ── Detection rate ──
    det = no = multi_c = 0
    for p in det_sample:
        fs = dumper.records.get(idx_of.get(p, -1), [])
        if len(fs) == 0:
            no += 1
        elif len(fs) == 1:
            det += 1
        else:
            multi_c += 1
    tot = det + no + multi_c
    print("\n=== [DeepStream] Face Detection ({} ảnh) ===".format(tot))
    print(f"  Detected 1 face : {det} ({100*det/max(tot,1):.1f}%)")
    print(f"  No face         : {no} ({100*no/max(tot,1):.1f}%)")
    print(f"  Multi face      : {multi_c}")

    # ── Same vs Diff similarity ──
    def cos(pr):
        e1, e2 = best_face(idx_of.get(pr[0], -1)), best_face(idx_of.get(pr[1], -1))
        if e1 is None or e2 is None:
            return None
        return float(np.dot(e1, e2))

    same = [s for s in map(cos, same_pairs) if s is not None]
    diff = [s for s in map(cos, diff_pairs) if s is not None]
    print("\n=== [DeepStream] Embedding similarity ===")
    print(f"  Same person (n={len(same)}): mean={np.mean(same):.3f} "
          f"min={np.min(same):.3f}" if same else "  Same: n=0")
    print(f"  Diff person (n={len(diff)}): mean={np.mean(diff):.3f} "
          f"max={np.max(diff):.3f}" if diff else "  Diff: n=0")
    if same and diff:
        best_thr, best_acc = 0, 0
        for thr in np.arange(0.0, 1.0, 0.01):
            tp = sum(1 for s in same if s >= thr)
            tn = sum(1 for s in diff if s < thr)
            acc = (tp + tn) / (len(same) + len(diff))
            if acc > best_acc:
                best_acc, best_thr = acc, thr
        cfg_thr = cfg["recognition"].get("face_threshold", 0.3)
        print(f"  Best threshold  : {best_thr:.2f} (acc={best_acc:.3f})")
        print(f"  Config face_threshold = {cfg_thr}")
    print("\nLưu ý: đây là số của ĐÚNG plugin nvdsfaceembed (ArcFace FP16 + "
          "CUDA align), so trực tiếp được với cell 19–20 của notebook (fallback "
          "InsightFace).")


# ───────────────────────── PLATE eval ──────────────────────────────────
def _load_ocr_text_gt(path):
    gt = {}
    if not os.path.exists(path):
        return gt
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                key = Path(parts[0]).name
                gt[key] = parts[1].strip()
                gt[Path(parts[0]).stem] = parts[1].strip()
    return gt


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1e-6)


def _run_plate(cfg, imgs, seqname, label, ocr_only=False):
    seqdir = SCRATCH / seqname
    print(f"Staging {len(imgs)} ảnh → {seqdir} ...")
    manifest = stage_images(imgs, seqdir)
    if ocr_only:
        pipeline, sink = build_ocr_pipeline(str(seqdir), len(manifest), cfg)
        probe_pad = sink.get_static_pad("sink")
    else:
        pipeline, sgie, sink = build_plate_pipeline(
            str(seqdir), len(manifest), cfg)
        probe_pad = sgie.get_static_pad("src")
    dumper = Dumper()
    probe_pad.add_probe(Gst.PadProbeType.BUFFER, dumper.plate_probe, None)
    print(f"Chạy DeepStream plate graph ({label}) trên {len(manifest)} ảnh ...")
    run_pipeline(pipeline, len(manifest), label)
    print(f"  frames probed = {dumper.frame_no} / {len(manifest)}")
    return manifest, dumper


def eval_plate(cfg, limit):
    det_dir = ROOT / "yolo_plate_data" / "images" / "val"
    det_lbl = ROOT / "yolo_plate_data" / "labels" / "val"
    ocr_dir = ROOT / "yolo_plate_ocr_data" / "images" / "val"
    ocr_gt = _load_ocr_text_gt(str(ROOT / "paddle_format" / "0rec_gt.txt"))
    rng = random.Random(2026)

    # ── PASS 1: Detection trên det-val (ảnh xe đầy đủ) ──
    imgs = sorted(glob.glob(str(det_dir / "*.jpg"))
                  + glob.glob(str(det_dir / "*.png")))
    if limit and len(imgs) > limit:
        imgs = rng.sample(imgs, limit)
    manifest, dumper = _run_plate(cfg, imgs, "seq_plate_det", "plate-det")
    sizes = [cv2.imread(p).shape[:2] for p in manifest]  # (h, w)

    # ── Detection (IoU≥0.5) ──
    tp = fp = fn = 0
    iou_thr = 0.5
    for i, p in enumerate(manifest):
        h, w = sizes[i]
        sx, sy = MUX_W / w, MUX_H / h
        gts = []
        lp = det_lbl / (Path(p).stem + ".txt")
        if lp.exists():
            for ln in lp.read_text().splitlines():
                parts = ln.split()
                if len(parts) >= 5:
                    cx, cy, bw, bh = map(float, parts[1:5])
                    gx1 = (cx - bw/2) * w * sx
                    gy1 = (cy - bh/2) * h * sy
                    gx2 = (cx + bw/2) * w * sx
                    gy2 = (cy + bh/2) * h * sy
                    gts.append([gx1, gy1, gx2, gy2])
        preds = [d["bbox"] for d in dumper.records.get(i, {}).get("dets", [])]
        matched = set()
        for pred in preds:
            best, bj = 0, -1
            for j, g in enumerate(gts):
                if j in matched:
                    continue
                v = _iou(pred, g)
                if v > best:
                    best, bj = v, j
            if best >= iou_thr:
                tp += 1
                matched.add(bj)
            else:
                fp += 1
        fn += len(gts) - len(matched)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"\n=== [DeepStream] Plate Detection (IoU≥{iou_thr}) ===")
    print(f"  TP={tp} FP={fp} FN={fn}")
    print(f"  Precision={prec:.3f}  Recall={rec:.3f}")

    # ── PASS 2: OCR trên ocr-val (crop biển số, GT text) ──
    ocr_imgs = sorted(glob.glob(str(ocr_dir / "*.jpg"))
                      + glob.glob(str(ocr_dir / "*.png")))
    ocr_imgs = [p for p in ocr_imgs
                if (ocr_gt.get(Path(p).name) or ocr_gt.get(Path(p).stem))]
    if limit and len(ocr_imgs) > limit:
        ocr_imgs = rng.sample(ocr_imgs, limit)
    o_manifest, o_dumper = _run_plate(
        cfg, ocr_imgs, "seq_plate_ocr", "plate-ocr", ocr_only=True)

    n = exact = 0
    char_ok = char_tot = 0
    for i, p in enumerate(o_manifest):
        key = Path(p).name
        gt = ocr_gt.get(key) or ocr_gt.get(Path(p).stem)
        if not gt:
            continue
        pred = o_dumper.records.get(i, {}).get("text", "")
        gtn = gt.replace("-", "").replace(".", "").upper()
        predn = pred.replace("-", "").replace(".", "").upper()
        n += 1
        if predn == gtn:
            exact += 1
        m = min(len(gtn), len(predn))
        char_ok += sum(1 for k in range(m) if gtn[k] == predn[k])
        char_tot += len(gtn)
        if os.environ.get("EVAL_DEBUG") and n <= 20:
            nd = len(o_dumper.records.get(i, {}).get("dets", []))
            print(f"    {Path(p).name:18s} GT={gtn:11s} PRED={predn:11s} "
                  f"ndet={nd} {'OK' if predn==gtn else ''}")
    if n:
        print(f"\n=== [DeepStream] Plate OCR ({n} ảnh có GT) ===")
        print(f"  Full-string acc : {exact}/{n} ({100*exact/n:.1f}%)")
        print(f"  Char acc        : {char_ok}/{char_tot} "
              f"({100*char_ok/max(char_tot,1):.1f}%)")
    else:
        print("\n⚠️  Không có ảnh nào khớp OCR text GT "
              "(paddle_format/0rec_gt.txt) với tên file det val.")


# ───────────────────────────── main ────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["face", "plate"])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--pairs", type=int, default=100)
    args = ap.parse_args()

    Gst.init(None)
    DeepStreamPipeline._register_local_plugins()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

    if args.mode == "face":
        eval_face(cfg, args.limit, args.pairs)
    else:
        eval_plate(cfg, args.limit)


if __name__ == "__main__":
    main()
