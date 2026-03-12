"""
engine.py — AI Inference

Khi có DeepStream:
  - PlateDetector KHÔNG dùng (detection qua nvinfer)
  - PlateOCR dùng cho crop từ DeepStream detection
  - FaceEngine dùng cho face camera frame

Khi fallback:
  - PlateDetector dùng ultralytics
  - PlateOCR dùng cho crop
  - FaceEngine dùng cho face camera
"""

import numpy as np
import cv2
import os
import logging

log = logging.getLogger("engine")


class PlateDetector:
    """YOLOv8n — chỉ dùng trong fallback mode (không có DeepStream)."""

    def __init__(self, model_path: str, imgsz: int = 640,
                 conf: float = 0.4, device: int = 0):
        from ultralytics import YOLO
        self.imgsz = imgsz
        self.conf = conf

        engine_path = model_path.rsplit(".", 1)[0] + ".engine"
        if model_path.endswith(".pt") and not os.path.exists(engine_path):
            log.info(f"Exporting → TensorRT FP16 (lần đầu)...")
            YOLO(model_path).export(format="engine", half=True,
                                    imgsz=imgsz, device=device)
            model_path = engine_path
        elif os.path.exists(engine_path):
            model_path = engine_path

        self.model = YOLO(model_path, task="detect")
        self.model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                   verbose=False)
        log.info(f"PlateDetector ready: {model_path}")

    def __call__(self, frame: np.ndarray) -> list:
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf,
                             verbose=False)
        out = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                out.append({"bbox": (x1, y1, x2, y2),
                            "conf": float(box.conf)})
        return out


class PlateOCR:
    """
    PaddleOCR 3.x (PP-OCRv5) / 2.x fallback.

    FIX: PP-OCRv5 predict() trả về list[OCRResult], mỗi OCRResult
    có thể có nhiều attribute name khác nhau tùy version.
    Thêm debug logging để xem cấu trúc thật.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = True):
        from paddleocr import PaddleOCR
        self._v3 = False
        try:
            self.ocr = PaddleOCR(
                lang=lang,
                device="gpu:0" if use_gpu else "cpu",
                use_textline_orientation=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False)
            self._v3 = True
        except TypeError:
            self.ocr = PaddleOCR(lang=lang, use_angle_cls=False,
                                 show_log=False, use_gpu=use_gpu)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._debug_done = False
        log.info(f"PlateOCR ready (v{'3.x' if self._v3 else '2.x'})")

    def __call__(self, plate_crop: np.ndarray) -> tuple:
        if plate_crop.size == 0:
            return ("", 0.0)
        
        h, w = plate_crop.shape[:2]
        if h < 48:
            scale = 48.0 / h
            plate_crop = cv2.resize(plate_crop, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_CUBIC)
 
        # ★ FIX v4: Multi-preprocessing — thử nhiều variant
        best_text, best_conf = "", 0.0
        for img in self._preprocess_variants(plate_crop):
            text, conf = (self._parse_v3(img) if self._v3
                          else self._parse_v2(img))
            if text and conf > best_conf:
                best_text, best_conf = text, conf
 
        return (best_text, best_conf)
 
    def _preprocess_variants(self, crop):
        """Trả về nhiều phiên bản preprocessed."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
 
        # Variant 1: CLAHE (tốt cho ảnh contrast thấp)
        enhanced = self._clahe.apply(gray)
        yield cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
 
        # Variant 2: Adaptive threshold (tốt ban đêm, ngược sáng)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 4)
        yield cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
 
        # Variant 3: Original (đôi khi ảnh gốc cho kết quả tốt nhất)
        yield crop

    def _parse_v3(self, img):
        """
        PP-OCRv5 result parsing.
        FIX: thêm debug + fallback cho nhiều attribute name.
        """
        results = self.ocr.predict(img)

        # ★ DEBUG: log cấu trúc kết quả 1 lần
        if not self._debug_done:
            self._debug_done = True
            log.info(f"OCR DEBUG: type(results)={type(results)} "
                     f"len={len(results) if results else 0}")
            if results:
                for i, res in enumerate(results):
                    log.info(f"OCR DEBUG: res[{i}] type={type(res)}")
                    # Log tất cả attributes
                    if hasattr(res, '__dict__'):
                        log.info(f"OCR DEBUG: res[{i}].__dict__ keys="
                                 f"{list(res.__dict__.keys())}")
                    if hasattr(res, 'keys'):
                        log.info(f"OCR DEBUG: res[{i}].keys()="
                                 f"{list(res.keys())}")
                    # Log dir() để tìm attribute đúng
                    attrs = [a for a in dir(res)
                             if not a.startswith('_')
                             and ('text' in a.lower()
                                  or 'rec' in a.lower()
                                  or 'score' in a.lower()
                                  or 'det' in a.lower())]
                    log.info(f"OCR DEBUG: relevant attrs={attrs}")
                    # Thử print raw
                    log.info(f"OCR DEBUG: str(res)={str(res)[:500]}")

        if not results:
            return ("", 0.0)

        texts, confs = [], []
        for res in results:
            # ── Thử nhiều cách parse ──

            # Cách 1: PP-OCRv5 mới nhất (rec_texts / rec_scores)
            rt = getattr(res, "rec_texts", None)
            rs = getattr(res, "rec_scores", None)
            if rt:
                texts.extend(rt)
                confs.extend(list(rs) if rs else [0.5] * len(rt))
                continue

            # Cách 2: Một số version dùng .text / .score
            if hasattr(res, 'text') and res.text:
                texts.append(res.text)
                confs.append(getattr(res, 'score', 0.5))
                continue

            # Cách 3: Dict-like result
            if isinstance(res, dict):
                if 'rec_texts' in res:
                    texts.extend(res['rec_texts'])
                    confs.extend(res.get('rec_scores', [0.5]*len(res['rec_texts'])))
                    continue
                if 'text' in res:
                    texts.append(res['text'])
                    confs.append(res.get('score', 0.5))
                    continue

            # Cách 4: PP-OCRv5 trả về list of dict per line
            #   [{"text": "99B1", "score": 0.95, "bbox": [...]}]
            if isinstance(res, list):
                for item in res:
                    if isinstance(item, dict) and 'text' in item:
                        texts.append(item['text'])
                        confs.append(item.get('score', 0.5))
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        # PaddleOCR 2.x style: [[bbox, (text, conf)], ...]
                        if isinstance(item[1], (list, tuple)):
                            texts.append(str(item[1][0]))
                            confs.append(float(item[1][1]))

            # Cách 5: Nếu res có attribute 'rec_text' (số ít)
            rt_single = getattr(res, 'rec_text', None)
            if rt_single:
                texts.append(rt_single)
                confs.append(getattr(res, 'rec_score', 0.5))

        if not texts:
            return ("", 0.0)

        combined = "".join(texts).replace(" ", "").upper()
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        log.debug(f"OCR parsed: texts={texts} → '{combined}' "
                  f"conf={avg_conf:.2f}")
        return (combined, avg_conf)

    def _parse_v2(self, img):
        results = self.ocr.ocr(img, cls=False)
        if not results or not results[0]:
            return ("", 0.0)
        texts, confs = [], []
        for line in results[0]:
            texts.append(line[1][0])
            confs.append(line[1][1])
        return ("".join(texts).replace(" ", "").upper(),
                sum(confs) / len(confs) if confs else 0.0)


class FaceEngine:
    """InsightFace buffalo_sc — detect + align + embed trong 1 call."""

    def __init__(self, model_pack: str = "buffalo_sc",
                 det_size: tuple = (640, 640)):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(
            name=model_pack,
            providers=["TensorrtExecutionProvider",
                       "CUDAExecutionProvider",
                       "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.app.get(np.zeros((det_size[1], det_size[0], 3), dtype=np.uint8))
        log.info(f"FaceEngine ready: {model_pack}")

    def __call__(self, frame: np.ndarray) -> list:
        """Detect + embed faces."""
        faces = self.app.get(frame)
        return [{
            "bbox": tuple(f.bbox.astype(int).tolist()),
            "conf": float(f.det_score),
            "embedding": f.normed_embedding,
        } for f in faces]

    @staticmethod
    def quality_ok(frame, bbox, blur_thr=35.0):
        """
        Boolean check.
        ★ v4: blur_thr hạ từ 80→35 cho RTMP stream (H.264 compression
        giảm Laplacian variance đáng kể so với camera trực tiếp).
        Thêm debug logging để biết chính xác tại sao fail.
        """
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            log.debug("FACE QUALITY: crop empty")
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_val = gray.mean()
        ok = blur_val >= blur_thr and 30 < mean_val < 230
        # ★ Log mỗi lần check — rất hữu ích khi debug
        log.info(f"FACE QUALITY: blur={blur_val:.1f} (thr={blur_thr}) "
                 f"brightness={mean_val:.1f} size={crop.shape[1]}x{crop.shape[0]} "
                 f"→ {'OK' if ok else 'FAIL'}"
                 f"{' [TOO_BLURRY]' if blur_val < blur_thr else ''}"
                 f"{' [TOO_DARK]' if mean_val <= 30 else ''}"
                 f"{' [TOO_BRIGHT]' if mean_val >= 230 else ''}")
        return ok
 
    @staticmethod
    def quality_score(frame, bbox, blur_thr=80.0):
        """
        ★ NEW v4: Trả về float 0.0~1.0 thay vì bool.
        Dùng cho weighted embedding averaging.
        """
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
 
        # Blur score (Laplacian variance, cap at 500)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(blur / 500.0, 1.0)
 
        # Brightness score (best at 128, penalty for too dark/bright)
        brightness = gray.mean()
        if brightness < 30 or brightness > 230:
            return 0.0
        bright_score = 1.0 - abs(brightness - 128) / 128.0
 
        # Size score (bigger face = better features)
        face_area = (x2 - x1) * (y2 - y1)
        size_score = min(face_area / 20000.0, 1.0)
 
        return blur_score * 0.5 + bright_score * 0.2 + size_score * 0.3