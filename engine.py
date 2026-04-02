
import numpy as np
import cv2
import os
import logging

log = logging.getLogger("engine")

# 36 classes: 0-9 = số, 10-35 = A-Z
CHAR_CLASSES = [str(i) for i in range(10)] + \
               [chr(c) for c in range(ord('A'), ord('Z') + 1)]


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


class PlateOCRYolo:
    """
    ★ YOLOv8n char detection — thay thế PaddleOCR.

    Detect từng ký tự trên crop biển số → sort theo vị trí → ghép chuỗi.
    Interface giữ nguyên: __call__(crop) → (text, conf)

    Ưu điểm vs PaddleOCR:
      - TensorRT 320px: ~2-5ms (vs 50-150ms cho 3 variant PaddleOCR)
      - Output chuẩn YOLO, không cần parse nhiều version
      - Handle biển 2 dòng bằng y-position clustering
    """

    def __init__(self, model_path: str, imgsz: int = 320,
                 conf: float = 0.3, device: int = 0):
        from ultralytics import YOLO
        self.imgsz = imgsz
        self.conf = conf

        # Auto-load .engine nếu có, hoặc export từ .pt
        engine_path = model_path.rsplit(".", 1)[0] + ".engine"
        if model_path.endswith(".pt") and not os.path.exists(engine_path):
            log.info(f"PlateOCR: Exporting → TensorRT FP16...")
            YOLO(model_path).export(format="engine", half=True,
                                    imgsz=imgsz, device=device)
            model_path = engine_path
        elif os.path.exists(engine_path):
            model_path = engine_path

        self.model = YOLO(model_path, task="detect")
        # Warmup
        self.model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                   verbose=False)
        log.info(f"PlateOCRYolo ready: {model_path} "
                 f"(imgsz={imgsz}, conf={conf})")

    def __call__(self, plate_crop: np.ndarray) -> tuple:
        """
        Detect ký tự + sort → chuỗi biển số.
        Returns: (text, avg_conf)

        Interface tương thích PlateOCR cũ — main.py không cần sửa.
        """
        if plate_crop.size == 0:
            return ("", 0.0)

        h, w = plate_crop.shape[:2]
        # Upscale crop nhỏ để YOLO detect tốt hơn
        if h < 48:
            scale = 48.0 / h
            plate_crop = cv2.resize(plate_crop, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_CUBIC)

        # ── Inference ──
        results = self.model(plate_crop, imgsz=self.imgsz,
                             conf=self.conf, verbose=False)

        chars = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls)
                if cls_id >= len(CHAR_CLASSES):
                    continue
                chars.append({
                    "char": CHAR_CLASSES[cls_id],
                    "conf": float(box.conf),
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                    "x1": x1,
                })

        if not chars:
            return ("", 0.0)

        # ── Sort: handle biển 2 dòng ──
        sorted_chars = self._sort_chars(chars, plate_crop.shape[0])

        text = "".join(c["char"] for c in sorted_chars)
        avg_conf = sum(c["conf"] for c in sorted_chars) / len(sorted_chars)

        log.debug(f"OCR YOLO: {len(sorted_chars)} chars → '{text}' "
                  f"conf={avg_conf:.2f}")
        return (text, avg_conf)

    @staticmethod
    def _sort_chars(chars: list, img_h: int) -> list:
        """
        Sort ký tự theo vị trí — handle biển 2 dòng.

        Biển xe máy VN 2 dòng: dòng 1 = tỉnh + series (29B1),
                                dòng 2 = số đăng ký (25739)
        Detect bằng y_center clustering:
          - Nếu khoảng cách y giữa ký tự cao/thấp nhất > 30% chiều cao ảnh
            → 2 dòng, sort mỗi dòng theo x riêng
          - Ngược lại → 1 dòng, sort theo x
        """
        ys = [c["cy"] for c in chars]
        y_range = max(ys) - min(ys)

        if y_range > img_h * 0.3 and len(chars) >= 4:
            # 2 dòng
            y_mid = (min(ys) + max(ys)) / 2
            line1 = sorted([c for c in chars if c["cy"] < y_mid],
                           key=lambda c: c["cx"])
            line2 = sorted([c for c in chars if c["cy"] >= y_mid],
                           key=lambda c: c["cx"])
            return line1 + line2
        else:
            # 1 dòng
            return sorted(chars, key=lambda c: c["cx"])


class PlateOCR:
    """
    PaddleOCR fallback — giữ lại để dùng khi chưa có YOLO OCR model.

    ★ DEPRECATED: Dùng PlateOCRYolo thay thế.
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
 
        best_text, best_conf = "", 0.0
        for img in self._preprocess_variants(plate_crop):
            text, conf = (self._parse_v3(img) if self._v3
                          else self._parse_v2(img))
            if text and conf > best_conf:
                best_text, best_conf = text, conf
 
        return (best_text, best_conf)
 
    def _preprocess_variants(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)
        yield cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 4)
        yield cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        yield crop

    def _parse_v3(self, img):
        results = self.ocr.predict(img)
        if not self._debug_done:
            self._debug_done = True
            log.info(f"OCR DEBUG: type(results)={type(results)} "
                     f"len={len(results) if results else 0}")

        if not results:
            return ("", 0.0)

        texts, confs = [], []
        for res in results:
            rt = getattr(res, "rec_texts", None)
            rs = getattr(res, "rec_scores", None)
            if rt:
                texts.extend(rt)
                confs.extend(list(rs) if rs else [0.5] * len(rt))
                continue
            if hasattr(res, 'text') and res.text:
                texts.append(res.text)
                confs.append(getattr(res, 'score', 0.5))
                continue
            if isinstance(res, dict):
                if 'rec_texts' in res:
                    texts.extend(res['rec_texts'])
                    confs.extend(res.get('rec_scores',
                                         [0.5]*len(res['rec_texts'])))
                    continue
            if isinstance(res, list):
                for item in res:
                    if isinstance(item, dict) and 'text' in item:
                        texts.append(item['text'])
                        confs.append(item.get('score', 0.5))
            rt_single = getattr(res, 'rec_text', None)
            if rt_single:
                texts.append(rt_single)
                confs.append(getattr(res, 'rec_score', 0.5))

        if not texts:
            return ("", 0.0)
        combined = "".join(texts).replace(" ", "").upper()
        avg_conf = sum(confs) / len(confs) if confs else 0.0
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
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            log.debug("FACE QUALITY: crop empty")
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_val = gray.mean()
        ok = blur_val >= blur_thr and 30 < mean_val < 230
        log.info(f"FACE QUALITY: blur={blur_val:.1f} (thr={blur_thr}) "
                 f"brightness={mean_val:.1f} size={crop.shape[1]}x{crop.shape[0]} "
                 f"→ {'OK' if ok else 'FAIL'}"
                 f"{' [TOO_BLURRY]' if blur_val < blur_thr else ''}"
                 f"{' [TOO_DARK]' if mean_val <= 30 else ''}"
                 f"{' [TOO_BRIGHT]' if mean_val >= 230 else ''}")
        return ok
 
    @staticmethod
    def quality_score(frame, bbox, blur_thr=80.0):
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(blur / 500.0, 1.0)
        brightness = gray.mean()
        if brightness < 30 or brightness > 230:
            return 0.0
        bright_score = 1.0 - abs(brightness - 128) / 128.0
        face_area = (x2 - x1) * (y2 - y1)
        size_score = min(face_area / 20000.0, 1.0)
        return blur_score * 0.5 + bright_score * 0.2 + size_score * 0.3