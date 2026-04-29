# Smart Parking System

Hệ thống bãi đỗ xe thông minh chạy trên **Jetson Orin Nano**, nhận diện **biển số + khuôn mặt** qua 2 camera RTMP, lưu dữ liệu xe ra/vào bằng **PostgreSQL + pgvector**, và phát dashboard real-time qua **FastAPI + WebSocket**.

Pipeline tối ưu GPU bằng **DeepStream** (zero-copy decode → nvinfer YOLO plate detect), fallback tự động về **GStreamer + Python inference** khi không có pyds.

---

## Kiến trúc

```
┌─ Plate cam ─┐                                            ┌─ YOLOv8n char OCR ─┐
│  iPhone RTMP│                                            │  (TensorRT FP16)   │
└─────────────┘                                            └────────────────────┘
       │                                                            ▲
       ▼                                                            │ crop
┌─────────────┐    ┌──────────────┐   ┌──────────────┐   ┌──────────┴────────┐
│   nginx     │───▶│  rtmpsrc     │──▶│  nvstreammux │──▶│  nvinfer (YOLOv8n │
│   RTMP      │    │  flvdemux    │   │  (batch=2)   │   │  plate detector)  │
└─────────────┘    │  h264parse   │   └──────┬───────┘   └──────┬────────────┘
       ▲           │  nvv4l2dec   │          │                   │
       │           └──────────────┘          ▼                   ▼
┌─ Face cam ──┐                         ┌────────────────────────────┐
│  iPhone RTMP│─────────────────────────▶   probe callback (Python)   │
└─────────────┘                         │   → extract frame+detections│
                                         └─────────┬──────────────────┘
                                                   │
                   ┌───────────────────────────────┼────────────────────────┐
                   ▼                               ▼                        ▼
         ┌───────────────────┐         ┌────────────────────┐   ┌────────────────────┐
         │ PlateOCRYolo      │◀──────  │ ThreadPoolExecutor │───▶│ InsightFace        │
         │ (char detect +    │         │ OCR ∥ Face parallel│   │ buffalo_sc         │
         │  2-line sort +    │         └────────────────────┘   │ (det + embed 512d) │
         │  format enforce)  │                                   └─────────┬──────────┘
         └─────────┬─────────┘                                             │
                   │                                                        │
                   ▼                                                        ▼
         ┌────────────────────┐                                ┌─────────────────────┐
         │  PlateValidator    │                                │  EmbeddingAvg       │
         │  (regex VN format) │                                │  (quality-weighted) │
         │  PlateVoter (N=5)  │                                │  n=3 frames         │
         └─────────┬──────────┘                                └─────────┬───────────┘
                   │                                                      │
                   └──────────────────────┬───────────────────────────────┘
                                          ▼
                          ┌────────────────────────────────┐
                          │ PostgreSQL + pgvector          │
                          │  • active(plate, embedding)    │
                          │  • parking_log (history)       │
                          │  • ivfflat cosine index        │
                          └──────────────┬─────────────────┘
                                         │
                          ┌──────────────▼─────────────────┐
                          │  FastAPI + WebSocket           │
                          │  /stream/{plate,face} + /ws    │
                          │  dashboard.html (real-time)    │
                          └────────────────────────────────┘
```

---

## Thành phần

| File | Vai trò |
|---|---|
| [main.py](main.py) | Orchestrator — khởi tạo model, pipeline, DB, web; chạy vòng lặp `process_entry` / `process_exit`. |
| [main2.py](main2.py) | Phiên bản alternate — rearrange thứ tự xử lý engine để giảm race condition khi parallel. |
| [pipeline.py](pipeline.py) | `DeepStreamPipeline` (nvstreammux + nvinfer + probe) và `StreamReader` (fallback GStreamer/OpenCV). |
| [engine.py](engine.py) | `PlateDetector`, `PlateOCRYolo` (★ char detect), `PlateOCR` (PaddleOCR fallback), `FaceEngine` (InsightFace). |
| [database.py](database.py) | `ParkingDB` — connection pool, cached stats, pgvector cosine search. |
| [web.py](web.py) | FastAPI app — REST + WebSocket + JPEG snapshot endpoint. |
| [templates/dashboard.html](templates/dashboard.html) | UI dashboard. |
| [configs/plate_det_config.txt](configs/plate_det_config.txt) | nvinfer config (engine, labels, YOLO parser lib). |
| [parsers/libnvds_yolov8_parser.so](parsers/) | Custom YOLOv8 output parser cho DeepStream. |
| [config.yaml](config.yaml) | Toàn bộ config runtime (camera, model, DB, web). |
| [setup.py](setup.py) | Verify GPU/TensorRT + init PostgreSQL + download models. |
| [wait_streams.py](wait_streams.py) | Utility chờ RTMP stream sẵn sàng trước khi khởi động. |

---

## Chế độ hoạt động

### ENTRY (xe vào)
1. `nvinfer` detect biển số trên plate stream → bbox + conf.
2. Crop biển số → **OCR + Face detect chạy song song** (ThreadPool, GPU release GIL).
3. OCR text → `enforce_plate_format` (fix confusion digit↔letter) → `PlateValidator` (regex VN) → `PlateVoter` (N=5 frames, majority ≥ 50%).
4. Face → quality gate (blur Laplacian, brightness, size) → `EmbeddingAvg` (weighted theo chất lượng, 3 frame).
5. Khi biển số vote ổn định **và** có embedding → `db.entry(plate, emb)` → WebSocket push event.

### EXIT (xe ra)
1. Detect + OCR biển số giống ENTRY.
2. `db.find_by_plate(plate)` → lấy embedding lúc vào.
3. So sánh cosine similarity với embedding live → nếu ≥ `face_threshold` (0.3) → `db.exit()`.
4. Log xuống bảng `parking_log` với duration + match_conf.

<!-- --- -->

<!-- ## Các tối ưu đã áp dụng

- **DeepStream zero-copy**: plate detection chạy trên GPU qua `nvinfer`, không copy frame ra CPU cho đến probe.
- **Early-skip trong probe** (`process_every_n`): bỏ batch trước `get_nvds_buf_surface` để tránh tốn cvtColor.
- **Parallel OCR + Face**: `ThreadPoolExecutor(max_workers=2)` — GPU inference release GIL nên true overlap.
- **TensorRT FP16**: `PlateDetector` và `PlateOCRYolo` tự export `.pt → .engine` ở lần chạy đầu.
- **YOLO OCR thay PaddleOCR**: ~2–5 ms/crop (vs 50–150 ms) + handle biển 2 dòng bằng residual-gap clustering.
- **PostgreSQL connection pool** + **cached stats** — không query count mỗi frame.
- **pgvector IVFFlat** (`lists=22`) cho cosine search — tự fallback sequential khi < 100 records.
- **Streaming JPEG**: downscale 1280px + quality 70, cache theo frame reference để không re-encode cùng frame.
- **Frame copy chỉ trong bbox** + JPEG encode quality giảm để đẩy web thread nhẹ hơn. -->

<!-- --- -->

## Cài đặt

```bash
# 1. Dependencies
pip install -r requirements.txt --break-system-packages

# 2. PostgreSQL + pgvector
sudo apt install postgresql postgresql-contrib postgresql-16-pgvector
python setup.py --init-db         # tạo user/db/extension

# 3. Verify GPU, TensorRT, models
python setup.py                   # download YOLOv8n + warmup face/ocr

# 4. (Optional) DeepStream SDK — có sẵn trong JetPack 6.x
#    pyds binding: cài từ NVIDIA hoặc pyds-ext
```

Models cần có trong [models/](models/):

```
plate_yolov8n.pt       → auto export .engine (TensorRT FP16)
plate_ocr_yolov8n.pt   → auto export .engine
```

---

## Chạy

```bash
python main.py --entry               # Entry mode + web dashboard
python main.py --exit                # Exit mode
python main.py --entry --no-show     # Headless (chỉ web)
python main.py --entry --no-web      # Chỉ CV2 window
python main.py --entry --debug       # Verbose logging
python main.py --benchmark video.mp4 # Benchmark từng stage
```

Dashboard: **http://jetson-ip:8080**

**Phím tắt trong CV2 window:**
- `Q` — thoát
- `M` — chuyển Entry ↔ Exit
- `S` — in stats

<!-- ---

## Config chính ([config.yaml](config.yaml))

```yaml
camera:
  plate: "rtmp://.../live/plate"    # RTMP từ iPhone qua nginx
  face:  "rtmp://.../live/face"
  hw_decode: true                   # GStreamer NVDEC
  process_every_n: 2                # Skip frame

deepstream:
  enabled: true                     # false → fallback GStreamer + Python
  plate_config: ./configs/plate_det_config.txt
  batch_size: 2

plate_ocr:
  backend: yolo                     # "yolo" (char detect) hoặc "paddle"
  model: ./models/plate_ocr_yolov8n.pt
  imgsz: 320
  conf: 0.4

face:
  model_pack: buffalo_sc
  blur_threshold: 10.0              # Laplacian variance gate

recognition:
  face_threshold: 0.3               # Cosine sim tối thiểu khi exit
  plate_vote_frames: 5
  face_avg_frames: 3

database:
  host: localhost
  dbname: parking
  max_capacity: 500
```

--- -->

## Schema DB

```sql
active(id, plate, embedding vector(512), entry_time, conf_plate, conf_face)
parking_log(id, plate, entry_time, exit_time, duration_min, match_conf)
```

Indexes: `idx_active_plate` (btree), `idx_active_embedding` (ivfflat, cosine).

<!-- ---

## Format biển số VN hỗ trợ

| Format | Ví dụ |
|---|---|
| Mới có series + dot | `99B1-257.39` |
| Cũ có dot | `29B-123.45` |
| 2 chữ có dot | `30AB-123.45` |
| Cũ không dot | `29B-12345` |
| Mới không dot | `51G1-23456` |
| 2 chữ không dot | `30AB-12345` |
| Nước ngoài | `80-123-NG-001` (NN/NG/QT/CV) |

Logic trong [main.py](main.py#L28) (`PlateValidator`) + [engine.py](engine.py#L29) (`enforce_plate_format`).

--- -->

## Endpoints web

| Path | Mô tả |
|---|---|
| `GET /` | Dashboard HTML |
| `GET /api/stats` | Current count, fps, mode, camera status |
| `GET /api/active` | Danh sách xe trong bãi |
| `GET /api/history` | Lịch sử ra/vào gần nhất |
| `GET /stream/{plate\|face}` | JPEG snapshot frame mới nhất (annotated) |
| `WS /ws` | Live event stream (entry/exit + crop base64) |

---

## Fallback khi không có DeepStream

Khi `pyds` không import được hoặc `deepstream.enabled: false`:
- `StreamReader` dùng GStreamer `uridecodebin → nvvidconv → appsink` cho mỗi camera.
- Plate detection chạy bằng `PlateDetector` (Ultralytics YOLOv8 + TensorRT engine).
- Mọi thứ khác giữ nguyên.

---

## Hai phiên bản main

- [main.py](main.py) — luồng chuẩn, parallel OCR ∥ Face trong cùng một request.
- [main2.py](main2.py) — rearrange thứ tự engine để giảm khả năng race condition khi parallel (xem commit `4284bc1`).
