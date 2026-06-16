# Smart Parking System

Hệ thống bãi đỗ xe thông minh chạy trên **Jetson Orin Nano (JetPack 6 / DeepStream 7.1)**. Nhận dạng **biển số** và **khuôn mặt** qua 2 camera RTMP độc lập, đối chiếu danh tính khi xe ra/vào, lưu trữ bằng **PostgreSQL + pgvector**, và phát dashboard real-time qua **FastAPI + WebSocket**.

Toàn bộ phần nặng (decode → detect → track → align → embed → OCR) chạy **trên GPU trong một đồ thị DeepStream duy nhất**. Python chỉ đọc metadata qua probe và xử lý logic nghiệp vụ (vote biển số, gom embedding theo identity, ghi DB). Khi không có `pyds`/DeepStream, hệ thống tự **fallback** về GStreamer + inference Python.

---

## Kiến trúc

Hệ thống dùng **hai pipeline DeepStream độc lập** (mỗi pipeline `batch=1`, nguồn riêng), không nhập chung streammux. Mỗi pipeline kết thúc bằng một `tee` tách 2 nhánh: **nhánh META** (fakesink + probe, không materialize ảnh) và **nhánh DISPLAY** (downscale GPU 640×360, throttle ~10 fps, appsink BGR cho web).

### Plate pipeline

```
RTMP ─► rtmpsrc/flvdemux/h264parse/nvv4l2dec
     ─► nvstreammux (batch=1)
     ─► nvinfer  plate_det      (YOLOv8n, gie-id=1, PGIE)        → bbox biển số
     ─► nvinfer  plate_ocr_sgie (YOLOv8n 36-class, gie-id=3, SGIE process-mode=2)
     ─► nvvideoconvert → capsfilter(RGBA)
     ─► tee ┬─ META    : fakesink + probe  (đọc biển số + crop on-demand cho OCR)
            └─ DISPLAY : queue → downscale → appsink (web snapshot)
```

- **PGIE `plate_det`**: YOLOv8n 1-class phát hiện biển số, parser `libnvds_yolov8_parser.so` (`NvDsInferParseYoloV8`).
- **SGIE `plate_ocr_sgie`**: chạy crop biển số qua YOLOv8n 36-class (0-9 + A-Z), parser `libnvds_plate_ocr_parser.so`. Mỗi ký tự là một obj_meta con (gie-id=3) gắn `parent` = obj_meta biển số.
- **Combine probe** (trên src pad của SGIE): gom các char obj_meta → **sort 2 dòng** (residual-gap clustering) → **enforce format VN** (sửa nhầm lẫn chữ↔số) → gắn chuỗi `PARKING.PLATE_TEXT_META` lên obj_meta biển số trước khi tới `tee`.

### Face pipeline

```
RTMP ─► rtmpsrc/flvdemux/h264parse/nvv4l2dec (+ flip-method rotation)
     ─► nvstreammux (batch=1)
     ─► nvinfer     face_det   (SCRFD, gie-id=2, output-tensor-meta=1)
     ─► nvdsscrfddec           (decode tensor SCRFD → obj_meta + LandmarksMeta)   [plugin local]
     ─► nvvideoconvert → capsfilter(RGBA)
     ─► nvtracker (NvDCF)       (gán object_id ổn định qua frame)
     ─► nvdsfaceembed          (CUDA 5-point align + TensorRT ArcFace → 512-d emb) [plugin local]
     ─► tee ┬─ META    : fakesink + probe  (đọc obj_meta + embedding + quality)
            └─ DISPLAY : queue → downscale → appsink (web snapshot)
```

- **PGIE `face_det`**: SCRFD xuất raw tensor (`output-tensor-meta=1`); parser `libnvds_scrfd_parser.so` là no-op. Decode + NMS thực sự nằm trong plugin `nvdsscrfddec` — threshold điều khiển bằng `face_embed_decode_conf_threshold` trong `config.yaml`.
- **`nvdsscrfddec`** (plugin local): decode tensor SCRFD thành obj_meta + 5 điểm landmark (`ParkingFaceLandmarksMeta`), đặt **trước** tracker để tracker thấy obj_meta hợp lệ.
- **`nvtracker` (NvDCF)**: gán `object_id` ổn định qua frame. Nhờ đó identity continuity do DeepStream lo — tầng Python bỏ hết logic cosine-sim / IoU bridge, chỉ còn gom embedding theo `object_id`.
- **`nvdsfaceembed`** (plugin local): CUDA warp-affine align khuôn mặt theo 5 landmark → TensorRT ArcFace (MobileFaceNet, 512-d). Tự tính quality (blur Laplacian + brightness), gắn `PARKING.FACE_EMBEDDING_META` (version 3) lên obj_meta. Hỗ trợ `face_embed_interval` để **chỉ embed lại 1 lần mỗi N+1 frame cho cùng track** (cắt ~67% ArcFace inference khi N=2).

### Tầng Python (nghiệp vụ)

```
probe META  ─►  PlateVoter (N=5, majority)            ─┐
            ─►  IdentityTracker (gom emb theo track,   ├─►  ParkingDB ─►  WebSocket
                 quality-weighted, tối đa 3 slot)      ─┘     (pgvector)      dashboard
```

---

## Cấu trúc thư mục

| Đường dẫn | Vai trò |
|---|---|
| [main.py](main.py) | Orchestrator. `ParkingSystem`, `PlateValidator`, `PlateVoter`, `IdentityTracker`; vòng lặp `process_entry` / `process_exit`; chế độ DeepStream và fallback. |
| [main2.py](main2.py) | Phiên bản alternate (rearrange thứ tự engine để giảm race condition). |
| [pipeline.py](pipeline.py) | `DeepStreamPipeline` (dựng 2 đồ thị plate/face, probe, combine OCR) và `StreamReader` (fallback GStreamer/OpenCV). |
| [engine.py](engine.py) | Model wrapper cho fallback path: `PlateDetector`, `PlateOCRYolo`, `PlateOCR` (PaddleOCR), `FaceEngine` (InsightFace). |
| [database.py](database.py) | `ParkingDB` — connection pool, pgvector cosine search, schema N-embedding/xe. |
| [web.py](web.py) | FastAPI app — REST + WebSocket + JPEG snapshot. |
| [templates/dashboard.html](templates/dashboard.html) | UI dashboard real-time. |
| [config.yaml](config.yaml) | Toàn bộ config runtime. |
| [configs/](configs/) | nvinfer/tracker configs: `plate_det_config.txt`, `plate_ocr_config.txt`, `face_det_config.txt`, `tracker_face_nvdcf.yml`. |
| [parsers/](parsers/) | Custom bbox parser (`.so` + `.cpp`): SCRFD, YOLOv8 plate, YOLOv8 plate-OCR. |
| [plugins/gst-nvdsscrfddec/](plugins/gst-nvdsscrfddec/) | GStreamer plugin: decode SCRFD tensor → obj_meta + landmarks. |
| [plugins/gst-nvdsfaceembed/](plugins/gst-nvdsfaceembed/) | GStreamer plugin: CUDA align + TensorRT ArcFace embedding. |
| [plugins/common/](plugins/common/) | Header meta dùng chung (`nvds_face_landmarks_meta.h`). |
| [models/](models/) | `.pt` / `.onnx` / `.engine` cho plate det, plate OCR, SCRFD, ArcFace. |
| [setup.py](setup.py) | Verify GPU/TensorRT + init PostgreSQL + chuẩn bị model. |
| [setup_rotation.py](setup_rotation.py) | Auto-detect góc xoay face cam, ghi `face_rotate` / `face_rotate_nv` vào config. |
| [wait_streams.py](wait_streams.py) | Chờ RTMP stream sẵn sàng trước khi khởi động. |
| [nginx_rtmp.conf](nginx_rtmp.conf) | Cấu hình nginx-rtmp (low-latency) cho 2 stream `live/plate` + `live/face`. |

---

## Luồng nghiệp vụ

### ENTRY (xe vào)
1. Face pipeline phát hiện + track + embed khuôn mặt; `IdentityTracker` gom embedding theo `object_id` (quality-weighted), commit slot khi đủ `identity_min_hits`.
2. Plate pipeline phát hiện biển số → OCR (SGIE) → combine probe ra chuỗi → `PlateValidator` (regex VN) → `PlateVoter` (N=5 frame, majority).
3. Khi biển số vote ổn định **và** có ≥1 embedding committed → `db.entry(plate, embeddings, qualities)` → push event qua WebSocket. Hỗ trợ **N embedding/xe** (xe chở nhiều người).

### EXIT (xe ra)
1. Detect + OCR biển số giống ENTRY.
2. `db.find_by_plate(plate)` → lấy danh sách embedding lúc vào.
3. So sánh cosine similarity giữa embedding live và các embedding lưu (pgvector) → nếu ≥ `face_threshold` → `db.exit()`.
4. Ghi `parking_log` (duration + match_conf).

---

## Cài đặt

### 1. Dependencies Python
```bash
pip install -r requirements.txt --break-system-packages
```

### 2. PostgreSQL + pgvector
```bash
sudo apt install postgresql postgresql-contrib postgresql-16-pgvector
python setup.py --init-db          # tạo user/db/extension + bảng
```

### 3. DeepStream & build plugin + parser
DeepStream 7.1 có sẵn trong JetPack 6 (xem `deepstream-7.1_*.deb`). `pyds` cài từ `pyds-*.whl` kèm repo.

```bash
# Parser (custom bbox)
make -C parsers

# Plugin GStreamer local (không cần install hệ thống — load qua GST_PLUGIN_PATH)
make -C plugins/gst-nvdsscrfddec
make -C plugins/gst-nvdsfaceembed
```
`pipeline.py` tự thêm 2 thư mục plugin vào `GST_PLUGIN_PATH` và scan registry lúc khởi động.

### 4. Models
Trong [models/](models/) cần có (đường dẫn `.engine`/`.onnx` khai báo trong `configs/*.txt`):

| Model | File |
|---|---|
| Plate detector (YOLOv8n) | `plate_yolov8n.{pt,onnx,engine}` |
| Plate OCR char detect (YOLOv8n 36-class) | `plate_ocr_yolov8n.{pt,onnx,engine}` |
| Face detector (SCRFD) | `face_det_scrfd_b1.onnx` → `face_det_scrfd_b1_fp16.engine` |
| Face embedding (ArcFace) | `face_embed_arcface.onnx` → `face_embed_arcface_fp16.engine` |

TensorRT engine build FP16 (`network-mode=2`). `setup.py` verify GPU/TensorRT và chuẩn bị model.

### 5. (Tùy chọn) Góc xoay face cam
```bash
python setup_rotation.py --write   # tự detect + ghi face_rotate/face_rotate_nv
```

### 6. nginx-rtmp (nguồn camera)
Push RTMP từ 2 iPhone tới `rtmp://<host>/live/plate` và `rtmp://<host>/live/face` (xem [nginx_rtmp.conf](nginx_rtmp.conf)).

---

## Chạy

```bash
python main.py --entry                # Entry mode + web dashboard
python main.py --exit                 # Exit mode
python main.py --entry --no-show      # Headless (chỉ web)
python main.py --entry --no-web       # Chỉ cv2 window
python main.py --entry --debug        # Verbose logging
python main.py --benchmark video.mp4  # Benchmark từng stage (--frames N)
```

Dashboard: **http://<jetson-ip>:8080**

---

## Cấu hình chính ([config.yaml](config.yaml))

```yaml
camera:
  plate: "rtmp://<host>/live/plate"
  face:  "rtmp://<host>/live/face"
  face_rotate_nv: 0            # nvvidconv flip-method (auto-set bởi setup_rotation.py)

deepstream:
  enabled: true                # false → fallback GStreamer + inference Python
  plate_enabled: true
  face_enabled: true
  face_embed_decode_conf_threshold: 0.50   # SCRFD score threshold (source of truth)
  face_embed_min_quality: 0.3              # quality gate trong plugin
  face_embed_interval: 2                   # embed 1/(N+1) frame mỗi track
  face_tracker_config: "./configs/tracker_face_nvdcf.yml"
  # face_det_interval / plate_det_interval  # nvinfer interval skip (0 = full)

recognition:
  face_threshold: 0.3          # cosine sim tối thiểu khi exit
  plate_vote_frames: 5
  max_identities: 3            # tối đa slot embedding/xe
  identity_min_hits: 2         # frame tối thiểu trước khi commit slot
  identity_stale_frames: 60    # dọn slot nếu track biến mất N frame
  plate_regex: "^\\d{2}[A-Z]\\d?-?\\d{3,5}\\.?\\d{0,2}$"

database:
  host: localhost
  dbname: parking
  max_capacity: 500
```

---

## Schema DB ([database.py](database.py))

```sql
active(id, plate UNIQUE, entry_time, conf_plate, conf_face)
active_faces(id, active_id → active(id) CASCADE, embedding vector(512), conf, quality)
parking_log(id, plate, entry_time, exit_time, duration_min, match_conf)
```

- **N embedding/xe**: mỗi xe (`active`) có nhiều dòng `active_faces` (xe chở nhiều người).
- Index: `idx_active_plate_unique` (unique btree), `idx_active_faces_embedding` (ivfflat cosine), `idx_active_faces_active_id`.

---

## Endpoints web ([web.py](web.py))

| Path | Mô tả |
|---|---|
| `GET /` | Dashboard HTML |
| `GET /api/stats` | Count, fps, mode, camera status |
| `GET /api/active` | Danh sách xe trong bãi |
| `GET /api/history` | Lịch sử ra/vào gần nhất |
| `GET /stream/{plate\|face}` | JPEG snapshot frame mới nhất (annotated) |
| `WS /ws` | Live event stream (entry/exit + crop base64) |

---

## Fallback khi không có DeepStream

Khi `pyds` không import được hoặc `deepstream.enabled: false`:
- `StreamReader` dùng GStreamer (`uridecodebin → nvvidconv → appsink`) cho mỗi camera.
- Plate: `PlateDetector` + `PlateOCRYolo` (Ultralytics YOLOv8 + TensorRT engine).
- Face: `FaceEngine` (InsightFace `buffalo_sc`, det + embed + quality gate trên CPU/GPU).
- Logic nghiệp vụ (vote, identity, DB, web) giữ nguyên.

---
