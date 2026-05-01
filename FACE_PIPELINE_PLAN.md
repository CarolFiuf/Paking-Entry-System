# Face Pipeline → DeepStream Migration Plan

**Mục tiêu**: Migrate face detection + embedding hiện đang chạy Python (InsightFace `buffalo_sc`) sang DeepStream PGIE+Tracker+SGIE, đồng thời chuyển từ "1 embedding/xe" → "multi-face embedding bank/xe" với match logic "exit pass khi 1 trong N embedding khớp".

**Hardware**: Jetson Orin Nano (Linux 5.15.148-tegra), DeepStream 7.1
**Author note**: Plan version v3.
- v1: bị sai architecture (giả định 1 camera).
- v2: sửa thành 2-cam, đề xuất bỏ alignment.
- v3 (current): user chọn GIỮ alignment qua **Cách C** — full GPU với `nvdspreprocess` + NPP affine warp + SGIE `input-tensor-from-meta=1`.

---

## 1. Bối cảnh

### Trạng thái hiện tại

- **2 camera RTMP** ([config.yaml:11-12](config.yaml#L11-L12)):
  - `source_id=0` = plate cam (chỉ thấy biển số)
  - `source_id=1` = face cam (chỉ thấy mặt người, **multi-face có thể xảy ra ở đây**)
- **DeepStream** đã enable cho plate detection: `nvinfer(plate, uid=1)` chạy trên cả 2 source ([pipeline.py:109-112](pipeline.py#L109-L112))
- **Face pipeline**: vẫn Python — `FaceEngine(buffalo_sc)` ở [engine.py:354-376](engine.py#L354-L376), gọi qua ThreadPoolExecutor ([main.py:312](main.py#L312))
- **Face cam có rotation**: auto-detect ở runtime ([main.py:485-502](main.py#L485-L502)) — nvinfer không tự rotate, nên phải xử lý trước PGIE_face
- **Embedding storage**: 1 vector/xe ở `active.embedding VECTOR(512)` ([database.py:104-113](database.py#L104-L113))
- **Embedding accumulator**: `EmbeddingAvg(n=3)` weighted-by-quality, 1 instance toàn cục, clear sau entry/exit success ([main.py:129-160](main.py#L129-L160))
- **Face match exit**: cosine sim với threshold 0.3 ([config.yaml:52](config.yaml#L52)), không dùng pgvector index dù `match_exit()` đã có sẵn ([database.py:218-250](database.py#L218-L250))
- **Cooldown 0.5s** sau mỗi entry/exit success ([main.py:632](main.py#L632))

### Yêu cầu

1. Tăng tốc face pipeline bằng DeepStream (TRT engine, GPU zero-copy)
2. Multi-face entry: lưu N embedding (mỗi mặt 1 embedding tích lũy độc lập, không trộn)
3. Exit pass khi candidate khớp với **bất kỳ** embedding đã lưu (driver/passenger đều exit được)

---

## 2. Kiến trúc đã chốt

```
src0(plate) → flvdemux → h264parse → nvv4l2dec → queue ───────────────────────┐
                                                                                │
src1(face)  → flvdemux → h264parse → nvv4l2dec → nvvidconv(flip-method=N) ─┐  │
                                                                            │  │
                                                                  mux.sink_0│  │mux.sink_1
                                                                            ↓  ↓
                                                                    nvstreammux(batch=2)
                                                                            ↓
                                                              nvinfer(PGIE_plate, uid=1)
                                                                            ↓
                                                              nvinfer(PGIE_face, uid=2,
                                                                      output-tensor-meta=1)
                                                                            ↓
                                                              [Probe A]: parse landmarks
                                                              từ frame_user_meta tensor,
                                                              attach NvDsUserMeta(landmarks)
                                                              vào obj_meta.obj_user_meta_list
                                                                            ↓
                                                              nvtracker(nvDCF, face only)
                                                                            ↓
                                                              nvdspreprocess(custom_lib=
                                                                  libnvds_face_align.so)
                                                              - đọc landmarks từ obj_user_meta
                                                              - NPP nppiWarpAffine_8u_C3R
                                                              - pack input tensor (N×3×112×112)
                                                                            ↓
                                                              nvinfer(SGIE_arcface, uid=3,
                                                                  input-tensor-from-meta=1,
                                                                  output-tensor-meta=1,
                                                                  operate-on-gie-id=2)
                                                                            ↓
                                                              [Probe B]: read 512-dim embedding
                                                              từ obj_user_meta của mỗi face
                                                                            ↓
                                                                    nvvidconv → RGBA
                                                                            ↓
                                                                  fakesink
```

### Lý do chọn single-pipeline (so với 2 sub-pipeline / nvstreamdemux)

- **Rotation per-source**: xử lý ở pad ingress (`nvvidconv flip-method=N` chỉ đặt trên branch src1 trước mux). Không cần demux.
- **Waste compute**: face PGIE chạy thừa trên src0 (~3ms/frame). SGIE chỉ kích hoạt khi face object tồn tại → src0 zero cost SGIE. Tracker với src0 không có face → near-zero.
- **Probe duy nhất**: filter bằng `frame_meta.source_id` + `obj_meta.unique_component_id`.
- **Đơn giản**: 1 GStreamer state machine, 1 sink, 1 probe, 1 mux.

### Filter logic trong 2 probes

**Probe A** (sau PGIE_face, trước nvtracker):
```
for frame_meta in batch (source_id==1 only):
    landmarks_per_obj = parse_scrfd_landmarks(frame_user_meta tensors)  # match by detection index
    for obj_meta in frame_meta.obj_meta_list:
        if obj_meta.unique_component_id == 2:  # PGIE_face
            attach NvDsUserMeta(LANDMARKS, float[10]) to obj_meta.obj_user_meta_list
```

**Probe B** (cuối pipeline):
```
for frame_meta in batch:
    if frame_meta.source_id == 0:    # plate cam
        for obj in objects:
            if obj.unique_component_id == 1:  # PGIE_plate
                → plate_detections
    elif frame_meta.source_id == 1:  # face cam
        for obj in objects:
            if obj.unique_component_id == 2:  # PGIE_face
                track_id  = obj.object_id      # từ nvtracker
                landmarks = read NvDsUserMeta(LANDMARKS) from obj_user_meta
                emb       = parse NvDsInferTensorMeta (uid=3) from obj_user_meta
                → faces with (bbox, conf, track_id, landmarks, emb)
```

---

## 3. Quyết định kỹ thuật chốt

| Vấn đề | Quyết định | Ghi chú |
|---|---|---|
| **Face detection model** | SCRFD-2.5GF (`det_2.5g.onnx` từ `~/.insightface/models/buffalo_sc/`) | Đã có sẵn, không cần download. Cần custom parser .so |
| **Face embedding model** | ArcFace MobileFaceNet (`w600k_mbf.onnx` từ buffalo_sc) | 512-dim, tương thích với embeddings đã lưu trong DB |
| **Face alignment** | **GIỮ qua Cách C: full GPU pipeline** | `nvdspreprocess` plugin + custom lib `libnvds_face_align.so` dùng NPP `nppiWarpAffine_8u_C3R` để warp 5 landmarks → 112x112 ArcFace template, output preprocessed tensor cho SGIE |
| **SGIE input mode** | `input-tensor-from-meta=1` | SGIE skip auto crop+resize, dùng tensor đã align bởi nvdspreprocess |
| **SGIE output parser** | KHÔNG cần custom .so | `network-type=100` + `output-tensor-meta=1` → tensor 512-dim attached vào `obj_user_meta` |
| **Landmarks attach** | Probe Python sau PGIE_face | PGIE bbox parser trả bbox; probe parse landmarks tensor (output-tensor-meta=1), match theo detection index, attach NvDsUserMeta vào obj_meta |
| **Tracker** | nvDCF | Track chỉ class face (class_id=0 ở PGIE_face) |
| **Pipeline topology** | Single pipeline + flip-method per-source ingress | Lý do ở §2 |
| **DB schema** | Bảng mới `active_face_embeddings` | Giữ `active.embedding` nullable 1-2 release để backward compat |
| **Branch** | `feature/deepstream-face-pipeline` | Không đi thẳng main |
| **Backward compat** | Feature flag `face.multi_embedding: bool` trong config | Cho phép rollback runtime |

---

## 4. Config values + lý do

```yaml
face:
  # Model & engine
  det_engine: ./models/face_det_scrfd_fp16.engine
  det_config: ./configs/face_det_config.txt
  embed_engine: ./models/face_embed_arcface_fp16.engine
  embed_config: ./configs/face_embed_config.txt
  preprocess_config: ./configs/face_preprocess_config.txt   # nvdspreprocess

  # Multi-embedding bank
  multi_embedding: true          # feature flag, rollback runtime
  k_max: 5                       # tối đa embeddings/xe — driver+passengers
  min_track_frames: 5            # ~0.17s @30fps; track phải sống ≥ ngần này mới đủ tin
  max_track_frames: 60           # ~2s; ngừng update embedding sau ngần này (tránh drift)
  track_idle_drop_frames: 30     # ~1s không update → drop track khỏi dict (chống dính xe sau)
  min_quality: 0.4               # quality score [0,1] threshold
  dedup_cos: 0.92                # 2 embedding mới giống cũ ≥ ngần này → bỏ (cùng người)
  blur_threshold: 10.0           # giữ nguyên hiện tại — Laplacian variance
  
  # Spatial gating: face cam khác plate cam, không có vehicle bbox
  # → dùng ROI cố định trên face cam (vùng giữa frame, nơi driver/passenger thường xuất hiện)
  roi_xyxy: [0.15, 0.10, 0.85, 0.90]   # tỉ lệ trên frame size; null = no ROI gating

recognition:
  face_threshold: 0.42                  # nâng từ 0.30 — bù false-positive khi N embeddings
  plate_vote_frames: 5                  # giữ nguyên
  face_avg_frames: 3                    # window size cho EmbeddingAvg per-track (KHÁC k_max)
  plate_face_max_gap_frames: 30         # ~1s; plate vote stable phải gần với face capture, tránh stale

tracker_nvdcf:
  config_file: ./configs/tracker_nvdcf.yml
  # Trong yml file:
  # maxShadowTrackingAge: 8             # giảm mạnh từ default ~30 → track chết nhanh, tránh re-assign
  # minTrackerConfidence: 0.5
  # maxTargetsPerStream: 20

camera:
  face_rotate: -1                       # -1 = auto-detect ở setup script, ghi đè sau
                                        # 0=no, 1=90CW, 2=180, 3=90CCW
                                        # nvvidconv flip-method: 0=none, 1=90CCW, 2=180, 3=90CW
                                        # mapping cv2 → nvvidconv ở engine.py setup
```

### Lý do chính

- **`min_track_frames=5`**: sweet spot. Trên 15 → rủi ro thật (vượt `maxShadowTrackingAge` của nvDCF + xa khoảng plate vote → app-level dính qua sync gap).
- **`max_track_frames=60`**: ngừng update sau 2s vì tài xế ngồi yên thì embedding sau giống cái đầu tiên — không cần cập nhật thêm, tránh drift do ánh sáng/góc đổi khi xe di chuyển.
- **`track_idle_drop_frames=30`**: 1 trong 3 invariant chống dính xe sau (xem §6).
- **`face_threshold=0.42`**: với N=5 embeddings, P(false-positive) ≈ 5×P(single). Threshold 0.30 hiện tại là ngưỡng cosine của ArcFace — quá lỏng. 0.42 là điểm tham chiếu khởi đầu, sẽ tune ở Phase 9.
- **`plate_face_max_gap_frames=30`**: 1 trong 3 invariant — plate voted xa khỏi face capture window → reject vote.
- **`maxShadowTrackingAge=8`** (nvDCF): giảm mạnh từ default ~30 để track chết nhanh, giảm xác suất re-assign track_id của xe trước cho xe sau.

---

## 5. 9 Phase

### Phase 1 — Extract ONNX & Build TRT Engines

**Mục tiêu**: 2 file `.engine` chạy được standalone, đo speed.

**Việc**:
1. Locate ONNX trong `~/.insightface/models/buffalo_sc/`:
   - `det_2.5g.onnx` (SCRFD detector, input 640x640)
   - `w600k_mbf.onnx` (ArcFace MobileFaceNet, input 112x112)
2. Copy vào `models/face_det_scrfd.onnx`, `models/face_embed_arcface.onnx`
3. Build TRT engine FP16:
   ```bash
   /usr/src/tensorrt/bin/trtexec \
     --onnx=models/face_det_scrfd.onnx \
     --saveEngine=models/face_det_scrfd_fp16.engine \
     --fp16 --workspace=2048

   /usr/src/tensorrt/bin/trtexec \
     --onnx=models/face_embed_arcface.onnx \
     --saveEngine=models/face_embed_arcface_fp16.engine \
     --fp16 --workspace=1024 \
     --minShapes=input.1:1x3x112x112 \
     --optShapes=input.1:8x3x112x112 \
     --maxShapes=input.1:16x3x112x112
   ```
4. ArcFace cần dynamic batch để SGIE batch nhiều face cùng lúc.

**Files thay đổi**: thêm 4 file vào `models/`

**Validation**:
- `trtexec --loadEngine=...` đo throughput
- Mục tiêu Orin Nano FP16: SCRFD-2.5GF ≤ 5ms, ArcFace ≤ 2ms@batch8

**Rollback**: xóa file. Pipeline cũ không bị ảnh hưởng.

**Effort**: 0.5 ngày

---

### Phase 2 — Custom SCRFD Parser (.so) + Landmarks Attach Probe

**Mục tiêu**: SCRFD output (9 tensors: 3 strides × {score, bbox, kps}) → bbox đi vào `NvDsObjectMeta` qua parser; 5 landmarks attach qua probe Python.

**Phần 2.1 — Bbox parser .so**:
1. Port từ open-source: tham khảo `marcoslucianops/DeepStream-Yolo-Face` hoặc DeepStream sample SCRFD parser.
2. Tạo `parsers/nvds_scrfd_parser.cpp`:
   - `NvDsInferParseCustomScrfd()` parse 9 output tensors theo anchor strides 8/16/32
   - Apply score threshold + NMS
   - Output `std::vector<NvDsInferParseObjectInfo>` (bbox+conf only)
3. Build:
   ```bash
   g++ -shared -fPIC -O2 -o parsers/libnvds_scrfd_parser.so \
     parsers/nvds_scrfd_parser.cpp \
     -I/opt/nvidia/deepstream/deepstream/sources/includes \
     -lnvinfer -lnvinfer_plugin
   ```
4. PGIE_face config sẽ có thêm `output-tensor-meta=1` để raw tensors vẫn available cho probe đọc landmarks.

**Phần 2.2 — Landmarks attach probe (Python)**:
1. Trong `pipeline.py` thêm probe trên source pad của `pgie_face` (CHẠY TRƯỚC nvtracker):
   ```python
   def _probe_attach_landmarks(self, pad, info, _):
       buf = info.get_buffer()
       batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
       l_frame = batch_meta.frame_meta_list
       while l_frame is not None:
           fm = pyds.NvDsFrameMeta.cast(l_frame.data)
           if fm.source_id != 1:
               l_frame = l_frame.next; continue
           
           # Đọc raw tensors từ frame_user_meta (PGIE uid=2 với output-tensor-meta=1)
           tensors = self._extract_tensor_meta(fm, gie_uid=2)
           if tensors is not None:
               landmarks_all = self._parse_scrfd_landmarks(tensors)  # list of (5,2) float
               # Match landmarks → obj_meta theo detection index
               # (parser SCRFD và probe phải dùng cùng thứ tự NMS)
               l_obj = fm.obj_meta_list; idx = 0
               while l_obj is not None and idx < len(landmarks_all):
                   om = pyds.NvDsObjectMeta.cast(l_obj.data)
                   if om.unique_component_id == 2:
                       self._attach_landmarks_user_meta(batch_meta, om, landmarks_all[idx])
                       idx += 1
                   l_obj = l_obj.next
           l_frame = l_frame.next
       return Gst.PadProbeReturn.OK
   ```
2. Hàm `_attach_landmarks_user_meta()` tạo `NvDsUserMeta`:
   ```python
   def _attach_landmarks_user_meta(self, batch_meta, obj_meta, landmarks_5x2):
       um = pyds.nvds_acquire_user_meta_from_pool(batch_meta)
       data = landmarks_5x2.flatten().astype(np.float32).tobytes()
       um.user_meta_data = data
       um.base_meta.meta_type = self.LANDMARKS_META_TYPE  # custom int, vd 0x10001000
       pyds.nvds_add_user_meta_to_obj(obj_meta, um)
   ```
3. nvdspreprocess (Phase 6) sẽ đọc lại `LANDMARKS_META_TYPE` từ `obj_user_meta` để warp.

**Files thay đổi**: thêm `parsers/nvds_scrfd_parser.cpp`, build `parsers/libnvds_scrfd_parser.so`. Sửa `pipeline.py` thêm probe A.

**Validation**:
- Smoke test PGIE standalone: in `n_detections` mỗi frame
- Print landmarks 5 điểm cho 1 face đứng yên — xem điểm nằm đúng vị trí mắt/mũi/miệng (overlay qua cv2.circle để debug)
- So với InsightFace cùng frame: `|n_dets_diff| ≤ 1` ≥ 90% frame; landmarks IoU >0.8

**Rollback**: nếu parser stuck quá 2 ngày → tạm dùng YOLOv8-face với landmarks output extension (cộng đồng có sẵn) + parser cũ `libnvds_yolov8_parser.so` đã có.

**Effort**: 1.5 ngày

---

### Phase 3 — Face Cam Rotation Setup-time

**Mục tiêu**: chuyển auto-detect rotation từ runtime sang setup-time, lưu vào config.

**Lý do**: nvinfer không tự rotate. Nếu PGIE_face chạy trước rotation → detect sai. Auto-detect runtime ([main.py:485-502](main.py#L485-L502)) không thể giữ trong DeepStream pipeline.

**Việc**:
1. Tạo `setup_rotation.py` — script CLI:
   - Connect face cam (RTMP từ config), grab vài frame
   - Thử 4 hướng: 0/90CW/180/90CCW
   - Chạy `FaceEngine` thử mỗi hướng, hướng nào detect được → ghi nhận
   - Print mapping: `cv2_code → nvvidconv flip-method`
   - Update `config.yaml: camera.face_rotate` (manual hoặc auto-write)
2. Sửa `pipeline.py`: đọc `camera.face_rotate`, áp dụng `nvvidconv flip-method` ở src1 ingress (xem Phase 4).
3. Loại bỏ `_rotate_face()` runtime ở [main.py:485-502](main.py#L485-L502) khi DeepStream mode (giữ cho fallback).

**Mapping code**:
| cv2 const | giá trị | nvvidconv flip-method |
|---|---|---|
| Không xoay | 0 (custom) | 0 |
| `ROTATE_90_CLOCKWISE` | 1 | 3 (90CW) |
| `ROTATE_180` | 2 | 2 |
| `ROTATE_90_COUNTERCLOCKWISE` | 3 | 1 (90CCW) |

**Files thay đổi**: thêm `setup_rotation.py`; sửa `pipeline.py`, `config.yaml`

**Validation**:
- Run `python setup_rotation.py` → in góc xoay đúng
- Frame face sau pipeline (probe extract) có orientation đúng (in 1 sample frame, eyes ở trên)

**Rollback**: giữ `_rotate_face()` runtime, bỏ `flip-method` ở pipeline. Fallback mode vẫn work.

**Effort**: 0.3 ngày

---

### Phase 4 — Restructure pipeline.py: PGIE_face + Probe A + tracker + nvdspreprocess + SGIE chain

**Mục tiêu**: pipeline chạy được với chain mới (Cách C — full GPU alignment), 2 probe attached đúng vị trí, metadata flow đúng theo source/uid.

**Việc**:
1. **Sửa `_add_rtmp_source()`** ([pipeline.py:159-211](pipeline.py#L159-L211)) — chỉ với src1:
   - Sau `nvv4l2decoder`, thêm `nvvideoconvert` với `flip-method=cfg.camera.face_rotate_nv`
   - Chain mới: `decoder → nvvidconv(flip) → queue → mux.sink_1`
2. **Sửa `_build_pipeline()`** ([pipeline.py:88-147](pipeline.py#L88-L147)):
   ```
   mux → nvinfer(plate, uid=1, config=plate_det_config.txt)
       → nvinfer(face, uid=2, config=face_det_config.txt,
                 output-tensor-meta=1)              # raw landmarks tensors cho Probe A
       → [Probe A on src pad: attach LANDMARKS user_meta vào obj_meta]
       → nvtracker(config=tracker_nvdcf.yml)
       → nvdspreprocess(config=face_preprocess_config.txt,
                        custom-lib=libnvds_face_align.so)
       → nvinfer(face_embed, uid=3, config=face_embed_config.txt,
                 input-tensor-from-meta=1, output-tensor-meta=1)
       → [Probe B on sink pad of fakesink (cuối pipeline)]
       → nvvideoconvert → capsfilter(RGBA) → fakesink
   ```
3. **Tạo configs/face_det_config.txt** (PGIE):
   ```ini
   [property]
   gpu-id=0
   net-scale-factor=0.0078431372
   offsets=127.5;127.5;127.5
   model-color-format=0
   onnx-file=../models/face_det_scrfd.onnx
   model-engine-file=../models/face_det_scrfd_fp16.engine
   batch-size=2
   network-mode=2
   num-detected-classes=1
   gie-unique-id=2
   process-mode=1
   network-type=0
   cluster-mode=2
   maintain-aspect-ratio=1
   output-tensor-meta=1                 # CẦN — Probe A đọc landmarks tensors
   parse-bbox-func-name=NvDsInferParseCustomScrfd
   custom-lib-path=../parsers/libnvds_scrfd_parser.so

   [class-attrs-all]
   pre-cluster-threshold=0.5
   nms-iou-threshold=0.4
   ```
4. **Đăng ký Probe A** trên src pad của `pgie_face`, TRƯỚC nvtracker:
   - Dùng `_probe_attach_landmarks()` (đã viết ở Phase 2.2)
   - Tracker phải nhận obj_meta đã có LANDMARKS user_meta để propagate qua frame (đảm bảo nvdspreprocess đọc được kể cả ở frame mà PGIE skip detect)
5. **Cấu hình nvdspreprocess** xem chi tiết ở Phase 6 (`face_preprocess_config.txt` + `libnvds_face_align.so`).
6. **SGIE config** xem chi tiết ở Phase 6 (`face_embed_config.txt` với `input-tensor-from-meta=1`).
7. **Đăng ký Probe B** trên sink pad fakesink:
   - Tách thành 2 hàm con: `_extract_plate_dets(obj_meta)`, `_extract_face_meta(obj_meta, frame_meta)`
   - Filter bằng `frame_meta.source_id` + `obj_meta.unique_component_id`
   - Source 0 + uid=1 → plate detections
   - Source 1 + uid=2 → face detections; đọc `obj_meta.object_id` (track_id) + parse NvDsInferTensorMeta uid=3 từ `obj_user_meta_list` để lấy embedding 512-dim
8. **Sửa state**: thay `_plate_detections` bằng cấu trúc tách bạch:
   ```python
   self._plate_dets = []           # source 0 only
   self._face_data  = []           # source 1 only — list of dict{bbox, conf, track_id, emb}
   ```
9. **API mới** `get_face_data() → list[dict]` thay cho `get_face_frame()` (giữ frame riêng cho web display).

**Files thay đổi**: `pipeline.py` (build + 2 probes), `configs/face_det_config.txt` (mới)

**Validation**:
- Pipeline khởi động không lỗi state ROAR (tất cả element link OK)
- Log mỗi 30 frames: `(n_plate_dets, n_face_dets, n_face_with_landmarks, n_face_with_embedding, embedding_norm)`
- `n_face_with_landmarks == n_face_dets` (Probe A attach đủ)
- `embedding_norm` ≈ 1.0 sau L2 normalize
- `n_face_with_embedding == n_face_dets` (mọi face đều có embedding)
- Track_id ổn định: 1 mặt đứng yên 50 frame → track_id không đổi

**Rollback**: comment chain face (PGIE_face → SGIE) → quay về plate-only. Fallback Python mode vẫn work.

**Effort**: 2 ngày (tăng từ v2 do thêm Probe A + nvdspreprocess wiring)

---

### Phase 5 — nvDCF Tracker Config

**Mục tiêu**: tracker config tối ưu cho face cam — chống re-assign track_id, chống ID switch.

**Việc**:
1. Copy template từ DeepStream samples:
   ```bash
   cp /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml \
      configs/tracker_nvdcf.yml
   ```
2. Sửa `configs/tracker_nvdcf.yml`:
   ```yaml
   BaseConfig:
     minDetectorConfidence: 0.5
   
   TargetManagement:
     maxTargetsPerStream: 20
     maxShadowTrackingAge: 8           # giảm mạnh từ 30 — chống re-assign
     probationAge: 3                    # track phải xác nhận trong 3 frame
     earlyTerminationAge: 1
   
   TrajectoryManagement:
     useUniqueID: 1                     # track_id unique trên toàn pipeline
   
   DataAssociator:
     thresholdMahalanobisDistMetric: 4.0
     minMatchingScore4Overall: 0.0
     minMatchingScore4SizeSimilarity: 0.6
     minMatchingScore4Iou: 0.0
     minTrackerConfidence: 0.5
   ```
3. (Optional) Set `trackingClassesList: [0]` nếu version DeepStream support — chỉ track class face.

**Files thay đổi**: `configs/tracker_nvdcf.yml` (mới)

**Validation**:
- 1 mặt đứng yên 50 frame → track_id không đổi
- 2 mặt cùng frame → 2 track_id phân biệt
- 1 mặt biến mất, 1 mặt mới xuất hiện sau 0.3s ở vị trí gần → track_id mới (không re-assign)

**Rollback**: dùng `tracker: "IOU"` simple Python fallback (đã mention trong [config.yaml:25](config.yaml#L25)).

**Effort**: 0.3 ngày

---

### Phase 6 — Custom nvdspreprocess Library (`libnvds_face_align.so`) + SGIE wiring

**Mục tiêu** (Cách C): viết custom nvdspreprocess C++ lib dùng NPP `nppiWarpAffine_8u_C3R` warp 5 landmarks → ArcFace 112×112 template trên GPU; pack tensor `(N, 3, 112, 112)` đẩy vào SGIE qua `input-tensor-from-meta=1`. Probe B đọc embedding từ `NvDsInferTensorMeta` của SGIE.

#### 6.1 ArcFace canonical template (5 keypoints, 112x112)

Hằng số kép (eyes, nose, mouth corners) — chuẩn InsightFace:
```cpp
// Order: left_eye, right_eye, nose, left_mouth, right_mouth (x,y)
const float ARCFACE_REF[5][2] = {
    {38.2946f, 51.6963f},
    {73.5318f, 51.5014f},
    {56.0252f, 71.7366f},
    {41.5493f, 92.3655f},
    {70.7299f, 92.2041f}
};
```

#### 6.2 Affine matrix solver (least-squares similarity)

OpenCV không có sẵn trên GPU lib path; tự viết solver 5-điểm similarity (4 tham số: scale·cos, scale·sin, tx, ty) qua normal equations — closed form, ~50 phép tính float, chạy trên CPU sau khi đọc landmarks (đủ rẻ với bbox count thấp). Kết quả là ma trận `2x3 float` truyền cho NPP.

```cpp
// Pseudocode
void compute_similarity_2x3(const float src[5][2], const float dst[5][2],
                             float M[2][3]) {
    // Solve [a -b tx; b a ty] * src = dst (least-squares)
    // Closed-form: see Umeyama (1991)
    // ...
}
```

#### 6.3 GPU warp + pack tensor

Pipeline trong `CustomTransformation()` callback của nvdspreprocess:
1. Lấy `NvBufSurface` của input frame (NV12 từ nvstreammux)
2. Convert NV12 → BGR/RGB qua `nvbufsurface` (hoặc dùng NPP `nppiNV12ToRGB_8u_P2C3R`)
3. Cho mỗi obj_meta của batch:
   - Đọc `LANDMARKS` user_meta (Probe A đã attach ở Phase 2.2)
   - Compute 2x3 affine matrix bằng `compute_similarity_2x3`
   - `nppiWarpAffine_8u_C3R(src, src_size, src_step, src_roi, dst_buf_112x112, ...)`
   - Normalize/quantize (ArcFace: `(x - 127.5) / 127.5` → float)
   - Permute HWC → CHW, copy vào batch tensor slot
4. Set `GstNvDsPreProcessTensor` meta: `(N, 3, 112, 112)` float32, attach vào batch

#### 6.4 File structure

```
preprocess/
  nvds_face_align.cpp           # main custom lib
  affine_solver.cpp             # similarity 2x3 solver
  Makefile
  libnvds_face_align.so         # build output
```

Build:
```bash
g++ -shared -fPIC -O2 \
  -I/opt/nvidia/deepstream/deepstream/sources/includes \
  -I/usr/local/cuda/include \
  -o preprocess/libnvds_face_align.so \
  preprocess/nvds_face_align.cpp \
  preprocess/affine_solver.cpp \
  -L/usr/local/cuda/lib64 -lnppig -lnppidei -lnppc \
  -lnvbufsurface -lnvdsgst_helper
```

#### 6.5 nvdspreprocess config (`configs/face_preprocess_config.txt`)

```ini
[property]
enable=1
target-unique-ids=3                 # gửi tensor cho SGIE uid=3
process-on-frame=0                  # process per-object (sau tracker)
unique-id=10
gpu-id=0

network-input-shape=16;3;112;112    # max batch 16 (cap K_max + others)
network-input-order=0               # NCHW
network-color-format=0              # RGB
processing-width=112
processing-height=112
scaling-buf-pool-size=8
tensor-buf-pool-size=8

custom-lib-path=../preprocess/libnvds_face_align.so
custom-tensor-preparation-function=CustomTensorPreparation

[user-configs]
# Custom keys mà lib đọc qua g_key_file_get_*
arcface-mean=127.5
arcface-std=127.5
landmarks-meta-type=268435457       # 0x10000001 — phải khớp Phase 2.2

[group-0]
src-ids=1                           # face cam only
operate-on-gie-id=2                 # PGIE_face
operate-on-class-ids=0
```

#### 6.6 SGIE config (`configs/face_embed_config.txt`)

```ini
[property]
gpu-id=0
model-engine-file=../models/face_embed_arcface_fp16.engine
onnx-file=../models/face_embed_arcface.onnx
batch-size=16
network-mode=2
gie-unique-id=3
operate-on-gie-id=2
operate-on-class-ids=0
process-mode=2                      # secondary
network-type=100                    # other (just emit tensor meta)

input-tensor-from-meta=1            # SKIP auto crop+resize — nhận tensor từ nvdspreprocess
output-tensor-meta=1                # publish embedding ra obj_user_meta

# KHÔNG cần:
# - net-scale-factor (đã norm trong custom lib)
# - offsets
# - model-color-format
# - infer-dims (đã set trong network-input-shape của preprocess)
```

#### 6.7 Probe B: đọc embedding từ NvDsInferTensorMeta

Trong `_extract_face_meta()`:
```python
def _extract_face_meta(obj_meta, frame_meta):
    emb = None
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        try:
            u = pyds.NvDsUserMeta.cast(l_user.data)
            if u.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                t = pyds.NvDsInferTensorMeta.cast(u.user_meta_data)
                if t.unique_id != 3:    # chỉ SGIE uid=3
                    l_user = l_user.next; continue
                import ctypes
                ptr = pyds.get_ptr(t.out_buf_ptrs_host[0])
                emb_arr = np.ctypeslib.as_array(
                    ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)),
                    shape=(512,)
                ).copy()
                n = np.linalg.norm(emb_arr)
                emb = emb_arr / n if n > 0 else emb_arr
                break
            l_user = l_user.next
        except StopIteration:
            break

    rect = obj_meta.rect_params
    return {
        "bbox": (int(rect.left), int(rect.top),
                 int(rect.left + rect.width),
                 int(rect.top + rect.height)),
        "conf": float(obj_meta.confidence),
        "track_id": int(obj_meta.object_id),
        "embedding": emb,
    }
```

**Files thay đổi/mới**:
- `preprocess/nvds_face_align.cpp` (mới, ~300-400 LOC)
- `preprocess/affine_solver.cpp` (mới, ~50 LOC)
- `preprocess/Makefile` (mới)
- `preprocess/libnvds_face_align.so` (build artifact)
- `configs/face_preprocess_config.txt` (mới)
- `configs/face_embed_config.txt` (mới — đã rút gọn so với Phase 4 v2)
- `pipeline.py` Probe B

**Validation**:
- Standalone test: dump warped 112x112 patch (qua hook trong custom lib khi `DEBUG=1`) → mở mắt/mũi/miệng đúng vị trí ArcFace template (overlay ARCFACE_REF lên patch)
- Embedding norm = 1.0 ± 1e-6
- A/B với InsightFace pipeline (cùng frame, cùng face): cosine ≥ 0.95 (vì cùng align + cùng model)
- Smoke test: 50 frame cùng người, cosine consecutive ≥ 0.85
- FPS pipeline 2 cam ≥ 25 (trên Orin Nano)

**Rollback ladder** (nếu Cách C stuck > 2 ngày):
1. Tạm bỏ alignment: SGIE chuyển `input-tensor-from-meta=0`, để SGIE tự crop+resize bbox → 112x112 (không align). Giữ pipeline GPU. Accuracy giảm ~3-5% cosine, vẫn chạy được prod.
2. Worst case: SGIE không chạy → Python fallback `face_eng()` trên crop từ bbox PGIE. Track_id từ nvDCF vẫn dùng.

**Effort**: 3 ngày (custom C++ lib + NPP wiring là phần khó nhất của plan)

---

### Phase 7 — DB Migration

**Mục tiêu**: schema hỗ trợ N embeddings/xe, zero-downtime migration.

**Việc**:
1. **Migration SQL**:
   ```sql
   CREATE TABLE active_face_embeddings (
       id          SERIAL PRIMARY KEY,
       active_id   INT NOT NULL REFERENCES active(id) ON DELETE CASCADE,
       embedding   VECTOR(512) NOT NULL,
       quality     REAL,
       track_id    INT,
       created_at  TIMESTAMPTZ DEFAULT NOW()
   );
   CREATE INDEX idx_afe_active ON active_face_embeddings(active_id);
   CREATE INDEX idx_afe_embed  ON active_face_embeddings 
       USING ivfflat (embedding vector_cosine_ops) WITH (lists = 22);

   -- Migrate dữ liệu hiện tại (1 emb/xe → 1 row trong bảng mới)
   INSERT INTO active_face_embeddings(active_id, embedding, quality)
   SELECT id, embedding, conf_face FROM active WHERE embedding IS NOT NULL;

   -- Backward compat
   ALTER TABLE active ALTER COLUMN embedding DROP NOT NULL;
   ```
2. **Sửa `database.py`**:
   - `entry(plate, embeddings: list[np.ndarray], qualities: list[float], track_ids: list[int], conf_plate, conf_face) → int`
     - Insert vào `active`, lấy `id`
     - Bulk insert vào `active_face_embeddings` (nhiều row)
     - Vẫn ghi `embedding[0]` vào `active.embedding` cho backward compat
   - `find_by_plate(plate) → dict | None`:
     - Trả về `{id, plate, embeddings: list[np.ndarray]}` (đổi field từ `embedding` → `embeddings`)
   - **THÊM** `match_exit_by_plate(plate, candidate_emb, threshold) → dict | None`:
     ```python
     SELECT a.id, a.plate, MIN(afe.embedding <=> %s::vector) AS dist
     FROM active a
     JOIN active_face_embeddings afe ON afe.active_id = a.id
     WHERE a.plate = %s
     GROUP BY a.id, a.plate
     HAVING MIN(afe.embedding <=> %s::vector) <= (1 - %s)
     LIMIT 1;
     ```
     - 1 query, dùng index, MIN distance = MAX similarity → "pass khi 1 trong N khớp"
   - `exit(record_id, match_conf)` không đổi (cascade DELETE qua FK xóa luôn embedding rows)

**Files thay đổi**: `database.py:97-216` (lớn), migration SQL chạy 1 lần

**Validation**:
- Unit test: insert xe với 3 embeddings → exit query với từng embedding pass
- Exit với embedding random → fail
- Migrate dry-run trên DB copy trước khi chạy prod
- Existing data: query `find_by_plate` cho 1 xe cũ → trả về list 1-element

**Rollback**:
```sql
DROP TABLE active_face_embeddings;
ALTER TABLE active ALTER COLUMN embedding SET NOT NULL;
```
Code cũ vẫn chạy nếu giữ feature flag.

**Effort**: 0.5 ngày

---

### Phase 8 — Per-Track Logic ở main.py + 3 Invariants

**Mục tiêu**: gộp tất cả Phase 1-7, viết logic tích lũy + lọc + ghi DB. **Quan trọng nhất** — chứa 3 invariant chống dính xe sau.

#### 8.1 Thay state singular → per-track dict

```python
# Cũ: self.face_avg = EmbeddingAvg(n=3)
# Mới:
self.face_tracks: dict[int, dict] = {}    # {track_id: {"avg": EmbeddingAvg, "last_update": int, "n_frames": int}}
self.last_plate_seen_frame = -1
self.cur_frame_id = 0
```

#### 8.2 Per-frame update trong `process_entry()`

```python
def process_entry(self, frame_plate, plate_dets, face_data):
    self.cur_frame_id += 1
    
    # ── Invariant 1: Track expiry ──
    expired = [tid for tid, t in self.face_tracks.items()
               if self.cur_frame_id - t["last_update"] > cfg.face.track_idle_drop_frames]
    for tid in expired:
        del self.face_tracks[tid]
    
    # 1) Plate detection
    if not plate_dets:
        return result
    self.last_plate_seen_frame = self.cur_frame_id    # cho invariant 3
    # ... best_p, crop, ocr ...
    
    # 2) Face accumulation per track
    if face_data:
        for face in face_data:
            tid = face["track_id"]
            if tid is None or face["embedding"] is None:
                continue
            
            # Spatial gating: ROI trên face cam
            if not in_roi(face["bbox"], frame_face.shape, cfg.face.roi_xyxy):
                continue
            
            # Quality gate
            ok, q = FaceEngine.quality(frame_face, face["bbox"], cfg.face.blur_threshold)
            if not ok or q < cfg.face.min_quality:
                continue
            
            # Frame cap (Invariant 2 phụ): max_track_frames
            t = self.face_tracks.setdefault(tid, {
                "avg": EmbeddingAvg(cfg.recognition.face_avg_frames),
                "last_update": self.cur_frame_id,
                "n_frames": 0,
            })
            if t["n_frames"] >= cfg.face.max_track_frames:
                t["last_update"] = self.cur_frame_id    # vẫn refresh để không bị expire
                continue
            
            t["avg"].update(face["embedding"], q)
            t["last_update"] = self.cur_frame_id
            t["n_frames"] += 1
    
    # 3) Vote
    stable = self.plate_voter.vote(plate)
    if not stable:
        return result
    
    # ── Invariant 3: Plate-Face sync ──
    # Vote stable PHẢI gần với face capture window
    has_recent_face = any(
        self.cur_frame_id - t["last_update"] <= cfg.recognition.plate_face_max_gap_frames
        for t in self.face_tracks.values()
    )
    if not has_recent_face:
        log.warning(f"ENTRY: plate '{stable}' voted nhưng face capture quá xa "
                    f"(>{cfg.recognition.plate_face_max_gap_frames} frames) — reject")
        self.plate_voter.clear()
        return result
    
    # 4) Build embedding bank
    candidates = [
        (tid, t["avg"]._latest, t["avg"]._latest_quality, t["n_frames"])
        for tid, t in self.face_tracks.items()
        if t["n_frames"] >= cfg.face.min_track_frames and t["avg"].ready
    ]
    if not candidates:
        log.debug("ENTRY: bank rỗng — chưa track nào đủ chín")
        return result
    
    # Sort by quality desc, dedup, cap
    candidates.sort(key=lambda x: x[2], reverse=True)
    bank = []
    for tid, emb, q, _ in candidates:
        if any(np.dot(emb, b[0]) > cfg.face.dedup_cos for b in bank):
            continue
        bank.append((emb, q, tid))
        if len(bank) >= cfg.face.k_max:
            break
    
    # 5) Insert
    code = self.db.entry(
        stable,
        embeddings=[b[0] for b in bank],
        qualities=[b[1] for b in bank],
        track_ids=[b[2] for b in bank],
        conf_plate=ocr_conf, conf_face=bank[0][1]
    )
    if code > 0:
        self.plate_voter.clear()
        self.face_tracks.clear()
        log.info(f"✅ ENTRY: {stable} bank={len(bank)} embeddings")
    
    return result
```

#### 8.3 Exit logic

```python
def process_exit(self, frame_face, frame_plate, plate_dets, face_data):
    self.cur_frame_id += 1
    
    # Track expiry (Invariant 1)
    expired = [tid for tid, t in self.face_tracks.items()
               if self.cur_frame_id - t["last_update"] > cfg.face.track_idle_drop_frames]
    for tid in expired:
        del self.face_tracks[tid]
    
    # ... plate detection, vote, ocr (giống entry) ...
    
    # Face accumulation (giống entry)
    # ...
    
    if not stable_plate:
        return result
    
    # Pick BEST track for matching (high quality, alive)
    candidates = sorted(
        [t for t in self.face_tracks.values() if t["avg"].ready],
        key=lambda t: t["avg"]._latest_quality, reverse=True
    )
    if not candidates:
        return result
    candidate_emb = candidates[0]["avg"]._latest
    
    # Match against bank in DB — 1 query, MIN distance
    match = self.db.match_exit_by_plate(
        plate=stable_plate,
        candidate_emb=candidate_emb,
        threshold=cfg.recognition.face_threshold
    )
    if not match:
        log.debug(f"EXIT: {stable_plate} — không embedding nào match candidate")
        # log similarity cao nhất để tune
        return result
    
    # Pass!
    self.db.exit(match["id"], match["sim"])
    self.plate_voter.clear()
    self.face_tracks.clear()
    log.info(f"✅ EXIT: {match['plate']} (sim={match['sim']:.3f})")
    return result
```

#### 8.4 Cooldown clear

Trong `_run_deepstream()` ([main.py:622-632](main.py#L622-L632)):
```python
if t0 < cooldown_until:
    result = {"ok": False}
    if t0 - cooldown_started > 1.0:    # cooldown vừa kết thúc
        self.face_tracks.clear()        # hard reset — Invariant 1 phụ
elif mode == "entry":
    ...
```

#### 8.5 Mode switch clear (đã có sẵn ở [main.py:651-652](main.py#L651-L652))

Bổ sung `self.face_tracks.clear()` cùng với `self.face_avg.clear()`.

#### 8.6 Sửa `EmbeddingAvg` ([main.py:129-160](main.py#L129-L160))

Thêm property `_latest_quality` (max quality trong buffer):
```python
@property
def _latest_quality(self):
    return max((q for _, q in self._buf), default=0.0)
```

#### 8.7 3 Invariants — tóm tắt

| # | Invariant | Implement | Chống cái gì |
|---|---|---|---|
| **1** | Track expiry | `track_idle_drop_frames=30` — drop track không update >1s | App-level dính xe sau |
| **2** | Frame cap | `max_track_frames=60` — ngừng update sau 2s | Embedding drift do ánh sáng/góc |
| **3** | Plate-Face sync | `plate_face_max_gap_frames=30` — reject vote nếu không có face fresh | Plate vote outdated → register nhầm xe |

Plus: hard reset `face_tracks.clear()` ở cooldown end + mode switch + entry/exit success.

**Files thay đổi**: `main.py:129-160` (EmbeddingAvg add property), `main.py:165-462` (state + process_entry + process_exit), `main.py:622-653` (run loop cooldown/mode handling), `pipeline.py` (đổi API trả `face_data` thay vì frame thuần)

**Validation**:
- E2E test với video 2 người vào xe → ≥ 2 embeddings được lưu, log `bank size`
- Exit với 1 trong 2 người → match pass
- Random face không phải 2 người trên → match fail
- **Test invariant 1**: xe A vào (face_tracks có 2 track), reject (vd full bãi) → đợi 2s → xe B vào → face_tracks chỉ chứa track của B
- **Test invariant 3**: simulate face cam disconnect 2s, plate cam vẫn hoạt động, vote stable → reject vote (no recent face)

**Rollback**: feature flag `face.multi_embedding: false` → giữ logic cũ với `EmbeddingAvg` singular (vẫn xài `face_data` từ DeepStream nhưng pick best, write 1 emb to DB như cũ).

**Effort**: 1 ngày

---

### Phase 9 — Tuning & Production

**Mục tiêu**: tune `face_threshold` và các tham số khác từ data thật.

**Việc**:
1. **Logging mọi exit attempt** vào `parking_log.match_conf` (cả pass+fail). Hiện tại chỉ log khi pass.
   - Sửa [main.py:447-453](main.py#L447-L453): log similarity ngay cả khi không match.
   - Hoặc tạo bảng riêng `exit_attempts(plate, sim, passed, ts)`.
2. **Chạy production 1-2 tuần**.
3. **Vẽ ROC**: plot histogram `match_conf` của pass vs fail. Chọn `face_threshold` tại điểm FPR ≤ 1%, TPR ≥ 95%.
4. **Tune `min_track_frames`, `k_max`, `dedup_cos`** dựa trên distribution thực tế:
   - Nếu bank size trung bình << k_max → giảm k_max
   - Nếu nhiều exit fail vì track không đủ chín → giảm min_track_frames
5. **Update [bench_full_pipeline_drops.csv](bench_full_pipeline_drops.csv)** với benchmark pipeline mới.
6. **Quyết định alignment** (Phase 10 optional):
   - Nếu accuracy đủ tốt (FPR ≤ 1%) → giữ nguyên
   - Nếu không → Phase 10 hybrid alignment (SCRFD landmarks → Python warp → Python embed)

**Files thay đổi**: log paths trong `main.py`, `database.py` (nếu thêm bảng `exit_attempts`)

**Validation**:
- ROC curve plot
- FPR ≤ 1%, TPR ≥ 95%
- FPS không suy giảm > 10% so với pipeline cũ

**Rollback**: revert threshold về giá trị cũ.

**Effort**: ongoing (2-3 tuần)

---

## 6. Tóm tắt 3 invariants chống dính xe sau

```
┌─────────────────────────────────────────────────────────────────┐
│ INVARIANT 1: Track Expiry                                       │
│   Per-frame: drop tracks không update > track_idle_drop_frames  │
│   Mục đích: dict không stale khi entry/exit fail                │
│                                                                 │
│ INVARIANT 2: Frame Cap                                          │
│   Per-track: ngừng update embedding sau max_track_frames        │
│   Mục đích: tránh drift do ánh sáng/góc khi xe đang di chuyển  │
│                                                                 │
│ INVARIANT 3: Plate-Face Sync                                    │
│   Reject plate vote nếu không có face track update gần đây      │
│   (gap > plate_face_max_gap_frames)                             │
│   Mục đích: tránh register plate cũ với face mới                │
└─────────────────────────────────────────────────────────────────┘
```

Plus hard resets:
- entry/exit success → `face_tracks.clear() + plate_voter.clear()`
- mode switch → `face_tracks.clear() + plate_voter.clear()`
- cooldown end → `face_tracks.clear()`

---

## 7. Mapping rotation cv2 → nvvidconv

| Hướng | cv2 const | cv2 code dùng trong main.py | nvvidconv flip-method |
|---|---|---|---|
| Không xoay | — | (không apply) | 0 |
| 90° CW | `ROTATE_90_CLOCKWISE` | 1 | 3 |
| 180° | `ROTATE_180` | 2 | 2 |
| 90° CCW | `ROTATE_90_COUNTERCLOCKWISE` | 3 | 1 |

(nvvidconv flip-method values: 0=identity, 1=90CCW, 2=180, 3=90CW, 4=horizontal, 5=upper-right-diagonal, 6=vertical, 7=upper-left-diagonal)

---

## 8. Tổng effort (v3 — Cách C)

| Phase | Effort | Risk |
|---|---|---|
| 1: Build engines | 0.5d | Low |
| 2: SCRFD parser .so + Landmarks attach probe | 1.5d | Med |
| 3: Rotation setup | 0.3d | Low |
| 4: Pipeline restructure (PGIE_face + Probe A + tracker + nvdspreprocess + SGIE + Probe B) | 2d | Med |
| 5: nvDCF config | 0.3d | Low |
| 6: `libnvds_face_align.so` (NPP warp) + SGIE wiring | 3d | **High** |
| 7: DB migration | 0.5d | Low |
| 8: Per-track logic + 3 invariants | 1d | High |
| 9: Tuning | 2-3 tuần | — |

**Total code**: ~9 ngày + 2-3 tuần tuning production.

Risk hotspot: Phase 6 (custom C++ NPP lib). Có rollback ladder rõ trong Phase 6.

---

## 9. Open decisions / pending

- [ ] Phase 7: chạy migration trực tiếp prod DB hay tạo DB test trước
- [ ] Phase 9: thêm bảng `exit_attempts` riêng cho ROC analysis hay chỉ dùng `parking_log.match_conf` extended

**Resolved trong v3**:
- Architecture: single pipeline + per-source `flip-method` ingress (đã chốt msg #8)
- Alignment: GIỮ qua Cách C (nvdspreprocess + NPP) (đã chốt msg #10-11)
- SCRFD parser source: port community + tự viết, user delegate (đã chốt msg #8)
- Branch: `feature/deepstream-face-pipeline` (đã chốt msg #8)
- Cách B fallback: kích hoạt nếu Cách C stuck > 2 ngày (xem rollback ladder Phase 6)

---

## 10. Files sẽ thay đổi (overall)

| File | Loại thay đổi |
|---|---|
| `models/face_det_scrfd.onnx` | Mới (copy from buffalo_sc cache) |
| `models/face_embed_arcface.onnx` | Mới (copy from buffalo_sc cache) |
| `models/face_det_scrfd_fp16.engine` | Mới (trtexec) |
| `models/face_embed_arcface_fp16.engine` | Mới (trtexec, dynamic batch 1-16) |
| `parsers/nvds_scrfd_parser.cpp` | Mới |
| `parsers/libnvds_scrfd_parser.so` | Mới (build) |
| `preprocess/nvds_face_align.cpp` | **Mới (Cách C — NPP affine warp)** |
| `preprocess/affine_solver.cpp` | **Mới (similarity 2x3)** |
| `preprocess/Makefile` | Mới |
| `preprocess/libnvds_face_align.so` | **Mới (build)** |
| `configs/face_det_config.txt` | Mới |
| `configs/face_preprocess_config.txt` | **Mới (Cách C)** |
| `configs/face_embed_config.txt` | Mới (input-tensor-from-meta=1) |
| `configs/tracker_nvdcf.yml` | Mới |
| `setup_rotation.py` | Mới |
| `pipeline.py` | Sửa lớn: ingress flip, chain mới, **2 probes (A + B)**, API mới |
| `main.py` | Sửa lớn: state per-track, 3 invariants, process_entry/exit |
| `database.py` | Sửa: schema mới, entry/find_by_plate/match_exit_by_plate |
| `config.yaml` | Thêm section `face.*` (gồm `preprocess_config`), sửa `recognition.face_threshold` |
| `engine.py` | Minor: helper detect rotation cho setup script |
| Migration SQL | Mới — chạy 1 lần |

---

## 11. Reference points trong code hiện tại

- Pipeline build: [pipeline.py:88-147](pipeline.py#L88-L147)
- RTMP source ingress: [pipeline.py:159-211](pipeline.py#L159-L211)
- Probe callback: [pipeline.py:238-310](pipeline.py#L238-L310)
- Plate config example: [configs/plate_det_config.txt](configs/plate_det_config.txt)
- YOLOv8 parser (reference): [parsers/nvds_yolov8_parser.cpp](parsers/nvds_yolov8_parser.cpp)
- FaceEngine current: [engine.py:354-411](engine.py#L354-L411)
- EmbeddingAvg current: [main.py:129-160](main.py#L129-L160)
- process_entry: [main.py:274-372](main.py#L274-L372)
- process_exit: [main.py:375-462](main.py#L375-L462)
- Cooldown logic: [main.py:622-632](main.py#L622-L632)
- Mode switch clear: [main.py:651-652](main.py#L651-L652)
- DB entry: [database.py:165-199](database.py#L165-L199)
- DB find_by_plate: [database.py:202-216](database.py#L202-L216)
- DB match_exit (existing, unused): [database.py:218-250](database.py#L218-L250)

---

**Last updated**: 2026-05-01
**Plan version**: v3 (Cách C — full GPU alignment qua nvdspreprocess + NPP `nppiWarpAffine_8u_C3R` + SGIE `input-tensor-from-meta=1`)
