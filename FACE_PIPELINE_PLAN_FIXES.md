# FACE_PIPELINE_PLAN.md - Fixes còn lại sau implementation

Mục tiêu: chỉ ghi những điểm còn cần sửa hoặc cần verify sau code hiện tại. Không nhắc lại các phase đã làm xong.

## Đã ổn, không còn là blocker

- Đã có DeepStream face chain: `PGIE face -> nvtracker -> nvdspreprocess -> SGIE embed`.
- Đã có `configs/face_preprocess_config.txt` và `preprocess/libnvds_face_align.so`.
- Đã có NPP ArcFace align trong `preprocess/nvds_face_align.cpp`.
- Đã có DB multi-embedding: `active_face_embeddings`, `entry()` nhận N embeddings, `match_exit_by_plate()`.
- Đã có landmark attach với `NvDsUserMeta`, `copy_func`, `release_func`.
- Đã có de-letterbox landmark từ SCRFD network coords về frame coords.
- DeepStream face chain không còn load InsightFace `FaceEngine` lúc startup.
- DeepStream path không còn rotate face frame lần hai.

## 1. Blocker: exit vẫn chỉ thử best track

Code hiện tại `_process_exit_ds()` dùng `_best_track_emb()`, tức chỉ verify một live face.

Plan cần giữ yêu cầu:

> Exit pass nếu any live face candidate khớp any stored embedding của cùng plate.

Cần thêm:

```yaml
face:
  exit_live_k_max: 5
```

Và sửa logic exit:

- Lấy top-K mature + fresh tracks.
- Gọi `db.match_exit_by_plate()` cho từng candidate.
- Pass nếu có bất kỳ match nào qua threshold.
- Log best sim dù pass/fail để tune threshold.

## 2. Blocker: mất landmark không được tạo zero embedding

Trong `preprocess/nvds_face_align.cpp`, nếu object thiếu landmark thì slot tensor đang để zero. SGIE vẫn có thể sinh embedding từ zero tensor, và app có thể tích lũy embedding rác.

Cần sửa hoặc ghi rõ trong plan:

- Nếu thiếu landmark: drop object khỏi preprocess batch, hoặc mark `align_ok=false` để Probe B/app bỏ qua.
- `_update_face_tracks()` nên yêu cầu `embedding is not None` và `align_ok/landmarks valid`.
- Bắt buộc test: landmark user meta còn tồn tại sau `nvtracker`.

Nếu tracker không giữ object user meta, phải reattach sau tracker bằng IoU hoặc chuyển điểm đặt `nvdspreprocess`.

## 3. Blocker: batch overflow phải đồng bộ ROI và tensor

Custom lib có `MAX_BATCH=16`. Khi `batch->units.size() > 16`, code clamp `n_units`, nhưng `roi_vector` downstream có thể vẫn chứa nhiều object hơn tensor slots.

Cần chốt một trong hai cách:

- Filter/drop object trước `nvdspreprocess` để số face <= `batch-size`.
- Hoặc trong custom preprocess cập nhật đồng bộ `seq_params.roi_vector` với tensor shape.

Không nên chỉ clamp tensor shape rồi để ROI metadata dài hơn.

## 4. Config mismatch: `face_threshold` vẫn là 0.3

`FACE_PIPELINE_PLAN.md` đề xuất `recognition.face_threshold=0.42`, nhưng `config.yaml` hiện vẫn là `0.3`.

Với multi-embedding, false-positive tăng theo số embedding lưu mỗi plate. Cần:

- Đổi default ban đầu lên `0.42`, hoặc
- Ghi rõ `0.3` chỉ là debug/dev, không dùng production.

Nên thêm bảng/log `exit_attempts` để tune ROC thay vì chỉ log pass vào `parking_log`.

## 5. Performance: DeepStream hot path vẫn dùng CPU face quality

DeepStream path không còn load InsightFace engine, nhưng `_update_face_tracks()` vẫn gọi static `FaceEngine.quality(frame_face, bbox, blur_thr)` trên frame CPU.

Chấp nhận được cho MVP, nhưng nếu mục tiêu là performance tốt nhất thì plan cần ghi rõ đây là nợ kỹ thuật:

- Thay bằng `landmark_geometry`/bbox size/detector conf.
- Nếu cần blur, chỉ crop nhỏ, không xử lý full frame.
- Không để CPU Laplacian quality nằm trên hot path DeepStream lâu dài.

## 6. Thiếu `max_track_frames`

Plan có `max_track_frames=60` để tránh embedding drift, nhưng code/config hiện chưa cap số frame update mỗi track.

Cần thêm:

```yaml
face:
  max_track_frames: 60
```

Và trong `_update_face_tracks()`:

- Nếu `n_frames >= max_track_frames`, chỉ update `last_update`, không update embedding average nữa.

## 7. Không nên hardcode frame size khi de-letterbox

`pipeline._probe_attach_landmarks()` đang backproject landmark về `1280x720`.

Plan nên yêu cầu lấy kích thước từ streammux/frame meta hoặc config duy nhất. Nếu sau này đổi mux width/height, landmark align sẽ lệch mà lỗi rất khó nhìn ra ngay.

Validation cần có:

- Overlay 5 landmarks lên frame thực tế.
- Test sau khi đổi mux size.

## 8. Các mục nên cập nhật lại trong FACE_PIPELINE_PLAN.md

- Phase 6.0 không còn là việc phải thêm mới; code đã có NPP path. Đổi thành mục validation/rollback.
- Landmark contract hiện thực tế là raw `float[10]`, không phải struct đầy đủ. Plan nên ghi đúng implementation, hoặc đổi code sang struct nếu muốn metadata giàu hơn.
- Acceptance criteria cần thêm:
  - Missing landmarks không tạo embedding hợp lệ.
  - Exit top-K live faces pass đúng.
  - Batch >16 faces không làm lệch ROI/tensor mapping.
  - A/B warped crop với InsightFace cosine >= 0.95.
  - Soak 30 phút không tăng memory đều.

## Thứ tự sửa khuyến nghị

1. Chặn zero embedding khi thiếu landmark.
2. Sửa exit top-K live candidates.
3. Đồng bộ batch overflow `MAX_BATCH`.
4. Chỉnh `face_threshold` và thêm logging/tuning.
5. Thêm `max_track_frames`.
6. Thay CPU quality bằng landmark geometry khi tối ưu performance.
