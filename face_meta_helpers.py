"""
face_meta_helpers.py

Helpers cho Probe A (Phase 2.2): decode SCRFD landmarks từ 9 raw output tensors
và attach vào NvDsObjectMeta qua NvDsUserMeta.

Logic decode PHẢI khớp 1-1 với parsers/nvds_scrfd_parser.cpp:
  - Iterate strides theo thứ tự [8, 16, 32]
  - Trong mỗi stride: y outer, x inner, anchor innermost
  - Threshold conf >= 0.5
  - NMS IoU > 0.4
  - Output sorted by conf DESC

Đảm bảo thứ tự kept detections giống parser → có thể attach landmarks[i]
vào obj_meta[i] theo index iteration.
"""

import numpy as np

# Custom user_meta type ID cho landmarks. Phải khớp với:
#   - configs/face_preprocess_config.txt: user-configs.landmarks-meta-type
#   - preprocess/nvds_face_align.cpp khi đọc user_meta
LANDMARKS_META_TYPE = 0x10001000

CONF_THRESH = 0.5
NMS_THRESH  = 0.4
NUM_ANCHORS = 2
STRIDES = (8, 16, 32)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU giữa 1 box a (4,) và N boxes b (N, 4)."""
    ix1 = np.maximum(a[0], b[:, 0])
    iy1 = np.maximum(a[1], b[:, 1])
    ix2 = np.minimum(a[2], b[:, 2])
    iy2 = np.minimum(a[3], b[:, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-6)


def _nms(boxes: np.ndarray, scores: np.ndarray,
         landmarks: np.ndarray, thresh: float) -> tuple:
    """
    NMS dùng cùng algo với parser: sort by conf desc, suppress IoU > thresh.
    Returns: (kept_boxes, kept_scores, kept_landmarks) — order = NMS-sorted.
    """
    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]
    landmarks = landmarks[order]

    suppressed = np.zeros(len(boxes), dtype=bool)
    keep = []
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(i)
        if i + 1 < len(boxes):
            ious = _iou(boxes[i], boxes[i + 1:])
            suppressed[i + 1:] = suppressed[i + 1:] | (ious > thresh)
    keep = np.array(keep, dtype=np.int32)
    return boxes[keep], scores[keep], landmarks[keep]


def decode_scrfd(layers_by_stride: dict, net_w: int, net_h: int,
                 conf_thresh: float = CONF_THRESH,
                 nms_thresh: float = NMS_THRESH) -> list:
    """
    Decode SCRFD output tensors → detections.

    Args:
        layers_by_stride: dict[int, dict] với keys = stride (8, 16, 32),
            mỗi value = {"score": (A,1), "bbox": (A,4), "kps": (A,10)}
            A = feat*feat*NUM_ANCHORS, feat = net_w/stride
        net_w, net_h: input network size (640, 640)
        conf_thresh, nms_thresh: phải = parser values

    Returns:
        list of dict: [{"bbox": (x1,y1,x2,y2), "conf": float,
                        "landmarks": np.ndarray (5,2)}]
        Order = parser NMS output order (high conf first).
    """
    all_boxes = []
    all_scores = []
    all_kps = []

    for stride in STRIDES:
        if stride not in layers_by_stride:
            continue
        L = layers_by_stride[stride]
        score = np.asarray(L["score"], dtype=np.float32).reshape(-1)
        bbox  = np.asarray(L["bbox"],  dtype=np.float32).reshape(-1, 4)
        kps   = np.asarray(L["kps"],   dtype=np.float32).reshape(-1, 10)

        feat = net_w // stride
        anchors = feat * feat * NUM_ANCHORS
        if score.size != anchors:
            raise ValueError(f"stride {stride}: score size {score.size} "
                             f"!= expected {anchors}")

        # Anchor centers — y outer, x inner, anchor innermost (match parser)
        ys = np.repeat(np.arange(feat, dtype=np.float32), feat) * stride
        xs = np.tile(np.arange(feat, dtype=np.float32), feat) * stride
        cx = np.repeat(xs, NUM_ANCHORS)
        cy = np.repeat(ys, NUM_ANCHORS)

        keep_mask = score >= conf_thresh
        if not keep_mask.any():
            continue

        idx = np.where(keep_mask)[0]
        s = score[idx]
        b = bbox[idx]
        k = kps[idx]
        cx_k = cx[idx]
        cy_k = cy[idx]

        x1 = cx_k - b[:, 0] * stride
        y1 = cy_k - b[:, 1] * stride
        x2 = cx_k + b[:, 2] * stride
        y2 = cy_k + b[:, 3] * stride

        # Landmarks: 5 keypoints, mỗi point (dx, dy) offset từ anchor center
        kx = cx_k[:, None] + k[:, 0::2] * stride   # (N, 5)
        ky = cy_k[:, None] + k[:, 1::2] * stride
        kps_xy = np.stack([kx, ky], axis=-1)        # (N, 5, 2)

        boxes = np.stack([x1, y1, x2, y2], axis=-1)  # (N, 4)
        all_boxes.append(boxes)
        all_scores.append(s)
        all_kps.append(kps_xy)

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    kps_arr = np.concatenate(all_kps, axis=0)

    # Clamp box vào network input — match parser
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, net_w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, net_h)

    kept_boxes, kept_scores, kept_kps = _nms(boxes, scores, kps_arr,
                                              nms_thresh)

    out = []
    for i in range(len(kept_boxes)):
        out.append({
            "bbox": tuple(float(v) for v in kept_boxes[i]),
            "conf": float(kept_scores[i]),
            "landmarks": kept_kps[i].astype(np.float32),
        })
    return out


# ─────────────────────────────────────────────────────────────
# pyds-specific helpers (chỉ chạy trong DeepStream pipeline)
# ─────────────────────────────────────────────────────────────

def extract_pgie_face_tensors(frame_meta, gie_uid: int = 2,
                               net_w: int = 640) -> dict:
    """
    Đọc 9 raw tensors từ frame_user_meta (PGIE_face với output-tensor-meta=1).
    Trả về dict {stride: {"score","bbox","kps"}}.

    Phải call sau khi PGIE_face publish tensor meta vào frame.
    """
    import pyds
    import ctypes

    layers_by_stride = {}
    score_by_anchors = {}
    bbox_by_anchors = {}
    kps_by_anchors = {}

    l_user = frame_meta.frame_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if (user_meta.base_meta.meta_type ==
                pyds.NVDSINFER_TENSOR_OUTPUT_META):
            tm = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            if tm.unique_id == gie_uid:
                for i in range(tm.num_output_layers):
                    li = pyds.get_nvds_LayerInfo(tm, i)
                    dims = li.dims
                    # last dim → kind; total/last_dim → anchors
                    last_dim = dims.d[dims.numDims - 1] if dims.numDims else 1
                    total = 1
                    for k in range(dims.numDims):
                        total *= dims.d[k]
                    anchors = total // last_dim if last_dim else 0

                    ptr = pyds.get_ptr(tm.out_buf_ptrs_host[i])
                    arr = np.ctypeslib.as_array(
                        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)),
                        shape=(total,)
                    ).copy()

                    if last_dim == 1:
                        score_by_anchors[anchors] = arr.reshape(anchors, 1)
                    elif last_dim == 4:
                        bbox_by_anchors[anchors] = arr.reshape(anchors, 4)
                    elif last_dim == 10:
                        kps_by_anchors[anchors] = arr.reshape(anchors, 10)
        l_user = l_user.next

    for stride in STRIDES:
        feat = net_w // stride
        anchors = feat * feat * NUM_ANCHORS
        if (anchors in score_by_anchors and anchors in bbox_by_anchors
                and anchors in kps_by_anchors):
            layers_by_stride[stride] = {
                "score": score_by_anchors[anchors],
                "bbox":  bbox_by_anchors[anchors],
                "kps":   kps_by_anchors[anchors],
            }
    return layers_by_stride


def attach_landmarks_user_meta(batch_meta, obj_meta,
                                landmarks_5x2: np.ndarray):
    """
    Attach NvDsUserMeta(LANDMARKS_META_TYPE) vào obj_meta.
    Data: 10 float32 = [x0,y0,x1,y1,...,x4,y4] (network-input coords).

    nvdspreprocess (Phase 6) sẽ đọc lại type này.
    """
    import pyds
    um = pyds.nvds_acquire_user_meta_from_pool(batch_meta)
    data = np.ascontiguousarray(landmarks_5x2.flatten(),
                                 dtype=np.float32).tobytes()
    # Lưu data — pyds yêu cầu ta giữ reference cho tới khi meta được consume
    # downstream. Workaround: encode vào bytes object, attach qua
    # set_user_copyfunc_release. Để đơn giản và an toàn, ta dùng dạng
    # base64 string trong user_meta (nvinfer/tracker không chạm).
    import base64
    um.user_meta_data = base64.b64encode(data).decode("ascii")
    um.base_meta.meta_type = LANDMARKS_META_TYPE
    pyds.nvds_add_user_meta_to_obj(obj_meta, um)


def read_landmarks_user_meta(obj_meta) -> np.ndarray | None:
    """
    Đọc landmarks (5,2) từ obj_user_meta_list.
    Trả None nếu không tìm thấy.
    """
    import pyds
    import base64
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        try:
            um = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if um.base_meta.meta_type == LANDMARKS_META_TYPE:
            try:
                raw = base64.b64decode(um.user_meta_data)
                arr = np.frombuffer(raw, dtype=np.float32).reshape(5, 2)
                return arr.copy()
            except Exception:
                return None
        l_user = l_user.next
    return None
