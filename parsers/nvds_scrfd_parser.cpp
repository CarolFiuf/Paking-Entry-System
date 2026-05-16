#include <algorithm>
#include <cmath>
#include <vector>

#include "nvdsinfer_custom_impl.h"
#include "scrfd_decode.h"

/**
 * SCRFD bbox parser (only) cho DeepStream nvinfer.
 *
 * Model: SCRFD-500MF với ONNX outputs đã sửa thành 3-D [batch, anchors, K]
 * qua scripts/fix_scrfd_onnx.py (xem face_det_scrfd_b1.onnx).
 *
 * Parser chỉ emit bbox sau khi lọc score + hình học landmark; nvdsfaceembed
 * decode lại landmarks từ frame_user_meta (output-tensor-meta=1) để align và
 * tạo embedding trên GPU. Logic decode/NMS dùng chung qua scrfd_decode.h.
 */

// Fallback nếu nvinfer config không khai báo pre-cluster-threshold (gần như
// không bao giờ xảy ra với face_det_config.txt). Giá trị này khớp với
// DEFAULT_DECODE_CONF_THRESH của plugin nvdsfaceembed để 2 bên đồng bộ.
static constexpr float DEFAULT_CONF_THRESH = 0.30f;

static int layer_total_elems(const NvDsInferLayerInfo& l) {
    int n = 1;
    for (uint32_t i = 0; i < l.inferDims.numDims; i++)
        n *= l.inferDims.d[i];
    return n;
}

static int layer_last_dim(const NvDsInferLayerInfo& l) {
    if (l.inferDims.numDims == 0) return 1;
    return l.inferDims.d[l.inferDims.numDims - 1];
}

static bool valid_face_geometry(const DecodedFace& d,
                                float net_w, float net_h) {
    float w = d.x2 - d.x1;
    float h = d.y2 - d.y1;
    if (!std::isfinite(w) || !std::isfinite(h) || w <= 0 || h <= 0)
        return false;

    float aspect = w / (h + 1e-6f);
    if (aspect < 0.45f || aspect > 1.45f)
        return false;

    float mx1 = d.x1 - 0.25f * w;
    float my1 = d.y1 - 0.25f * h;
    float mx2 = d.x2 + 0.25f * w;
    float my2 = d.y2 + 0.25f * h;
    for (int i = 0; i < 5; i++) {
        float x = d.lm[i][0], y = d.lm[i][1];
        if (!std::isfinite(x) || !std::isfinite(y))
            return false;
        if (x < mx1 || x > mx2 || y < my1 || y > my2)
            return false;
        if (x < -16.f || x > net_w + 16.f ||
            y < -16.f || y > net_h + 16.f)
            return false;
    }

    const float *le = d.lm[0], *re = d.lm[1], *nose = d.lm[2];
    const float *lm = d.lm[3], *rm = d.lm[4];
    float eye_dx = re[0] - le[0];
    float mouth_dx = rm[0] - lm[0];
    if (eye_dx <= 0 || mouth_dx <= 0)
        return false;

    float eye_dist = std::hypot(re[0] - le[0], re[1] - le[1]);
    float mouth_dist = std::hypot(rm[0] - lm[0], rm[1] - lm[1]);
    if (eye_dist < 0.12f * w || eye_dist > 0.75f * w)
        return false;
    if (mouth_dist < 0.08f * w || mouth_dist > 0.75f * w)
        return false;

    float eye_y = 0.5f * (le[1] + re[1]);
    float mouth_y = 0.5f * (lm[1] + rm[1]);
    if (nose[1] <= eye_y - 0.15f * h || nose[1] >= mouth_y + 0.25f * h)
        return false;
    if (mouth_y <= eye_y + 0.10f * h)
        return false;

    float min_eye_x = std::min(le[0], re[0]);
    float max_eye_x = std::max(le[0], re[0]);
    if (nose[0] < min_eye_x - 0.35f * w ||
        nose[0] > max_eye_x + 0.35f * w)
        return false;

    return true;
}

extern "C" bool NvDsInferParseCustomScrfd(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // Lấy thẳng từ nvinfer config (pre-cluster-threshold). Plugin
    // nvdsfaceembed cũng đọc giá trị này qua pipeline.py để đảm bảo
    // decode hai bên dùng chung threshold.
    float conf_thresh = detectionParams.perClassPreclusterThreshold.empty()
        ? DEFAULT_CONF_THRESH
        : detectionParams.perClassPreclusterThreshold[0];

    ScrfdLayers layers;
    for (const auto& layer : outputLayersInfo) {
        if (!layer.buffer) continue;
        int last_dim = layer_last_dim(layer);
        int total    = layer_total_elems(layer);
        if (last_dim <= 0 || total <= 0) continue;
        int anchors  = total / last_dim;

        const float* buf = static_cast<const float*>(layer.buffer);
        if (last_dim == 1)        layers.scores[anchors] = buf;
        else if (last_dim == 4)   layers.boxes[anchors]  = buf;
        else if (last_dim == 10)  layers.kps[anchors]    = buf;
    }

    std::vector<DecodedFace> raw;
    scrfd_decode(layers, (int)networkInfo.width, (int)networkInfo.height,
                 conf_thresh, raw);

    std::vector<DecodedFace> valid;
    valid.reserve(raw.size());
    for (const auto& d : raw) {
        if (valid_face_geometry(d, networkInfo.width, networkInfo.height))
            valid.push_back(d);
    }

    auto kept = scrfd_nms(valid);

    for (const auto& d : kept) {
        NvDsInferParseObjectInfo obj{};
        obj.classId             = 0;
        obj.detectionConfidence = d.conf;
        obj.left                = std::max(0.f, d.x1);
        obj.top                 = std::max(0.f, d.y1);
        obj.width  = std::min((float)networkInfo.width,  d.x2) - obj.left;
        obj.height = std::min((float)networkInfo.height, d.y2) - obj.top;
        if (obj.width > 0 && obj.height > 0)
            objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomScrfd);
