#include <cstring>
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>
#include "nvdsinfer_custom_impl.h"

/**
 * SCRFD bbox parser (only) cho DeepStream nvinfer.
 *
 * Model: SCRFD-500MF với ONNX outputs đã sửa thành 3-D [batch, anchors, K]
 * qua scripts/fix_scrfd_onnx.py (xem face_det_scrfd_b1.onnx).
 *
 * Parser chỉ emit bbox sau khi lọc score + hình học landmark; nvdsfaceembed
 * decode lại landmarks từ frame_user_meta (output-tensor-meta=1) để align và
 * tạo embedding trên GPU.
 */

static constexpr float MIN_CONF_THRESH = 0.50f;
static constexpr float NMS_THRESH  = 0.4f;
static constexpr int   NUM_ANCHORS = 2;

struct Detection {
    float x1, y1, x2, y2, conf;
    float lm[5][2];
};

static float iou(const Detection& a, const Detection& b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

static std::vector<Detection> nms(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.conf > b.conf;
              });
    std::vector<Detection> kept;
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (!suppressed[j] && iou(dets[i], dets[j]) > NMS_THRESH)
                suppressed[j] = true;
        }
    }
    return kept;
}

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

static bool valid_face_geometry(const Detection& d,
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
    float conf_thresh = MIN_CONF_THRESH;
    if (!detectionParams.perClassPreclusterThreshold.empty()) {
        conf_thresh = std::max(
            conf_thresh, detectionParams.perClassPreclusterThreshold[0]);
    }

    std::map<int, const float*> score_by_anchors;
    std::map<int, const float*> bbox_by_anchors;
    std::map<int, const float*> kps_by_anchors;

    for (const auto& layer : outputLayersInfo) {
        if (!layer.buffer) continue;
        int last_dim = layer_last_dim(layer);
        int total    = layer_total_elems(layer);
        if (last_dim <= 0 || total <= 0) continue;
        int anchors  = total / last_dim;

        const float* buf = static_cast<const float*>(layer.buffer);
        if (last_dim == 1)        score_by_anchors[anchors] = buf;
        else if (last_dim == 4)   bbox_by_anchors[anchors]  = buf;
        else if (last_dim == 10)  kps_by_anchors[anchors]   = buf;
    }

    const int strides[] = {8, 16, 32};
    std::vector<Detection> all_dets;

    for (int stride : strides) {
        int feat = static_cast<int>(networkInfo.width) / stride;
        int anchors = feat * feat * NUM_ANCHORS;

        auto sit = score_by_anchors.find(anchors);
        auto bit = bbox_by_anchors.find(anchors);
        auto kit = kps_by_anchors.find(anchors);
        if (sit == score_by_anchors.end() || bit == bbox_by_anchors.end() ||
                kit == kps_by_anchors.end())
            continue;

        const float* score = sit->second;
        const float* bbox  = bit->second;
        const float* kps   = kit->second;

        for (int i = 0; i < anchors; i++) {
            float s = score[i];
            if (s < conf_thresh) continue;
            int loc = i / NUM_ANCHORS;
            int y   = loc / feat;
            int x   = loc % feat;
            float cx = x * stride;
            float cy = y * stride;
            Detection d;
            d.x1   = cx - bbox[i * 4 + 0] * stride;
            d.y1   = cy - bbox[i * 4 + 1] * stride;
            d.x2   = cx + bbox[i * 4 + 2] * stride;
            d.y2   = cy + bbox[i * 4 + 3] * stride;
            d.conf = s;
            for (int p = 0; p < 5; p++) {
                d.lm[p][0] = cx + kps[i * 10 + p * 2 + 0] * stride;
                d.lm[p][1] = cy + kps[i * 10 + p * 2 + 1] * stride;
            }
            if (valid_face_geometry(d, networkInfo.width,
                                    networkInfo.height)) {
                all_dets.push_back(d);
            }
        }
    }

    auto kept = nms(all_dets);

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
