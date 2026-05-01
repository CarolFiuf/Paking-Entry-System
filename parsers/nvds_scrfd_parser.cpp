#include <cstring>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>
#include "nvdsinfer_custom_impl.h"

/**
 * SCRFD bbox parser cho DeepStream nvinfer (network-type=0 detector).
 *
 * Model: SCRFD-2.5GF (det_2.5g.onnx) — InsightFace buffalo_m pack.
 * Input: 640x640.
 * Output (9 layers, batch=1 view):
 *   stride 8 :  score (12800,1), bbox (12800,4), kps (12800,10)   feat 80x80
 *   stride 16:  score (3200, 1), bbox (3200, 4), kps (3200, 10)   feat 40x40
 *   stride 32:  score (800,  1), bbox (800,  4), kps (800,  10)   feat 20x20
 *   NUM_ANCHORS_PER_LOC = 2
 *
 * Bbox encoding: (l, t, r, b) là khoảng cách (theo stride units) từ anchor
 * center → 4 cạnh. anchor_centers = (x*stride, y*stride), iteration y-major.
 *
 * Parser này chỉ output bbox + conf vào NvDsInferParseObjectInfo.
 * Landmarks sẽ được attach bởi Probe A trong Python (đọc lại 9 tensors qua
 * output-tensor-meta=1). Để Probe A match landmarks → obj_meta theo index,
 * cả parser và probe phải emit detections theo CÙNG ORDER:
 *   1) Iterate strides theo thứ tự [8, 16, 32]
 *   2) Trong mỗi stride: y outer, x inner, anchor innermost
 *   3) Score threshold + NMS (sort by conf desc, IoU > NMS_THRESH suppress)
 *   4) Output kept detections theo NMS-sorted order
 */

static constexpr float CONF_THRESH = 0.5f;
static constexpr float NMS_THRESH  = 0.4f;
static constexpr int   NUM_ANCHORS = 2;

struct Detection {
    float x1, y1, x2, y2, conf;
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

extern "C" bool NvDsInferParseCustomScrfd(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // Map: anchors_count → (score_buf, bbox_buf)
    // anchors_count = H*W*NUM_ANCHORS, dùng để derive stride.
    std::map<int, const float*> score_by_anchors;
    std::map<int, const float*> bbox_by_anchors;

    for (const auto& layer : outputLayersInfo) {
        if (!layer.buffer) continue;
        int last_dim = layer_last_dim(layer);
        int total    = layer_total_elems(layer);
        if (last_dim <= 0 || total <= 0) continue;
        int anchors  = total / last_dim;

        const float* buf = static_cast<const float*>(layer.buffer);
        if (last_dim == 1)        score_by_anchors[anchors] = buf;
        else if (last_dim == 4)   bbox_by_anchors[anchors]  = buf;
        // last_dim == 10 → landmarks, parser bỏ qua
    }

    // Strides expected: [8, 16, 32]. feat_size = networkInfo.width / stride.
    const int strides[] = {8, 16, 32};
    std::vector<Detection> all_dets;

    for (int stride : strides) {
        int feat = static_cast<int>(networkInfo.width) / stride;
        int anchors = feat * feat * NUM_ANCHORS;

        auto sit = score_by_anchors.find(anchors);
        auto bit = bbox_by_anchors.find(anchors);
        if (sit == score_by_anchors.end() || bit == bbox_by_anchors.end())
            continue;

        const float* score = sit->second;
        const float* bbox  = bit->second;

        for (int i = 0; i < anchors; i++) {
            float s = score[i];
            if (s < CONF_THRESH) continue;

            int loc = i / NUM_ANCHORS;     // H*W index
            int y   = loc / feat;
            int x   = loc % feat;
            float cx = x * stride;
            float cy = y * stride;

            float l = bbox[i * 4 + 0];
            float t = bbox[i * 4 + 1];
            float r = bbox[i * 4 + 2];
            float b = bbox[i * 4 + 3];

            Detection d;
            d.x1   = cx - l * stride;
            d.y1   = cy - t * stride;
            d.x2   = cx + r * stride;
            d.y2   = cy + b * stride;
            d.conf = s;
            all_dets.push_back(d);
        }
    }

    auto kept = nms(all_dets);

    for (const auto& d : kept) {
        NvDsInferParseObjectInfo obj;
        obj.classId             = 0;          // single class: face
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
