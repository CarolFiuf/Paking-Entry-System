#include <cstring>
#include <cassert>
#include <algorithm>
#include <vector>
#include "nvdsinfer_custom_impl.h"

/**
 * YOLOv8 output parser cho DeepStream nvinfer.
 * Output shape: [batch, (4+num_classes), 8400]
 *   - 4 = cx, cy, w, h (normalized by input_size)
 *   - num_classes = 1 (license_plate)
 *   - 8400 = anchors
 */

static constexpr int NUM_CLASSES = 1;
static constexpr float CONF_THRESH = 0.4f;
static constexpr float NMS_THRESH = 0.45f;

struct Detection {
    float x1, y1, x2, y2, conf;
    int cls;
};

static float iou(const Detection& a, const Detection& b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, ix2-ix1) * std::max(0.f, iy2-iy1);
    float area_a = (a.x2-a.x1)*(a.y2-a.y1);
    float area_b = (b.x2-b.x1)*(b.y2-b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

static std::vector<Detection> nms(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(),
              [](auto& a, auto& b){ return a.conf > b.conf; });
    std::vector<Detection> result;
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i+1; j < dets.size(); j++) {
            if (!suppressed[j] && iou(dets[i], dets[j]) > NMS_THRESH)
                suppressed[j] = true;
        }
    }
    return result;
}

extern "C" bool NvDsInferParseYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // YOLOv8 output: [5, 8400] where 5 = cx,cy,w,h,conf(class0)
    const float* output = (const float*)outputLayersInfo[0].buffer;
    const int dims = 4 + NUM_CLASSES;  // 5
    const int anchors = 8400;

    std::vector<Detection> dets;

    for (int i = 0; i < anchors; i++) {
        // YOLOv8 output layout: [dims][anchors] (transposed)
        float cx   = output[0 * anchors + i];
        float cy   = output[1 * anchors + i];
        float w    = output[2 * anchors + i];
        float h    = output[3 * anchors + i];

        // Class scores start at index 4
        float max_conf = 0;
        int max_cls = 0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            float score = output[(4 + c) * anchors + i];
            if (score > max_conf) {
                max_conf = score;
                max_cls = c;
            }
        }

        if (max_conf < CONF_THRESH) continue;

        Detection d;
        d.x1 = cx - w / 2.0f;
        d.y1 = cy - h / 2.0f;
        d.x2 = cx + w / 2.0f;
        d.y2 = cy + h / 2.0f;
        d.conf = max_conf;
        d.cls = max_cls;
        dets.push_back(d);
    }

    auto kept = nms(dets);

    for (auto& d : kept) {
        NvDsInferParseObjectInfo obj;
        obj.classId = d.cls;
        obj.detectionConfidence = d.conf;

        // Clamp to network input dims
        obj.left   = std::max(0.f, d.x1);
        obj.top    = std::max(0.f, d.y1);
        obj.width  = std::min((float)networkInfo.width, d.x2) - obj.left;
        obj.height = std::min((float)networkInfo.height, d.y2) - obj.top;

        if (obj.width > 0 && obj.height > 0)
            objectList.push_back(obj);
    }

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV8);