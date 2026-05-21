#include <cstring>
#include <cassert>
#include <algorithm>
#include <vector>
#include "nvdsinfer_custom_impl.h"

/**
 * YOLOv8 character OCR parser cho DeepStream nvinfer (SGIE).
 *
 * Model: plate_ocr_yolov8n.engine — 36 class (0-9 + A-Z), imgsz=320.
 * Output: [1, 4+36, N_anchors] = [1, 40, 2100] cho imgsz=320.
 * Layout giống YOLOv8 ultralytics export: [dims, anchors].
 *
 * Khác parser plate detector:
 *  - NUM_CLASSES = 36
 *  - N_anchors detect runtime từ layer dims (không hardcode 8400)
 *  - NMS dùng inter-class hợp lý: cùng class NMS chuẩn, khác class chỉ
 *    suppress khi overlap rất cao (ký tự liền kề có thể chạm nhẹ).
 *  - Threshold đọc từ detectionParams.perClassPreclusterThreshold[0]
 *    (class-attrs-all trong config) thay vì hardcode.
 */

static constexpr int NUM_CLASSES = 36;
static constexpr float DEFAULT_CONF_THRESH = 0.3f;
static constexpr float NMS_THRESH_SAME = 0.45f;
static constexpr float NMS_THRESH_DIFF = 0.85f;

struct Detection {
    float x1, y1, x2, y2, conf;
    int cls;
};

static inline float iou(const Detection& a, const Detection& b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
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
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            float ov = iou(dets[i], dets[j]);
            float thr = (dets[i].cls == dets[j].cls)
                ? NMS_THRESH_SAME : NMS_THRESH_DIFF;
            if (ov > thr) suppressed[j] = true;
        }
    }
    return result;
}

extern "C" bool NvDsInferParseYoloV8PlateOCR(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;

    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const float* output = (const float*)layer.buffer;
    if (!output) return false;

    const int dims = 4 + NUM_CLASSES;

    // Detect N_anchors runtime: total elements / dims.
    int total = 1;
    for (unsigned d = 0; d < layer.inferDims.numDims; d++)
        total *= layer.inferDims.d[d];
    int anchors = total / dims;
    if (anchors <= 0) return false;

    // Threshold: ưu tiên config (class-attrs-all → perClassPreclusterThreshold[0]).
    float conf_thr = DEFAULT_CONF_THRESH;
    if (!detectionParams.perClassPreclusterThreshold.empty()) {
        conf_thr = detectionParams.perClassPreclusterThreshold[0];
    }

    std::vector<Detection> dets;
    dets.reserve(64);

    for (int i = 0; i < anchors; i++) {
        float cx = output[0 * anchors + i];
        float cy = output[1 * anchors + i];
        float w  = output[2 * anchors + i];
        float h  = output[3 * anchors + i];

        float max_conf = 0;
        int max_cls = 0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            float score = output[(4 + c) * anchors + i];
            if (score > max_conf) {
                max_conf = score;
                max_cls = c;
            }
        }
        if (max_conf < conf_thr) continue;

        Detection d;
        d.x1 = cx - w * 0.5f;
        d.y1 = cy - h * 0.5f;
        d.x2 = cx + w * 0.5f;
        d.y2 = cy + h * 0.5f;
        d.conf = max_conf;
        d.cls = max_cls;
        dets.push_back(d);
    }

    auto kept = nms(dets);

    for (auto& d : kept) {
        NvDsInferParseObjectInfo obj;
        obj.classId = d.cls;
        obj.detectionConfidence = d.conf;
        obj.left   = std::max(0.f, d.x1);
        obj.top    = std::max(0.f, d.y1);
        obj.width  = std::min((float)networkInfo.width,  d.x2) - obj.left;
        obj.height = std::min((float)networkInfo.height, d.y2) - obj.top;
        if (obj.width > 0 && obj.height > 0)
            objectList.push_back(obj);
    }

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV8PlateOCR);
