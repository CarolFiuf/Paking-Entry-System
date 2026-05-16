#include "scrfd_decode.h"

#include <algorithm>
#include <cmath>

float face_iou(const DecodedFace &a, const DecodedFace &b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

void scrfd_decode(const ScrfdLayers &layers, int net_w, int net_h,
                  float conf_thresh, std::vector<DecodedFace> &out) {
    (void)net_h;
    out.clear();

    const int strides[] = {8, 16, 32};
    for (int stride : strides) {
        int feat = net_w / stride;
        int anchors = feat * feat * SCRFD_NUM_ANCHORS;

        auto sit = layers.scores.find(anchors);
        auto bit = layers.boxes.find(anchors);
        auto kit = layers.kps.find(anchors);
        if (sit == layers.scores.end() || bit == layers.boxes.end() ||
                kit == layers.kps.end())
            continue;

        const float *S = sit->second;
        const float *B = bit->second;
        const float *K = kit->second;

        for (int i = 0; i < anchors; i++) {
            float score = S[i];
            if (score < conf_thresh) continue;
            int loc = i / SCRFD_NUM_ANCHORS;
            int y = loc / feat;
            int x = loc % feat;
            float cx = x * stride;
            float cy = y * stride;
            DecodedFace d{};
            d.conf = score;
            d.x1 = cx - B[i * 4 + 0] * stride;
            d.y1 = cy - B[i * 4 + 1] * stride;
            d.x2 = cx + B[i * 4 + 2] * stride;
            d.y2 = cy + B[i * 4 + 3] * stride;
            for (int p = 0; p < 5; p++) {
                d.lm[p][0] = cx + K[i * 10 + p * 2 + 0] * stride;
                d.lm[p][1] = cy + K[i * 10 + p * 2 + 1] * stride;
            }
            out.push_back(d);
        }
    }
}

std::vector<DecodedFace> scrfd_nms(std::vector<DecodedFace> &dets,
                                   float iou_thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const DecodedFace &a, const DecodedFace &b) {
                  return a.conf > b.conf;
              });
    std::vector<DecodedFace> kept;
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (!suppressed[j] && face_iou(dets[i], dets[j]) > iou_thresh)
                suppressed[j] = true;
        }
    }
    return kept;
}
