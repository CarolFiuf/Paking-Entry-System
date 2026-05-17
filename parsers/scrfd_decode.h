#pragma once

#include <map>
#include <vector>

struct DecodedFace {
    float x1, y1, x2, y2, conf;
    float lm[5][2];
};

constexpr int SCRFD_NUM_ANCHORS = 2;
constexpr float SCRFD_NMS_IOU = 0.4f;

float face_iou(const DecodedFace &a, const DecodedFace &b);

struct ScrfdLayers {
    std::map<int, const float *> scores;
    std::map<int, const float *> boxes;
    std::map<int, const float *> kps;
};

void scrfd_decode(const ScrfdLayers &layers, int net_w, int net_h,
                  float conf_thresh, std::vector<DecodedFace> &out);

std::vector<DecodedFace> scrfd_nms(std::vector<DecodedFace> &dets,
                                   float iou_thresh = SCRFD_NMS_IOU);

// Loại false positive bằng kiểm tra hình học landmark (mắt/mũi/miệng).
// Toạ độ phải còn trong net coords (chưa transform về frame).
bool valid_face_geometry(const DecodedFace &d, float net_w, float net_h);
