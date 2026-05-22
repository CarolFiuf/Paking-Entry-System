#ifndef NVDS_FACE_LANDMARKS_META_H
#define NVDS_FACE_LANDMARKS_META_H

#include <stdint.h>

// User-meta type attached lên NvDsObjectMeta bởi nvdsscrfddec (face detection
// decoder) hoặc nhánh decode trong nvdsfaceembed monolithic. Carrier dữ liệu
// landmarks + bbox đã transform về frame coord, để nvdsfaceembed downstream
// align ArcFace mà không phải re-decode SCRFD tensor.
//
// Lifecycle:
//   scrfddec / monolithic decode → tạo ParkingFaceLandmarksMeta, gắn vào
//   obj_meta->obj_user_meta_list. Tracker (nvtracker) đặt giữa scrfddec và
//   embed plugin sẽ preserve obj_user_meta_list (verified DS 7.1).
//   nvdsfaceembed đọc meta này, dùng landmarks để align face vào template
//   ArcFace 112×112, attach ParkingFaceEmbeddingMeta (chỉ embedding+quality)
//   trở lại cùng obj_meta.
//
// version=1 hiện tại. Khi đổi layout, bump version + cập nhật Python ctypes.

#define PARKING_FACE_LANDMARKS_META_DESC "PARKING.FACE_LANDMARKS_META"

// Bit flags
#define PARKING_FACE_LM_FLAG_VALID 0x1u

typedef struct {
    uint32_t version;       // = 1
    uint32_t flags;         // bit0 = VALID
    float    landmarks[10]; // 5 (x,y) tại frame coord (đã revert letterbox)
    float    bbox[4];       // x1,y1,x2,y2 đã clip vào frame
    float    det_conf;      // SCRFD anchor score (post-NMS)
} ParkingFaceLandmarksMeta;

#endif
