#ifndef NVDS_FACE_EMBED_META_H
#define NVDS_FACE_EMBED_META_H

#include <stdint.h>

// Slim embedding meta — sau khi split scrfddec/embed, landmarks + bbox sống
// trong ParkingFaceLandmarksMeta (plugins/common/nvds_face_landmarks_meta.h).
// Plugin embed chỉ owns embedding + quality.
//
// version=3 vì layout thay đổi (bỏ landmarks[10] + bbox[4] vs v2). Python
// ctypes phải match version này để tránh đọc nhầm offset.

#define PARKING_FACE_EMBED_META_DESC "PARKING.FACE_EMBEDDING_META"
#define PARKING_FACE_EMBED_DIMS 512

#define PARKING_FACE_EMB_FLAG_VALID 0x1u

typedef struct {
    uint32_t version;       // = 3
    uint32_t dims;          // = 512
    uint32_t flags;         // bit0 = VALID
    uint32_t reserved;
    float    embedding[PARKING_FACE_EMBED_DIMS];
    float    quality;       // [0,1] (=0 nếu plugin quality kernel reject)
} ParkingFaceEmbeddingMeta;

#endif
