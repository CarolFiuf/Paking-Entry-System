#ifndef NVDS_FACE_EMBED_META_H
#define NVDS_FACE_EMBED_META_H

#include <stdint.h>

#define PARKING_FACE_EMBED_META_DESC "PARKING.FACE_EMBEDDING_META"
#define PARKING_FACE_EMBED_DIMS 512

typedef struct {
    uint32_t version;
    uint32_t dims;
    uint32_t flags;
    uint32_t reserved;
    float embedding[PARKING_FACE_EMBED_DIMS];
    float landmarks[10];
    float bbox[4];
    float quality;
} ParkingFaceEmbeddingMeta;

#endif
