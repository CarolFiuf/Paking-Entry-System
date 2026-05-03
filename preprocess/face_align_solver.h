// Pure CPU 2D similarity solver (rotation + uniform scale + translation).
// Closed-form least-squares for 5-point face landmark alignment.
//
// Used by nvds_face_align.cpp to compute 2x3 affine matrix M such that
// M * src_landmarks ≈ dst_landmarks (ARCFACE_REF canonical 5 points).
// Forward map src→dst (NPP nppiWarpAffine convention).

#ifndef FACE_ALIGN_SOLVER_H
#define FACE_ALIGN_SOLVER_H

namespace face_align {

// ArcFace canonical 5-keypoint template @ 112x112 (InsightFace standard).
// Order: left_eye, right_eye, nose, left_mouth, right_mouth.
extern const float ARCFACE_REF[5][2];

// Compute 2x3 similarity matrix M = [[a, -b, tx], [b, a, ty]] mapping
// src 5x2 → dst 5x2 in least-squares sense.
//
// Returns the 6 values in row-major: out_M[0..5] = a, -b, tx, b, a, ty.
// Suitable for direct use as NPP nppiWarpAffine_8u_C3R coefficients.
void compute_similarity_2x3(const float src[5][2],
                             const float dst[5][2],
                             double out_M[6]);

}  // namespace face_align

#endif
