#include "face_align_solver.h"

namespace face_align {

const float ARCFACE_REF[5][2] = {
    {38.2946f, 51.6963f},  // left_eye
    {73.5318f, 51.5014f},  // right_eye
    {56.0252f, 71.7366f},  // nose
    {41.5493f, 92.3655f},  // left_mouth
    {70.7299f, 92.2041f},  // right_mouth
};

// Closed-form 2D similarity (Kabsch / Umeyama, no reflection).
// Model: dst = s * R * src + t, where R is 2D rotation (no flip).
// Parameterize as M = [[a, -b, tx], [b, a, ty]] where s = sqrt(a^2+b^2).
//
// Solve via:
//   sxx = sum(src_x * src_x + src_y * src_y) (centered)
//   sdc = sum(src_x * dst_x + src_y * dst_y) (centered, "cosine-like")
//   sds = sum(src_x * dst_y - src_y * dst_x) (centered, "sine-like")
//   a   = sdc / sxx
//   b   = sds / sxx
//   tx  = dst_mean_x - (a*src_mean_x - b*src_mean_y)
//   ty  = dst_mean_y - (b*src_mean_x + a*src_mean_y)
void compute_similarity_2x3(const float src[5][2],
                             const float dst[5][2],
                             double out_M[6]) {
    double smx = 0, smy = 0, dmx = 0, dmy = 0;
    for (int i = 0; i < 5; i++) {
        smx += src[i][0]; smy += src[i][1];
        dmx += dst[i][0]; dmy += dst[i][1];
    }
    smx /= 5.0; smy /= 5.0; dmx /= 5.0; dmy /= 5.0;

    double sxx = 0, sdc = 0, sds = 0;
    for (int i = 0; i < 5; i++) {
        double sx = src[i][0] - smx, sy = src[i][1] - smy;
        double dx = dst[i][0] - dmx, dy = dst[i][1] - dmy;
        sxx += sx * sx + sy * sy;
        sdc += sx * dx + sy * dy;
        sds += sx * dy - sy * dx;
    }

    double a, b;
    if (sxx > 1e-12) {
        a = sdc / sxx;
        b = sds / sxx;
    } else {
        // Degenerate: all src points coincident → identity-ish fallback.
        a = 1.0; b = 0.0;
    }
    double tx = dmx - (a * smx - b * smy);
    double ty = dmy - (b * smx + a * smy);

    out_M[0] = a;  out_M[1] = -b; out_M[2] = tx;
    out_M[3] = b;  out_M[4] =  a; out_M[5] = ty;
}

}  // namespace face_align
