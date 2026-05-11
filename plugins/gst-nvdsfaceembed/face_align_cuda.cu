#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

static constexpr int FACE_W = 112;
static constexpr int FACE_H = 112;
static constexpr int FACE_TENSOR_FLOATS = 3 * FACE_W * FACE_H;

__global__ void face_align_rgba_chw_kernel(const uint8_t *rgba, int src_w,
                                           int src_h, int src_pitch,
                                           float *dst, int slot, float m0,
                                           float m1, float m2, float m3,
                                           float m4, float m5) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= FACE_W || y >= FACE_H) return;

    float sx = m0 * x + m1 * y + m2;
    float sy = m3 * x + m4 * y + m5;
    float rgb[3] = {0.0f, 0.0f, 0.0f};

    if (isfinite(sx) && isfinite(sy) &&
        sx >= 0.0f && sy >= 0.0f &&
        sx <= (float)(src_w - 1) && sy <= (float)(src_h - 1)) {
        int x0 = (int)floorf(sx);
        int y0 = (int)floorf(sy);
        int x1 = min(x0 + 1, src_w - 1);
        int y1 = min(y0 + 1, src_h - 1);
        float ax = sx - (float)x0;
        float ay = sy - (float)y0;

        const uint8_t *p00 = rgba + y0 * src_pitch + x0 * 4;
        const uint8_t *p01 = rgba + y0 * src_pitch + x1 * 4;
        const uint8_t *p10 = rgba + y1 * src_pitch + x0 * 4;
        const uint8_t *p11 = rgba + y1 * src_pitch + x1 * 4;

        for (int c = 0; c < 3; c++) {
            float v0 = (1.0f - ax) * (float)p00[c] + ax * (float)p01[c];
            float v1 = (1.0f - ax) * (float)p10[c] + ax * (float)p11[c];
            rgb[c] = (1.0f - ay) * v0 + ay * v1;
        }
    }

    int pix = y * FACE_W + x;
    float *slot_ptr = dst + slot * FACE_TENSOR_FLOATS;
    slot_ptr[0 * FACE_W * FACE_H + pix] = (rgb[0] - 127.5f) / 127.5f;
    slot_ptr[1 * FACE_W * FACE_H + pix] = (rgb[1] - 127.5f) / 127.5f;
    slot_ptr[2 * FACE_W * FACE_H + pix] = (rgb[2] - 127.5f) / 127.5f;
}

extern "C" cudaError_t launch_face_align_rgba_chw_kernel(
    const uint8_t *rgba, int src_w, int src_h, int src_pitch, float *dst,
    int slot, const double coeffs[6], cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((FACE_W + block.x - 1) / block.x,
              (FACE_H + block.y - 1) / block.y);
    face_align_rgba_chw_kernel<<<grid, block, 0, stream>>>(
        rgba, src_w, src_h, src_pitch, dst, slot, (float)coeffs[0],
        (float)coeffs[1], (float)coeffs[2], (float)coeffs[3],
        (float)coeffs[4], (float)coeffs[5]);
    return cudaGetLastError();
}
