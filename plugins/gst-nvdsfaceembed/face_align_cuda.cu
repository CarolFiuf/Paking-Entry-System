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

// Per-face quality: Laplacian variance (proxy for blur) + mean brightness,
// đọc kênh G của aligned CHW tensor (đã normalize về [-1,1]). Output 2 float
// mỗi face: [blur_var, mean_brightness] trên thang [0,255].
__global__ void face_quality_kernel(const float *aligned, float *out,
                                    int batch_size) {
    int face = blockIdx.x;
    if (face >= batch_size) return;
    int tid = threadIdx.x;

    // Channel G nằm ở offset 1 * 112*112 trong CHW của face.
    const float *g = aligned + face * FACE_TENSOR_FLOATS +
                     1 * FACE_W * FACE_H;

    __shared__ float s_sum[256];
    __shared__ float s_lap2[256];

    float local_sum = 0.0f;
    float local_lap2 = 0.0f;

    // Interior 110×110 = 12100 pixel (bỏ 1 pixel viền cho stencil 4-neighbor).
    const int N_INTERIOR = 110 * 110;
    for (int idx = tid; idx < N_INTERIOR; idx += 256) {
        int yi = (idx / 110) + 1;
        int xi = (idx % 110) + 1;

        // Đưa về [0, 255]: orig = norm * 127.5 + 127.5
        float c  = g[yi       * FACE_W + xi]     * 127.5f + 127.5f;
        float up = g[(yi - 1) * FACE_W + xi]     * 127.5f + 127.5f;
        float dn = g[(yi + 1) * FACE_W + xi]     * 127.5f + 127.5f;
        float lf = g[yi       * FACE_W + xi - 1] * 127.5f + 127.5f;
        float rt = g[yi       * FACE_W + xi + 1] * 127.5f + 127.5f;

        float lap = 4.0f * c - up - dn - lf - rt;
        local_sum  += c;
        local_lap2 += lap * lap;
    }

    s_sum[tid] = local_sum;
    s_lap2[tid] = local_lap2;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid]  += s_sum[tid + s];
            s_lap2[tid] += s_lap2[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float n_px = (float)N_INTERIOR;
        float mean = s_sum[0] / n_px;
        // Laplacian high-pass có E[L] ≈ 0, dùng E[L^2] làm variance approx.
        float var = s_lap2[0] / n_px;
        out[face * 2 + 0] = var;
        out[face * 2 + 1] = mean;
    }
}

extern "C" cudaError_t launch_face_quality_kernel(
    const float *aligned, float *out, int batch_size, cudaStream_t stream) {
    if (batch_size <= 0) return cudaSuccess;
    dim3 block(256);
    dim3 grid(batch_size);
    face_quality_kernel<<<grid, block, 0, stream>>>(aligned, out, batch_size);
    return cudaGetLastError();
}
