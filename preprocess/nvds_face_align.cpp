// libnvds_face_align.so — custom nvdspreprocess library for face alignment.
//
// For each face obj_meta:
//   1) Read 5 landmarks from user_meta (attached by Probe A in Python).
//   2) Compute 2x3 similarity matrix mapping landmarks → ARCFACE_REF.
//   3) NPP NV12→RGB on the input full frame (cached per frame).
//   4) NPP nppiWarpAffine_8u_C3R → 112x112 RGB warped patch.
//   5) NPP C3→P3 split + Convert 8u→32f + per-channel normalize
//      (x-127.5)/127.5, written CHW directly into the output tensor slot.
//
// SGIE downstream consumes via input-tensor-from-meta=1.
//
// On Jetson, DeepStream buffers are usually NVBUF_MEM_SURFACE_ARRAY and
// dataPtr is NULL. We map the input GstBuffer's NvBufSurface to CUDA via EGL,
// following the allocator pattern used by gst-nvinfer-custom.

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaEGL.h>
#include <gst/gst.h>
#include <npp.h>
#include <nppi_color_conversion.h>
#include <nppi_geometry_transforms.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_arithmetic_and_logical_operations.h>

#include "nvdspreprocess_lib.h"
#include "nvdspreprocess_meta.h"
#include "nvbufsurface.h"

#include "face_align_solver.h"

// LANDMARKS user_meta type — must match Python LANDMARKS_META_TYPE in
// face_meta_helpers.py. Not in any DS enum so we use a custom int.
static constexpr int LANDMARKS_META_TYPE = 0x10001000;

static constexpr int FACE_W = 112;
static constexpr int FACE_H = 112;
static constexpr int TENSOR_SLOT_FLOATS = 3 * FACE_W * FACE_H;
// Must match network-input-shape[0] in face_preprocess_config.txt — tensor
// pool buffer is sized for this many slots; n_units must not exceed it.
static constexpr int MAX_BATCH = 16;

struct CustomCtx {
    cudaStream_t stream;
    NppStreamContext npp_ctx;

    // Full-frame RGB scratch (HWC interleaved). Reallocated when input size
    // grows. Cache key = input dataPtr pointer.
    Npp8u *rgb_buf;
    int rgb_alloc_w;
    int rgb_alloc_h;
    int rgb_w;
    int rgb_h;

    // Per-face scratch (fixed at 112×112).
    Npp8u *warp_buf;        // 3-channel interleaved RGB after warp
    Npp8u *planar_8u[3];    // each is 112×112 single-channel

    int    debug_call_count;
};

static bool ensure_rgb_buf(CustomCtx *ctx, int W, int H) {
    if (ctx->rgb_buf && W <= ctx->rgb_alloc_w && H <= ctx->rgb_alloc_h)
        return true;
    if (ctx->rgb_buf) {
        cudaFree(ctx->rgb_buf);
        ctx->rgb_buf = nullptr;
    }
    cudaError_t err = cudaMalloc(&ctx->rgb_buf, (size_t)W * H * 3);
    if (err != cudaSuccess) {
        printf("[face_align] cudaMalloc rgb_buf %dx%d failed: %s\n",
                W, H, cudaGetErrorString(err));
        return false;
    }
    ctx->rgb_alloc_w = W;
    ctx->rgb_alloc_h = H;
    return true;
}

static bool nv12_planes_to_rgb(CustomCtx *ctx, int W, int H, int colorFormat,
                               int srcStep,
                               const Npp8u *y_plane, const Npp8u *uv_plane) {
    if (colorFormat != NVBUF_COLOR_FORMAT_NV12 &&
        colorFormat != NVBUF_COLOR_FORMAT_NV12_ER) {
        printf("[face_align] unsupported colorFormat=%d (expected NV12)\n",
               colorFormat);
        return false;
    }
    if (!y_plane || !uv_plane || srcStep <= 0) {
        printf("[face_align] invalid NV12 input (y=%p uv=%p pitch=%d)\n",
               (void *)y_plane, (void *)uv_plane, srcStep);
        return false;
    }
    if (!ensure_rgb_buf(ctx, W, H)) return false;

    const Npp8u *planes[2] = {y_plane, uv_plane};
    NppStatus s = nppiNV12ToRGB_8u_P2C3R_Ctx(
        planes, srcStep, ctx->rgb_buf, W * 3, NppiSize{W, H}, ctx->npp_ctx);
    if (s != NPP_SUCCESS) {
        printf("[face_align] nppiNV12ToRGB err=%d\n", (int)s);
        return false;
    }

    ctx->rgb_w = W;
    ctx->rgb_h = H;
    return true;
}

// Convert input batch frame NV12 → device RGB. Supports both CUDA-addressable
// dataPtr and Jetson NVBUF_MEM_SURFACE_ARRAY via EGL/CUDA mapping.
static bool nv12_to_rgb_from_batch(CustomCtx *ctx,
                                   NvDsPreProcessBatch *batch,
                                   guint batch_index) {
    if (!batch || !batch->inbuf) return false;

    GstMapInfo in_map = GST_MAP_INFO_INIT;
    if (!gst_buffer_map(batch->inbuf, &in_map, GST_MAP_READ)) {
        printf("[face_align] gst_buffer_map(inbuf) failed\n");
        return false;
    }

    NvBufSurface *surf = (NvBufSurface *)in_map.data;
    if (!surf || batch_index >= surf->numFilled) {
        printf("[face_align] invalid NvBufSurface or batch_index=%u\n",
               batch_index);
        gst_buffer_unmap(batch->inbuf, &in_map);
        return false;
    }

    NvBufSurfaceParams *sp = surf->surfaceList + batch_index;
    bool ok = false;

    if (sp->dataPtr) {
        // dGPU / CUDA_UNIFIED path: dataPtr addressable from CUDA, planes
        // contiguous via planeParams offsets, pitch from planeParams.
        auto *base = (Npp8u *)sp->dataPtr;
        const Npp8u *y_plane = base + sp->planeParams.offset[0];
        const Npp8u *uv_plane = base + sp->planeParams.offset[1];
        ok = nv12_planes_to_rgb(ctx, (int)sp->width, (int)sp->height,
                                 (int)sp->colorFormat,
                                 (int)sp->planeParams.pitch[0],
                                 y_plane, uv_plane);
    } else if (surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
        // Jetson NVMM path: dataPtr=NULL. Map surface to EGLImage, then
        // register with CUDA driver to get device pointers per plane.
        if (NvBufSurfaceMapEglImage(surf, batch_index) != 0) {
            printf("[face_align] NvBufSurfaceMapEglImage failed\n");
            gst_buffer_unmap(batch->inbuf, &in_map);
            return false;
        }

        CUgraphicsResource cuda_resource = nullptr;
        CUeglFrame egl_frame;
        CUresult cuerr = cuGraphicsEGLRegisterImage(
            &cuda_resource, sp->mappedAddr.eglImage,
            CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
        if (cuerr == CUDA_SUCCESS) {
            cuerr = cuGraphicsResourceGetMappedEglFrame(
                &egl_frame, cuda_resource, 0, 0);
        }

        if (cuerr != CUDA_SUCCESS) {
            printf("[face_align] CUDA EGL register/get-frame err=%d\n",
                   (int)cuerr);
        } else if (egl_frame.frameType != CU_EGL_FRAME_TYPE_PITCH) {
            // BLOCK_LINEAR returns CUarray instead of pPitch — NPP needs
            // pitched device pointers, abort instead of guessing.
            printf("[face_align] unsupported frameType=%d (expect PITCH)\n",
                   (int)egl_frame.frameType);
        } else if (egl_frame.planeCount < 2 ||
                   !egl_frame.frame.pPitch[0] ||
                   !egl_frame.frame.pPitch[1]) {
            // NV12 must expose Y + UV separately. Abort rather than guess
            // a fallback offset — Y/UV are independent allocations on Jetson.
            printf("[face_align] EGL frame planeCount=%d Y=%p UV=%p invalid "
                   "for NV12\n",
                   egl_frame.planeCount,
                   egl_frame.frame.pPitch[0], egl_frame.frame.pPitch[1]);
        } else {
            // EGL pitch may differ from surf->planeParams.pitch[0] due to
            // driver realignment on map; always use egl_frame.pitch for NPP.
            const Npp8u *y_plane = (const Npp8u *)egl_frame.frame.pPitch[0];
            const Npp8u *uv_plane = (const Npp8u *)egl_frame.frame.pPitch[1];
            ok = nv12_planes_to_rgb(ctx,
                                     (int)egl_frame.width,
                                     (int)egl_frame.height,
                                     (int)sp->colorFormat,
                                     (int)egl_frame.pitch,
                                     y_plane, uv_plane);
            // NPP work above is enqueued on ctx->stream; finish before
            // unregistering the EGL resource.
            cudaStreamSynchronize(ctx->stream);
        }

        if (cuda_resource) {
            cuGraphicsUnregisterResource(cuda_resource);
        }
        NvBufSurfaceUnMapEglImage(surf, batch_index);
    } else {
        printf("[face_align] no dataPtr and unsupported memType=%d\n",
               (int)surf->memType);
    }

    gst_buffer_unmap(batch->inbuf, &in_map);
    return ok;
}

static bool read_landmarks(NvDsObjectMeta *obj_meta, float out_lm[5][2]) {
    if (!obj_meta) return false;
    NvDsMetaList *l = obj_meta->obj_user_meta_list;
    while (l) {
        NvDsUserMeta *um = (NvDsUserMeta *)l->data;
        if (um && um->base_meta.meta_type == (NvDsMetaType)LANDMARKS_META_TYPE
                && um->user_meta_data) {
            const float *src = (const float *)um->user_meta_data;
            for (int i = 0; i < 5; i++) {
                out_lm[i][0] = src[i * 2 + 0];
                out_lm[i][1] = src[i * 2 + 1];
            }
            return true;
        }
        l = l->next;
    }
    return false;
}

extern "C"
CustomCtx *initLib(CustomInitParams initparams) {
    auto *ctx = new CustomCtx;
    std::memset(ctx, 0, sizeof(*ctx));

    cuInit(0);

    cudaError_t err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess) {
        printf("[face_align] cudaStreamCreate failed: %s\n",
                cudaGetErrorString(err));
        delete ctx;
        return nullptr;
    }
    nppGetStreamContext(&ctx->npp_ctx);
    ctx->npp_ctx.hStream = ctx->stream;

    if (cudaMalloc(&ctx->warp_buf, FACE_W * FACE_H * 3) != cudaSuccess) {
        printf("[face_align] cudaMalloc warp_buf failed\n");
        cudaStreamDestroy(ctx->stream);
        delete ctx;
        return nullptr;
    }
    for (int c = 0; c < 3; c++) {
        if (cudaMalloc(&ctx->planar_8u[c], FACE_W * FACE_H) != cudaSuccess) {
            printf("[face_align] cudaMalloc planar_8u[%d] failed\n", c);
            cudaFree(ctx->warp_buf);
            for (int k = 0; k < c; k++) cudaFree(ctx->planar_8u[k]);
            cudaStreamDestroy(ctx->stream);
            delete ctx;
            return nullptr;
        }
    }

    printf("[face_align] initLib OK (NPP path active, "
           "ARCFACE_REF 5-keypoint, 112x112 RGB float CHW)\n");
    return ctx;
}

extern "C"
void deInitLib(CustomCtx *ctx) {
    if (!ctx) return;
    if (ctx->rgb_buf)   cudaFree(ctx->rgb_buf);
    if (ctx->warp_buf)  cudaFree(ctx->warp_buf);
    for (int c = 0; c < 3; c++) {
        if (ctx->planar_8u[c]) cudaFree(ctx->planar_8u[c]);
    }
    cudaStreamDestroy(ctx->stream);
    delete ctx;
}

// Symbol name MUST match custom-input-transformation-function in
// face_preprocess_config.txt. Framework dlsym() the exact string — the
// "Async" suffix also tells gst-nvdspreprocess to wait on params.sync_obj
// before invoking CustomTensorPreparation (see gstnvdspreprocess.cpp:1653).
//
// NOTE: We don't actually consume the scaling-pool output (converted_frame_ptr).
// We still must perform the transform because the framework expects out_surf
// to be filled — skipping it triggers downstream pool errors.
extern "C"
NvDsPreProcessStatus CustomAsyncTransformation(NvBufSurface *in_surf,
                                                NvBufSurface *out_surf,
                                                CustomTransformParams &params) {
    NvBufSurfTransform_Error err =
        NvBufSurfTransformSetSessionParams(&params.transform_config_params);
    if (err != NvBufSurfTransformError_Success) {
        printf("[face_align] SetSessionParams err=%d\n", err);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }
    err = NvBufSurfTransformAsync(in_surf, out_surf,
                                    &params.transform_params,
                                    &params.sync_obj);
    if (err != NvBufSurfTransformError_Success) {
        printf("[face_align] NvBufSurfTransformAsync err=%d\n", err);
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
    }
    return NVDSPREPROCESS_SUCCESS;
}

extern "C"
NvDsPreProcessStatus CustomTensorPreparation(CustomCtx *ctx,
                                             NvDsPreProcessBatch *batch,
                                             NvDsPreProcessCustomBuf *&buf,
                                             CustomTensorParams &tensorParam,
                                             NvDsPreProcessAcquirer *acquirer) {
    buf = acquirer->acquire();
    if (!buf || !buf->memory_ptr) {
        printf("[face_align] acquire returned NULL buffer\n");
        return NVDSPREPROCESS_RESOURCE_ERROR;
    }

    size_t n_units = batch->units.size();
    if (n_units == 0) {
        tensorParam.params.network_input_shape[0] = 0;
        return NVDSPREPROCESS_SUCCESS;
    }
    if (n_units > (size_t)MAX_BATCH) {
        printf("[face_align] WARN batch=%zu > MAX_BATCH=%d; clamping (extra "
               "faces dropped). Bump network-input-shape[0] in config.\n",
               n_units, MAX_BATCH);
        n_units = MAX_BATCH;
    }

    Npp32f *tensor = (Npp32f *)buf->memory_ptr;

    // Zero-fill so missing-landmark slots are deterministic (SGIE will still
    // emit some embedding; downstream filter by track_frames count).
    cudaMemsetAsync(tensor, 0, n_units * TENSOR_SLOT_FLOATS * sizeof(float),
                     ctx->stream);

    GstBuffer *cached_inbuf = nullptr;
    guint cached_batch_index = G_MAXUINT;
    int   n_aligned = 0;
    int   n_lm_missing = 0;

    for (size_t i = 0; i < n_units; i++) {
        auto &unit = batch->units[i];
        // process-on-frame=0 mode → framework populates unit.obj_meta. Skip
        // unit nếu null thay vì rơi vào roi_meta.object_meta (có thể là object
        // khác/stale → match landmark sai sang face khác).
        if (!unit.obj_meta) {
            n_lm_missing++;
            continue;
        }

        float lm[5][2];
        if (!read_landmarks(unit.obj_meta, lm)) {
            n_lm_missing++;
            continue;
        }

        // 1) NV12 → RGB once per unique input frame.
        if (batch->inbuf != cached_inbuf ||
                unit.batch_index != cached_batch_index) {
            if (!nv12_to_rgb_from_batch(ctx, batch, unit.batch_index)) {
                continue;
            }
            cached_inbuf = batch->inbuf;
            cached_batch_index = unit.batch_index;
        }

        // 2) Compute similarity 2x3 (forward map src→dst per NPP convention).
        double M[6];
        face_align::compute_similarity_2x3(
            lm, face_align::ARCFACE_REF, M);
        const double coeffs[2][3] = {
            {M[0], M[1], M[2]},
            {M[3], M[4], M[5]},
        };

        // 3) NPP affine warp full-frame RGB → 112x112 RGB (HWC).
        NppiSize srcSize = {ctx->rgb_w, ctx->rgb_h};
        NppiRect srcROI  = {0, 0, ctx->rgb_w, ctx->rgb_h};
        NppiRect dstROI  = {0, 0, FACE_W, FACE_H};

        NppStatus s = nppiWarpAffine_8u_C3R_Ctx(
            ctx->rgb_buf, srcSize, ctx->rgb_w * 3, srcROI,
            ctx->warp_buf, FACE_W * 3, dstROI,
            coeffs, NPPI_INTER_LINEAR, ctx->npp_ctx);
        if (s != NPP_SUCCESS) {
            printf("[face_align] warp err=%d\n", (int)s);
            continue;
        }

        // 4) HWC → planar uint8 (CHW preliminary).
        Npp8u *aDst[3] = {
            ctx->planar_8u[0], ctx->planar_8u[1], ctx->planar_8u[2]
        };
        s = nppiCopy_8u_C3P3R_Ctx(ctx->warp_buf, FACE_W * 3,
                                   aDst, FACE_W,
                                   NppiSize{FACE_W, FACE_H}, ctx->npp_ctx);
        if (s != NPP_SUCCESS) {
            printf("[face_align] C3P3R err=%d\n", (int)s);
            continue;
        }

        // 5) Per channel: convert 8u→32f directly into tensor slot, then
        //    in-place normalize (x - 127.5) * (1/127.5).
        Npp32f *slot = tensor + i * TENSOR_SLOT_FLOATS;
        const int   dst_step_f32 = FACE_W * (int)sizeof(float);
        const NppiSize plane_size = {FACE_W, FACE_H};

        for (int c = 0; c < 3; c++) {
            Npp32f *plane = slot + c * FACE_W * FACE_H;
            nppiConvert_8u32f_C1R_Ctx(ctx->planar_8u[c], FACE_W,
                                       plane, dst_step_f32,
                                       plane_size, ctx->npp_ctx);
            nppiSubC_32f_C1IR_Ctx(127.5f, plane, dst_step_f32,
                                   plane_size, ctx->npp_ctx);
            nppiMulC_32f_C1IR_Ctx(1.0f / 127.5f, plane, dst_step_f32,
                                   plane_size, ctx->npp_ctx);
        }

        n_aligned++;
    }

    cudaError_t cerr = cudaStreamSynchronize(ctx->stream);
    if (cerr != cudaSuccess) {
        printf("[face_align] streamSync err=%s\n", cudaGetErrorString(cerr));
        acquirer->release(buf);
        return NVDSPREPROCESS_CUDA_ERROR;
    }

    tensorParam.params.network_input_shape[0] = (int)n_units;

    if ((++ctx->debug_call_count % 60) == 0) {
        printf("[face_align] batch=%zu aligned=%d no_lm=%d (every 60 calls)\n",
                n_units, n_aligned, n_lm_missing);
    }
    return NVDSPREPROCESS_SUCCESS;
}
