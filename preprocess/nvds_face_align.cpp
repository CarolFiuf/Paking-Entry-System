// libnvds_face_align.so — custom nvdspreprocess library for face alignment.
//
// For each face obj_meta:
//   1) Decode SCRFD raw tensor meta from face_det frame_user_meta.
//   2) IoU-match decoded bbox to obj_meta, recover 5 landmarks in frame coords.
//   3) Compute 2x3 similarity matrix mapping landmarks → ARCFACE_REF.
//   4) NPP NV12→RGB on the input full frame (cached per frame).
//   5) NPP affine warp → 112x112 RGB warped patch.
//   6) NPP C3→P3 split + Convert 8u→32f + per-channel normalize
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
#include <cmath>
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
#include "nvdsmeta.h"
#include "nvdsinfer.h"
#include "gstnvdsinfer.h"
#include <map>
#include <algorithm>

#include "face_align_solver.h"

// SCRFD network input. Frame size lấy runtime từ unit.input_surf_params,
// không hardcode để theo streammux output bất kỳ.
static constexpr int   NET_W = 640;
static constexpr int   NET_H = 640;
// Margin (network px) cho phép landmark lệch ngoài frame nhỏ — SCRFD đôi khi
// extrapolate mép trán/cằm ra ngoài; vượt margin này coi như invalid.
static constexpr float LM_MARGIN_PX = 8.0f;

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
    bool   debug_zero_tensor;
    bool   detach_roi_object_meta;
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

// ─────────────────────────────────────────────────────────────────────
// SCRFD raw-tensor decoder + IoU matcher.
// Đọc 9 raw tensor từ frame_user_meta (output-tensor-meta=1), decode &
// NMS giống parser bbox; mỗi detection kèm 5 landmarks. Sau đó IoU-match
// với obj_meta đang xét (bbox đã ở frame coords sau nvinfer transform)
// để gán landmark đúng.
// ─────────────────────────────────────────────────────────────────────
struct DecodedFace {
    float x1, y1, x2, y2;       // network coords (640x640)
    float conf;
    float lm[5][2];             // network coords
};

static constexpr float DECODE_CONF_THRESH = 0.30f;
static constexpr float DECODE_NMS_IOU     = 0.4f;
static constexpr int   DECODE_NUM_ANCHORS = 2;

static float face_iou(const DecodedFace& a, const DecodedFace& b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    float A = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    float B = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    return inter / (A + B - inter + 1e-6f);
}

// Decode SCRFD output 9 tensors → kept faces (network coords).
static void decode_scrfd_from_frame(NvDsFrameMeta *fm, int gie_uid,
                                     int net_w,
                                     std::vector<DecodedFace> &out) {
    out.clear();
    if (!fm) return;

    std::map<int, const float*> sc, bb, kp;
    for (NvDsMetaList *l = fm->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *)l->data;
        if (!um || um->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
            continue;
        NvDsInferTensorMeta *tm =
            (NvDsInferTensorMeta *)um->user_meta_data;
        if (!tm || (int)tm->unique_id != gie_uid) continue;

        for (unsigned int i = 0; i < tm->num_output_layers; i++) {
            NvDsInferLayerInfo &li = tm->output_layers_info[i];
            int total = 1;
            for (unsigned int k = 0; k < li.inferDims.numDims; k++)
                total *= li.inferDims.d[k];
            int last = li.inferDims.numDims ?
                li.inferDims.d[li.inferDims.numDims - 1] : 1;
            if (last <= 0 || total <= 0) continue;
            int anchors = total / last;
            const float *buf = (const float *)tm->out_buf_ptrs_host[i];
            if (!buf) continue;
            if      (last == 1)  sc[anchors] = buf;
            else if (last == 4)  bb[anchors] = buf;
            else if (last == 10) kp[anchors] = buf;
        }
    }

    const int strides[] = {8, 16, 32};
    std::vector<DecodedFace> all;
    for (int stride : strides) {
        int feat = net_w / stride;
        int anchors = feat * feat * DECODE_NUM_ANCHORS;
        auto sit = sc.find(anchors);
        auto bit = bb.find(anchors);
        auto kit = kp.find(anchors);
        if (sit == sc.end() || bit == bb.end() || kit == kp.end()) continue;
        const float *S = sit->second, *B = bit->second, *K = kit->second;
        for (int i = 0; i < anchors; i++) {
            float s = S[i];
            if (s < DECODE_CONF_THRESH) continue;
            int loc = i / DECODE_NUM_ANCHORS;
            int y = loc / feat, x = loc % feat;
            float cx = x * stride, cy = y * stride;
            DecodedFace d;
            d.conf = s;
            d.x1 = cx - B[i*4+0] * stride;
            d.y1 = cy - B[i*4+1] * stride;
            d.x2 = cx + B[i*4+2] * stride;
            d.y2 = cy + B[i*4+3] * stride;
            for (int p = 0; p < 5; p++) {
                d.lm[p][0] = cx + K[i*10 + p*2 + 0] * stride;
                d.lm[p][1] = cy + K[i*10 + p*2 + 1] * stride;
            }
            all.push_back(d);
        }
    }

    // NMS sort-by-conf desc.
    std::sort(all.begin(), all.end(),
              [](const DecodedFace& a, const DecodedFace& b){
                  return a.conf > b.conf;
              });
    std::vector<bool> sup(all.size(), false);
    for (size_t i = 0; i < all.size(); i++) {
        if (sup[i]) continue;
        out.push_back(all[i]);
        for (size_t j = i + 1; j < all.size(); j++)
            if (!sup[j] && face_iou(all[i], all[j]) > DECODE_NMS_IOU)
                sup[j] = true;
    }
}

// Convert decoded face (network coords) → frame coords using SCRFD letterbox.
static void net_to_frame_lm(const DecodedFace &d, int net_w, int net_h,
                            int frame_w, int frame_h,
                            float out_lm[5][2],
                            float out_box[4]) {
    float scale = std::min((float)net_w / frame_w, (float)net_h / frame_h);
    float pad_x = (net_w - frame_w * scale) * 0.5f;
    float pad_y = (net_h - frame_h * scale) * 0.5f;
    out_box[0] = (d.x1 - pad_x) / scale;
    out_box[1] = (d.y1 - pad_y) / scale;
    out_box[2] = (d.x2 - pad_x) / scale;
    out_box[3] = (d.y2 - pad_y) / scale;
    for (int i = 0; i < 5; i++) {
        out_lm[i][0] = (d.lm[i][0] - pad_x) / scale;
        out_lm[i][1] = (d.lm[i][1] - pad_y) / scale;
    }
}

// Find decoded face whose bbox best matches obj_meta's rect_params (frame
// coords). Returns true + landmarks (frame coords, validated) on success.
static bool match_landmarks_for_obj(const std::vector<DecodedFace> &faces,
                                     int net_w, int net_h,
                                     int frame_w, int frame_h,
                                     const NvOSD_RectParams &rp,
                                     float out_lm[5][2]) {
    if (faces.empty()) return false;

    float best_iou = 0.0f;
    int   best_idx = -1;
    float best_lm[5][2];

    DecodedFace obj_box;
    obj_box.x1 = rp.left;
    obj_box.y1 = rp.top;
    obj_box.x2 = rp.left + rp.width;
    obj_box.y2 = rp.top + rp.height;

    for (size_t i = 0; i < faces.size(); i++) {
        float box[4];
        float lm[5][2];
        net_to_frame_lm(faces[i], net_w, net_h, frame_w, frame_h, lm, box);
        DecodedFace fb; fb.x1 = box[0]; fb.y1 = box[1];
                       fb.x2 = box[2]; fb.y2 = box[3];
        float v = face_iou(obj_box, fb);
        if (v > best_iou) {
            best_iou = v;
            best_idx = (int)i;
            std::memcpy(best_lm, lm, sizeof(lm));
        }
    }
    if (best_idx < 0 || best_iou < 0.3f) return false;

    // Validate: finite + within frame ± margin.
    for (int i = 0; i < 5; i++) {
        float x = best_lm[i][0], y = best_lm[i][1];
        if (!std::isfinite(x) || !std::isfinite(y)
                || x < -LM_MARGIN_PX || x > frame_w + LM_MARGIN_PX
                || y < -LM_MARGIN_PX || y > frame_h + LM_MARGIN_PX) {
            return false;
        }
        out_lm[i][0] = x;
        out_lm[i][1] = y;
    }
    return true;
}

static void detach_roi_object_meta(CustomTensorParams &tensorParam) {
    // Gst-nvinfer's input-tensor-from-meta output path wraps tensor output
    // in NVDS_ROI_META. Its ROI release function deletes roi.object_meta.
    // nvdspreprocess object mode points roi.object_meta at the real
    // NvDsObjectMeta owned by NvDsBatchMeta, so leaving it non-null causes
    // invalid free/double-free. Keep ROI bbox/frame info, but drop this
    // non-owning pointer before metadata is attached downstream.
    for (auto &roi : tensorParam.seq_params.roi_vector) {
        roi.object_meta = nullptr;
    }
}

extern "C"
CustomCtx *initLib(CustomInitParams initparams) {
    auto *ctx = new CustomCtx;
    std::memset(ctx, 0, sizeof(*ctx));
    auto debug_it = initparams.user_configs.find("debug-zero-tensor");
    if (debug_it != initparams.user_configs.end()) {
        const std::string &v = debug_it->second;
        ctx->debug_zero_tensor =
            (v == "1" || v == "true" || v == "TRUE" || v == "yes");
    }
    auto detach_it = initparams.user_configs.find("detach-roi-object-meta");
    if (detach_it != initparams.user_configs.end()) {
        const std::string &v = detach_it->second;
        ctx->detach_roi_object_meta =
            (v == "1" || v == "true" || v == "TRUE" || v == "yes");
    }

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
           "ARCFACE_REF 5-keypoint, 112x112 RGB float CHW, "
           "debug-zero-tensor=%d, detach-roi-object-meta=%d)\n",
           ctx->debug_zero_tensor ? 1 : 0,
           ctx->detach_roi_object_meta ? 1 : 0);
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
    static int prep_call_no = 0;
    int prep_my_call = ++prep_call_no;
    if (prep_my_call <= 5 || prep_my_call % 60 == 0) {
        std::printf("[face_align] prep #%d n_units=%zu\n",
                    prep_my_call, n_units);
        std::fflush(stdout);
    }
    if (n_units == 0) {
        tensorParam.params.network_input_shape[0] = 0;
        if (ctx->detach_roi_object_meta)
            detach_roi_object_meta(tensorParam);
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
    if (ctx->debug_zero_tensor) {
        cudaError_t cerr = cudaStreamSynchronize(ctx->stream);
        if (cerr != cudaSuccess) {
            printf("[face_align] streamSync err=%s\n",
                   cudaGetErrorString(cerr));
            acquirer->release(buf);
            return NVDSPREPROCESS_CUDA_ERROR;
        }
        tensorParam.params.network_input_shape[0] = (int)n_units;
        if (ctx->detach_roi_object_meta)
            detach_roi_object_meta(tensorParam);
        if (prep_my_call <= 5 || (++ctx->debug_call_count % 60) == 0) {
            std::printf("[face_align] prep #%d zero-tensor debug: batch=%zu\n",
                        prep_my_call, n_units);
            std::fflush(stdout);
        }
        return NVDSPREPROCESS_SUCCESS;
    }

    GstBuffer *cached_inbuf = nullptr;
    guint cached_batch_index = G_MAXUINT;
    int   n_aligned = 0;
    int   n_lm_missing = 0;

    // Cache decoded faces per frame_meta (keyed by pointer) — decode 1 lần
    // mỗi unique frame, share giữa các unit thuộc cùng frame.
    std::map<NvDsFrameMeta*, std::vector<DecodedFace>> faces_by_frame;

    for (size_t i = 0; i < n_units; i++) {
        auto &unit = batch->units[i];
        NvDsObjectMeta *om = unit.roi_meta.object_meta;
        NvDsFrameMeta  *fm = unit.frame_meta;
        if (!om || !fm) {
            n_lm_missing++;
            continue;
        }

        int frame_w = unit.input_surf_params ?
            (int)unit.input_surf_params->width : 0;
        int frame_h = unit.input_surf_params ?
            (int)unit.input_surf_params->height : 0;
        if (frame_w <= 0 || frame_h <= 0) {
            n_lm_missing++;
            continue;
        }

        // Decode 9 raw tensors → faces (network coords); cache per frame.
        auto it = faces_by_frame.find(fm);
        if (it == faces_by_frame.end()) {
            std::vector<DecodedFace> faces;
            // gie-unique-id của face_det = 2 (xem face_det_config.txt).
            decode_scrfd_from_frame(fm, /*gie_uid=*/2, NET_W, faces);
            it = faces_by_frame.emplace(fm, std::move(faces)).first;
            if (prep_my_call <= 5) {
                std::printf("[face_align] prep #%d frame=%p decoded %zu "
                            "faces\n", prep_my_call, (void*)fm,
                            it->second.size());
                std::fflush(stdout);
            }
        }

        float lm[5][2];
        if (!match_landmarks_for_obj(it->second, NET_W, NET_H,
                                      frame_w, frame_h,
                                      om->rect_params, lm)) {
            if (prep_my_call <= 5) {
                std::printf("[face_align] prep #%d unit[%zu] no IoU match "
                            "(rect=%.0f,%.0f,%.0fx%.0f decoded=%zu)\n",
                            prep_my_call, i,
                            om->rect_params.left, om->rect_params.top,
                            om->rect_params.width, om->rect_params.height,
                            it->second.size());
                std::fflush(stdout);
            }
            n_lm_missing++;
            continue;
        }
        if (prep_my_call <= 5) {
            std::printf("[face_align] prep #%d unit[%zu] lm0=(%.1f,%.1f) "
                        "rect=(%.0f,%.0f,%.0fx%.0f)\n",
                        prep_my_call, i, lm[0][0], lm[0][1],
                        om->rect_params.left, om->rect_params.top,
                        om->rect_params.width, om->rect_params.height);
            std::fflush(stdout);
        }

        // DEBUG: bypass NV12→RGB + warp completely; chỉ test xem pipeline
        // có ổn không khi face_align không đụng GPU. Tensor slot đã zero-fill.
        n_aligned++;
        continue;

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
        // Compute BACKWARD similarity (ARCFACE_REF → lm) cho nppiWarpAffineBack.
        // Backward variant nhận thẳng map dst→src, không invert nội bộ — ổn
        // định số hơn forward+invert (FORWARD scale ~0.35 × invert → coeffs
        // lớn → kernel sample địa chỉ wild → cudaErrorIllegalAddress).
        double M[6];
        face_align::compute_similarity_2x3(
            face_align::ARCFACE_REF, lm, M);
        bool m_ok = true;
        for (int k = 0; k < 6; k++) {
            if (!std::isfinite(M[k])) { m_ok = false; break; }
        }
        if (!m_ok) {
            if (prep_my_call <= 5) {
                std::printf("[face_align] prep #%d unit[%zu] M has NaN/Inf — "
                            "skip\n", prep_my_call, i);
                std::fflush(stdout);
            }
            n_lm_missing++;
            continue;
        }
        const double coeffs[2][3] = {
            {M[0], M[1], M[2]},
            {M[3], M[4], M[5]},
        };

        // 3) NPP affine warp full-frame RGB → 112x112 RGB (HWC).
        NppiSize srcSize = {ctx->rgb_w, ctx->rgb_h};
        NppiRect srcROI  = {0, 0, ctx->rgb_w, ctx->rgb_h};
        NppiRect dstROI  = {0, 0, FACE_W, FACE_H};

        NppStatus s = nppiWarpAffineBack_8u_C3R_Ctx(
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
    if (ctx->detach_roi_object_meta)
        detach_roi_object_meta(tensorParam);

    if (prep_my_call <= 5 || (++ctx->debug_call_count % 60) == 0) {
        std::printf("[face_align] prep #%d done: batch=%zu aligned=%d "
                    "no_lm=%d\n",
                    prep_my_call, n_units, n_aligned, n_lm_missing);
        std::fflush(stdout);
    }
    return NVDSPREPROCESS_SUCCESS;
}
