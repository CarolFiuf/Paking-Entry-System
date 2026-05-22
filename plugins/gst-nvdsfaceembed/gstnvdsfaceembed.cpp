#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>
#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <NvInfer.h>

#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvbufsurface.h"
#include "nvdsmeta.h"

#include "face_align_solver.h"
#include "nvds_face_embed_meta.h"
#include "nvds_face_landmarks_meta.h"

extern "C" cudaError_t launch_face_align_rgba_chw_kernel(
    const uint8_t *rgba, int src_w, int src_h, int src_pitch, float *dst,
    int slot, const double coeffs[6], cudaStream_t stream);

extern "C" cudaError_t launch_face_quality_kernel(
    const float *aligned, float *out, int batch_size, cudaStream_t stream);

#define PACKAGE "nvdsfaceembed"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "Parking face aligned TensorRT embedding plugin"
#define BINARY_PACKAGE "parking-system"
#define URL "local"

#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"

static constexpr int FACE_W = 112;
static constexpr int FACE_H = 112;
static constexpr int FACE_TENSOR_FLOATS = 3 * FACE_W * FACE_H;
static constexpr int EMBED_DIMS = PARKING_FACE_EMBED_DIMS;

GST_DEBUG_CATEGORY_STATIC(gst_nvdsfaceembed_debug);
#define GST_CAT_DEFAULT gst_nvdsfaceembed_debug

typedef struct _GstNvDsFaceEmbed GstNvDsFaceEmbed;
typedef struct _GstNvDsFaceEmbedClass GstNvDsFaceEmbedClass;

struct _GstNvDsFaceEmbed {
    GstBaseTransform base_trans;

    guint gpu_id;
    guint unique_id;
    guint face_gie_id;
    gint source_id;
    guint max_batch_size;
    guint debug_interval;
    gboolean align_on_gpu;
    gboolean allow_cpu_fallback;
    gchar *engine_file;

    cudaStream_t stream;
    float *trt_input_host;
    float *trt_input_dev;
    float *trt_output_dev;
    float *trt_output_host;
    // Quality output: 2 float/face [blur_var, mean_brightness], thang [0,255].
    float *quality_dev;
    float *quality_host;
    gfloat min_quality;
    gfloat blur_threshold;

    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
    gchar *input_name;
    gchar *output_name;

    NvDsMetaType embed_meta_type;
    NvDsMetaType landmarks_meta_type;
    guint64 call_count;
    guint64 stat_frames;
    guint64 stat_objs;
    guint64 stat_lm;
    guint64 stat_emb;
    guint64 stat_gpu;
    guint64 stat_cpu;
    guint64 stat_gpu_fallback;
    guint64 stat_interval_skips;

    // interval = N → skip embed cho 1 track nếu nó vừa được embed trong N
    // frame gần đây (yêu cầu nvtracker upstream gán object_id ổn định).
    // 0 = embed mọi frame như cũ. last_embed_frame map track_id → frame_idx
    // tại lần embed gần nhất; được purge định kỳ trong transform_ip.
    guint interval;
    guint64 frame_counter;
    std::unordered_map<guint64, guint64> *last_embed_frame;
};

struct _GstNvDsFaceEmbedClass {
    GstBaseTransformClass parent_class;
};

#define GST_TYPE_NVDSFACEEMBED (gst_nvdsfaceembed_get_type())
#define GST_NVDSFACEEMBED(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVDSFACEEMBED, GstNvDsFaceEmbed))

G_DEFINE_TYPE(GstNvDsFaceEmbed, gst_nvdsfaceembed, GST_TYPE_BASE_TRANSFORM);

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            g_printerr("[nvdsfaceembed][TRT] %s\n", msg);
        }
    }
};

static TrtLogger g_trt_logger;

struct FaceJob {
    NvDsFrameMeta *frame_meta;
    NvDsObjectMeta *obj_meta;
    guint batch_id;
    float lm[5][2];
    float bbox[4];
};

enum {
    PROP_0,
    PROP_GPU_ID,
    PROP_UNIQUE_ID,
    PROP_FACE_GIE_ID,
    PROP_SOURCE_ID,
    PROP_ENGINE_FILE,
    PROP_BATCH_SIZE,
    PROP_DEBUG_INTERVAL,
    PROP_ALIGN_ON_GPU,
    PROP_ALLOW_CPU_FALLBACK,
    PROP_MIN_QUALITY,
    PROP_BLUR_THRESHOLD,
    PROP_INTERVAL
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
        GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA }")));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
        GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA }")));

static bool align_face_cpu_rgba_to_slot(const guint8 *rgba, int width,
                                        int height, int pitch,
                                        float *dst_base, const float lm[5][2],
                                        int slot) {
    if (!rgba || !dst_base || width <= 1 || height <= 1 || pitch <= 0)
        return false;

    double M[6];
    face_align::compute_similarity_2x3(face_align::ARCFACE_REF, lm, M);
    for (double v : M) {
        if (!std::isfinite(v)) return false;
    }

    float *slot_ptr = dst_base + slot * FACE_TENSOR_FLOATS;
    for (int y = 0; y < FACE_H; y++) {
        for (int x = 0; x < FACE_W; x++) {
            double sx = M[0] * x + M[1] * y + M[2];
            double sy = M[3] * x + M[4] * y + M[5];
            float rgb[3] = {0.0f, 0.0f, 0.0f};
            if (std::isfinite(sx) && std::isfinite(sy) &&
                sx >= 0.0 && sy >= 0.0 &&
                sx <= (double)(width - 1) && sy <= (double)(height - 1)) {
                int x0 = (int)std::floor(sx);
                int y0 = (int)std::floor(sy);
                int x1 = std::min(x0 + 1, width - 1);
                int y1 = std::min(y0 + 1, height - 1);
                float ax = (float)(sx - x0);
                float ay = (float)(sy - y0);
                const guint8 *p00 = rgba + y0 * pitch + x0 * 4;
                const guint8 *p01 = rgba + y0 * pitch + x1 * 4;
                const guint8 *p10 = rgba + y1 * pitch + x0 * 4;
                const guint8 *p11 = rgba + y1 * pitch + x1 * 4;
                for (int c = 0; c < 3; c++) {
                    float v0 = (1.0f - ax) * p00[c] + ax * p01[c];
                    float v1 = (1.0f - ax) * p10[c] + ax * p11[c];
                    rgb[c] = (1.0f - ay) * v0 + ay * v1;
                }
            }
            int pix = y * FACE_W + x;
            slot_ptr[0 * FACE_W * FACE_H + pix] =
                (rgb[0] - 127.5f) / 127.5f;
            slot_ptr[1 * FACE_W * FACE_H + pix] =
                (rgb[1] - 127.5f) / 127.5f;
            slot_ptr[2 * FACE_W * FACE_H + pix] =
                (rgb[2] - 127.5f) / 127.5f;
        }
    }
    return true;
}

struct MappedGpuFrame {
    guint batch_id = G_MAXUINT;
    CUgraphicsResource resource = nullptr;
    const guint8 *rgba = nullptr;
    int width = 0;
    int height = 0;
    int pitch = 0;
};

static void unmap_gpu_frame(NvBufSurface *surface, MappedGpuFrame &mapped) {
    if (mapped.resource) {
        cuGraphicsUnregisterResource(mapped.resource);
        mapped.resource = nullptr;
    }
    if (surface && mapped.batch_id != G_MAXUINT) {
        NvBufSurfaceUnMapEglImage(surface, mapped.batch_id);
    }
    mapped.batch_id = G_MAXUINT;
    mapped.rgba = nullptr;
    mapped.width = mapped.height = mapped.pitch = 0;
}

static bool map_gpu_rgba_frame(GstNvDsFaceEmbed *self, NvBufSurface *surface,
                               guint batch_id, MappedGpuFrame &out) {
    if (!surface || batch_id >= surface->numFilled) return false;

    NvBufSurfaceParams *sp = surface->surfaceList + batch_id;
    if (sp->colorFormat != NVBUF_COLOR_FORMAT_RGBA) {
        GST_WARNING_OBJECT(self, "expected RGBA surface for GPU align, got %d",
                           (int)sp->colorFormat);
        return false;
    }
    if (NvBufSurfaceMapEglImage(surface, batch_id) != 0) {
        GST_WARNING_OBJECT(self, "NvBufSurfaceMapEglImage failed");
        return false;
    }

    out.batch_id = batch_id;
    CUeglFrame egl_frame;
    CUresult cuerr = cuGraphicsEGLRegisterImage(
        &out.resource, sp->mappedAddr.eglImage,
        CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (cuerr == CUDA_SUCCESS) {
        cuerr = cuGraphicsResourceGetMappedEglFrame(
            &egl_frame, out.resource, 0, 0);
    }

    if (cuerr != CUDA_SUCCESS) {
        GST_WARNING_OBJECT(self, "CUDA EGL map failed: %d", (int)cuerr);
        unmap_gpu_frame(surface, out);
        return false;
    }
    if (egl_frame.frameType != CU_EGL_FRAME_TYPE_PITCH ||
        egl_frame.planeCount < 1 || !egl_frame.frame.pPitch[0]) {
        GST_WARNING_OBJECT(self, "unsupported EGL frame type=%d planes=%d",
                           (int)egl_frame.frameType,
                           (int)egl_frame.planeCount);
        unmap_gpu_frame(surface, out);
        return false;
    }

    out.rgba = (const guint8 *)egl_frame.frame.pPitch[0];
    out.width = (int)egl_frame.width;
    out.height = (int)egl_frame.height;
    out.pitch = (int)egl_frame.pitch;
    if (out.width <= 0) out.width = (int)sp->width;
    if (out.height <= 0) out.height = (int)sp->height;
    if (out.pitch <= 0) out.pitch = (int)sp->planeParams.pitch[0];
    return out.rgba && out.width > 1 && out.height > 1 && out.pitch > 0;
}

static MappedGpuFrame *get_mapped_gpu_frame(
    GstNvDsFaceEmbed *self, NvBufSurface *surface,
    std::vector<MappedGpuFrame> &mapped_frames, guint batch_id) {
    for (auto &m : mapped_frames) {
        if (m.batch_id == batch_id) return &m;
    }
    MappedGpuFrame mapped;
    if (!map_gpu_rgba_frame(self, surface, batch_id, mapped)) return nullptr;
    mapped_frames.push_back(mapped);
    return &mapped_frames.back();
}

static bool align_face_gpu_rgba_to_slot(GstNvDsFaceEmbed *self,
                                        const MappedGpuFrame &frame,
                                        const float lm[5][2], int slot) {
    double M[6];
    face_align::compute_similarity_2x3(face_align::ARCFACE_REF, lm, M);
    for (double v : M) {
        if (!std::isfinite(v)) return false;
    }

    cudaError_t err = launch_face_align_rgba_chw_kernel(
        frame.rgba, frame.width, frame.height, frame.pitch,
        self->trt_input_dev, slot, M, self->stream);
    if (err != cudaSuccess) {
        GST_WARNING_OBJECT(self, "GPU face align launch failed: %s",
                           cudaGetErrorString(err));
        return false;
    }
    return true;
}

static gpointer face_embed_meta_copy(gpointer data, gpointer user_data) {
    NvDsUserMeta *src_um = (NvDsUserMeta *)data;
    if (!src_um || !src_um->user_meta_data) return nullptr;
    auto *dst = (ParkingFaceEmbeddingMeta *)g_malloc0(
        sizeof(ParkingFaceEmbeddingMeta));
    std::memcpy(dst, src_um->user_meta_data,
                sizeof(ParkingFaceEmbeddingMeta));
    return (gpointer)dst;
}

static void face_embed_meta_release(gpointer data, gpointer user_data) {
    NvDsUserMeta *um = (NvDsUserMeta *)data;
    if (um && um->user_meta_data) {
        g_free(um->user_meta_data);
        um->user_meta_data = nullptr;
    }
}

// Quality score [0, 1] gộp từ Laplacian variance, mean brightness, bbox area.
// Match công thức Python FaceEngine.quality để Python/Plugin đồng nhất.
static float compute_quality_score(float blur_var, float mean,
                                   float bbox_w, float bbox_h,
                                   float blur_thr) {
    if (!std::isfinite(blur_var) || !std::isfinite(mean)) return 0.0f;
    if (blur_var < blur_thr) return 0.0f;
    if (mean <= 30.0f || mean >= 230.0f) return 0.0f;
    float blur_s = std::min(blur_var / 500.0f, 1.0f);
    float bright_s = 1.0f - std::fabs(mean - 128.0f) / 128.0f;
    float area = std::max(0.0f, bbox_w) * std::max(0.0f, bbox_h);
    float size_s = std::min(area / 20000.0f, 1.0f);
    return blur_s * 0.5f + bright_s * 0.2f + size_s * 0.3f;
}

static void attach_embedding_meta(GstNvDsFaceEmbed *self,
                                  NvDsBatchMeta *batch_meta,
                                  NvDsObjectMeta *obj_meta,
                                  const float *embedding,
                                  float quality) {
    if (!batch_meta || !obj_meta || !embedding) return;

    auto *payload = (ParkingFaceEmbeddingMeta *)g_malloc0(
        sizeof(ParkingFaceEmbeddingMeta));
    payload->version = 3;
    payload->dims = EMBED_DIMS;
    payload->flags = PARKING_FACE_EMB_FLAG_VALID;

    float norm = 0.0f;
    for (int i = 0; i < EMBED_DIMS; i++) norm += embedding[i] * embedding[i];
    norm = std::sqrt(norm);
    for (int i = 0; i < EMBED_DIMS; i++) {
        payload->embedding[i] = norm > 1e-6f ? embedding[i] / norm
                                             : embedding[i];
    }
    payload->quality = quality;

    NvDsUserMeta *um = nvds_acquire_user_meta_from_pool(batch_meta);
    if (!um) {
        g_free(payload);
        return;
    }
    um->user_meta_data = payload;
    um->base_meta.meta_type = self->embed_meta_type;
    um->base_meta.copy_func = face_embed_meta_copy;
    um->base_meta.release_func = face_embed_meta_release;
    nvds_add_user_meta_to_obj(obj_meta, um);
}

static bool run_trt(GstNvDsFaceEmbed *self, int batch_size) {
    if (batch_size <= 0) return true;
    nvinfer1::Dims4 input_dims(batch_size, 3, FACE_H, FACE_W);
    if (!self->context->setInputShape(self->input_name, input_dims)) {
        GST_ERROR_OBJECT(self, "TensorRT setInputShape failed for batch=%d",
                         batch_size);
        return false;
    }
    if (!self->context->setTensorAddress(self->input_name,
                                         self->trt_input_dev) ||
        !self->context->setTensorAddress(self->output_name,
                                         self->trt_output_dev)) {
        GST_ERROR_OBJECT(self, "TensorRT setTensorAddress failed");
        return false;
    }
    if (!self->context->enqueueV3(self->stream)) {
        GST_ERROR_OBJECT(self, "TensorRT enqueueV3 failed");
        return false;
    }
    cudaError_t err = cudaMemcpyAsync(self->trt_output_host,
                                      self->trt_output_dev,
                                      (size_t)batch_size * EMBED_DIMS *
                                          sizeof(float),
                                      cudaMemcpyDeviceToHost, self->stream);
    if (err != cudaSuccess) {
        GST_ERROR_OBJECT(self, "cudaMemcpyAsync output failed: %s",
                         cudaGetErrorString(err));
        return false;
    }
    err = cudaStreamSynchronize(self->stream);
    if (err != cudaSuccess) {
        GST_ERROR_OBJECT(self, "TensorRT stream sync failed: %s",
                         cudaGetErrorString(err));
        return false;
    }
    return true;
}

static bool load_engine(GstNvDsFaceEmbed *self) {
    if (!self->engine_file || !self->engine_file[0]) {
        GST_ERROR_OBJECT(self, "engine-file is required");
        return false;
    }
    std::ifstream f(self->engine_file, std::ios::binary | std::ios::ate);
    if (!f) {
        GST_ERROR_OBJECT(self, "cannot open engine-file=%s",
                         self->engine_file);
        return false;
    }
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> engine_data((size_t)size);
    if (!f.read(engine_data.data(), size)) {
        GST_ERROR_OBJECT(self, "cannot read engine-file=%s",
                         self->engine_file);
        return false;
    }

    self->runtime = nvinfer1::createInferRuntime(g_trt_logger);
    if (!self->runtime) return false;
    self->engine = self->runtime->deserializeCudaEngine(
        engine_data.data(), engine_data.size());
    if (!self->engine) return false;
    self->context = self->engine->createExecutionContext();
    if (!self->context) return false;

    const char *input_name = nullptr;
    const char *output_name = nullptr;
    int nb_io = self->engine->getNbIOTensors();
    for (int i = 0; i < nb_io; i++) {
        const char *name = self->engine->getIOTensorName(i);
        if (!name) continue;
        nvinfer1::TensorIOMode mode = self->engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT && !input_name) {
            input_name = name;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT && !output_name) {
            output_name = name;
        }
    }
    if (!input_name || !output_name) {
        GST_ERROR_OBJECT(self, "cannot discover TensorRT input/output names");
        return false;
    }
    self->input_name = g_strdup(input_name);
    self->output_name = g_strdup(output_name);
    GST_INFO_OBJECT(self, "TensorRT engine loaded: input=%s output=%s",
                    self->input_name, self->output_name);
    return true;
}

static gboolean gst_nvdsfaceembed_start(GstBaseTransform *btrans) {
    GstNvDsFaceEmbed *self = GST_NVDSFACEEMBED(btrans);
    cuInit(0);
    if (cudaSetDevice(self->gpu_id) != cudaSuccess) return FALSE;
    if (cudaStreamCreate(&self->stream) != cudaSuccess) return FALSE;

    if (cudaMalloc(&self->trt_input_dev,
                   (size_t)self->max_batch_size * FACE_TENSOR_FLOATS *
                       sizeof(float)) != cudaSuccess)
        return FALSE;
    if (cudaMalloc(&self->trt_output_dev,
                   (size_t)self->max_batch_size * EMBED_DIMS *
                       sizeof(float)) != cudaSuccess)
        return FALSE;
    if (cudaMallocHost(&self->trt_output_host,
                       (size_t)self->max_batch_size * EMBED_DIMS *
                           sizeof(float)) != cudaSuccess)
        return FALSE;
    if (cudaMallocHost(&self->trt_input_host,
                       (size_t)self->max_batch_size * FACE_TENSOR_FLOATS *
                           sizeof(float)) != cudaSuccess)
        return FALSE;
    // 2 float/face: [blur_var, mean_brightness] thang [0,255].
    if (cudaMalloc(&self->quality_dev,
                   (size_t)self->max_batch_size * 2 *
                       sizeof(float)) != cudaSuccess)
        return FALSE;
    if (cudaMallocHost(&self->quality_host,
                       (size_t)self->max_batch_size * 2 *
                           sizeof(float)) != cudaSuccess)
        return FALSE;

    self->embed_meta_type = nvds_get_user_meta_type(
        (gchar *)PARKING_FACE_EMBED_META_DESC);
    self->landmarks_meta_type = nvds_get_user_meta_type(
        (gchar *)PARKING_FACE_LANDMARKS_META_DESC);
    if (!load_engine(self)) return FALSE;

    // Map track_id → last embed frame_counter cho interval skip.
    // GObject zero-init the struct nên alloc bằng new ở start để tránh
    // gọi constructor trên memory chưa init.
    if (!self->last_embed_frame) {
        self->last_embed_frame = new std::unordered_map<guint64, guint64>();
    }
    self->frame_counter = 0;
    self->stat_interval_skips = 0;

    GST_INFO_OBJECT(self, "started engine=%s face-gie-id=%u source-id=%d "
                    "batch=%u align-on-gpu=%d cpu-fallback=%d "
                    "embed-meta-type=%d lm-meta-type=%d interval=%u",
                    self->engine_file, self->face_gie_id, self->source_id,
                    self->max_batch_size,
                    self->align_on_gpu ? 1 : 0,
                    self->allow_cpu_fallback ? 1 : 0,
                    self->embed_meta_type, self->landmarks_meta_type,
                    self->interval);
    return TRUE;
}

static gboolean gst_nvdsfaceembed_stop(GstBaseTransform *btrans) {
    GstNvDsFaceEmbed *self = GST_NVDSFACEEMBED(btrans);
    if (self->context) { delete self->context; self->context = nullptr; }
    if (self->engine) { delete self->engine; self->engine = nullptr; }
    if (self->runtime) { delete self->runtime; self->runtime = nullptr; }
    if (self->input_name) { g_free(self->input_name); self->input_name = nullptr; }
    if (self->output_name) { g_free(self->output_name); self->output_name = nullptr; }

    if (self->trt_output_host) {
        cudaFreeHost(self->trt_output_host);
        self->trt_output_host = nullptr;
    }
    if (self->trt_input_host) {
        cudaFreeHost(self->trt_input_host);
        self->trt_input_host = nullptr;
    }
    if (self->trt_output_dev) {
        cudaFree(self->trt_output_dev);
        self->trt_output_dev = nullptr;
    }
    if (self->trt_input_dev) {
        cudaFree(self->trt_input_dev);
        self->trt_input_dev = nullptr;
    }
    if (self->quality_host) {
        cudaFreeHost(self->quality_host);
        self->quality_host = nullptr;
    }
    if (self->quality_dev) {
        cudaFree(self->quality_dev);
        self->quality_dev = nullptr;
    }
    if (self->stream) {
        cudaStreamDestroy(self->stream);
        self->stream = nullptr;
    }
    if (self->last_embed_frame) {
        delete self->last_embed_frame;
        self->last_embed_frame = nullptr;
    }
    return TRUE;
}

static bool process_jobs_chunk_cpu(GstNvDsFaceEmbed *self,
                                   NvBufSurface *surface,
                                   NvDsBatchMeta *batch_meta,
                                   const std::vector<FaceJob> &jobs,
                                   size_t begin, size_t end) {
    guint mapped_batch_id = G_MAXUINT;
    const guint8 *rgba = nullptr;
    int width = 0, height = 0, pitch = 0;
    int slot = 0;

    for (size_t i = begin; i < end; i++, slot++) {
        const FaceJob &job = jobs[i];
        if (job.batch_id != mapped_batch_id) {
            if (mapped_batch_id != G_MAXUINT) {
                NvBufSurfaceUnMap(surface, mapped_batch_id, 0);
                mapped_batch_id = G_MAXUINT;
            }
            if (job.batch_id >= surface->numFilled) return false;
            NvBufSurfaceParams *sp = surface->surfaceList + job.batch_id;
            if (sp->colorFormat != NVBUF_COLOR_FORMAT_RGBA) {
                GST_WARNING_OBJECT(self, "expected RGBA surface, got %d",
                                   (int)sp->colorFormat);
                return false;
            }
            if (NvBufSurfaceMap(surface, job.batch_id, 0,
                                NVBUF_MAP_READ) != 0) {
                GST_WARNING_OBJECT(self, "NvBufSurfaceMap CPU failed");
                return false;
            }
            if (surface->memType == NVBUF_MEM_SURFACE_ARRAY)
                NvBufSurfaceSyncForCpu(surface, job.batch_id, 0);
            rgba = (const guint8 *)sp->mappedAddr.addr[0];
            width = (int)sp->width;
            height = (int)sp->height;
            pitch = (int)sp->planeParams.pitch[0];
            mapped_batch_id = job.batch_id;
        }
        if (!align_face_cpu_rgba_to_slot(rgba, width, height, pitch,
                                         self->trt_input_host, job.lm, slot)) {
            if (mapped_batch_id != G_MAXUINT)
                NvBufSurfaceUnMap(surface, mapped_batch_id, 0);
            return false;
        }
    }
    if (mapped_batch_id != G_MAXUINT)
        NvBufSurfaceUnMap(surface, mapped_batch_id, 0);

    int batch_size = (int)(end - begin);
    cudaError_t err = cudaMemcpyAsync(
        self->trt_input_dev, self->trt_input_host,
        (size_t)batch_size * FACE_TENSOR_FLOATS * sizeof(float),
        cudaMemcpyHostToDevice, self->stream);
    if (err != cudaSuccess) {
        GST_WARNING_OBJECT(self, "input cudaMemcpyAsync failed: %s",
                           cudaGetErrorString(err));
        return false;
    }

    err = launch_face_quality_kernel(self->trt_input_dev, self->quality_dev,
                                     batch_size, self->stream);
    if (err != cudaSuccess) {
        GST_WARNING_OBJECT(self, "quality kernel failed: %s",
                           cudaGetErrorString(err));
        return false;
    }
    err = cudaMemcpyAsync(self->quality_host, self->quality_dev,
                          (size_t)batch_size * 2 * sizeof(float),
                          cudaMemcpyDeviceToHost, self->stream);
    if (err != cudaSuccess) {
        GST_WARNING_OBJECT(self, "quality memcpy failed: %s",
                           cudaGetErrorString(err));
        return false;
    }

    if (!run_trt(self, batch_size)) return false;

    for (int i = 0; i < batch_size; i++) {
        const FaceJob &job = jobs[begin + i];
        const float *emb = self->trt_output_host + i * EMBED_DIMS;
        float blur_var = self->quality_host[i * 2 + 0];
        float mean_b   = self->quality_host[i * 2 + 1];
        float bw = job.bbox[2] - job.bbox[0];
        float bh = job.bbox[3] - job.bbox[1];
        float quality = compute_quality_score(blur_var, mean_b, bw, bh,
                                              self->blur_threshold);
        // Cap rate embed cho track theo interval — đánh dấu đã embed ngay
        // cả khi quality không pass (tránh spam ArcFace frame kế tiếp cho
        // track đang ở góc xấu).
        if (self->last_embed_frame &&
            job.obj_meta->object_id != UNTRACKED_OBJECT_ID) {
            (*self->last_embed_frame)[(guint64)job.obj_meta->object_id] =
                self->frame_counter;
        }
        if (quality < self->min_quality) continue;
        attach_embedding_meta(self, batch_meta, job.obj_meta, emb, quality);
        self->stat_emb++;
        self->stat_cpu++;
    }
    return true;
}

static bool process_jobs_chunk_gpu(GstNvDsFaceEmbed *self,
                                   NvBufSurface *surface,
                                   NvDsBatchMeta *batch_meta,
                                   const std::vector<FaceJob> &jobs,
                                   size_t begin, size_t end) {
    std::vector<MappedGpuFrame> mapped_frames;
    mapped_frames.reserve(2);

    int slot = 0;
    bool ok = true;
    for (size_t i = begin; i < end; i++, slot++) {
        const FaceJob &job = jobs[i];
        MappedGpuFrame *mapped = get_mapped_gpu_frame(
            self, surface, mapped_frames, job.batch_id);
        if (!mapped ||
            !align_face_gpu_rgba_to_slot(self, *mapped, job.lm, slot)) {
            ok = false;
            break;
        }
    }

    int batch_size = (int)(end - begin);
    if (ok) {
        cudaError_t err = launch_face_quality_kernel(self->trt_input_dev,
                                                     self->quality_dev,
                                                     batch_size,
                                                     self->stream);
        if (err != cudaSuccess) {
            GST_WARNING_OBJECT(self, "quality kernel failed: %s",
                               cudaGetErrorString(err));
            ok = false;
        }
    }
    if (ok) {
        cudaError_t err = cudaMemcpyAsync(
            self->quality_host, self->quality_dev,
            (size_t)batch_size * 2 * sizeof(float),
            cudaMemcpyDeviceToHost, self->stream);
        if (err != cudaSuccess) {
            GST_WARNING_OBJECT(self, "quality memcpy failed: %s",
                               cudaGetErrorString(err));
            ok = false;
        }
    }
    if (ok) ok = run_trt(self, batch_size);

    if (!ok) {
        cudaStreamSynchronize(self->stream);
    }
    for (auto &m : mapped_frames) {
        unmap_gpu_frame(surface, m);
    }
    if (!ok) return false;

    for (int i = 0; i < batch_size; i++) {
        const FaceJob &job = jobs[begin + i];
        const float *emb = self->trt_output_host + i * EMBED_DIMS;
        float blur_var = self->quality_host[i * 2 + 0];
        float mean_b   = self->quality_host[i * 2 + 1];
        float bw = job.bbox[2] - job.bbox[0];
        float bh = job.bbox[3] - job.bbox[1];
        float quality = compute_quality_score(blur_var, mean_b, bw, bh,
                                              self->blur_threshold);
        if (self->last_embed_frame &&
            job.obj_meta->object_id != UNTRACKED_OBJECT_ID) {
            (*self->last_embed_frame)[(guint64)job.obj_meta->object_id] =
                self->frame_counter;
        }
        if (quality < self->min_quality) continue;
        attach_embedding_meta(self, batch_meta, job.obj_meta, emb, quality);
        self->stat_emb++;
        self->stat_gpu++;
    }
    return true;
}

static bool process_jobs_chunk(GstNvDsFaceEmbed *self, NvBufSurface *surface,
                               NvDsBatchMeta *batch_meta,
                               const std::vector<FaceJob> &jobs,
                               size_t begin, size_t end) {
    if (self->align_on_gpu) {
        if (process_jobs_chunk_gpu(self, surface, batch_meta, jobs,
                                   begin, end)) {
            return true;
        }
        self->stat_gpu_fallback++;
        if (!self->allow_cpu_fallback) return false;
        GST_WARNING_OBJECT(self, "GPU align failed; falling back to CPU");
    }
    return process_jobs_chunk_cpu(self, surface, batch_meta, jobs, begin, end);
}

static GstFlowReturn gst_nvdsfaceembed_transform_ip(GstBaseTransform *btrans,
                                                    GstBuffer *buf) {
    GstNvDsFaceEmbed *self = GST_NVDSFACEEMBED(btrans);
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_FLOW_OK;

    GstMapInfo map = GST_MAP_INFO_INIT;
    if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
        GST_WARNING_OBJECT(self, "gst_buffer_map failed");
        return GST_FLOW_OK;
    }
    NvBufSurface *surface = (NvBufSurface *)map.data;
    if (!surface) {
        gst_buffer_unmap(buf, &map);
        return GST_FLOW_OK;
    }

    // Frame counter cho interval skip + purge map định kỳ. Tracker upstream
    // (giữa scrfddec và embed) đã gán object_id ổn định cho track committed
    // qua probationAge → interval skip hoạt động thật sự.
    self->frame_counter++;
    if (self->last_embed_frame &&
        self->interval > 0 &&
        (self->frame_counter & 0xFF) == 0) {
        const guint64 ttl = 300;
        guint64 cutoff = self->frame_counter > ttl
            ? self->frame_counter - ttl : 0;
        for (auto it = self->last_embed_frame->begin();
             it != self->last_embed_frame->end(); ) {
            if (it->second < cutoff)
                it = self->last_embed_frame->erase(it);
            else
                ++it;
        }
    }

    std::vector<FaceJob> jobs;

    for (NvDsMetaList *lf = batch_meta->frame_meta_list; lf; lf = lf->next) {
        NvDsFrameMeta *fm = (NvDsFrameMeta *)lf->data;
        if (!fm) continue;
        if (self->source_id >= 0 && (gint)fm->source_id != self->source_id)
            continue;
        self->stat_frames++;

        // Iterate obj_meta_list, đọc LandmarksMeta do nvdsscrfddec attach
        // (đã survive tracker). Tracker đã gán object_id ổn định cho track
        // committed; obj còn trong probation hoặc tracker-predicted có
        // object_id == UNTRACKED_OBJECT_ID.
        for (NvDsMetaList *lo = fm->obj_meta_list; lo; lo = lo->next) {
            NvDsObjectMeta *om = (NvDsObjectMeta *)lo->data;
            if (!om) continue;
            if ((guint)om->unique_component_id != self->face_gie_id) continue;
            self->stat_objs++;

            // Tìm LandmarksMeta đã attach upstream. Tracker-predicted obj
            // (det miss frame này) không có meta → skip silently.
            ParkingFaceLandmarksMeta *lm_meta = nullptr;
            for (NvDsMetaList *lu = om->obj_user_meta_list; lu; lu = lu->next) {
                NvDsUserMeta *um = (NvDsUserMeta *)lu->data;
                if (!um) continue;
                if (um->base_meta.meta_type == self->landmarks_meta_type) {
                    lm_meta = (ParkingFaceLandmarksMeta *)um->user_meta_data;
                    break;
                }
            }
            if (!lm_meta || !(lm_meta->flags & PARKING_FACE_LM_FLAG_VALID))
                continue;

            // Interval skip — bây giờ object_id đã được tracker gán cho track
            // đã commit (probationAge=2). Track mới hoặc untracked vẫn embed
            // mỗi frame để build best-of-N.
            if (self->interval > 0 && self->last_embed_frame &&
                om->object_id != UNTRACKED_OBJECT_ID) {
                guint64 tid = (guint64)om->object_id;
                auto it = self->last_embed_frame->find(tid);
                if (it != self->last_embed_frame->end() &&
                    self->frame_counter - it->second <= self->interval) {
                    self->stat_interval_skips++;
                    continue;
                }
            }

            FaceJob job{};
            job.frame_meta = fm;
            job.obj_meta = om;
            job.batch_id = fm->batch_id;
            for (int i = 0; i < 5; i++) {
                job.lm[i][0] = lm_meta->landmarks[i * 2 + 0];
                job.lm[i][1] = lm_meta->landmarks[i * 2 + 1];
            }
            for (int i = 0; i < 4; i++) {
                job.bbox[i] = lm_meta->bbox[i];
            }
            jobs.push_back(job);
            self->stat_lm++;
        }
    }

    for (size_t begin = 0; begin < jobs.size(); begin += self->max_batch_size) {
        size_t end = std::min(jobs.size(), begin + self->max_batch_size);
        if (!process_jobs_chunk(self, surface, batch_meta, jobs, begin, end)) {
            GST_WARNING_OBJECT(self, "failed to process face embedding chunk");
            break;
        }
    }

    gst_buffer_unmap(buf, &map);

    self->call_count++;
    if (self->debug_interval > 0 &&
        self->call_count % self->debug_interval == 0) {
        GST_INFO_OBJECT(self, "stats frames=%lu objs=%lu lm=%lu emb=%lu "
                        "interval_skips=%lu",
                        self->stat_frames, self->stat_objs, self->stat_lm,
                        self->stat_emb, self->stat_interval_skips);
        GST_INFO_OBJECT(self, "align backend gpu=%lu cpu=%lu fallback=%lu",
                        self->stat_gpu, self->stat_cpu,
                        self->stat_gpu_fallback);
        self->stat_frames = 0;
        self->stat_objs = 0;
        self->stat_lm = 0;
        self->stat_emb = 0;
        self->stat_gpu = 0;
        self->stat_cpu = 0;
        self->stat_gpu_fallback = 0;
        self->stat_interval_skips = 0;
    }
    return GST_FLOW_OK;
}

static void gst_nvdsfaceembed_set_property(GObject *object, guint prop_id,
                                           const GValue *value,
                                           GParamSpec *pspec) {
    GstNvDsFaceEmbed *self = GST_NVDSFACEEMBED(object);
    switch (prop_id) {
        case PROP_GPU_ID:
            self->gpu_id = g_value_get_uint(value);
            break;
        case PROP_UNIQUE_ID:
            self->unique_id = g_value_get_uint(value);
            break;
        case PROP_FACE_GIE_ID:
            self->face_gie_id = g_value_get_uint(value);
            break;
        case PROP_SOURCE_ID:
            self->source_id = g_value_get_int(value);
            break;
        case PROP_ENGINE_FILE:
            g_free(self->engine_file);
            self->engine_file = g_value_dup_string(value);
            break;
        case PROP_BATCH_SIZE:
            self->max_batch_size = g_value_get_uint(value);
            break;
        case PROP_DEBUG_INTERVAL:
            self->debug_interval = g_value_get_uint(value);
            break;
        case PROP_ALIGN_ON_GPU:
            self->align_on_gpu = g_value_get_boolean(value);
            break;
        case PROP_ALLOW_CPU_FALLBACK:
            self->allow_cpu_fallback = g_value_get_boolean(value);
            break;
        case PROP_MIN_QUALITY:
            self->min_quality = (gfloat)g_value_get_double(value);
            break;
        case PROP_BLUR_THRESHOLD:
            self->blur_threshold = (gfloat)g_value_get_double(value);
            break;
        case PROP_INTERVAL:
            self->interval = g_value_get_uint(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void gst_nvdsfaceembed_get_property(GObject *object, guint prop_id,
                                           GValue *value,
                                           GParamSpec *pspec) {
    GstNvDsFaceEmbed *self = GST_NVDSFACEEMBED(object);
    switch (prop_id) {
        case PROP_GPU_ID:
            g_value_set_uint(value, self->gpu_id);
            break;
        case PROP_UNIQUE_ID:
            g_value_set_uint(value, self->unique_id);
            break;
        case PROP_FACE_GIE_ID:
            g_value_set_uint(value, self->face_gie_id);
            break;
        case PROP_SOURCE_ID:
            g_value_set_int(value, self->source_id);
            break;
        case PROP_ENGINE_FILE:
            g_value_set_string(value, self->engine_file);
            break;
        case PROP_BATCH_SIZE:
            g_value_set_uint(value, self->max_batch_size);
            break;
        case PROP_DEBUG_INTERVAL:
            g_value_set_uint(value, self->debug_interval);
            break;
        case PROP_ALIGN_ON_GPU:
            g_value_set_boolean(value, self->align_on_gpu);
            break;
        case PROP_ALLOW_CPU_FALLBACK:
            g_value_set_boolean(value, self->allow_cpu_fallback);
            break;
        case PROP_MIN_QUALITY:
            g_value_set_double(value, self->min_quality);
            break;
        case PROP_BLUR_THRESHOLD:
            g_value_set_double(value, self->blur_threshold);
            break;
        case PROP_INTERVAL:
            g_value_set_uint(value, self->interval);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void gst_nvdsfaceembed_finalize(GObject *object) {
    GstNvDsFaceEmbed *self = GST_NVDSFACEEMBED(object);
    g_free(self->engine_file);
    G_OBJECT_CLASS(gst_nvdsfaceembed_parent_class)->finalize(object);
}

static void gst_nvdsfaceembed_init(GstNvDsFaceEmbed *self) {
    GstBaseTransform *btrans = GST_BASE_TRANSFORM(self);
    gst_base_transform_set_in_place(btrans, TRUE);
    gst_base_transform_set_passthrough(btrans, TRUE);

    self->gpu_id = 0;
    self->unique_id = 7;
    self->face_gie_id = 2;
    self->source_id = 1;
    self->max_batch_size = 16;
    self->debug_interval = 150;
    self->align_on_gpu = TRUE;
    self->allow_cpu_fallback = TRUE;
    // Default: vẫn attach mọi face (Python filter), không skip embedding nào.
    self->min_quality = 0.0f;
    // Match Python FaceEngine.quality.blur_thr default.
    self->blur_threshold = 10.0f;
    // interval=0 = embed mọi frame (giữ behavior cũ khi không có nvtracker).
    self->interval = 0;
    self->frame_counter = 0;
    self->last_embed_frame = nullptr;       // alloc trong start
    self->stat_interval_skips = 0;
    self->engine_file = g_strdup(
        "/home/somethink/parking_system/models/face_embed_arcface_fp16.engine");
}

static void gst_nvdsfaceembed_class_init(GstNvDsFaceEmbedClass *klass) {
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass *base_class = GST_BASE_TRANSFORM_CLASS(klass);

    gobject_class->set_property = gst_nvdsfaceembed_set_property;
    gobject_class->get_property = gst_nvdsfaceembed_get_property;
    gobject_class->finalize = gst_nvdsfaceembed_finalize;
    base_class->start = gst_nvdsfaceembed_start;
    base_class->stop = gst_nvdsfaceembed_stop;
    base_class->transform_ip = gst_nvdsfaceembed_transform_ip;

    g_object_class_install_property(
        gobject_class, PROP_GPU_ID,
        g_param_spec_uint("gpu-id", "GPU ID", "GPU device ID", 0, G_MAXUINT,
                          0, (GParamFlags)(G_PARAM_READWRITE |
                                            G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(
        gobject_class, PROP_UNIQUE_ID,
        g_param_spec_uint("unique-id", "Unique ID", "Element unique ID", 0,
                          G_MAXUINT, 7,
                          (GParamFlags)(G_PARAM_READWRITE |
                                        G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(
        gobject_class, PROP_FACE_GIE_ID,
        g_param_spec_uint("face-gie-id", "Face GIE ID",
                          "unique_component_id of face detector", 0,
                          G_MAXUINT, 2,
                          (GParamFlags)(G_PARAM_READWRITE |
                                        G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(
        gobject_class, PROP_SOURCE_ID,
        g_param_spec_int("source-id", "Source ID",
                         "DeepStream source_id to process; -1 means all", -1,
                         G_MAXINT, 1,
                         (GParamFlags)(G_PARAM_READWRITE |
                                       G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(
        gobject_class, PROP_ENGINE_FILE,
        g_param_spec_string("engine-file", "Engine file",
                            "TensorRT ArcFace engine path",
                            "/home/somethink/parking_system/models/"
                            "face_embed_arcface_fp16.engine",
                            (GParamFlags)(G_PARAM_READWRITE |
                                          G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(
        gobject_class, PROP_BATCH_SIZE,
        g_param_spec_uint("batch-size", "Batch size",
                          "Maximum ArcFace batch size", 1, 256, 16,
                          (GParamFlags)(G_PARAM_READWRITE |
                                        G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(
        gobject_class, PROP_DEBUG_INTERVAL,
        g_param_spec_uint("debug-interval", "Debug interval",
                          "Log stats every N transform calls; 0 disables", 0,
                          G_MAXUINT, 150,
                          (GParamFlags)(G_PARAM_READWRITE |
                                        G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(
        gobject_class, PROP_ALIGN_ON_GPU,
        g_param_spec_boolean("align-on-gpu", "Align on GPU",
                             "Use CUDA EGL RGBA alignment into TensorRT input",
                             TRUE,
                             (GParamFlags)(G_PARAM_READWRITE |
                                           G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(
        gobject_class, PROP_ALLOW_CPU_FALLBACK,
        g_param_spec_boolean("allow-cpu-fallback", "Allow CPU fallback",
                             "Fallback to CPU RGBA alignment if GPU map fails",
                             TRUE,
                             (GParamFlags)(G_PARAM_READWRITE |
                                           G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(
        gobject_class, PROP_MIN_QUALITY,
        g_param_spec_double("min-quality", "Min quality",
                            "Skip attach embedding meta if computed quality "
                            "score < this threshold (0..1). 0 = always attach.",
                            0.0, 1.0, 0.0,
                            (GParamFlags)(G_PARAM_READWRITE |
                                          G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(
        gobject_class, PROP_BLUR_THRESHOLD,
        g_param_spec_double("blur-threshold", "Blur threshold",
                            "Minimum Laplacian variance to accept face (0..)",
                            0.0, 10000.0, 10.0,
                            (GParamFlags)(G_PARAM_READWRITE |
                                          G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(
        gobject_class, PROP_INTERVAL,
        g_param_spec_uint("interval", "Embed interval per track",
                          "Skip embedding for a tracked face if it was "
                          "embedded within the last N frames "
                          "(requires nvtracker upstream). 0 = embed every "
                          "frame as before.",
                          0, 1000, 0,
                          (GParamFlags)(G_PARAM_READWRITE |
                                        G_PARAM_STATIC_STRINGS)));

    gst_element_class_add_pad_template(
        element_class, gst_static_pad_template_get(&src_template));
    gst_element_class_add_pad_template(
        element_class, gst_static_pad_template_get(&sink_template));
    gst_element_class_set_details_simple(
        element_class, "Parking NvDs Face Embed", "DeepStream",
        "Aligned ArcFace TensorRT embedding attached as NvDsObject user meta",
        "parking-system");

    GST_DEBUG_CATEGORY_INIT(gst_nvdsfaceembed_debug, "nvdsfaceembed", 0,
                            "nvdsfaceembed plugin");
}

static gboolean plugin_init(GstPlugin *plugin) {
    return gst_element_register(plugin, "nvdsfaceembed", GST_RANK_PRIMARY,
                                GST_TYPE_NVDSFACEEMBED);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, nvdsfaceembed,
                  DESCRIPTION, plugin_init, VERSION, LICENSE, BINARY_PACKAGE,
                  URL)
