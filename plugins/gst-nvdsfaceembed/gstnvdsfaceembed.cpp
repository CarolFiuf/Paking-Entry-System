#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
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
#include "scrfd_decode.h"

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
static constexpr float DEFAULT_DECODE_CONF_THRESH = 0.30f;

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
    guint net_width;
    guint net_height;
    guint min_object_width;
    guint min_object_height;
    guint debug_interval;
    gfloat decode_conf_threshold;
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
    guint64 call_count;
    guint64 stat_frames;
    guint64 stat_objs;
    guint64 stat_lm;
    guint64 stat_emb;
    guint64 stat_gpu;
    guint64 stat_cpu;
    guint64 stat_gpu_fallback;
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
    PROP_NET_WIDTH,
    PROP_NET_HEIGHT,
    PROP_MIN_OBJECT_WIDTH,
    PROP_MIN_OBJECT_HEIGHT,
    PROP_DEBUG_INTERVAL,
    PROP_DECODE_CONF_THRESHOLD,
    PROP_ALIGN_ON_GPU,
    PROP_ALLOW_CPU_FALLBACK,
    PROP_MIN_QUALITY,
    PROP_BLUR_THRESHOLD
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
        GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA }")));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
        GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA }")));

static void decode_scrfd_from_frame(NvDsFrameMeta *fm, int gie_uid, int net_w,
                                    int net_h, float conf_threshold,
                                    std::vector<DecodedFace> &out) {
    out.clear();
    if (!fm) return;

    ScrfdLayers layers;
    for (NvDsMetaList *l = fm->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *)l->data;
        if (!um || um->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
            continue;
        NvDsInferTensorMeta *tm = (NvDsInferTensorMeta *)um->user_meta_data;
        if (!tm || (int)tm->unique_id != gie_uid) continue;

        for (unsigned int i = 0; i < tm->num_output_layers; i++) {
            NvDsInferLayerInfo &li = tm->output_layers_info[i];
            int total = 1;
            for (unsigned int d = 0; d < li.inferDims.numDims; d++)
                total *= li.inferDims.d[d];
            int last = li.inferDims.numDims ?
                li.inferDims.d[li.inferDims.numDims - 1] : 1;
            if (total <= 0 || last <= 0) continue;
            int anchors = total / last;
            const float *buf = (const float *)tm->out_buf_ptrs_host[i];
            if (!buf) continue;
            if (last == 1) layers.scores[anchors] = buf;
            else if (last == 4) layers.boxes[anchors] = buf;
            else if (last == 10) layers.kps[anchors] = buf;
        }
    }

    std::vector<DecodedFace> raw;
    scrfd_decode(layers, net_w, net_h, conf_threshold, raw);
    out = scrfd_nms(raw);
}

static void net_to_frame_lm(const DecodedFace &d, int net_w, int net_h,
                            int frame_w, int frame_h, float out_lm[5][2],
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
                                  const float lm[5][2],
                                  const float bbox[4],
                                  float quality) {
    if (!batch_meta || !obj_meta || !embedding) return;

    auto *payload = (ParkingFaceEmbeddingMeta *)g_malloc0(
        sizeof(ParkingFaceEmbeddingMeta));
    payload->version = 2;
    payload->dims = EMBED_DIMS;
    payload->flags = 1;

    float norm = 0.0f;
    for (int i = 0; i < EMBED_DIMS; i++) norm += embedding[i] * embedding[i];
    norm = std::sqrt(norm);
    for (int i = 0; i < EMBED_DIMS; i++) {
        payload->embedding[i] = norm > 1e-6f ? embedding[i] / norm
                                             : embedding[i];
    }
    for (int i = 0; i < 5; i++) {
        payload->landmarks[i * 2 + 0] = lm[i][0];
        payload->landmarks[i * 2 + 1] = lm[i][1];
    }
    for (int i = 0; i < 4; i++) {
        payload->bbox[i] = bbox[i];
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
    if (!load_engine(self)) return FALSE;

    GST_INFO_OBJECT(self, "started engine=%s face-gie-id=%u source-id=%d "
                    "batch=%u decode-conf=%.2f align-on-gpu=%d "
                    "cpu-fallback=%d meta-type=%d",
                    self->engine_file, self->face_gie_id, self->source_id,
                    self->max_batch_size, self->decode_conf_threshold,
                    self->align_on_gpu ? 1 : 0,
                    self->allow_cpu_fallback ? 1 : 0, self->embed_meta_type);
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
        if (quality < self->min_quality) continue;
        attach_embedding_meta(self, batch_meta, job.obj_meta, emb, job.lm,
                              job.bbox, quality);
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
        if (quality < self->min_quality) continue;
        attach_embedding_meta(self, batch_meta, job.obj_meta, emb, job.lm,
                              job.bbox, quality);
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

    std::vector<FaceJob> jobs;

    for (NvDsMetaList *lf = batch_meta->frame_meta_list; lf; lf = lf->next) {
        NvDsFrameMeta *fm = (NvDsFrameMeta *)lf->data;
        if (!fm) continue;
        if (self->source_id >= 0 && (gint)fm->source_id != self->source_id)
            continue;
        self->stat_frames++;

        int frame_w = 0;
        int frame_h = 0;
        if (fm->batch_id < surface->numFilled) {
            frame_w = (int)surface->surfaceList[fm->batch_id].width;
            frame_h = (int)surface->surfaceList[fm->batch_id].height;
        }
        if (frame_w <= 0 || frame_h <= 0) {
            frame_w = (int)fm->source_frame_width;
            frame_h = (int)fm->source_frame_height;
        }
        if (frame_w <= 0 || frame_h <= 0)
            continue;

        // Decode SCRFD đúng 1 lần / frame, sau đó tạo obj_meta + job align
        // trong cùng vòng. Parser là no-op nên nvinfer không sinh obj_meta.
        std::vector<DecodedFace> raw_faces;
        decode_scrfd_from_frame(fm, self->face_gie_id,
                                (int)self->net_width,
                                (int)self->net_height,
                                self->decode_conf_threshold, raw_faces);

        std::vector<DecodedFace> valid_faces;
        valid_faces.reserve(raw_faces.size());
        for (const auto &d : raw_faces) {
            self->stat_objs++;
            if (valid_face_geometry(d, (float)self->net_width,
                                    (float)self->net_height))
                valid_faces.push_back(d);
        }
        auto kept_faces = scrfd_nms(valid_faces);

        for (const auto &d : kept_faces) {
            float lm[5][2];
            float box[4];
            net_to_frame_lm(d, (int)self->net_width, (int)self->net_height,
                            frame_w, frame_h, lm, box);

            float x1 = std::max(0.0f, box[0]);
            float y1 = std::max(0.0f, box[1]);
            float x2 = std::min((float)frame_w, box[2]);
            float y2 = std::min((float)frame_h, box[3]);
            if (!std::isfinite(x1) || !std::isfinite(y1) ||
                !std::isfinite(x2) || !std::isfinite(y2))
                continue;
            float bw = x2 - x1;
            float bh = y2 - y1;
            if (bw < (float)self->min_object_width ||
                bh < (float)self->min_object_height)
                continue;

            bool lm_in_frame = true;
            for (int i = 0; i < 5; i++) {
                if (!std::isfinite(lm[i][0]) || !std::isfinite(lm[i][1])) {
                    lm_in_frame = false;
                    break;
                }
            }
            if (!lm_in_frame) continue;

            NvDsObjectMeta *om = nvds_acquire_obj_meta_from_pool(batch_meta);
            if (!om) continue;
            om->unique_component_id = self->face_gie_id;
            om->class_id = 0;
            om->object_id = UNTRACKED_OBJECT_ID;
            om->confidence = d.conf;
            om->tracker_confidence = 0.0f;
            om->rect_params.left = x1;
            om->rect_params.top = y1;
            om->rect_params.width = bw;
            om->rect_params.height = bh;
            om->rect_params.border_width = 0;
            om->rect_params.has_bg_color = 0;
            om->detector_bbox_info.org_bbox_coords.left = x1;
            om->detector_bbox_info.org_bbox_coords.top = y1;
            om->detector_bbox_info.org_bbox_coords.width = bw;
            om->detector_bbox_info.org_bbox_coords.height = bh;
            g_strlcpy(om->obj_label, "face", MAX_LABEL_SIZE);
            nvds_add_obj_meta_to_frame(fm, om, nullptr);

            FaceJob job{};
            job.frame_meta = fm;
            job.obj_meta = om;
            job.batch_id = fm->batch_id;
            float box_out[4] = {x1, y1, x2, y2};
            std::memcpy(job.lm, lm, sizeof(lm));
            std::memcpy(job.bbox, box_out, sizeof(box_out));
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
        GST_INFO_OBJECT(self, "stats frames=%lu objs=%lu lm=%lu emb=%lu",
                        self->stat_frames, self->stat_objs, self->stat_lm,
                        self->stat_emb);
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
        case PROP_NET_WIDTH:
            self->net_width = g_value_get_uint(value);
            break;
        case PROP_NET_HEIGHT:
            self->net_height = g_value_get_uint(value);
            break;
        case PROP_MIN_OBJECT_WIDTH:
            self->min_object_width = g_value_get_uint(value);
            break;
        case PROP_MIN_OBJECT_HEIGHT:
            self->min_object_height = g_value_get_uint(value);
            break;
        case PROP_DEBUG_INTERVAL:
            self->debug_interval = g_value_get_uint(value);
            break;
        case PROP_DECODE_CONF_THRESHOLD:
            self->decode_conf_threshold = (gfloat)g_value_get_double(value);
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
        case PROP_NET_WIDTH:
            g_value_set_uint(value, self->net_width);
            break;
        case PROP_NET_HEIGHT:
            g_value_set_uint(value, self->net_height);
            break;
        case PROP_MIN_OBJECT_WIDTH:
            g_value_set_uint(value, self->min_object_width);
            break;
        case PROP_MIN_OBJECT_HEIGHT:
            g_value_set_uint(value, self->min_object_height);
            break;
        case PROP_DEBUG_INTERVAL:
            g_value_set_uint(value, self->debug_interval);
            break;
        case PROP_DECODE_CONF_THRESHOLD:
            g_value_set_double(value, self->decode_conf_threshold);
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
    self->net_width = 640;
    self->net_height = 640;
    self->min_object_width = 32;
    self->min_object_height = 32;
    self->debug_interval = 150;
    self->decode_conf_threshold = DEFAULT_DECODE_CONF_THRESH;
    self->align_on_gpu = TRUE;
    self->allow_cpu_fallback = TRUE;
    // Default: vẫn attach mọi face (Python filter), không skip embedding nào.
    self->min_quality = 0.0f;
    // Match Python FaceEngine.quality.blur_thr default.
    self->blur_threshold = 10.0f;
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
        gobject_class, PROP_NET_WIDTH,
        g_param_spec_uint("net-width", "Detector net width",
                          "SCRFD detector network width", 1, G_MAXUINT, 640,
                          (GParamFlags)(G_PARAM_READWRITE |
                                        G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(
        gobject_class, PROP_NET_HEIGHT,
        g_param_spec_uint("net-height", "Detector net height",
                          "SCRFD detector network height", 1, G_MAXUINT, 640,
                          (GParamFlags)(G_PARAM_READWRITE |
                                        G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(
        gobject_class, PROP_MIN_OBJECT_WIDTH,
        g_param_spec_uint("input-object-min-width", "Min object width",
                          "Minimum face width to embed", 1, G_MAXUINT, 32,
                          (GParamFlags)(G_PARAM_READWRITE |
                                        G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(
        gobject_class, PROP_MIN_OBJECT_HEIGHT,
        g_param_spec_uint("input-object-min-height", "Min object height",
                          "Minimum face height to embed", 1, G_MAXUINT, 32,
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
        gobject_class, PROP_DECODE_CONF_THRESHOLD,
        g_param_spec_double("decode-conf-threshold",
                            "Decode confidence threshold",
                            "SCRFD confidence threshold used when decoding "
                            "landmarks from tensor meta", 0.0, 1.0,
                            DEFAULT_DECODE_CONF_THRESH,
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
