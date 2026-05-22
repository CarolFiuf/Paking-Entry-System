// gstnvdsscrfddec — DeepStream plugin: decode raw SCRFD tensors thành
// NvDsObjectMeta + attach ParkingFaceLandmarksMeta.
//
// Vai trò: vốn là một phần của gst-nvdsfaceembed monolithic. Tách ra để
// nvtracker chèn được GIỮA scrfddec và embed → object_id ổn định trước khi
// embed plugin chạy → interval skip thật sự hiệu lực.
//
// Pipeline:
//   nvinfer(SCRFD, parser stub) → nvdsscrfddec → nvvideoconvert(RGBA)
//   → nvtracker → nvdsfaceembed → ...
//
// Input: NvDsBatchMeta với frame_user_meta_list chứa NVDSINFER_TENSOR_OUTPUT_META
//        của SCRFD (9 layers × {score, bbox, kps}).
// Output (in-place, passthrough): mỗi face hợp lệ tạo 1 NvDsObjectMeta
//        (unique_component_id=face_gie_id, object_id=UNTRACKED) + 1
//        ParkingFaceLandmarksMeta gắn vào obj_user_meta_list.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvbufsurface.h"
#include "nvdsmeta.h"

#include "nvds_face_landmarks_meta.h"
#include "scrfd_decode.h"

#define PACKAGE "nvdsscrfddec"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "Parking SCRFD tensor decoder → obj_meta + landmarks meta"
#define BINARY_PACKAGE "parking-system"
#define URL "local"

#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"

static constexpr float DEFAULT_DECODE_CONF_THRESH = 0.50f;

GST_DEBUG_CATEGORY_STATIC(gst_nvdsscrfddec_debug);
#define GST_CAT_DEFAULT gst_nvdsscrfddec_debug

typedef struct _GstNvDsScrfdDec GstNvDsScrfdDec;
typedef struct _GstNvDsScrfdDecClass GstNvDsScrfdDecClass;

struct _GstNvDsScrfdDec {
    GstBaseTransform base_trans;

    guint gpu_id;
    guint unique_id;
    guint face_gie_id;
    gint source_id;
    guint net_width;
    guint net_height;
    guint min_object_width;
    guint min_object_height;
    guint debug_interval;
    gfloat decode_conf_threshold;

    NvDsMetaType landmarks_meta_type;

    guint64 call_count;
    guint64 stat_frames;
    guint64 stat_decoded;
    guint64 stat_kept;
};

struct _GstNvDsScrfdDecClass {
    GstBaseTransformClass parent_class;
};

#define GST_TYPE_NVDSSCRFDDEC (gst_nvdsscrfddec_get_type())
#define GST_NVDSSCRFDDEC(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVDSSCRFDDEC, GstNvDsScrfdDec))

G_DEFINE_TYPE(GstNvDsScrfdDec, gst_nvdsscrfddec, GST_TYPE_BASE_TRANSFORM);

enum {
    PROP_0,
    PROP_GPU_ID,
    PROP_UNIQUE_ID,
    PROP_FACE_GIE_ID,
    PROP_SOURCE_ID,
    PROP_NET_WIDTH,
    PROP_NET_HEIGHT,
    PROP_MIN_OBJECT_WIDTH,
    PROP_MIN_OBJECT_HEIGHT,
    PROP_DEBUG_INTERVAL,
    PROP_DECODE_CONF_THRESHOLD,
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

static gpointer face_landmarks_meta_copy(gpointer data, gpointer user_data) {
    NvDsUserMeta *src_um = (NvDsUserMeta *)data;
    if (!src_um || !src_um->user_meta_data) return nullptr;
    auto *dst = (ParkingFaceLandmarksMeta *)g_malloc0(
        sizeof(ParkingFaceLandmarksMeta));
    std::memcpy(dst, src_um->user_meta_data,
                sizeof(ParkingFaceLandmarksMeta));
    return (gpointer)dst;
}

static void face_landmarks_meta_release(gpointer data, gpointer user_data) {
    NvDsUserMeta *um = (NvDsUserMeta *)data;
    if (um && um->user_meta_data) {
        g_free(um->user_meta_data);
        um->user_meta_data = nullptr;
    }
}

static void attach_landmarks_meta(GstNvDsScrfdDec *self,
                                  NvDsBatchMeta *batch_meta,
                                  NvDsObjectMeta *obj_meta,
                                  const float lm[5][2],
                                  const float bbox[4],
                                  float det_conf) {
    auto *payload = (ParkingFaceLandmarksMeta *)g_malloc0(
        sizeof(ParkingFaceLandmarksMeta));
    payload->version = 1;
    payload->flags = PARKING_FACE_LM_FLAG_VALID;
    for (int i = 0; i < 5; i++) {
        payload->landmarks[i * 2 + 0] = lm[i][0];
        payload->landmarks[i * 2 + 1] = lm[i][1];
    }
    for (int i = 0; i < 4; i++) {
        payload->bbox[i] = bbox[i];
    }
    payload->det_conf = det_conf;

    NvDsUserMeta *um = nvds_acquire_user_meta_from_pool(batch_meta);
    if (!um) {
        g_free(payload);
        return;
    }
    um->user_meta_data = payload;
    um->base_meta.meta_type = self->landmarks_meta_type;
    um->base_meta.copy_func = face_landmarks_meta_copy;
    um->base_meta.release_func = face_landmarks_meta_release;
    nvds_add_user_meta_to_obj(obj_meta, um);
}

static gboolean gst_nvdsscrfddec_start(GstBaseTransform *btrans) {
    GstNvDsScrfdDec *self = GST_NVDSSCRFDDEC(btrans);
    self->landmarks_meta_type = nvds_get_user_meta_type(
        (gchar *)PARKING_FACE_LANDMARKS_META_DESC);
    GST_INFO_OBJECT(self, "started face-gie-id=%u source-id=%d net=%ux%u "
                    "conf=%.2f min=%ux%u meta-type=%d",
                    self->face_gie_id, self->source_id,
                    self->net_width, self->net_height,
                    self->decode_conf_threshold,
                    self->min_object_width, self->min_object_height,
                    self->landmarks_meta_type);
    return TRUE;
}

static gboolean gst_nvdsscrfddec_stop(GstBaseTransform *btrans) {
    return TRUE;
}

static GstFlowReturn gst_nvdsscrfddec_transform_ip(GstBaseTransform *btrans,
                                                   GstBuffer *buf) {
    GstNvDsScrfdDec *self = GST_NVDSSCRFDDEC(btrans);
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_FLOW_OK;

    // Map buffer để đọc NvBufSurface.surfaceList[batch_id].width/height —
    // đây là kích thước mux output (vd 1280×720) mà nvinfer SCRFD chạy trên.
    // Dùng source_frame_width của NvDsFrameMeta sẽ SAI vì nvstreammux set
    // field đó bằng kích thước source camera (vd 1920×1080 → khác mux output)
    // → bbox decode lệch coord so với RGBA surface mà embed downstream dùng
    // để align.
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
        // Fallback (không lý tưởng) — chỉ đúng khi mux config = source size.
        if (frame_w <= 0 || frame_h <= 0) {
            frame_w = (int)fm->source_frame_width;
            frame_h = (int)fm->source_frame_height;
        }
        if (frame_w <= 0 || frame_h <= 0) continue;

        std::vector<DecodedFace> raw_faces;
        decode_scrfd_from_frame(fm, self->face_gie_id,
                                (int)self->net_width,
                                (int)self->net_height,
                                self->decode_conf_threshold, raw_faces);

        std::vector<DecodedFace> valid_faces;
        valid_faces.reserve(raw_faces.size());
        for (const auto &d : raw_faces) {
            self->stat_decoded++;
            if (valid_face_geometry(d, (float)self->net_width,
                                    (float)self->net_height))
                valid_faces.push_back(d);
        }
        auto kept_faces = scrfd_nms(valid_faces);

        nvds_acquire_meta_lock(batch_meta);
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

            float box_out[4] = {x1, y1, x2, y2};
            attach_landmarks_meta(self, batch_meta, om, lm, box_out, d.conf);
            self->stat_kept++;
        }
        nvds_release_meta_lock(batch_meta);
    }

    gst_buffer_unmap(buf, &map);

    self->call_count++;
    if (self->debug_interval > 0 &&
        self->call_count % self->debug_interval == 0) {
        GST_INFO_OBJECT(self, "stats frames=%lu decoded=%lu kept=%lu",
                        self->stat_frames, self->stat_decoded,
                        self->stat_kept);
        self->stat_frames = 0;
        self->stat_decoded = 0;
        self->stat_kept = 0;
    }
    return GST_FLOW_OK;
}

static void gst_nvdsscrfddec_set_property(GObject *object, guint prop_id,
                                          const GValue *value,
                                          GParamSpec *pspec) {
    GstNvDsScrfdDec *self = GST_NVDSSCRFDDEC(object);
    switch (prop_id) {
        case PROP_GPU_ID:
            self->gpu_id = g_value_get_uint(value); break;
        case PROP_UNIQUE_ID:
            self->unique_id = g_value_get_uint(value); break;
        case PROP_FACE_GIE_ID:
            self->face_gie_id = g_value_get_uint(value); break;
        case PROP_SOURCE_ID:
            self->source_id = g_value_get_int(value); break;
        case PROP_NET_WIDTH:
            self->net_width = g_value_get_uint(value); break;
        case PROP_NET_HEIGHT:
            self->net_height = g_value_get_uint(value); break;
        case PROP_MIN_OBJECT_WIDTH:
            self->min_object_width = g_value_get_uint(value); break;
        case PROP_MIN_OBJECT_HEIGHT:
            self->min_object_height = g_value_get_uint(value); break;
        case PROP_DEBUG_INTERVAL:
            self->debug_interval = g_value_get_uint(value); break;
        case PROP_DECODE_CONF_THRESHOLD:
            self->decode_conf_threshold = (gfloat)g_value_get_double(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void gst_nvdsscrfddec_get_property(GObject *object, guint prop_id,
                                          GValue *value, GParamSpec *pspec) {
    GstNvDsScrfdDec *self = GST_NVDSSCRFDDEC(object);
    switch (prop_id) {
        case PROP_GPU_ID:
            g_value_set_uint(value, self->gpu_id); break;
        case PROP_UNIQUE_ID:
            g_value_set_uint(value, self->unique_id); break;
        case PROP_FACE_GIE_ID:
            g_value_set_uint(value, self->face_gie_id); break;
        case PROP_SOURCE_ID:
            g_value_set_int(value, self->source_id); break;
        case PROP_NET_WIDTH:
            g_value_set_uint(value, self->net_width); break;
        case PROP_NET_HEIGHT:
            g_value_set_uint(value, self->net_height); break;
        case PROP_MIN_OBJECT_WIDTH:
            g_value_set_uint(value, self->min_object_width); break;
        case PROP_MIN_OBJECT_HEIGHT:
            g_value_set_uint(value, self->min_object_height); break;
        case PROP_DEBUG_INTERVAL:
            g_value_set_uint(value, self->debug_interval); break;
        case PROP_DECODE_CONF_THRESHOLD:
            g_value_set_double(value, self->decode_conf_threshold); break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void gst_nvdsscrfddec_init(GstNvDsScrfdDec *self) {
    GstBaseTransform *btrans = GST_BASE_TRANSFORM(self);
    gst_base_transform_set_in_place(btrans, TRUE);
    gst_base_transform_set_passthrough(btrans, TRUE);

    self->gpu_id = 0;
    self->unique_id = 8;
    self->face_gie_id = 2;
    self->source_id = 0;
    self->net_width = 640;
    self->net_height = 640;
    self->min_object_width = 32;
    self->min_object_height = 32;
    self->debug_interval = 150;
    self->decode_conf_threshold = DEFAULT_DECODE_CONF_THRESH;
}

static void gst_nvdsscrfddec_class_init(GstNvDsScrfdDecClass *klass) {
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass *base_class = GST_BASE_TRANSFORM_CLASS(klass);

    gobject_class->set_property = gst_nvdsscrfddec_set_property;
    gobject_class->get_property = gst_nvdsscrfddec_get_property;
    base_class->start = gst_nvdsscrfddec_start;
    base_class->stop = gst_nvdsscrfddec_stop;
    base_class->transform_ip = gst_nvdsscrfddec_transform_ip;

    g_object_class_install_property(gobject_class, PROP_GPU_ID,
        g_param_spec_uint("gpu-id", "GPU ID", "GPU device ID",
                          0, G_MAXUINT, 0,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_UNIQUE_ID,
        g_param_spec_uint("unique-id", "Unique ID", "Element unique ID",
                          0, G_MAXUINT, 8,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_FACE_GIE_ID,
        g_param_spec_uint("face-gie-id", "Face GIE ID",
                          "unique_component_id của nvinfer SCRFD upstream",
                          0, G_MAXUINT, 2,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_SOURCE_ID,
        g_param_spec_int("source-id", "Source ID",
                         "Stream source_id to process; -1 = all",
                         -1, G_MAXINT, 0,
                         (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_NET_WIDTH,
        g_param_spec_uint("net-width", "Net width",
                          "SCRFD detector network input width",
                          1, G_MAXUINT, 640,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_NET_HEIGHT,
        g_param_spec_uint("net-height", "Net height",
                          "SCRFD detector network input height",
                          1, G_MAXUINT, 640,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_MIN_OBJECT_WIDTH,
        g_param_spec_uint("input-object-min-width", "Min object width",
                          "Skip face nhỏ hơn (post-NMS bbox width)",
                          1, G_MAXUINT, 32,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_MIN_OBJECT_HEIGHT,
        g_param_spec_uint("input-object-min-height", "Min object height",
                          "Skip face nhỏ hơn (post-NMS bbox height)",
                          1, G_MAXUINT, 32,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_DEBUG_INTERVAL,
        g_param_spec_uint("debug-interval", "Debug interval",
                          "Log stats mỗi N transform_ip calls; 0 = disable",
                          0, G_MAXUINT, 150,
                          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_DECODE_CONF_THRESHOLD,
        g_param_spec_double("decode-conf-threshold",
                            "SCRFD decode confidence threshold",
                            "Anchor score gate khi decode SCRFD tensor",
                            0.0, 1.0, DEFAULT_DECODE_CONF_THRESH,
                            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_add_pad_template(
        element_class, gst_static_pad_template_get(&src_template));
    gst_element_class_add_pad_template(
        element_class, gst_static_pad_template_get(&sink_template));
    gst_element_class_set_details_simple(
        element_class, "Parking NvDs SCRFD Decoder", "DeepStream",
        "Decode SCRFD tensor → NvDsObjectMeta + ParkingFaceLandmarksMeta",
        "parking-system");

    GST_DEBUG_CATEGORY_INIT(gst_nvdsscrfddec_debug, "nvdsscrfddec", 0,
                            "nvdsscrfddec plugin");
}

static gboolean plugin_init(GstPlugin *plugin) {
    return gst_element_register(plugin, "nvdsscrfddec", GST_RANK_PRIMARY,
                                GST_TYPE_NVDSSCRFDDEC);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, nvdsscrfddec,
                  DESCRIPTION, plugin_init, VERSION, LICENSE, BINARY_PACKAGE,
                  URL)
