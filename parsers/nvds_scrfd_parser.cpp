#include <vector>

#include "nvdsinfer_custom_impl.h"

/**
 * SCRFD parser stub.
 *
 * Toàn bộ decode + landmark + obj_meta giờ do plugin gst-nvdsfaceembed làm
 * trong 1 lượt duy nhất (decode SCRFD → tạo obj_meta + landmark user meta →
 * align → ArcFace). nvinfer chỉ chạy TRT engine và đẩy tensor xuống qua
 * output-tensor-meta=1; parser callback này được giữ lại chỉ để thoả interface
 * của nvinfer (cluster-mode=4 + parse-bbox-func-name).
 *
 * Trả về empty objectList → nvinfer không tự sinh NvDsObjectMeta.
 */
extern "C" bool NvDsInferParseCustomScrfd(
    std::vector<NvDsInferLayerInfo> const& /*outputLayersInfo*/,
    NvDsInferNetworkInfo const& /*networkInfo*/,
    NvDsInferParseDetectionParams const& /*detectionParams*/,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    objectList.clear();
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomScrfd);
