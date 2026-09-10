/**
 * Host registration for the Ascend 950 packed-C8 DRAM -> HBM copy op.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../op_kernel/a5_kvcache_scatter_copy_c8_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t SOURCE_TOKEN_IDS = 0;
constexpr size_t DESTINATION_SLOTS = 1;
constexpr size_t COPY_COUNTS = 2;
constexpr size_t HBM_BLOCK_TABLE = 3;
constexpr size_t DRAM_BLOCK_TABLE = 4;
constexpr size_t HBM_KV = 5;
constexpr size_t DRAM_KV = 6;

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t KV_HEADS = 1;
constexpr int64_t PACKED_ROW_BYTES = 656;
constexpr int64_t MAX_COPY_CAP = 8192;
constexpr int64_t MAX_SOURCE_TOKENS = 1 << 18;

bool IsPackedCache(const gert::Shape &shape)
{
    return shape.GetDimNum() == 4 && shape.GetDim(0) > 0 &&
        shape.GetDim(1) == BLOCK_SIZE && shape.GetDim(2) == KV_HEADS &&
        shape.GetDim(3) == PACKED_ROW_BYTES;
}

bool GetCopyMetadataShape(
    const gert::Shape &shape,
    int64_t &batch,
    int64_t &capacity)
{
    if (shape.GetDimNum() == 2) {
        batch = shape.GetDim(0);
        capacity = shape.GetDim(1);
        return true;
    }
    if (shape.GetDimNum() == 3 && shape.GetDim(1) == 1) {
        batch = shape.GetDim(0);
        capacity = shape.GetDim(2);
        return true;
    }
    return false;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingA5KvcacheScatterCopyC8(
    gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = SOURCE_TOKEN_IDS; index <= DRAM_KV; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    for (size_t index = SOURCE_TOKEN_IDS; index <= DRAM_BLOCK_TABLE; ++index) {
        if (context->GetInputDesc(index)->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }
    if (context->GetInputDesc(HBM_KV)->GetDataType() != ge::DT_INT8 ||
        context->GetInputDesc(DRAM_KV)->GetDataType() != ge::DT_INT8) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape sourceIds =
        context->GetInputShape(SOURCE_TOKEN_IDS)->GetStorageShape();
    const gert::Shape destinationSlots =
        context->GetInputShape(DESTINATION_SLOTS)->GetStorageShape();
    const gert::Shape copyCounts =
        context->GetInputShape(COPY_COUNTS)->GetStorageShape();
    const gert::Shape hbmTable =
        context->GetInputShape(HBM_BLOCK_TABLE)->GetStorageShape();
    const gert::Shape dramTable =
        context->GetInputShape(DRAM_BLOCK_TABLE)->GetStorageShape();
    const gert::Shape hbmKv =
        context->GetInputShape(HBM_KV)->GetStorageShape();
    const gert::Shape dramKv =
        context->GetInputShape(DRAM_KV)->GetStorageShape();

    int64_t sourceBatch = 0;
    int64_t sourceCapacity = 0;
    int64_t destinationBatch = 0;
    int64_t destinationCapacity = 0;
    if (!GetCopyMetadataShape(sourceIds, sourceBatch, sourceCapacity) ||
        !GetCopyMetadataShape(
            destinationSlots, destinationBatch, destinationCapacity) ||
        copyCounts.GetDimNum() != 1 ||
        hbmTable.GetDimNum() != 2 || dramTable.GetDimNum() != 2 ||
        !IsPackedCache(hbmKv) || !IsPackedCache(dramKv)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = copyCounts.GetDim(0);
    if (batch <= 0 || sourceBatch != batch || destinationBatch != batch ||
        sourceCapacity <= 0 || sourceCapacity > MAX_COPY_CAP ||
        destinationCapacity != sourceCapacity ||
        hbmTable.GetDim(0) != batch || dramTable.GetDim(0) != batch ||
        hbmTable.GetDim(1) <= 0 || dramTable.GetDim(1) <= 0 ||
        dramTable.GetDim(1) * BLOCK_SIZE > MAX_SOURCE_TOKENS) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aivCount == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t totalPairSlots =
        static_cast<uint64_t>(batch) * sourceCapacity;
    const uint32_t usedCoreNum = static_cast<uint32_t>(
        totalPairSlots < aivCount ? totalPairSlots : aivCount);
    auto *tiling = context->GetTilingData<
        A5KvcacheScatterCopyC8TilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->copyCap = static_cast<uint32_t>(sourceCapacity);
    tiling->hbmMaxBlockNum = static_cast<uint32_t>(hbmTable.GetDim(1));
    tiling->dramMaxBlockNum = static_cast<uint32_t>(dramTable.GetDim(1));
    tiling->packedRowBytes = PACKED_ROW_BYTES;
    tiling->totalPairSlots = totalPairSlots;
    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
static ge::graphStatus InferPackedScatterShape(
    gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(HBM_KV) == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(HBM_KV);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferPackedScatterDataType(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_INT8);
    return ge::GRAPH_SUCCESS;
}

class A5KvcacheScatterCopyC8 : public OpDef {
public:
    explicit A5KvcacheScatterCopyC8(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> bytes = {ge::DT_INT8};
        const std::vector<ge::DataType> ints = {ge::DT_INT32};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};

        this->Input("source_token_ids").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("destination_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("copy_counts").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("hbm_block_table").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("dram_block_table").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("hbm_kv").ParamType(REQUIRED).DataType(bytes).Format(formats);
        this->Input("dram_kv").ParamType(REQUIRED).DataType(bytes).Format(formats);
        this->Output("hbm_kv_out").ParamType(REQUIRED).DataType(bytes).Format(formats);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore()
            .SetTiling(optiling::TilingA5KvcacheScatterCopyC8)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5KvcacheScatterCopyC8);

IMPL_OP_INFERSHAPE(A5KvcacheScatterCopyC8)
    .InferShape(InferPackedScatterShape)
    .InferDataType(InferPackedScatterDataType);
} // namespace ops
