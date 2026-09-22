/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "a5_quant_lightning_indexer_c8_tiling.h"
#include <algorithm>
#include "../op_kernel/a5_quant_lightning_indexer_c8_template_tiling_key.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

ge::graphStatus A5QuantLightningIndexerC8Tiling::GetNpuInfo(QLI8TilingInfo &tilingInfo) const
{
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("A5QuantLightningIndexerC8", "opName got from TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    tilingInfo.opName = context_->GetNodeName();
    tilingInfo.platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(tilingInfo.platformInfo == nullptr, OPS_LOG_E(tilingInfo.opName, "GetPlatformInfo is nullptr."),
               return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo.platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OPS_ERR_IF(aicNum == 0 || aivNum == 0, OPS_LOG_E(tilingInfo.opName, "num of core obtained is 0."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr,
               OPS_LOG_E(tilingInfo.opName, "workspace size buffer is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
               OPS_LOG_E(tilingInfo.opName, "raw tiling data is nullptr."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus A5QuantLightningIndexerC8Tiling::GetTensorInfo(QLI8TilingInfo &tilingInfo) const
{
    auto &op = tilingInfo.opParamInfo;
    op.query.desc = context_->GetInputDesc(QLI8_QUERY_INDEX);
    op.query.shape = context_->GetInputShape(QLI8_QUERY_INDEX);
    op.key.desc = context_->GetInputDesc(QLI8_KEY_INDEX);
    op.key.shape = context_->GetInputShape(QLI8_KEY_INDEX);
    op.weights.desc = context_->GetInputDesc(QLI8_WEIGHTS_INDEX);
    op.weights.shape = context_->GetInputShape(QLI8_WEIGHTS_INDEX);
    op.queryDequantScale.desc = context_->GetInputDesc(QLI8_QUERY_DEQUANT_SCALE_INDEX);
    op.queryDequantScale.shape = context_->GetInputShape(QLI8_QUERY_DEQUANT_SCALE_INDEX);
    op.keyDequantScale.desc = context_->GetInputDesc(QLI8_KEY_DEQUANT_SCALE_INDEX);
    op.keyDequantScale.shape = context_->GetInputShape(QLI8_KEY_DEQUANT_SCALE_INDEX);
    op.actualSeqLengthsQ.desc = context_->GetInputDesc(QLI8_ACTUAL_SEQ_Q_INDEX);
    op.actualSeqLengthsQ.tensor = context_->GetInputTensor(QLI8_ACTUAL_SEQ_Q_INDEX);
    op.actualSeqLengthsK.desc = context_->GetInputDesc(QLI8_ACTUAL_SEQ_K_INDEX);
    op.actualSeqLengthsK.tensor = context_->GetInputTensor(QLI8_ACTUAL_SEQ_K_INDEX);
    op.blockTable.desc = context_->GetInputDesc(QLI8_BLOCK_TABLE_INDEX);
    op.blockTable.tensor = context_->GetInputTensor(QLI8_BLOCK_TABLE_INDEX);
    op.sparseIndices.desc = context_->GetOutputDesc(QLI8_SPARSE_INDICES_INDEX);
    op.sparseIndices.shape = context_->GetOutputShape(QLI8_SPARSE_INDICES_INDEX);

    OPS_ERR_IF(op.query.desc == nullptr || op.query.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "query desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.key.desc == nullptr || op.key.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "key desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.weights.desc == nullptr || op.weights.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "weights desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.queryDequantScale.desc == nullptr || op.queryDequantScale.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "query_dequant_scale desc/shape is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.keyDequantScale.desc == nullptr || op.keyDequantScale.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "key_dequant_scale desc/shape is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.actualSeqLengthsQ.desc == nullptr || op.actualSeqLengthsQ.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "actual_seq_lengths_query desc/tensor is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.actualSeqLengthsK.desc == nullptr || op.actualSeqLengthsK.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "actual_seq_lengths_key desc/tensor is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.blockTable.desc == nullptr || op.blockTable.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "block_table desc/tensor is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.sparseIndices.desc == nullptr || op.sparseIndices.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "sparse_indices desc/shape is nullptr."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus A5QuantLightningIndexerC8Tiling::CheckDtype(const QLI8TilingInfo &tilingInfo) const
{
    const auto &op = tilingInfo.opParamInfo;
    OPS_ERR_IF(op.query.desc->GetDataType() != ge::DT_FLOAT8_E4M3FN ||
                   op.key.desc->GetDataType() != ge::DT_FLOAT8_E4M3FN,
               OPS_LOG_E(tilingInfo.opName, "query/key dtype must be float8_e4m3fn."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.weights.desc->GetDataType() != ge::DT_FLOAT,
               OPS_LOG_E(tilingInfo.opName, "weights dtype must be fp32."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.queryDequantScale.desc->GetDataType() != ge::DT_FLOAT ||
                   op.keyDequantScale.desc->GetDataType() != ge::DT_FLOAT,
               OPS_LOG_E(tilingInfo.opName, "dequant scales dtype must be fp32."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32 ||
                   op.actualSeqLengthsK.desc->GetDataType() != ge::DT_INT32 ||
                   op.blockTable.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "per-request metadata inputs dtype must be int32."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.sparseIndices.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "sparse_indices dtype must be int32."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus A5QuantLightningIndexerC8Tiling::CheckShape(QLI8TilingInfo &tilingInfo) const
{
    const auto &op = tilingInfo.opParamInfo;
    const auto &qShape = op.query.shape->GetStorageShape();
    const auto &kShape = op.key.shape->GetStorageShape();
    const auto &wShape = op.weights.shape->GetStorageShape();
    const auto &qScaleShape = op.queryDequantScale.shape->GetStorageShape();
    const auto &kScaleShape = op.keyDequantScale.shape->GetStorageShape();
    const auto &seqQShape = op.actualSeqLengthsQ.tensor->GetStorageShape();
    const auto &seqShape = op.actualSeqLengthsK.tensor->GetStorageShape();
    const auto &blockShape = op.blockTable.tensor->GetStorageShape();
    const auto &indexOutShape = op.sparseIndices.shape->GetStorageShape();

    OPS_ERR_IF(qShape.GetDimNum() != QLI8_DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "query must be TND [B, N, 128], where N is 8/16/24/32/64."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDimNum() != QLI8_DIM_NUM_FOUR,
               OPS_LOG_E(tilingInfo.opName, "key must be PA_BSND [num_blocks, block_size, 1, 128]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(wShape.GetDimNum() != QLI8_DIM_NUM_TWO || qScaleShape.GetDimNum() != QLI8_DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "weights/query_dequant_scale must be [B, N]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kScaleShape.GetDimNum() != QLI8_DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "key_dequant_scale must be [num_blocks, block_size, 1]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(seqQShape.GetDimNum() != QLI8_DIM_NUM_ONE || seqShape.GetDimNum() != QLI8_DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "per-request seq lengths must be rank 1."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(blockShape.GetDimNum() != QLI8_DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "block_table must be rank 2."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(indexOutShape.GetDimNum() != QLI8_DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "sparse_indices must be [B, 1, 2048]."),
               return ge::GRAPH_FAILED);

    tilingInfo.bSize = static_cast<uint32_t>(qShape.GetDim(0));
    tilingInfo.gSize = static_cast<uint32_t>(qShape.GetDim(QLI8_DIM_IDX_ONE));
    tilingInfo.blockSize = static_cast<uint32_t>(kShape.GetDim(QLI8_DIM_IDX_ONE));
    tilingInfo.maxBlockNumPerBatch = static_cast<uint32_t>(blockShape.GetDim(QLI8_DIM_IDX_ONE));
    tilingInfo.s2Size = tilingInfo.blockSize * tilingInfo.maxBlockNumPerBatch;

    OPS_ERR_IF(tilingInfo.bSize == 0, OPS_LOG_E(tilingInfo.opName, "batch size must be > 0."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.gSize != 8 && tilingInfo.gSize != 16 && tilingInfo.gSize != 24 &&
                   tilingInfo.gSize != 32 && tilingInfo.gSize != 64,
               OPS_LOG_E(tilingInfo.opName, "decode query N must be 8/16/24/32/64."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(seqQShape.GetShapeSize() != tilingInfo.bSize ||
                   seqShape.GetShapeSize() != tilingInfo.bSize ||
                   blockShape.GetDim(0) != tilingInfo.bSize,
               OPS_LOG_E(tilingInfo.opName, "query and all per-request metadata batch dimensions must match."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDim(0) == 0, OPS_LOG_E(tilingInfo.opName, "key num_blocks must be > 0."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.blockSize != QLI8_BLOCK_SIZE,
               OPS_LOG_E(tilingInfo.opName, "key block_size must be 128."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDim(QLI8_DIM_IDX_TWO) != QLI8_DECODE_N2,
               OPS_LOG_E(tilingInfo.opName, "key N2 must be 1."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(qShape.GetDim(QLI8_DIM_IDX_TWO) != QLI8_HEAD_DIM || kShape.GetDim(QLI8_DIM_IDX_THREE) != QLI8_HEAD_DIM,
               OPS_LOG_E(tilingInfo.opName, "head_dim must be 128."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(wShape.GetDim(0) != tilingInfo.bSize || wShape.GetDim(1) != tilingInfo.gSize ||
                   qScaleShape.GetDim(0) != tilingInfo.bSize || qScaleShape.GetDim(1) != tilingInfo.gSize,
               OPS_LOG_E(tilingInfo.opName, "weights/query_dequant_scale must match query [B, N]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kScaleShape.GetDim(0) != kShape.GetDim(0) ||
                   kScaleShape.GetDim(QLI8_DIM_IDX_ONE) != tilingInfo.blockSize ||
                   kScaleShape.GetDim(QLI8_DIM_IDX_TWO) != QLI8_DECODE_N2,
               OPS_LOG_E(tilingInfo.opName, "key_dequant_scale must match key [num_blocks, 128, 1]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(indexOutShape.GetDim(0) != tilingInfo.bSize || indexOutShape.GetDim(1) != QLI8_DECODE_N2 ||
                   indexOutShape.GetDim(2) != QLI8_SPARSE_COUNT,
               OPS_LOG_E(tilingInfo.opName, "sparse_indices must have shape [B, 1, 2048]."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus A5QuantLightningIndexerC8Tiling::ParseAndCheck(QLI8TilingInfo &tilingInfo)
{
    if (GetNpuInfo(tilingInfo) != ge::GRAPH_SUCCESS || GetTensorInfo(tilingInfo) != ge::GRAPH_SUCCESS ||
        CheckDtype(tilingInfo) != ge::GRAPH_SUCCESS || CheckShape(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus A5QuantLightningIndexerC8Tiling::DoTiling(QLI8TilingInfo *tilingInfo)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    // workspace 与 kernel Init 的 GM 布局严格一致：
    //   libapi + aicNum*scoreGm + aicNum*ldScoreGm + aicNum*ldIndexGm
    // scoreGm：每 AIC 核存整个 s2 的 score，s1BaseSize*Align(s2,128)*sizeof(uint16)
    // ldScoreGm/ldIndexGm：每核至多 2 个跨核行的 LD 数据，
    //   s1BaseSize*Align(sparseCount,16)*2（slot=2 固定）。
    uint32_t s1BaseSize = (QLI8_M_BASE_SIZE + tilingInfo->gSize - 1) / tilingInfo->gSize;
    uint64_t alignS2 = ((uint64_t)tilingInfo->s2Size + QLI8_S2_BASE_SIZE - 1) / QLI8_S2_BASE_SIZE *
                       QLI8_S2_BASE_SIZE;
    uint64_t scoreSize = (uint64_t)s1BaseSize * alignS2 * sizeof(uint16_t);
    uint64_t ldScoreSize = (uint64_t)s1BaseSize * QLI8_TOP_K_COUNT_ALIGN_16 * 2 * sizeof(uint16_t);
    uint64_t ldIndexSize = (uint64_t)s1BaseSize * QLI8_TOP_K_COUNT_ALIGN_16 * 2 * sizeof(int32_t);
    uint64_t workspaceSize = (uint64_t)ascendcPlatform.GetLibApiWorkSpaceSize() +
                             (uint64_t)aicNum * (scoreSize + ldScoreSize + ldIndexSize);
    context_->GetWorkspaceSizes(1)[0] = static_cast<size_t>(workspaceSize);

    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_gSize(tilingInfo->gSize);
    tilingData_.set_s1Size(1);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_sparseCount(QLI8_SPARSE_COUNT);
    tilingData_.set_keyStride0(QLI8_KEY_STRIDE0);
    tilingData_.set_keyDequantScaleStride0(QLI8_KEY_DEQUANT_SCALE_STRIDE0);
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_sparseMode(3);
    tilingData_.set_cmpRatio(1);
    tilingData_.set_batchSupperFlag(0);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // ops.json 中以 uint8 顶替 float8_e4m3fn（msopgen 类型表不支持 fp8），
    // tiling key 的 DT 取 ge::DT_UINT8，与模板声明保持一致。
    uint32_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint32_t>(ge::DT_UINT8));
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForA5QuantLightningIndexerC8(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForA5QuantLightningIndexerC8(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR("A5QuantLightningIndexerC8", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    QLI8TilingInfo liInfo;
    A5QuantLightningIndexerC8Tiling liTiling(context);
    if (liTiling.ParseAndCheck(liInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return liTiling.DoTiling(&liInfo);
}

IMPL_OP_OPTILING(A5QuantLightningIndexerC8)
    .Tiling(TilingForA5QuantLightningIndexerC8)
    .TilingParse<QLI8CompileInfo>(TilingPrepareForA5QuantLightningIndexerC8);

} // namespace optiling
