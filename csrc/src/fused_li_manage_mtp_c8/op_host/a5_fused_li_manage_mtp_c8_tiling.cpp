/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "a5_fused_li_manage_mtp_c8_tiling.h"
#include <algorithm>
#include "../op_kernel/a5_fused_li_manage_mtp_c8_template_tiling_key.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

ge::graphStatus A5FusedLiManageMtpC8Tiling::GetNpuInfo(LIMtpC8TilingInfo &tilingInfo) const
{
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("A5FusedLiManageMtpC8", "opName got from TilingContext is nullptr.");
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

ge::graphStatus A5FusedLiManageMtpC8Tiling::GetTensorInfo(LIMtpC8TilingInfo &tilingInfo) const
{
    auto &op = tilingInfo.opParamInfo;
    op.query.desc = context_->GetInputDesc(QUERY_INDEX);
    op.query.shape = context_->GetInputShape(QUERY_INDEX);
    op.key.desc = context_->GetInputDesc(KEY_INDEX);
    op.key.shape = context_->GetInputShape(KEY_INDEX);
    op.weights.desc = context_->GetInputDesc(WEIGHTS_INDEX);
    op.weights.shape = context_->GetInputShape(WEIGHTS_INDEX);
    op.queryDequantScale.desc = context_->GetInputDesc(QUERY_DEQUANT_SCALE_INDEX);
    op.queryDequantScale.shape = context_->GetInputShape(QUERY_DEQUANT_SCALE_INDEX);
    op.keyDequantScale.desc = context_->GetInputDesc(KEY_DEQUANT_SCALE_INDEX);
    op.keyDequantScale.shape = context_->GetInputShape(KEY_DEQUANT_SCALE_INDEX);
    op.actualSeqLengthsQ.desc = context_->GetInputDesc(ACTUAL_SEQ_Q_INDEX);
    op.actualSeqLengthsQ.tensor = context_->GetInputTensor(ACTUAL_SEQ_Q_INDEX);
    op.reqPoolEntries.desc = context_->GetInputDesc(REQ_POOL_ENTRIES_INDEX);
    op.reqPoolEntries.tensor = context_->GetInputTensor(REQ_POOL_ENTRIES_INDEX);
    op.cacheSlots.desc = context_->GetInputDesc(CACHE_SLOTS_INDEX);
    op.cacheSlots.shape = context_->GetInputShape(CACHE_SLOTS_INDEX);
    op.cacheTokens.desc = context_->GetInputDesc(CACHE_TOKENS_INDEX);
    op.cacheTokens.tensor = context_->GetInputTensor(CACHE_TOKENS_INDEX);
    op.candidateLens.desc = context_->GetInputDesc(CANDIDATE_LENS_INDEX);
    op.candidateLens.tensor = context_->GetInputTensor(CANDIDATE_LENS_INDEX);
    op.blockTable.desc = context_->GetInputDesc(BLOCK_TABLE_INDEX);
    op.blockTable.tensor = context_->GetInputTensor(BLOCK_TABLE_INDEX);
    op.topkDestSlotsOut.desc = context_->GetOutputDesc(TOPK_DEST_SLOTS_INDEX);
    op.topkDestSlotsOut.shape = context_->GetOutputShape(TOPK_DEST_SLOTS_INDEX);
    op.missSourceIdsOut.desc = context_->GetOutputDesc(MISS_SOURCE_IDS_INDEX);
    op.missSourceIdsOut.shape = context_->GetOutputShape(MISS_SOURCE_IDS_INDEX);
    op.missDestSlotsOut.desc = context_->GetOutputDesc(MISS_DEST_SLOTS_INDEX);
    op.missDestSlotsOut.shape = context_->GetOutputShape(MISS_DEST_SLOTS_INDEX);
    op.missCountsOut.desc = context_->GetOutputDesc(MISS_COUNTS_INDEX);
    op.missCountsOut.shape = context_->GetOutputShape(MISS_COUNTS_INDEX);
    op.cacheSlotsOut.desc = context_->GetOutputDesc(CACHE_SLOTS_OUT_INDEX);
    op.cacheSlotsOut.shape = context_->GetOutputShape(CACHE_SLOTS_OUT_INDEX);

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
    OPS_ERR_IF(op.reqPoolEntries.desc == nullptr || op.reqPoolEntries.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "req_pool_entries desc/tensor is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheSlots.desc == nullptr || op.cacheSlots.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "cache_slots_pool desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheTokens.desc == nullptr || op.cacheTokens.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "cache_tokens desc/tensor is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.candidateLens.desc == nullptr || op.candidateLens.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "candidate_lens desc/tensor is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.blockTable.desc == nullptr || op.blockTable.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "block_table desc/tensor is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.topkDestSlotsOut.desc == nullptr || op.topkDestSlotsOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "topk_destination_slots desc/shape is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.missSourceIdsOut.desc == nullptr || op.missSourceIdsOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "miss_source_ids desc/shape is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.missDestSlotsOut.desc == nullptr || op.missDestSlotsOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "miss_destination_slots desc/shape is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.missCountsOut.desc == nullptr || op.missCountsOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "miss_counts desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheSlotsOut.desc == nullptr || op.cacheSlotsOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "cache_slots_alias desc/shape is nullptr."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus A5FusedLiManageMtpC8Tiling::CheckDtype(const LIMtpC8TilingInfo &tilingInfo) const
{
    const auto &op = tilingInfo.opParamInfo;
    OPS_ERR_IF(op.query.desc->GetDataType() != ge::DT_FLOAT8_E4M3FN ||
                   op.key.desc->GetDataType() != ge::DT_FLOAT8_E4M3FN,
               OPS_LOG_E(tilingInfo.opName, "query/key dtype must be float8_e4m3fn."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.weights.desc->GetDataType() != ge::DT_BF16,
               OPS_LOG_E(tilingInfo.opName, "weights dtype must be bf16."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.queryDequantScale.desc->GetDataType() != ge::DT_FLOAT ||
                   op.keyDequantScale.desc->GetDataType() != ge::DT_FLOAT,
               OPS_LOG_E(tilingInfo.opName, "dequant scales dtype must be fp32."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32 ||
                   op.reqPoolEntries.desc->GetDataType() != ge::DT_INT32 ||
                   op.cacheSlots.desc->GetDataType() != ge::DT_INT32 ||
                   op.cacheTokens.desc->GetDataType() != ge::DT_INT32 ||
                   op.candidateLens.desc->GetDataType() != ge::DT_INT32 ||
                   op.blockTable.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "metadata inputs dtype must be int32."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.topkDestSlotsOut.desc->GetDataType() != ge::DT_INT32 ||
                   op.missSourceIdsOut.desc->GetDataType() != ge::DT_INT32 ||
                   op.missDestSlotsOut.desc->GetDataType() != ge::DT_INT32 ||
                   op.missCountsOut.desc->GetDataType() != ge::DT_INT32 ||
                   op.cacheSlotsOut.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "outputs dtype must be int32."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus A5FusedLiManageMtpC8Tiling::CheckShape(LIMtpC8TilingInfo &tilingInfo) const
{
    const auto &op = tilingInfo.opParamInfo;
    const auto &qShape = op.query.shape->GetStorageShape();
    const auto &kShape = op.key.shape->GetStorageShape();
    const auto &wShape = op.weights.shape->GetStorageShape();
    const auto &qScaleShape = op.queryDequantScale.shape->GetStorageShape();
    const auto &kScaleShape = op.keyDequantScale.shape->GetStorageShape();
    const auto &seqQShape = op.actualSeqLengthsQ.tensor->GetStorageShape();
    const auto &reqPoolShape = op.reqPoolEntries.tensor->GetStorageShape();
    const auto &cacheShape = op.cacheSlots.shape->GetStorageShape();
    const auto &cacheTokensShape = op.cacheTokens.tensor->GetStorageShape();
    const auto &candidateShape = op.candidateLens.tensor->GetStorageShape();
    const auto &blockShape = op.blockTable.tensor->GetStorageShape();
    const auto &topkSlotsOutShape = op.topkDestSlotsOut.shape->GetStorageShape();
    const auto &missSourceOutShape = op.missSourceIdsOut.shape->GetStorageShape();
    const auto &missDestOutShape = op.missDestSlotsOut.shape->GetStorageShape();
    const auto &missCountOutShape = op.missCountsOut.shape->GetStorageShape();
    const auto &cacheSlotsOutShape = op.cacheSlotsOut.shape->GetStorageShape();

    OPS_ERR_IF(qShape.GetDimNum() != DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "query must be TND [T, N1, 128], where N1 is 32/64."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDimNum() != DIM_NUM_FOUR,
               OPS_LOG_E(tilingInfo.opName, "key must be PA_BSND [num_blocks, block_size, 1, 128]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(wShape.GetDimNum() != DIM_NUM_TWO || qScaleShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "weights/query_dequant_scale must be [T, N1]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kScaleShape.GetDimNum() != DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "key_dequant_scale must be [num_blocks, block_size, 1]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "cache_slots_pool must be [pool_size, source_capacity]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(seqQShape.GetDimNum() != DIM_NUM_ONE || reqPoolShape.GetDimNum() != DIM_NUM_ONE ||
                   cacheTokensShape.GetDimNum() != DIM_NUM_ONE || candidateShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "per-request metadata must be rank 1."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(blockShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "block_table must be rank 2."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(topkSlotsOutShape.GetDimNum() != DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "topk_destination_slots must be [T, 1, 2048]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missSourceOutShape.GetDimNum() != DIM_NUM_TWO || missDestOutShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "miss_source_ids/miss_destination_slots must be [B, 8192]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missCountOutShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "miss_counts must be [B]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheSlotsOutShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "cache_slots_alias must be rank 2."),
               return ge::GRAPH_FAILED);

    tilingInfo.bSize = static_cast<uint32_t>(seqQShape.GetDim(0));
    tilingInfo.tSize = static_cast<uint32_t>(qShape.GetDim(0));
    tilingInfo.n1Size = static_cast<uint32_t>(qShape.GetDim(DIM_IDX_ONE));
    tilingInfo.blockSize = static_cast<uint32_t>(kShape.GetDim(DIM_IDX_ONE));
    tilingInfo.maxBlockNumPerBatch = static_cast<uint32_t>(blockShape.GetDim(DIM_IDX_ONE));
    tilingInfo.s2Size = tilingInfo.blockSize * tilingInfo.maxBlockNumPerBatch;
    tilingInfo.poolSize = static_cast<uint32_t>(cacheShape.GetDim(0));
    tilingInfo.cacheSlotsSize = static_cast<uint32_t>(cacheShape.GetDim(1));

    OPS_ERR_IF(tilingInfo.bSize == 0, OPS_LOG_E(tilingInfo.opName, "batch size must be > 0."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.n1Size != 32 && tilingInfo.n1Size != 64,
               OPS_LOG_E(tilingInfo.opName, "MTP query N1 must be 32/64."),
               return ge::GRAPH_FAILED);
    // MTP：T 为 packed query 总数，每请求 2~4 个
    OPS_ERR_IF(tilingInfo.tSize < tilingInfo.bSize * MTP_MIN_QUERY_COUNT ||
                   tilingInfo.tSize > tilingInfo.bSize * MTP_MAX_QUERY_COUNT,
               OPS_LOG_E(tilingInfo.opName,
                         "packed query count T must be in [2B, 4B]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(reqPoolShape.GetShapeSize() != tilingInfo.bSize ||
                   cacheTokensShape.GetShapeSize() != tilingInfo.bSize ||
                   candidateShape.GetShapeSize() != tilingInfo.bSize ||
                   blockShape.GetDim(0) != tilingInfo.bSize,
               OPS_LOG_E(tilingInfo.opName,
                         "query and all per-request metadata batch dimensions must match."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.poolSize == 0 || tilingInfo.cacheSlotsSize == 0,
               OPS_LOG_E(tilingInfo.opName, "cache_slots_pool dimensions must be positive."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDim(0) == 0, OPS_LOG_E(tilingInfo.opName, "key num_blocks must be > 0."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.blockSize != DECODE_BLOCK_SIZE,
               OPS_LOG_E(tilingInfo.opName, "key block_size must be 128."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDim(DIM_IDX_TWO) != DECODE_N2,
               OPS_LOG_E(tilingInfo.opName, "key N2 must be 1."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.s2Size != tilingInfo.cacheSlotsSize ||
                   tilingInfo.cacheSlotsSize > MAX_CACHE_SLOTS_SIZE,
               OPS_LOG_E(tilingInfo.opName,
                         "source capacity must equal block-table capacity and be <= 2^18."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(qShape.GetDim(DIM_IDX_TWO) != DECODE_HEAD_DIM || kShape.GetDim(DIM_IDX_THREE) != DECODE_HEAD_DIM,
               OPS_LOG_E(tilingInfo.opName, "head_dim must be 128."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(wShape.GetDim(0) != tilingInfo.tSize || wShape.GetDim(1) != tilingInfo.n1Size ||
                   qScaleShape.GetDim(0) != tilingInfo.tSize || qScaleShape.GetDim(1) != tilingInfo.n1Size,
               OPS_LOG_E(tilingInfo.opName, "weights/query_dequant_scale must match query [T, N1]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kScaleShape.GetDim(0) != kShape.GetDim(0) ||
                   kScaleShape.GetDim(DIM_IDX_ONE) != tilingInfo.blockSize ||
                   kScaleShape.GetDim(DIM_IDX_TWO) != DECODE_N2,
               OPS_LOG_E(tilingInfo.opName, "key_dequant_scale must match key [num_blocks, 128, 1]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(topkSlotsOutShape.GetDim(0) != tilingInfo.tSize ||
                   topkSlotsOutShape.GetDim(1) != DECODE_N2 ||
                   topkSlotsOutShape.GetDim(2) != DECODE_SPARSE_COUNT,
               OPS_LOG_E(tilingInfo.opName, "topk_destination_slots must have shape [T, 1, 2048]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missSourceOutShape.GetDim(0) != tilingInfo.bSize ||
                   missSourceOutShape.GetDim(1) != MTP_UNION_CAPACITY ||
                   missDestOutShape.GetDim(0) != tilingInfo.bSize ||
                   missDestOutShape.GetDim(1) != MTP_UNION_CAPACITY,
               OPS_LOG_E(tilingInfo.opName, "miss outputs must have shape [B, 8192]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missCountOutShape.GetDim(0) != tilingInfo.bSize,
               OPS_LOG_E(tilingInfo.opName, "miss_counts must have shape [B]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheSlotsOutShape.GetDim(0) != cacheShape.GetDim(0) ||
                   cacheSlotsOutShape.GetDim(1) != cacheShape.GetDim(1),
               OPS_LOG_E(tilingInfo.opName, "cache_slots_alias must match cache_slots_pool."),
               return ge::GRAPH_FAILED);

    // TND 布局：T 必须等于 actual_seq_lengths_query 的末元素（累计前缀和）
    OPS_ERR_IF(seqQShape.GetShapeSize() != tilingInfo.bSize,
               OPS_LOG_E(tilingInfo.opName,
                         "actual_seq_lengths_query must have size B."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus A5FusedLiManageMtpC8Tiling::ParseAndCheck(LIMtpC8TilingInfo &tilingInfo)
{
    if (GetNpuInfo(tilingInfo) != ge::GRAPH_SUCCESS || GetTensorInfo(tilingInfo) != ge::GRAPH_SUCCESS ||
        CheckDtype(tilingInfo) != ge::GRAPH_SUCCESS || CheckShape(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus A5FusedLiManageMtpC8Tiling::DoTiling(LIMtpC8TilingInfo *tilingInfo)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    constexpr uint32_t S1_BASE_SIZE = 4;
    constexpr uint32_t S2_BASE_SIZE = 128;
    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // score 区：按 s1BaseSize 行 × 全部 s2 块
    workspaceSize += S1_BASE_SIZE *
        ((tilingInfo->s2Size + S2_BASE_SIZE - 1) / S2_BASE_SIZE) *
        S2_BASE_SIZE * sizeof(uint16_t) * aicNum;
    // topk 区：T 行 × 2048 × 4B（每请求 packed query 的 token 行）
    workspaceSize += tilingInfo->tSize * DECODE_SPARSE_COUNT * sizeof(int32_t);
    context_->GetWorkspaceSizes(1)[0] = workspaceSize;

    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_tSize(tilingInfo->tSize);
    tilingData_.set_n2Size(DECODE_N2);
    tilingData_.set_gSize(tilingInfo->n1Size);
    tilingData_.set_s1Size(1);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_sparseCount(DECODE_SPARSE_COUNT);
    tilingData_.set_keyStride0(KEY_STRIDE0);
    tilingData_.set_keyDequantScaleStride0(KEY_DEQUANT_SCALE_STRIDE0);
    tilingData_.set_usedCoreNum(aicNum);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_sparseMode(3);
    tilingData_.set_poolSize(tilingInfo->poolSize);
    tilingData_.set_cacheSlotsSize(tilingInfo->cacheSlotsSize);
    tilingData_.set_packedQueryCount(tilingInfo->tSize);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // ops.json 中以 uint8 顶替 float8_e4m3fn（msopgen 类型表不支持 fp8），
    // tiling key 的 DT 取 ge::DT_UINT8，与模板声明保持一致。
    uint32_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint32_t>(ge::DT_UINT8));
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForA5FusedLiManageMtpC8(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForA5FusedLiManageMtpC8(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR("A5FusedLiManageMtpC8", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    LIMtpC8TilingInfo liInfo;
    A5FusedLiManageMtpC8Tiling liTiling(context);
    if (liTiling.ParseAndCheck(liInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return liTiling.DoTiling(&liInfo);
}

IMPL_OP_OPTILING(A5FusedLiManageMtpC8)
    .Tiling(TilingForA5FusedLiManageMtpC8)
    .TilingParse<LIMtpC8CompileInfo>(TilingPrepareForA5FusedLiManageMtpC8);

} // namespace optiling
