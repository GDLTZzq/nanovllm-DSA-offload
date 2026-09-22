/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <register/op_impl_registry.h>
#include "a5_sfa_shared/ops_log_compat.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5;
constexpr uint32_t CACHE_SLOTS_INDEX = 7;
constexpr int64_t DECODE_SPARSE_COUNT = 2048;
constexpr int64_t MTP_UNION_CAPACITY = DECODE_SPARSE_COUNT * 4;

static ge::graphStatus InferShapeA5FusedLiManageMtpC8(
    gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("A5FusedLiManageMtpC8", "InferShapeContext is nullptr."),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    const gert::Shape *actualQShape = context->GetInputShape(ACTUAL_SEQ_Q_INDEX);
    const gert::Shape *cacheShape = context->GetInputShape(CACHE_SLOTS_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, actualQShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, cacheShape, return ge::GRAPH_FAILED);

    gert::Shape *topkSlotsShape = context->GetOutputShape(0);
    gert::Shape *missSourceShape = context->GetOutputShape(1);
    gert::Shape *missDestShape = context->GetOutputShape(2);
    gert::Shape *missCountShape = context->GetOutputShape(3);
    gert::Shape *cacheAliasShape = context->GetOutputShape(4);
    OPS_LOG_E_IF_NULL(context, topkSlotsShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, missSourceShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, missDestShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, missCountShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, cacheAliasShape, return ge::GRAPH_FAILED);

    OPS_ERR_IF(queryShape->GetDimNum() != 3 || actualQShape->GetDimNum() != 1,
               OPS_LOG_E(context, "query/actual_seq_lengths_query ranks must be 3/1."),
               return ge::GRAPH_FAILED);
    const int64_t packedQueries = queryShape->GetDim(0);
    const int64_t batch = actualQShape->GetDim(0);
    topkSlotsShape->SetDimNum(3);
    topkSlotsShape->SetDim(0, packedQueries);
    topkSlotsShape->SetDim(1, 1);
    topkSlotsShape->SetDim(2, DECODE_SPARSE_COUNT);
    missSourceShape->SetDimNum(2);
    missSourceShape->SetDim(0, batch);
    missSourceShape->SetDim(1, MTP_UNION_CAPACITY);
    *missDestShape = *missSourceShape;
    missCountShape->SetDimNum(1);
    missCountShape->SetDim(0, batch);
    *cacheAliasShape = *cacheShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeA5FusedLiManageMtpC8(
    gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("A5FusedLiManageMtpC8", "InferDataTypeContext is nullptr."),
               return ge::GRAPH_FAILED);
    for (uint32_t index = 0; index < 5; ++index) {
        context->SetOutputDataType(index, ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(A5FusedLiManageMtpC8)
    .InferShape(InferShapeA5FusedLiManageMtpC8)
    .InferDataType(InferDataTypeA5FusedLiManageMtpC8);
} // namespace ops
