/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <register/op_impl_registry.h>
#include "a5_sfa_shared/ops_log_compat.h"

using namespace ge;

namespace ops {
constexpr uint32_t QLI8_QUERY_INDEX = 0;
constexpr uint32_t QLI8_KEY_INDEX = 1;
constexpr int64_t QLI8_SPARSE_COUNT = 2048;
constexpr int64_t QLI8_K_HEAD_NUM = 1;

static ge::graphStatus InferShapeA5QuantLightningIndexerC8(
    gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("A5QuantLightningIndexerC8", "InferShapeContext is nullptr."),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QLI8_QUERY_INDEX);
    const gert::Shape *keyShape = context->GetInputShape(QLI8_KEY_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, keyShape, return ge::GRAPH_FAILED);

    gert::Shape *outShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, outShape, return ge::GRAPH_FAILED);

    OPS_ERR_IF(queryShape->GetDimNum() != 3 || keyShape->GetDimNum() != 4,
               OPS_LOG_E(context, "query/key ranks must be 3/4 (TND query, PA_BSND key)."),
               return ge::GRAPH_FAILED);
    outShape->SetDimNum(3);
    outShape->SetDim(0, queryShape->GetDim(0));
    outShape->SetDim(1, QLI8_K_HEAD_NUM);
    outShape->SetDim(2, QLI8_SPARSE_COUNT);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeA5QuantLightningIndexerC8(
    gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("A5QuantLightningIndexerC8", "InferDataTypeContext is nullptr."),
               return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(A5QuantLightningIndexerC8)
    .InferShape(InferShapeA5QuantLightningIndexerC8)
    .InferDataType(InferDataTypeA5QuantLightningIndexerC8);
} // namespace ops
