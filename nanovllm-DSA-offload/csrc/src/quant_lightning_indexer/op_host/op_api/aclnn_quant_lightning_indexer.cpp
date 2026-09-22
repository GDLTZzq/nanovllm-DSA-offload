/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include <string.h>
#include "graph/types.h"
#include "aclnn_quant_lightning_indexer.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerQuantLightningIndexerGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *queryDequantScale, const aclTensor *keyDequantScale,
    const aclTensor *actualSeqLengthsQueryOptional, const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional, int64_t queryQuantMode, int64_t keyQuantMode,
    char *layoutQueryOptional, char *layoutKeyOptional, int64_t sparseCount, int64_t sparseMode,
    int64_t preTokens, int64_t nextTokens, int64_t keyStride0, int64_t keyDequantScaleStride0,
    const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerQuantLightningIndexer(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                 const aclrtStream stream);

aclnnStatus aclnnQuantLightningIndexerGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional,
    int64_t queryQuantMode,
    int64_t keyQuantMode,
    char *layoutQueryOptional,
    char *layoutKeyOptional,
    int64_t sparseCount,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    if (query == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Query pointer is null, cannot get data type!");
        return ge::GRAPH_FAILED;
    }

    int64_t keyStride0 = -1;
    int64_t keyDequantScaleStride0 = -1;
    if (!IsContiguous(key)) {
        auto keyStride = key->GetViewStrides();
        keyStride0 = keyStride[0];
    }
    if (!IsContiguous(keyDequantScale)) {
        auto keyScaleStride = keyDequantScale->GetViewStrides();
        keyDequantScaleStride0 = keyScaleStride[0];
    }

    return aclnnInnerQuantLightningIndexerGetWorkspaceSize(
        query, key, weights, queryDequantScale, keyDequantScale, actualSeqLengthsQueryOptional,
        actualSeqLengthsKeyOptional, blockTableOptional, queryQuantMode, keyQuantMode,
        layoutQueryOptional, layoutKeyOptional, sparseCount, sparseMode, preTokens, nextTokens,
        keyStride0, keyDequantScaleStride0, out, workspaceSize, executor);
}

aclnnStatus aclnnQuantLightningIndexer(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    return aclnnInnerQuantLightningIndexer(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
