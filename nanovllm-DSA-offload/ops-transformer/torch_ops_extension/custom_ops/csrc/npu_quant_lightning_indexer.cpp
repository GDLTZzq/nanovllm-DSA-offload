/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

constexpr int64_t KEY_N_DIM = 2;  // PA_BSND key 形状 (blocks, block_size, N, D) 的 N 轴

// 工具函数，推导输出 sparse_indices / sparse_values 的 shape 与 dtype
//   - sparse_indices: INT32，BSND -> [B, S, N(k), sparse_count]；TND -> [T, N(k), sparse_count]
//   - sparse_values: FLOAT，return_value=False 时为空 [0]
std::tuple<at::Tensor, at::Tensor> construct_qli_output_tensors(const at::Tensor &query, const at::Tensor &key,
                                                                c10::string_view layoutQuery,
                                                                c10::string_view layoutKey, int64_t sparseCount,
                                                                bool returnValue)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(layoutQuery == "TND" || layoutQuery == "BSND",
                "layout_query must be TND or BSND, but got ", layoutQuery, ".");
    TORCH_CHECK(layoutKey == "PA_BSND" || layoutKey == "TND",
                "layout_key must be PA_BSND or TND, but got ", layoutKey, ".");
    TORCH_CHECK(sparseCount > 0, "sparse_count must be greater than 0, but got ", sparseCount, ".");

    int64_t keyHeadNum = (layoutKey == "TND") ? key.size(1) : key.size(KEY_N_DIM);
    c10::SmallVector<int64_t, 4> outSize;
    if (layoutQuery == "BSND") {
        outSize = {query.size(0), query.size(1), keyHeadNum, sparseCount};
    } else {
        outSize = {query.size(0), keyHeadNum, sparseCount};
    }

    at::Tensor sparseIndices = at::empty(outSize, query.options().dtype(at::kInt));
    at::Tensor sparseValues =
        returnValue ? at::empty(outSize, query.options().dtype(at::kFloat))
                    : at::empty({0}, query.options().dtype(at::kFloat));
    return std::tuple<at::Tensor, at::Tensor>(sparseIndices, sparseValues);
}

// NPU 设备前向实现（函数形参顺序 = schema 顺序）
std::tuple<at::Tensor, at::Tensor> npu_quant_lightning_indexer_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &queryDequantScale, const at::Tensor &keyDequantScale,
    const c10::optional<at::Tensor> &actualSeqLengthsQuery, const c10::optional<at::Tensor> &actualSeqLengthsKey,
    const c10::optional<at::Tensor> &blockTable, const c10::optional<at::Tensor> &metadata,
    int64_t queryQuantMode, int64_t keyQuantMode, c10::string_view layoutQuery, c10::string_view layoutKey,
    int64_t sparseCount, int64_t sparseMode, int64_t preTokens, int64_t nextTokens, int64_t cmpRatio,
    bool returnValue)
{
    TORCH_CHECK(weights.numel() > 0, "Tensor weights is empty.");
    TORCH_CHECK(queryDequantScale.numel() > 0, "Tensor query_dequant_scale is empty.");
    TORCH_CHECK(keyDequantScale.numel() > 0, "Tensor key_dequant_scale is empty.");

    std::tuple<at::Tensor, at::Tensor> outputs =
        construct_qli_output_tensors(query, key, layoutQuery, layoutKey, sparseCount, returnValue);
    at::Tensor sparseIndices = std::get<0>(outputs);
    at::Tensor sparseValues = std::get<1>(outputs);

    std::string layoutQueryStr(layoutQuery);
    std::string layoutKeyStr(layoutKey);
    char *layoutQueryPtr = const_cast<char *>(layoutQueryStr.c_str());
    char *layoutKeyPtr = const_cast<char *>(layoutKeyStr.c_str());

    // EXEC_NPU_CMD_V1 实参顺序 = aclnnQuantLightningIndexerGetWorkspaceSize 声明顺序
    // （输入 -> 属性 -> 输出），key_stride0 / key_dequant_scale_stride0 走默认 0
    int64_t keyStride0 = 0;
    int64_t keyDequantScaleStride0 = 0;
    EXEC_NPU_CMD_V1(aclnnQuantLightningIndexer, query, key, weights, queryDequantScale, keyDequantScale,
                    actualSeqLengthsQuery, actualSeqLengthsKey, blockTable, metadata, queryQuantMode, keyQuantMode,
                    layoutQueryPtr, layoutKeyPtr, sparseCount, sparseMode, preTokens, nextTokens, cmpRatio,
                    returnValue, keyStride0, keyDequantScaleStride0, sparseIndices, sparseValues);

    return outputs;
}

// META 设备前向实现
std::tuple<at::Tensor, at::Tensor> npu_quant_lightning_indexer_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &queryDequantScale, const at::Tensor &keyDequantScale,
    const c10::optional<at::Tensor> &actualSeqLengthsQuery, const c10::optional<at::Tensor> &actualSeqLengthsKey,
    const c10::optional<at::Tensor> &blockTable, const c10::optional<at::Tensor> &metadata,
    int64_t queryQuantMode, int64_t keyQuantMode, c10::string_view layoutQuery, c10::string_view layoutKey,
    int64_t sparseCount, int64_t sparseMode, int64_t preTokens, int64_t nextTokens, int64_t cmpRatio,
    bool returnValue)
{
    return construct_qli_output_tensors(query, key, layoutQuery, layoutKey, sparseCount, returnValue);
}
} // namespace custom

// NPU 设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_quant_lightning_indexer", &custom::npu_quant_lightning_indexer_npu);
}

// META 设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_quant_lightning_indexer", &custom::npu_quant_lightning_indexer_meta);
}
