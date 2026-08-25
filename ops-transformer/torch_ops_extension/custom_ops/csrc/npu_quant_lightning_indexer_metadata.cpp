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

// metadata 输出按 quant_lightning_indexer_metadata.h 的 QLI_META_SIZE=1024 个 int32
constexpr int64_t QLI_METADATA_ELEM_NUM = 1024;

at::Tensor npu_quant_lightning_indexer_metadata_npu(
    const c10::optional<at::Tensor> &actualSeqLengthsQuery, const c10::optional<at::Tensor> &actualSeqLengthsKey,
    int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t queryQuantMode, int64_t keyQuantMode,
    int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK, c10::string_view layoutQuery,
    c10::string_view layoutKey, int64_t sparseCount, int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    int64_t cmpRatio, c10::string_view device)
{
    c10::Device outputDevice(torch_npu::utils::get_npu_device_type(), 0);
    at::Tensor output = torch::empty({QLI_METADATA_ELEM_NUM}, torch::dtype(torch::kInt32).device(outputDevice));

    std::string layoutQueryStr(layoutQuery);
    std::string layoutKeyStr(layoutKey);
    char *layoutQueryPtr = const_cast<char *>(layoutQueryStr.c_str());
    char *layoutKeyPtr = const_cast<char *>(layoutKeyStr.c_str());

    // EXEC_NPU_CMD_V1 实参顺序 = aclnnQuantLightningIndexerMetadataGetWorkspaceSize 声明顺序；
    // aic_core_num / aiv_core_num / soc_version 由 aclnn 层从平台信息自动填充
    EXEC_NPU_CMD_V1(aclnnQuantLightningIndexerMetadata, actualSeqLengthsQuery, actualSeqLengthsKey, numHeadsQ,
                    numHeadsK, headDim, queryQuantMode, keyQuantMode, batchSize, maxSeqlenQ, maxSeqlenK,
                    layoutQueryPtr, layoutKeyPtr, sparseCount, sparseMode, preTokens, nextTokens, cmpRatio, output);

    return output;
}

at::Tensor npu_quant_lightning_indexer_metadata_meta(
    const c10::optional<at::Tensor> &actualSeqLengthsQuery, const c10::optional<at::Tensor> &actualSeqLengthsKey,
    int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t queryQuantMode, int64_t keyQuantMode,
    int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK, c10::string_view layoutQuery,
    c10::string_view layoutKey, int64_t sparseCount, int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    int64_t cmpRatio, c10::string_view device)
{
    return torch::empty({QLI_METADATA_ELEM_NUM}, torch::dtype(torch::kInt32));
}
} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_quant_lightning_indexer_metadata", &custom::npu_quant_lightning_indexer_metadata_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_quant_lightning_indexer_metadata", &custom::npu_quant_lightning_indexer_metadata_meta);
}
