/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file a5_quant_lightning_indexer_c8.cpp
 * \brief Decode-only fused C8 QuantLightningIndexer kernel entry.
 *
 * 单算子融合：官方 QuantLightningIndexer 依赖独立 AICPU metadata 算子做
 * BalanceSchedule + SplitFD 分核；本算子把这些调度逻辑搬进内核
 * （arch35/quant_lightning_indexer_kernel.h::ComputeSplitInfo，on-device
 * 分核），不新增 metadata 输入、无 sparseValues 输出，一个算子完成
 * "元数据调度 + top-k 稀疏索引"两个阶段。
 *
 * 固定组合（与官方 DT_FLOAT8_E4M3FN 分支一致）：
 *   fp8_e4m3fn query/key（ops.json 以 uint8 顶替）+ fp32 weights
 *   + fp32 scales + fp32 QK 累加 + uint16 score + int32 输出索引。
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "a5_quant_lightning_indexer_c8_template_tiling_key.h"
#include "arch35/quant_lightning_indexer_kernel.h"

using namespace QLIKernel;

#define INVOKE_QLI_C8_OP_IMPL(templateClass, ...)                                                          \
    do {                                                                                                    \
        templateClass<QLIType<__VA_ARGS__>> op;                                                             \
        GET_TILING_DATA_WITH_STRUCT(QLITilingData, tiling_data_in, tiling);                                 \
        const QLITilingData *__restrict tiling_data = &tiling_data_in;                                      \
        op.Init(query, key, weights, queryScale, keyScale, actualSeqLengthsQ, actualSeqLengthsK, blocktable, \
                sparseIndices, user, tiling_data, &tPipe);                                                  \
        op.Process();                                                                                       \
    } while (0)

template <int DT>
__global__ __aicore__ void a5_quant_lightning_indexer_c8(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                                         __gm__ uint8_t *weights, __gm__ uint8_t *queryScale,
                                                         __gm__ uint8_t *keyScale, __gm__ uint8_t *actualSeqLengthsQ,
                                                         __gm__ uint8_t *actualSeqLengthsK, __gm__ uint8_t *blocktable,
                                                         __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                                         __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    static_assert(DT == LI_C8_TPL_UINT8,
                  "A5QuantLightningIndexerC8 tiling key must be the uint8 storage id");
    INVOKE_QLI_C8_OP_IMPL(QLIPreload, fp8_e4m3fn_t, fp8_e4m3fn_t, float, uint16_t, int32_t,
                          1, LI_LAYOUT::TND, LI_LAYOUT::PA_BSND);
}
