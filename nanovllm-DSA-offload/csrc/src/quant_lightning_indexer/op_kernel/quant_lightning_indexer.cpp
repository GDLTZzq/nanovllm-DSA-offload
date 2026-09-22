/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "quant_lightning_indexer_template_tiling_key.h"

#if (__CCE_AICORE__ == 310)
    #include "arch35/quant_lightning_indexer_kernel.h"

#else
    #include "arch22/quant_lightning_indexer_kernel.h"
#endif

using namespace QLIKernel;

// =====================================================================================
// [QuantLI vs LI 差异点 #1] Kernel 入口参数
// =====================================================================================
// LI 的 kernel (lightning_indexer.cpp L39-45):
//   参数: query, key, weights, actualSeqLenQ, actualSeqLenK, blocktable,
//         prevTopkIndices, sparseIndices, sparseValues, workspace, tiling
//   Init: op.Init(query, key, weights, actualSeqLenQ, actualSeqLenK, blocktable,
//                 prevTopkIndices, sparseIndices, sparseValues, user, tiling_data, &tPipe)
//
// QuantLI (本文件):
//   参数: 多了 queryScale, keyScale              ← 反量化scale输入
//         少了 prevTopkIndices, sparseValues      ← 无GVR/value输出
//   模板: 多了 Q_LAYOUT_T, K_LAYOUT_T 泛型       ← Q和K的layout可以不同(BSND/TND/PA)
//   类型: QLIType 比 LIType 多了 W_T, SCALE_T, QK_T, SCORE_T 四个类型参数
//         (因为量化路径需要: weight精度、反量化scale精度、QK累加类型、score存储类型)
// =====================================================================================

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                         \
    do {                                                                                                     \
        templateClass<QLIType<__VA_ARGS__>> op;                                                              \
        GET_TILING_DATA_WITH_STRUCT(QLITilingData, tiling_data_in, tiling);                                  \
        const QLITilingData *__restrict tiling_data = &tiling_data_in;                                       \
        op.Init(query, key, weights, queryScale, keyScale, actualSeqLengthsQ, actualSeqLengthsK, blocktable, \
                sparseIndices, user, tiling_data, &tPipe);                                                   \
        op.Process();                                                                                        \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int Q_LAYOUT_T, int K_LAYOUT_T>
__global__ __aicore__ void quant_lightning_indexer(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                                   __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale,
                                                   __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK,
                                                   __gm__ uint8_t *blocktable, __gm__ uint8_t *sparseIndices,
                                                   __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

// =====================================================================================
// [QuantLI vs LI 差异点 #2] dtype 分支 & 模板参数展开
// =====================================================================================
// LI (lightning_indexer.cpp L50-66):
//   仅支持 BF16 / FP16:
//     L52-53: bfloat16_t, bfloat16_t, int32_t, PAGE_ATTN, LAYOUT, K_LAYOUT, DT_W_FLAG
//             → 7个模板参数, 无量化相关类型
//     L54-55: half, half, int32_t, ...
//   非A5平台走 LightningIndexerKernel (half/bf16)
//
// QuantLI (本文件):
//   支持 fp8_e4m3fn / hifp8 / int8:
//     每种组合需要显式指定:
//       Q_T(fp8/int8) + K_T(fp8/int8) + OUT_T(int32)
//       + PAGE_ATTN + Q_LAYOUT + K_LAYOUT
//       + W_T(half/bf16) + SCALE_T(half/bf16/float32) + QK_T(int32/float32) + SCORE_T(uint16/uint32)
//       → 10个模板参数!
//   非A5平台走 QLIPreload (仅int8兜底)
// =====================================================================================
#if (__CCE_AICORE__ == 310)
    if (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN) {
        if (ORIG_DTYPE_WEIGHTS == DT_FLOAT16) {
            INVOKE_LI_NO_KFC_OP_IMPL(QuantLightningIndexerKernel, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t,
                PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
                half, half, float32_t, uint16_t);
        } else {
            INVOKE_LI_NO_KFC_OP_IMPL(QuantLightningIndexerKernel, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t,
                PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
                bfloat16_t, float32_t, float32_t, uint16_t);
        }
    } else if (ORIG_DTYPE_QUERY == DT_HIFLOAT8) {
        INVOKE_LI_NO_KFC_OP_IMPL(QuantLightningIndexerKernel, hifloat8, hifloat8, int32_t,
                                PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
                                bfloat16_t, float32_t, float32_t, uint16_t);
    } else {
        INVOKE_LI_NO_KFC_OP_IMPL(QuantLightningIndexerKernel, int8_t, int8_t, int32_t,
                                PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
                                half, half, int32_t, uint32_t);
    }
#else
    INVOKE_LI_NO_KFC_OP_IMPL(QLIPreload, int8_t, int8_t, int32_t,
                                PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T));
#endif
}
