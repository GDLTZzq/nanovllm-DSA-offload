/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Stage 4: official LightningIndexer scheduling, union eviction, and cache update.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "fused_li_manage_mtp_template_tiling_key.h"
#include "lightning_indexer_kernel.h"
#include "fused_li_manage_mtp_union.h"
#include "fused_li_manage_mtp_workspace.h"

using namespace LIKernel;
using namespace AscendC;

#define LI_MTP_COPY_TILING()                                                                                           \
    GET_TILING_DATA_WITH_STRUCT(FusedLiManageTilingData, tiling_data_in, tiling);                                      \
    const FusedLiManageTilingData *__restrict tiling_data = &tiling_data_in

#define INVOKE_LI_MTP_TOPK(...)                                                                                        \
    do {                                                                                                               \
        LI_MTP_COPY_TILING();                                                                                           \
        __gm__ uint8_t *unionPair0 =                                                                                    \
            user + MtpWorkspace::Pair0Offset(GetBlockNum(), tiling_data->n1Size);                                      \
        __gm__ uint8_t *unionPair1 =                                                                                    \
            user + MtpWorkspace::Pair1Offset(GetBlockNum(), tiling_data->n1Size, tiling_data->bSize);                  \
        __gm__ uint8_t *scoreScratch =                                                                                  \
            user + MtpWorkspace::ScoreOffset(GetBlockNum(), tiling_data->n1Size, tiling_data->bSize);                  \
        __gm__ uint8_t *thresholdScratch =                                                                              \
            user + MtpWorkspace::ThresholdOffset(GetBlockNum(), tiling_data->n1Size, tiling_data->bSize,               \
                                                   tiling_data->cacheSlotsSize);                                         \
        LIPreload<LIType<__VA_ARGS__, int32_t, true, LI_LAYOUT::BSND, LI_LAYOUT::PA_BSND>> op;                          \
        op.Init(query, key, weights, reqPoolEntries, cacheSlots, nullptr, actualSeqLengths, blockTable,                 \
                topkSourceIds, topkSlots, unionPair0, unionPair1, scoreScratch, thresholdScratch,                       \
                user, tiling_data, &tPipe);                                                                             \
        op.Process();                                                                                                   \
        if ASCEND_IS_AIV {                                                                                              \
            SyncAll();                                                                                                  \
            if ((GetBlockIdx() & 1U) == 0U) {                                                                          \
                tPipe.Reset();                                                                                          \
                MtpUnion::MtpMissUnion unionOp;                                                                         \
                unionOp.Init(unionPair0, unionPair1, actualSeqLengths, cacheSlots, cacheTokens,                         \
                             reqPoolEntries, scoreScratch, thresholdScratch, missSrcIds,                               \
                             missDstSlots, missCount, topkSourceIds, topkSlots,                                        \
                             tiling_data->bSize, tiling_data->s2Size,                                                  \
                             tiling_data->cacheSlotsSize, &tPipe);                                                       \
                unionOp.Process(GetBlockIdx() / 2U, GetBlockNum());                                                     \
            }                                                                                                           \
        }                                                                                                               \
    } while (0)

template <int DT>
__global__ __aicore__ void nanovllm_fused_li_manage_mtp(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
    __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheSlots,
    __gm__ uint8_t *cacheTokens, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *topkSlots,
    __gm__ uint8_t *topkSourceIds, __gm__ uint8_t *missSrcIds,
    __gm__ uint8_t *missDstSlots, __gm__ uint8_t *missCount,
    __gm__ uint8_t *cacheSlotsOut, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)
#else
    TPipe tPipe;
    (void)cacheSlotsOut;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if constexpr (DT == LI_MTP_TPL_FP16) {
        INVOKE_LI_MTP_TOPK(half, half);
    } else {
        INVOKE_LI_MTP_TOPK(bfloat16_t, bfloat16_t);
    }
#endif
}
