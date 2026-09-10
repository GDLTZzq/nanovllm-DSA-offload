/**
 * Stage-1 A5 C8 MTP3 LightningIndexer.
 *
 * The Stage-1 work unit is one (request, gS1 tile). The stable path uses two
 * query rows per tile; high-batch N=32 uses one four-query tile per request.
 * The Arch35 payload TopK carries
 * (slot14, source18), classifies every route as miss-prefix/hit-suffix and
 * preserves old hit slots without changing cache_slots_pool.
 */

#ifndef A5_FUSED_LI_MANAGE_MTP_C8_QLI_GL_H
#define A5_FUSED_LI_MANAGE_MTP_C8_QLI_GL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "a5_fused_li_manage_mtp_c8_tiling.h"
#include "a5_fused_li_manage_mtp_c8_workspace.h"

using LIC8TilingData = A5FusedLiManageMtpC8TilingData;
#define A5_MTP_CLASSIFY_ONLY 1
#include "arch35_payload_c8/lightning_indexer_common.h"
#include "arch35_payload_c8/lightning_indexer_service_cube.h"
#include "arch35_payload_c8/lightning_indexer_service_vector.h"

namespace a5_fused_li_manage_mtp_c8_gl {
using namespace AscendC;
using namespace QLICommon;
using namespace QLIKernel;

constexpr uint32_t QLI_BLOCK_SIZE = 128;
constexpr uint32_t QLI_HEAD_DIM = 128;
constexpr uint32_t QLI_TOPK = 2048;
constexpr uint32_t QUERY_COUNT = 4;
constexpr uint32_t QLI_ROUTE_CAPACITY = QUERY_COUNT * QLI_TOPK;

using C8MtpQliType = QLIType<
    fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t, true,
    LI_LAYOUT::TND, LI_LAYOUT::PA_BSND,
    bfloat16_t, float32_t, float32_t, uint16_t>;

class QuantLiMtpPhase {
public:
    __aicore__ inline QuantLiMtpPhase(
        TPipe *pipe, const A5FusedLiManageMtpC8TilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR query, GM_ADDR key, GM_ADDR weights,
        GM_ADDR queryDequantScale, GM_ADDR keyDequantScale,
        GM_ADDR actualSeqLengthsQuery, GM_ADDR cacheTokens,
        GM_ADDR candidateLens, GM_ADDR reqPoolEntries,
        GM_ADDR cacheSlotsPool, GM_ADDR blockTable,
        GM_ADDR routePairRows, GM_ADDR topkSourceIds, GM_ADDR topkSlots,
        GM_ADDR routeThresholds, GM_ADDR routeMissCounts,
        GM_ADDR userWorkspace)
    {
        subBlockIdx_ = GetBlockIdx();
        if ASCEND_IS_AIV {
            aiCoreIdx_ = subBlockIdx_ / 2U;
        } else {
            aiCoreIdx_ = subBlockIdx_;
        }

        actualSeqLengthsQueryGm_.SetGlobalBuffer(
            (__gm__ int32_t *)actualSeqLengthsQuery);
        cacheTokensGm_.SetGlobalBuffer((__gm__ int32_t *)cacheTokens);
        candidateLensGm_.SetGlobalBuffer((__gm__ int32_t *)candidateLens);
        reqPoolEntriesGm_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        cacheSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);
        blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);

        constInfo_.batchSize = tiling_->batchSize;
        constInfo_.tSize = tiling_->packedQueryCount;
        constInfo_.gSize = tiling_->indexHeads;
        constInfo_.qHeadNum = tiling_->indexHeads;
        constInfo_.kHeadNum = 1;
        constInfo_.headDim = QLI_HEAD_DIM;
        constInfo_.sparseCount = QLI_TOPK;
        constInfo_.kSeqSize = tiling_->maxCandidateLen;
        constInfo_.qSeqSize = QUERY_COUNT;
        constInfo_.kCacheBlockSize = QLI_BLOCK_SIZE;
        constInfo_.maxBlockNumPerBatch = tiling_->maxBlockNumPerBatch;
        constInfo_.outputLayout = LI_LAYOUT::TND;
        constInfo_.attenMaskFlag = true;
        constInfo_.isAccumSeqS1 = true;
        // Match the host-selected adaptive schedule. The shared cube and
        // TopK paths support both shapes; ProcessVec1 explicitly handles the
        // two rows assigned to each AIV when s1BaseSize is four.
        constInfo_.s1BaseSize = tiling_->queryTileSize;
        constInfo_.mBaseSize =
            constInfo_.s1BaseSize * tiling_->indexHeads;
        constInfo_.s2BaseSize = QLI_BLOCK_SIZE;
        constInfo_.keyStride0 = tiling_->keyStride;
        constInfo_.keyDequantScaleStride0 = tiling_->scaleStride;
        constInfo_.poolSize = tiling_->poolSize;
        constInfo_.cacheSlotsSize = tiling_->sourceCapacity;
        constInfo_.setL2DisableFlag = false;

        scoreWorkspaceBaseGm_.SetGlobalBuffer(
            (__gm__ uint16_t *)userWorkspace);

        if ASCEND_IS_AIV {
            weightsGm_.SetGlobalBuffer((__gm__ bfloat16_t *)weights);
            queryScaleGm_.SetGlobalBuffer((__gm__ float *)queryDequantScale);
            keyScaleGm_.SetGlobalBuffer((__gm__ float *)keyDequantScale);
            routePairRowsGm_.SetGlobalBuffer(
                (__gm__ int32_t *)routePairRows);
            topkSourceIdsGm_.SetGlobalBuffer(
                (__gm__ int32_t *)topkSourceIds);
            topkSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)topkSlots);
            routeMissCountsGm_.SetGlobalBuffer(
                (__gm__ int32_t *)routeMissCounts);
            routeThresholdsGm_.SetGlobalBuffer(
                (__gm__ uint16_t *)routeThresholds);
            vectorService_.InitParams(constInfo_, tiling_);
            vectorService_.InitVecInputTensor(
                weightsGm_, queryScaleGm_, keyScaleGm_,
                routePairRowsGm_, blockTableGm_, cacheSlotsGm_,
                topkSlotsGm_, routeMissCountsGm_);
            vectorService_.InitMtpTopkSourceTensor(topkSourceIdsGm_);
            vectorService_.InitMtpThresholdTensor(routeThresholdsGm_);
            vectorService_.InitBuffers(pipe_);
        } else {
            queryGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)query);
            keyGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)key);
            matmulService_.InitParams(constInfo_);
            matmulService_.InitMm1GlobalTensor(
                blockTableGm_, keyGm_, queryGm_);
            matmulService_.InitBuffers(pipe_);
        }
    }

    __aicore__ inline void Process()
    {
        if (tiling_->queryTileSize != 2U &&
            tiling_->queryTileSize != QUERY_COUNT) {
            return;
        }
        const uint32_t queryTileCount =
            QUERY_COUNT / tiling_->queryTileSize;
        const uint32_t taskCount = tiling_->batchSize * queryTileCount;
        // 小 batch 打分拆分：所有 task 全部 active，且每 task 至少能分到 2 段
        // （scoringCoreNum/taskCount >= 2）。满足则把每条 row 的 S2 打分拆到
        // 多个 MIX 对上并行，SyncAll 后仍由 task 归属核跑整行 topk。
        if (tiling_->splitEnable != 0U && taskCount >= 1U &&
            taskCount * 2U <= tiling_->scoringCoreNum &&
            AllTasksActive(queryTileCount)) {
            ProcessSplitScore(queryTileCount, taskCount);
            return;
        }
        bool hasActiveTask = false;
        for (uint32_t task = aiCoreIdx_; task < taskCount;
             task += tiling_->usedCoreNum) {
            const uint32_t batch = task / queryTileCount;
            if (IsActiveRequest(batch)) {
                hasActiveTask = true;
            }
        }
        if (!hasActiveTask) {
            return;
        }

        if ASCEND_IS_AIV {
            vectorService_.AllocEventID();
            CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                ConstInfo::CROSS_VC_EVENT);
            CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                ConstInfo::CROSS_VC_EVENT + 1U);
        } else {
            matmulService_.AllocEventID();
        }

        uint32_t globalLoop = 0;
        for (uint32_t task = aiCoreIdx_; task < taskCount;
             task += tiling_->usedCoreNum) {
            const uint32_t batch = task / queryTileCount;
            const uint32_t gS1 = task % queryTileCount;
            if (!IsActiveRequest(batch)) {
                continue;
            }
            const uint32_t queryBegin = batch * QUERY_COUNT;
            const uint32_t candidate = static_cast<uint32_t>(
                candidateLensGm_.GetValue(batch));
            const uint32_t loopCount = candidate / QLI_BLOCK_SIZE;
            if ASCEND_IS_AIV {
                const uint64_t requestScoreBase =
                    static_cast<uint64_t>(batch) *
                        (tiling_->scoreWorkspaceStride / sizeof(uint16_t));
                const uint64_t routeScoreBase =
                    static_cast<uint64_t>(gS1) *
                        constInfo_.s1BaseSize * tiling_->sourceCapacity;
                vectorService_.InitVecWorkspaceTensor(
                    scoreWorkspaceBaseGm_[
                        requestScoreBase + routeScoreBase]);
            }
            for (uint32_t s2 = 0; s2 < loopCount; ++s2, ++globalLoop) {
                RunInfo run{};
                run.loop = globalLoop;
                run.bN2Idx = batch;
                run.bIdx = batch;
                run.n2Idx = 0;
                run.gS1Idx = gS1;
                run.s2Idx = s2;
                run.s2Start = 0;
                run.validS2Len = candidate;
                run.kScaleLoop = s2 / 16U + 1U;
                run.cacheRowIdx = static_cast<uint32_t>(
                    reqPoolEntriesGm_.GetValue(batch));
                run.cacheTokenCount = static_cast<uint32_t>(
                    cacheTokensGm_.GetValue(batch));
                run.actS1Size = QUERY_COUNT;
                run.actS2Size = candidate;
                run.actS2SizeOrig = candidate;
                run.actMBaseSize = constInfo_.mBaseSize;
                run.actualSingleProcessSInnerSize = QLI_BLOCK_SIZE;
                run.actualSingleProcessSInnerSizeAlign = QLI_BLOCK_SIZE;
                run.tensorQueryOffset =
                    static_cast<uint64_t>(queryBegin) *
                        tiling_->indexHeads * QLI_HEAD_DIM +
                    static_cast<uint64_t>(gS1) * constInfo_.mBaseSize *
                        QLI_HEAD_DIM;
                run.tensorKeyOffset =
                    static_cast<uint64_t>(s2) * QLI_BLOCK_SIZE * QLI_HEAD_DIM;
                run.tensorKeyScaleOffset =
                    static_cast<uint64_t>(s2) * QLI_BLOCK_SIZE;
                run.tensorWeightsOffset =
                    static_cast<uint64_t>(queryBegin) * tiling_->indexHeads;
                run.indiceOutOffset =
                    static_cast<uint64_t>(queryBegin) * QLI_TOPK;
                run.isFirstS2InnerLoop = s2 == 0;
                run.isLastS2InnerLoop = s2 + 1U == loopCount;
                run.isAllLoopEnd = false;
                run.isValid = true;

                if ASCEND_IS_AIC {
                    matmulService_.ComputeMm1(run);
                } else {
                    vectorService_.ProcessVec1(run);
                    if (run.isLastS2InnerLoop) {
                        const bool finalTask =
                            task + tiling_->usedCoreNum >= taskCount;
                        vectorService_.ProcessTopK(run, finalTask);
                    }
                }
            }
        }

        if ASCEND_IS_AIV {
            vectorService_.FreeEventID();
        } else {
            matmulService_.FreeEventID();
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                ConstInfo::CROSS_VC_EVENT);
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                ConstInfo::CROSS_VC_EVENT + 1U);
        }
    }

private:
    __aicore__ inline bool AllTasksActive(uint32_t queryTileCount)
    {
        const uint32_t taskCount = tiling_->batchSize * queryTileCount;
        for (uint32_t task = 0; task < taskCount; ++task) {
            const uint32_t batch = task / queryTileCount;
            if (!IsActiveRequest(batch)) {
                return false;
            }
        }
        return true;
    }

    // 与整行版 Process() 的 run 填充逐字段一致。打分拆分把同一 (task, s2) 的
    // run 原样喂给 ComputeMm1/ProcessVec1，只把 run.loop 换成按对内的顺序号
    // （决定 mm1 pingpong slot 与 CROSS 事件奇偶握手）。分数 GM 写址只依赖
    // (scoreGm 基址 + s2 + blockId)，与 loop 无关，故拆分后每块写址与整行版一致。
    __aicore__ inline void FillRun(RunInfo &run, uint32_t batch, uint32_t gS1,
                                   uint32_t candidate, uint32_t s2,
                                   uint32_t s2Start, uint32_t loopOrd,
                                   bool isFirst, bool isLast)
    {
        const uint32_t queryBegin = batch * QUERY_COUNT;
        run.loop = loopOrd;
        run.bN2Idx = batch;
        run.bIdx = batch;
        run.n2Idx = 0;
        run.gS1Idx = gS1;
        run.s2Idx = s2;
        run.s2Start = s2Start;
        run.validS2Len = candidate;
        run.kScaleLoop = s2 / 16U + 1U;
        run.cacheRowIdx = static_cast<uint32_t>(
            reqPoolEntriesGm_.GetValue(batch));
        run.cacheTokenCount = static_cast<uint32_t>(
            cacheTokensGm_.GetValue(batch));
        run.actS1Size = QUERY_COUNT;
        run.actS2Size = candidate;
        run.actS2SizeOrig = candidate;
        run.actMBaseSize = constInfo_.mBaseSize;
        run.actualSingleProcessSInnerSize = QLI_BLOCK_SIZE;
        run.actualSingleProcessSInnerSizeAlign = QLI_BLOCK_SIZE;
        run.tensorQueryOffset =
            static_cast<uint64_t>(queryBegin) * tiling_->indexHeads *
                QLI_HEAD_DIM +
            static_cast<uint64_t>(gS1) * constInfo_.mBaseSize * QLI_HEAD_DIM;
        run.tensorKeyOffset =
            static_cast<uint64_t>(s2) * QLI_BLOCK_SIZE * QLI_HEAD_DIM;
        run.tensorKeyScaleOffset =
            static_cast<uint64_t>(s2) * QLI_BLOCK_SIZE;
        run.tensorWeightsOffset =
            static_cast<uint64_t>(queryBegin) * tiling_->indexHeads;
        run.indiceOutOffset = static_cast<uint64_t>(queryBegin) * QLI_TOPK;
        run.isFirstS2InnerLoop = isFirst;
        run.isLastS2InnerLoop = isLast;
        run.isAllLoopEnd = false;
        run.isValid = true;
    }

    // 小 batch 打分跨核拆分：phase A 把每条 task 的 loopCount 个 S2 块按 16 块
    // 对齐切成 chunkCount = scoringCoreNum/taskCount 段，段 (p, c) 由
    // pair g = c*taskCount + p 处理（c=0 即 owner pair p）。AIC/AIV 共 driver，
    // 每对只跑自己连续一段，run.loop 用段内顺序号，事件握手与整行版逐块自洽。
    // phase A 在所有 AIV 上跑完（每个 AIV 一个 Alloc/Free 会话）后：AIV FreeEventID
    // 关会话 -> SyncAll（assistant 列段分数对 owner 可见）-> owner pair(g<taskCount)
    // 开第二个会话对整行跑 ProcessTopK -> FreeEventID。classify/publish 语义不变。
    __aicore__ inline void ProcessSplitScore(uint32_t queryTileCount,
                                             uint32_t taskCount)
    {
        const uint32_t chunkCount = tiling_->scoringCoreNum / taskCount;
        const uint32_t g = aiCoreIdx_;
        const uint32_t p = g % taskCount;
        const uint32_t c = g / taskCount;
        const uint32_t batch = p / queryTileCount;
        const uint32_t gS1 = p % queryTileCount;
        const uint32_t candidate = static_cast<uint32_t>(
            candidateLensGm_.GetValue(batch));
        const uint32_t loopCount = candidate / QLI_BLOCK_SIZE;

        if ASCEND_IS_AIV {
            vectorService_.AllocEventID();
            CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                ConstInfo::CROSS_VC_EVENT);
            CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                ConstInfo::CROSS_VC_EVENT + 1U);
        } else {
            matmulService_.AllocEventID();
        }

        if (c < chunkCount) {
            const uint32_t segLenRaw =
                (loopCount + chunkCount - 1U) / chunkCount;
            const uint32_t segStride =
                ((segLenRaw + 15U) / 16U) * 16U;
            const uint32_t segStart = c * segStride;
            if (segStart < loopCount) {
                const uint32_t segEnd =
                    (c + 1U) * segStride < loopCount
                        ? (c + 1U) * segStride
                        : loopCount;
                if ASCEND_IS_AIV {
                    const uint64_t requestScoreBase =
                        static_cast<uint64_t>(batch) *
                            (tiling_->scoreWorkspaceStride /
                             sizeof(uint16_t));
                    const uint64_t routeScoreBase =
                        static_cast<uint64_t>(gS1) *
                            constInfo_.s1BaseSize *
                            tiling_->sourceCapacity;
                    vectorService_.InitVecWorkspaceTensor(
                        scoreWorkspaceBaseGm_[requestScoreBase +
                                              routeScoreBase]);
                }
                uint32_t loopOrd = 0;
                for (uint32_t s2 = segStart; s2 < segEnd; ++s2, ++loopOrd) {
                    RunInfo run{};
                    FillRun(run, batch, gS1, candidate, s2, segStart,
                            loopOrd, s2 == segStart, s2 + 1U == segEnd);
                    if ASCEND_IS_AIC {
                        matmulService_.ComputeMm1(run);
                    } else {
                        vectorService_.ProcessVec1(run);
                    }
                }
            }
        }

        if ASCEND_IS_AIV {
            // phase A 会话先关闭再 SyncAll，复刻独立引擎已验证的邻接
            // ProcessMain(FreeEventID) -> ProcessDecode(SyncAll)。每块 score 已在
            // ProcessVec1:555 的 WaitFlag<V_MTE3> 前写完 GM；FreeEventID 等待每块
            // 末尾 line-565 的 VEC1_MTE3_V credit，故此处所有 AIV 的列段分数都在 GM。
            // 屏障使 assistant 对(g>=taskCount) 写的列段对 owner 对 AIV 的 MTE2 可见。
            // partner AIC 的收尾 CrossCoreWaitFlag(CROSS_VC+0/+1) 只依赖两 AIV phase A
            // 各 parity 最后一块的 line-566 set（本 AIV 到此前已产生），AIC 独立退出，
            // 不参与本屏障——AIC 侧无死锁。AIC 的 phase A 会话(Alloc/Free)在其分支已关。
            vectorService_.FreeEventID();
            AscendC::SyncAll();
            if (g < taskCount) {
                // owner pair 两个 AIV 各自对自辖半幅 route 行做整行 topk。
                // c=0 段开始时已把 scoreGm 基址设到本 task 行区，SyncAll 后全列可见。
                // ProcessTopK 首行 WaitFlag<TOPK_V_MTE2>(service_vector.h:1062) 依赖
                // AllocEventID 的预置 credit，故 topk 须在第二个 Alloc/Free 会话内跑。
                vectorService_.AllocEventID();
                RunInfo run{};
                FillRun(run, batch, gS1, candidate, loopCount - 1U, 0U, 0U,
                        false, true);
                vectorService_.ProcessTopK(run, true);
                vectorService_.FreeEventID();
            }
        } else {
            matmulService_.FreeEventID();
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                ConstInfo::CROSS_VC_EVENT);
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                ConstInfo::CROSS_VC_EVENT + 1U);
        }
    }

    __aicore__ inline bool IsActiveRequest(uint32_t batch)
    {
        const int32_t expectedQueryEnd =
            static_cast<int32_t>((batch + 1U) * QUERY_COUNT);
        const int32_t queryEnd =
            actualSeqLengthsQueryGm_.GetValue(batch);
        const int32_t budget = cacheTokensGm_.GetValue(batch);
        const int32_t candidate = candidateLensGm_.GetValue(batch);
        const int32_t poolRow = reqPoolEntriesGm_.GetValue(batch);
        return queryEnd == expectedQueryEnd && budget >= 8192 &&
            budget <= 16256 && candidate >= budget &&
            candidate <= static_cast<int32_t>(tiling_->maxCandidateLen) &&
            candidate % static_cast<int32_t>(QLI_BLOCK_SIZE) == 0 &&
            poolRow >= 0 && poolRow < static_cast<int32_t>(tiling_->poolSize);
    }

    TPipe *pipe_;
    const A5FusedLiManageMtpC8TilingData *tiling_;
    uint32_t subBlockIdx_ = 0;
    uint32_t aiCoreIdx_ = 0;
    ConstInfo constInfo_{};
    QLIMatmul<C8MtpQliType> matmulService_;
    QLIVector<C8MtpQliType> vectorService_;
    GlobalTensor<fp8_e4m3fn_t> queryGm_;
    GlobalTensor<fp8_e4m3fn_t> keyGm_;
    GlobalTensor<bfloat16_t> weightsGm_;
    GlobalTensor<float> queryScaleGm_;
    GlobalTensor<float> keyScaleGm_;
    GlobalTensor<int32_t> actualSeqLengthsQueryGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> cacheTokensGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> cacheSlotsGm_;
    GlobalTensor<int32_t> routePairRowsGm_;
    GlobalTensor<int32_t> topkSourceIdsGm_;
    GlobalTensor<int32_t> topkSlotsGm_;
    GlobalTensor<int32_t> routeMissCountsGm_;
    GlobalTensor<uint16_t> routeThresholdsGm_;
    GlobalTensor<uint16_t> scoreWorkspaceBaseGm_;
};
} // namespace a5_fused_li_manage_mtp_c8_gl

#endif
