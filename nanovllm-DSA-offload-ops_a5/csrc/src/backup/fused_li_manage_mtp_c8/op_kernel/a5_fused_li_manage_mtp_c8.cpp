/** Stage 4: complete one-kernel C8 MTP3 LI and cache management. */

// 调试二分：=1 时跳过 stage2（OrderedMissUnion），仅验证 stage1（QLI 打分 +
// LD 归并 + classify+publish）是否单独崩溃。裸 winner 探针完成，恢复 0 跑整机。
// 2026-09-04 stage1 内容修复（Branch1），开 stage2 无打印整机验证 507015 是否随
// stage1 干净而消失。
#define C8_MTP_BISECT_SKIP_STAGE2 0
// 调试二分：=1 时打开 kernel 内 printf 阶段标记（[C8BISECT]/[ST2]），定位崩溃
// 阶段。DIAG：官方测试首启即挂起（stage2 首次真跑），临时置 1 定位卡点，恢复 0。
#define C8_MTP_BISECT_PRINT 0
// 调试二分：=1 时跳过 ProcessLD（LD 归并读 + LdTopK），隔离「cube/vec1/topk写/事件」
// 与「LD 归并」。定位到根因后恢复为 0。
#define C8_MTP_BISECT_SKIP_LD 0
// 调试二分：=1 时跳过 ProcessTopK（非 LD publish + LD 部分写），只剩 cube 打分 +
// ProcessVec1 + 事件 + SyncAll。定位到根因后恢复为 0。
#define C8_MTP_BISECT_SKIP_TOPK 0
// 调试二分：=1 时跳过 cube Fixpipe（L0C->UB），验证 M=128 dualDst 写是否越界。
// 定位到根因后恢复为 0。
#define C8_MTP_BISECT_SKIP_FIXPIPE 0
// 调试二分：=1 时跳过 AIV ProcessVec1 的向量算术（Cast+BatchMul），保留
// weight/qScale/kScale 加载、事件、score GM 写。定位到根因后恢复为 0。
#define C8_MTP_BISECT_SKIP_VEC1 0
// 调试二分：=1 时 AIV 完全跳过 ProcessVec1（含 loads/GM 写），仅保留跨核同步
// （等 C 核 mm1 + 通知 C 核可复用）。定位到根因后恢复为 0。
#define C8_MTP_BISECT_SKIP_PV1_ENTIRE 0
// 调试二分：=1 时 AIC/AIV 都完全跳过 ProcessMain 的整个 bN2/gS1/s2 循环（含
// ComputeMm1 与跨核 CROSS 事件），仅剩 ProcessDecode（InitLDBuffers/ICachePreLoad/
// SyncAll）+ entry SyncAll。判定：仍 507015 → 崩在 ProcessDecode/entry 同步；
// 不崩 → 崩在 ProcessMain 循环（ComputeMm1 的 MTE2 加载 或 CalcRunInfo/CROSS 事件）。
#define C8_MTP_BISECT_SKIP_PROCESS_MAIN 0
// 一次性诊断：=1 时 classify 每核打印 [MCDIAG]（mc 读在打印之前，打印不影响计数），
// stage2 打 [UDIAG]（各 route 长度/去重/受害者）。定位 classify AR 计数问题用，完事回 0。
// 已修复（ClearSpr+绝对值，2026-09-03）：残留 AR 位移导致 content 垃圾 + 计数时序。
// 2026-09-03 验证 stage2 dedup race 修复：恢复 0，跑无打印整机验证。
// DIAG(mixed 2row 2026-09-04)：置 1 打开 [MCDIAG]/[CQ0..6]/[LDFILL]，定位 q1/q3 发布
// 内容错（payload 截断 vs classify）。定位后恢复 0。
// 2026-09-04 根因=Branch2(whole-row greedy) row1 处理，已强制 Branch1 修复。恢复 0
// 跑无打印整机（开 stage2）验证 507015 是否随 stage1 干净而消失。
#define C8_MTP_DIAG_MC 0

#include "kernel_operator.h"
#include "a5_fused_li_manage_mtp_c8_tiling.h"
#include "a5_fused_li_manage_mtp_c8_qli.h"
#include "a5_fused_li_manage_mtp_c8_union.h"
#include "a5_fused_li_manage_mtp_c8_workspace.h"

extern "C" __global__ __aicore__ void
a5_fused_li_manage_mtp_c8(
    GM_ADDR query,
    GM_ADDR key,
    GM_ADDR weights,
    GM_ADDR queryDequantScale,
    GM_ADDR keyDequantScale,
    GM_ADDR actualSeqLengthsQuery,
    GM_ADDR reqPoolEntries,
    GM_ADDR cacheSlotsPool,
    GM_ADDR cacheTokens,
    GM_ADDR candidateLens,
    GM_ADDR blockTable,
    GM_ADDR topkSourceIds,
    GM_ADDR topkDestinationSlots,
    GM_ADDR topkMissCount,
    GM_ADDR missSourceIds,
    GM_ADDR missDestinationSlots,
    GM_ADDR missCounts,
    GM_ADDR cacheSlotsAlias,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)cacheSlotsAlias;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(A5FusedLiManageMtpC8TilingData);
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    using namespace a5_fused_li_manage_mtp_c8_workspace;
    const uint64_t scoreStride = tilingData.scoreWorkspaceStride;
    const uint64_t batchSize = tilingData.batchSize;
    // The helpers in the shared workspace header are intentionally host-side
    // constexpr functions for tiling.  Expand the same arithmetic here so an
    // __aicore__ kernel never calls a host function.
    const uint64_t routePairOffset = scoreStride * batchSize;
    const uint64_t routePairBytes =
        batchSize * UNION_CAPACITY * 2U * sizeof(int32_t);
    const uint64_t routeThresholdOffset =
        routePairOffset + routePairBytes;
    const uint64_t routeThresholdBytes =
        batchSize * ROUTES * THRESHOLD_STRIDE * sizeof(uint16_t);
    GM_ADDR routePairRows =
        userWorkspace + routePairOffset;
    GM_ADDR routeThresholds =
        userWorkspace + routeThresholdOffset;
    // QLI LD 归并工作区（ldScore/ldIndex）紧随 thresholds，host 按 workspace.h
    // 的 LdScoreBytes/LdIndexBytes 分配。
    GM_ADDR ldWorkspace =
        userWorkspace + routeThresholdOffset + routeThresholdBytes;
    // The public per-query output is also the Stage-1 -> Stage-2 hand-off.
    // This keeps one authoritative count and avoids a redundant GM copy.
    GM_ADDR routeMissCounts = topkMissCount;

    a5_fused_li_manage_mtp_c8_impl::QuantLiMtpPhase qli(
        &pipe, &tilingData);
    qli.Init(
        query, key, weights, queryDequantScale, keyDequantScale,
        actualSeqLengthsQuery, cacheTokens, candidateLens,
        reqPoolEntries, cacheSlotsPool, blockTable,
        routePairRows, topkSourceIds, topkDestinationSlots,
        routeThresholds, routeMissCounts,
        userWorkspace, ldWorkspace);
#if C8_MTP_BISECT_PRINT
    if (GetBlockIdx() == 0U) {
        if ASCEND_IS_AIV {
            AscendC::printf("[C8BISECT] core0(AIV) entry qli.Process enter\n");
        } else {
            AscendC::printf("[C8BISECT] core0(AIC) entry qli.Process enter\n");
        }
    }
#endif
    qli.Process();
#if C8_MTP_BISECT_PRINT
    if (GetBlockIdx() == 0U) {
        if ASCEND_IS_AIV {
            AscendC::printf("[C8BISECT] core0(AIV) entry qli.Process done\n");
        } else {
            AscendC::printf("[C8BISECT] core0(AIC) entry qli.Process done\n");
        }
    }
#endif

    if ASCEND_IS_AIV {
        // The (request, gS1) QLI task owners publish route pairs/counts with
        // MTE3. Every request owner consumes all four routes through MTE2 only
        // after this global barrier.
#if C8_MTP_BISECT_PRINT
        if (GetBlockIdx() == 0U) {
            AscendC::printf("[C8BISECT] core0(AIV) entry SyncAll enter\n");
        }
#endif
        AscendC::SyncAll();
#if C8_MTP_BISECT_PRINT
        if (GetBlockIdx() == 0U) {
            AscendC::printf("[C8BISECT] core0(AIV) entry SyncAll done\n");
        }
#endif
#if C8_MTP_BISECT_SKIP_STAGE2
        (void)0;  // BISECT：stage2 跳过，stage1 已跑完；输出区保持 caller 毒值。
#else
        if ((GetBlockIdx() & 1U) == 0U) {
#if C8_MTP_BISECT_PRINT
            if (GetBlockIdx() == 0U) {
                AscendC::printf("[ST2] S0 stage2 enter, reset\n");
            }
#endif
            pipe.Reset();
            a5_fused_li_manage_mtp_c8_impl::OrderedMissUnion unionOp;
            unionOp.Init(
                routePairRows, routeThresholds, routeMissCounts,
                userWorkspace, candidateLens, reqPoolEntries,
                cacheSlotsPool, missSourceIds, missDestinationSlots,
                missCounts, topkSourceIds, topkDestinationSlots,
                cacheTokens,
                tilingData.sourceCapacity, tilingData.batchSize, &pipe);
#if C8_MTP_BISECT_PRINT
            if (GetBlockIdx() == 0U) {
                AscendC::printf("[ST2] S1 init done\n");
            }
#endif
            unionOp.Process(GetBlockIdx() / 2U,
                            tilingData.usedCoreNum);
#if C8_MTP_BISECT_PRINT
            if (GetBlockIdx() == 0U) {
                AscendC::printf("[ST2] S2 stage2 done\n");
            }
#endif
        }
#endif  // C8_MTP_BISECT_SKIP_STAGE2
    }
}
