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
 * \file quant_lightning_indexer_kernel.h
 * \brief
 */

#ifndef quant_lightning_indexer_KERNEL_H
#define quant_lightning_indexer_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_common.h"
#include "quant_lightning_indexer_service_vector.h"
#include "quant_lightning_indexer_service_cube.h"

namespace QLIKernel {
using namespace QLICommon;
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;   // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;    // S2方向循环的结束Idx
    uint32_t actS1Size = 1ULL;  // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0ULL;
    uint32_t actS2SizeOrig = 0ULL;//压缩前s2
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false;  // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;            // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;          // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;         // S2方向循环的尾基本块大小
    bool isNeedLD = false;     // 该基本块是否需要LD
};

template <typename QLIT>
class QLIPreload {
public:
    __aicore__ inline QLIPreload(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengthsK, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *sparseIndices,
                                __gm__ uint8_t *workspace, const QLITilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    using Q_T = typename QLIT::queryType;
    using K_T = typename QLIT::keyType;
    using OUT_T = typename QLIT::outputType;
    static constexpr bool PAGE_ATTENTION = QLIT::pageAttention;
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIT::keyLayout;

    using SCORE_T = typename QLIT::scoreType;

    QLIMatmul<QLIT> matmulService;
    QLIVector<QLIT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 256;
    // 官方 arch35 的 S1 基本块大小：Branch 2 对齐官方时 s1BaseSize=S1_BASE_SIZE，
    // tSize≤64 减半 → s1BaseSize=2、mBaseSize=2×gSize。
    static constexpr uint32_t S1_BASE_SIZE = 4;
    static constexpr uint32_t S2_BASE_SIZE = 128;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;

    static constexpr int64_t LD_PREFETCH_LEN = 2;
    // for workspace double
    static constexpr uint32_t WS_DOUBLE = 2;

    // MIX_AIC_1_2 下 AIC=24、AIV=48（编译期常量，与 GetBlockNum() 自洽）
    static constexpr uint32_t AIC_CORE_NUM = 24;
    static constexpr uint32_t AIV_CORE_NUM = 48;

    // Branch 1（S2 跨核切分 + LD）仅在整数核数能均分时才有收益：
    // rowK=floor(24/rowsCount)>=2 ⇔ rowsCount<=12 → 临界路径 ≤ 256 块 < 官方整行 512 块。
    // rowsCount 13~24 时 rowK=1 → 多数行单核、临界路径退化为整行 512 块，还白付 LD 开销，
    // 与官方"整行无 LD"结构无异（官方此处还因 s1BaseSize 更小流量更省）→ 直接走官方整行逻辑。
    static constexpr uint32_t BRANCH1_MAX_ROWS = AIC_CORE_NUM / 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t keyScaleCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;
    bool isUsedCoreEqZero = false;
    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<float> weightsGm;
    GlobalTensor<float> qScaleGm;
    GlobalTensor<float> kScaleGm;

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;

    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGm;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    QLICommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    QLICommon::SplitCoreInfo splitCoreInfo{};
    QLICommon::LdSplitCoreInfo ldInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const QLITilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK);
    // ================================Split Core================================
    // on-device 分核：复刻 AICPU metadata 的 BalanceSchedule + SplitFD，融合进内核。
    // 所有核（AIC/AIV）跑同一确定性算法，各自取自己的切片，无需跨核同步、不新增输入。
    __aicore__ inline void ComputeSplitInfo(uint32_t cubeCoreIdx, uint32_t vecCoreIdx);
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig);
    // (bN2, gS1) 行游标：按 bN2 主序/gS1 次序列举所有非空行
    __aicore__ inline void InitRowCursor(uint32_t &bN2, uint32_t &gS1);
    __aicore__ inline uint32_t RowCursorInfo(uint32_t bN2, uint32_t gS1, uint32_t &w, uint32_t &tailM);
    __aicore__ inline bool AdvanceRowCursor(uint32_t &bN2, uint32_t &gS1);
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                            QLICommon::RunInfo runInfo, uint32_t qScaleLoop,
                                            uint32_t kScaleLoop);
    __aicore__ inline void ProcessDecode();
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &actualSeqLengthsGm, uint32_t defaultSeqLen);
    __aicore__ inline uint32_t GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                            GlobalTensor<uint32_t> &actualSeqLengthsGm, uint32_t defaultSeqLen, uint32_t cmpRatio);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size, uint32_t &actS2SizeOrig);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, QLICommon::RunInfo &runInfo,
                                       uint32_t qScaleLoop, uint32_t kScaleLoop);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::InitTilingData(const QLITilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.attenMaskFlag = (tilingData->sparseMode == 3);
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->sparseCount;
    constInfo.cmpRatio = tilingData->cmpRatio;
    constInfo.batchSupperFlag = tilingData->batchSupperFlag;
    constInfo.keyStride0 = tilingData->keyStride0;
    constInfo.keyDequantScaleStride0 = tilingData->keyDequantScaleStride0;
    constInfo.outputLayout = Q_LAYOUT_T;  // 输出和输入形状一致
    if (Q_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if (K_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;

    constInfo.mBaseSize = M_BASE_SIZE;
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.s1BaseSize = (constInfo.mBaseSize + constInfo.gSize - 1) / constInfo.gSize;
    // workspace 步长固定用 host 分配值，不随 Branch 2 的对齐覆盖变化
    constInfo.s1BaseSizeWs = constInfo.s1BaseSize;
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                          __gm__ uint8_t *actualSeqLengthsK)
{
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = (constInfo.batchSupperFlag) ? constInfo.batchSize + 1 : constInfo.batchSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (actualSeqLengthsK == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsK, constInfo.actualLenDims);
    }
}

template <typename QLIT>
__aicore__ inline uint32_t QLIPreload<QLIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                             GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                             uint32_t defaultSeqLen)
{
    bIdx = (constInfo.batchSupperFlag)? bIdx + 1 : bIdx; // 如果为B+1情况，则向后移动一位
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq && bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename QLIT>
__aicore__ inline uint32_t QLIPreload<QLIT>::GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                             GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                             uint32_t defaultSeqLen, uint32_t cmpRatio)
{
    if (actualLenDims == 0) {
        return defaultSeqLen * cmpRatio;
    } else if (isAccumSeq && bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size, uint32_t &actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    actS2SizeOrig =
        GetActualSeqLenKey(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, actualSeqLengthsGm, constInfo.kSeqSize, constInfo.cmpRatio); // 压缩前的actS2Size
    actS2Size = actS2SizeOrig / constInfo.cmpRatio;   // 真实使用的压缩后S2长度    
}

template <typename QLIT>
__aicore__ inline uint32_t QLIPreload<QLIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                     uint32_t actS2SizeOrig)
{
    if (actS2SizeOrig / constInfo.cmpRatio == 0) {
        return 0;
    }
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    int32_t validS2LenBase = static_cast<int32_t>(actS2SizeOrig) - static_cast<int32_t>(actS1Size);    // 压缩前的validS2LenBase
    int32_t validS2Len = (static_cast<int32_t>(s1Offset) + validS2LenBase + static_cast<int32_t>(constInfo.s1BaseSize)) / static_cast<int32_t>(constInfo.cmpRatio);  
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2SizeOrig) / constInfo.cmpRatio);
    validS2Len = Max(validS2Len, 1);
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::InitRowCursor(uint32_t &bN2, uint32_t &gS1)
{
    bN2 = 0;
    gS1 = 0;
    uint32_t totalBN2 = (uint32_t)(constInfo.batchSize * constInfo.kHeadNum);
    while (bN2 < totalBN2) {
        uint32_t bIdx = bN2 / constInfo.kHeadNum;
        uint32_t actS1Size, actS2Size, actS2SizeOrig;
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        if (actS1Size != 0 && actS2Size != 0) {
            return;
        }
        bN2++;
    }
    gS1 = 0;
}

template <typename QLIT>
__aicore__ inline uint32_t QLIPreload<QLIT>::RowCursorInfo(uint32_t bN2, uint32_t gS1, uint32_t &w, uint32_t &tailM)
{
    uint32_t bIdx = bN2 / constInfo.kHeadNum;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
    if (constInfo.attenMaskFlag) {
        w = GetS2BaseBlockNumOnMask(gS1, actS1Size, actS2SizeOrig);
    } else {
        w = (actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    // 该 gS1 行在 S1 轴的行数（与内核 curS1ProcNum 一致），LD 归约的 m 轴大小
    uint32_t gS1Num = (actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    uint32_t lastRow = gS1Num - 1;
    uint32_t mSize = (gS1 == lastRow) ? (actS1Size * constInfo.gSize - gS1 * constInfo.mBaseSize)
                                      : constInfo.mBaseSize;
    tailM = mSize / constInfo.gSize;
    return w;
}

template <typename QLIT>
__aicore__ inline bool QLIPreload<QLIT>::AdvanceRowCursor(uint32_t &bN2, uint32_t &gS1)
{
    gS1++;
    uint32_t totalBN2 = (uint32_t)(constInfo.batchSize * constInfo.kHeadNum);
    while (bN2 < totalBN2) {
        uint32_t bIdx = bN2 / constInfo.kHeadNum;
        uint32_t actS1Size, actS2Size, actS2SizeOrig;
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        uint32_t gS1Num = (actS1Size == 0 || actS2Size == 0)
                              ? 0
                              : (actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
        if (gS1 < gS1Num) {
            return true;
        }
        bN2++;
        gS1 = 0;
    }
    return false;
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::ComputeSplitInfo(uint32_t cubeCoreIdx, uint32_t vecCoreIdx)
{
    // ===== Pass A：统计总行数与总块数（全部按 bN2 主序/gS1 次序）=====
    uint32_t rowsCount = 0;
    uint32_t totalBlocks = 0;
    uint32_t rowW[AIC_CORE_NUM], rowB[AIC_CORE_NUM], rowG[AIC_CORE_NUM], rowTailM[AIC_CORE_NUM];
    uint32_t totalBN2 = (uint32_t)(constInfo.batchSize * constInfo.kHeadNum);
    for (uint32_t bN2 = 0; bN2 < totalBN2; bN2++) {
        uint32_t bIdx = bN2 / constInfo.kHeadNum;
        uint32_t actS1Size, actS2Size, actS2SizeOrig;
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        if (actS1Size == 0 || actS2Size == 0) {
            continue;
        }
        uint32_t gS1Num = (actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
        uint32_t lastRow = gS1Num - 1;
        for (uint32_t gS1 = 0; gS1 < gS1Num; gS1++) {
            uint32_t w;
            if (constInfo.attenMaskFlag) {
                w = GetS2BaseBlockNumOnMask(gS1, actS1Size, actS2SizeOrig);
            } else {
                w = (actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
            }
            if (rowsCount < AIC_CORE_NUM) {
                rowW[rowsCount] = w;
                rowB[rowsCount] = bN2;
                rowG[rowsCount] = gS1;
                uint32_t mSize = (gS1 == lastRow) ? (actS1Size * constInfo.gSize - gS1 * constInfo.mBaseSize)
                                                  : constInfo.mBaseSize;
                rowTailM[rowsCount] = mSize / constInfo.gSize;
            }
            rowsCount++;
            totalBlocks += w;
        }
    }
    if (totalBlocks == 0) {
        // 全空 case：清理输出
        isUsedCoreEqZero = true;
        splitCoreInfo.isCoreEnable = false;
        return;
    }

    if (rowsCount <= BRANCH1_MAX_ROWS) {
        // ===== Apportionment：行数 ≤ 12，把每行按块数比例切给多个核 =====
        // 每行 k_R 份（1 ≤ k_R ≤ w_R，保证每行至少一个核、行内切点跨核数不超过块数）
        uint32_t rowK[AIC_CORE_NUM];
        uint32_t sumK = 0;
        for (uint32_t i = 0; i < rowsCount; i++) {
            uint32_t k = (uint32_t)((uint64_t)rowW[i] * AIC_CORE_NUM / totalBlocks);
            if (k == 0) {
                k = 1;
            }
            if (k > rowW[i]) {
                k = rowW[i];
            }
            rowK[i] = k;
            sumK += k;
        }
        // 补齐到 AIC_CORE_NUM：按最大分数余数加给未饱和行
        while (sumK < AIC_CORE_NUM) {
            int32_t best = -1;
            uint32_t bestFrac = 0;
            for (uint32_t i = 0; i < rowsCount; i++) {
                if (rowK[i] < rowW[i]) {
                    uint32_t frac = (uint32_t)(((uint64_t)rowW[i] * AIC_CORE_NUM) % totalBlocks);
                    if (best < 0 || frac > bestFrac) {
                        best = (int32_t)i;
                        bestFrac = frac;
                    }
                }
            }
            if (best < 0) {
                break;  // 全部行已饱和（块数不足），剩余核闲置
            }
            rowK[best]++;
            sumK++;
        }
        // 超出则从最大 k_R 行扣回（≥1 保底）
        while (sumK > AIC_CORE_NUM) {
            int32_t best = -1;
            uint32_t bestK = 0;
            for (uint32_t i = 0; i < rowsCount; i++) {
                if (rowK[i] > 1 && rowK[i] > bestK) {
                    best = (int32_t)i;
                    bestK = rowK[i];
                }
            }
            if (best < 0) {
                break;
            }
            rowK[best]--;
            sumK--;
        }
        uint32_t usedCoreNumLocal = sumK;

        // LD slot 分配：被切分行（k_R ≥ 2）按行序占连续 slot；rowSlot[i] = 行 i 的首个 slot
        uint32_t rowSlot[AIC_CORE_NUM];
        uint32_t fdCount = 0;
        uint32_t fdBN2[AIC_CORE_NUM], fdM[AIC_CORE_NUM], fdW[AIC_CORE_NUM];
        uint32_t fdSplitNum[AIC_CORE_NUM], fdMSize[AIC_CORE_NUM];
        uint32_t slotBase = 0;
        for (uint32_t i = 0; i < rowsCount; i++) {
            if (rowK[i] >= 2) {
                fdBN2[fdCount] = rowB[i];
                fdM[fdCount] = rowG[i];
                fdW[fdCount] = slotBase;
                fdSplitNum[fdCount] = rowK[i];
                fdMSize[fdCount] = rowTailM[i];
                rowSlot[i] = slotBase;
                slotBase += rowK[i];
                fdCount++;
            } else {
                rowSlot[i] = 0;
            }
        }

        // 本核（AIC 索引 cubeCoreIdx）对应的行内片段 (R, p)
        uint32_t R = rowsCount;
        uint32_t p = 0;
        if (cubeCoreIdx < usedCoreNumLocal) {
            uint32_t cum = 0;
            for (uint32_t i = 0; i < rowsCount; i++) {
                if (cubeCoreIdx < cum + rowK[i]) {
                    R = i;
                    p = cubeCoreIdx - cum;
                    break;
                }
                cum += rowK[i];
            }
        }
        if (cubeCoreIdx >= usedCoreNumLocal) {
            splitCoreInfo.isCoreEnable = false;
            if ASCEND_IS_AIV {
                ldInfo.isLdCoreEnable = false;
            }
            return;
        }
        splitCoreInfo.isCoreEnable = true;
        splitCoreInfo.bN2Start = splitCoreInfo.bN2End = rowB[R];
        splitCoreInfo.gS1Start = splitCoreInfo.gS1End = rowG[R];
        if (rowK[R] >= 2) {
            splitCoreInfo.s2Start = (p * rowW[R]) / rowK[R];
            splitCoreInfo.s2End = ((p + 1) * rowW[R]) / rowK[R] - 1;
            // 该片段写入的 LD slot（写侧 offset = slot * s1BaseSize*T + rowIdx*T）
            ldInfo.saveWorkSpaceIdx = rowSlot[R] + p;
        } else {
            splitCoreInfo.s2Start = 0;
            splitCoreInfo.s2End = rowW[R] - 1;
            ldInfo.saveWorkSpaceIdx = 0;
        }

        if ASCEND_IS_AIV {
            // ===== SplitFD：把 fd 归约任务负载均衡分给 48 个 AIV =====
            if (fdCount == 0) {
                ldInfo.isLdCoreEnable = false;
                return;
            }
            uint64_t totalFDLoad = 0;
            for (uint32_t i = 0; i < fdCount; i++) {
                totalFDLoad += (uint64_t)fdSplitNum[i] * fdMSize[i];
            }
            uint64_t averageLoad = (totalFDLoad + AIV_CORE_NUM - 1) / AIV_CORE_NUM;
            uint32_t fdIdx[AIV_CORE_NUM], fdMStart[AIV_CORE_NUM], fdMNum[AIV_CORE_NUM];
            uint32_t curCoreIndex = 0;
            for (uint32_t i = 0; i < fdCount; i++) {
                uint32_t curFDVectorNum = (uint32_t)((uint64_t)fdSplitNum[i] * fdMSize[i] / averageLoad);
                curFDVectorNum = Max(1U, curFDVectorNum);
                uint32_t curAveMSize = (fdMSize[i] + curFDVectorNum - 1) / curFDVectorNum;
                curFDVectorNum = (fdMSize[i] + curAveMSize - 1) / curAveMSize;
                for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
                    if (curCoreIndex >= AIV_CORE_NUM) {
                        break;
                    }
                    fdIdx[curCoreIndex] = i;
                    fdMStart[curCoreIndex] = vid * curAveMSize;
                    fdMNum[curCoreIndex] =
                        (vid < curFDVectorNum - 1) ? curAveMSize : fdMSize[i] - vid * curAveMSize;
                    curCoreIndex++;
                }
            }
            if (vecCoreIdx >= curCoreIndex) {
                ldInfo.isLdCoreEnable = false;
                return;
            }
            uint32_t fi = fdIdx[vecCoreIdx];
            ldInfo.isLdCoreEnable = true;
            ldInfo.bn2Idx = fdBN2[fi];
            ldInfo.bIdx = ldInfo.bn2Idx / constInfo.kHeadNum;
            ldInfo.n2Idx = ldInfo.bn2Idx % constInfo.kHeadNum;
            ldInfo.mIdx = fdM[fi];
            ldInfo.workspaceIdx = fdW[fi];
            ldInfo.workspaceNum = fdSplitNum[fi];
            ldInfo.mStart = fdMStart[vecCoreIdx];
            ldInfo.mNum = fdMNum[vecCoreIdx];
            uint64_t actualSeqQPrefixSum = 0;
            if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
                uint32_t actualSeqLengthsGmQIdx = (constInfo.batchSupperFlag) ? ldInfo.bIdx : ldInfo.bIdx - 1;
                actualSeqQPrefixSum = (ldInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(actualSeqLengthsGmQIdx);
            } else {  // BSND
                actualSeqQPrefixSum = (ldInfo.bIdx <= 0) ? 0 : (uint64_t)ldInfo.bIdx * constInfo.qSeqSize;
            }
            ldInfo.indiceOutCoreOffset = actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount +
                                         ldInfo.n2Idx * constInfo.sparseCount +
                                         ldInfo.mIdx * constInfo.s1BaseSize * constInfo.kHeadNum * constInfo.sparseCount;
        }
        return;
    }

    // ===== Branch 2 = 官方整行逻辑：base size 对齐官方 arch35 =====
    // 官方：s1BaseSize = S1_BASE_SIZE(4)，tSize≤64 减半 → decode 各 batch s1BaseSize=2、
    // mBaseSize=s1BaseSize×gSize（heads=32→64、heads=64→128）。host workspace 仍按
    // s1BaseSizeWs（旧值）分配，计算值只影响 score/UB 布局（qkVLstride/dstNdStride 均
    // ∝ mBaseSize 同式缩放、resMm1Buf_ 随之缩小），每核 GM segment 步长由 s1BaseSizeWs 钉死。
    constInfo.s1BaseSize = (constInfo.batchSize <= 64) ? (S1_BASE_SIZE / 2) : S1_BASE_SIZE;
    constInfo.mBaseSize = constInfo.s1BaseSize * (uint32_t)constInfo.gSize;
    constInfo.isWholeRowGreedy = true;
    // mBaseSize 变化会改 gS1Num（行结构），按新 base size 重算 rowsCount/totalBlocks，
    // 贪心分组必须按新行结构枚举（decode 各请求 actS1Size=1 → gS1Num 恒为 1，不变）。
    rowsCount = 0;
    totalBlocks = 0;
    for (uint32_t bN2 = 0; bN2 < totalBN2; bN2++) {
        uint32_t bIdx = bN2 / constInfo.kHeadNum;
        uint32_t actS1Size, actS2Size, actS2SizeOrig;
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        if (actS1Size == 0 || actS2Size == 0) {
            continue;
        }
        uint32_t gS1Num = (actS1Size * (uint32_t)constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
        for (uint32_t gS1 = 0; gS1 < gS1Num; gS1++) {
            uint32_t w;
            if (constInfo.attenMaskFlag) {
                w = GetS2BaseBlockNumOnMask(gS1, actS1Size, actS2SizeOrig);
            } else {
                w = (actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
            }
            rowsCount++;
            totalBlocks += w;
        }
    }

    // ===== Whole-row grouping：行数 > 12，整行贪心分组（无跨核行 → 无 LD）=====
    // 每核目标 = 剩余块数 / 剩余核数（动态），maxTake 保证每核至少给后续核留 1 行，
    // 最后一个核（coresLeft==1）maxTake==全部剩余行 —— 从结构上杜绝行被悬空。
    uint32_t bN2 = 0;
    uint32_t gS1 = 0;
    InitRowCursor(bN2, gS1);
    uint32_t rowIdx = 0;
    uint32_t blocksLeft = totalBlocks;
    for (uint32_t curCore = 0; curCore < AIC_CORE_NUM && rowIdx < rowsCount; curCore++) {
        uint32_t coresLeft = AIC_CORE_NUM - curCore;                 // 含当前核
        uint32_t target = (blocksLeft + coresLeft - 1) / coresLeft;  // 本核目标块数
        // 至少给后续每核留 1 行；行数 < 剩余核数时无"留行"约束，取全部剩余（weight 上限自会兜底）
        uint32_t rowsRemaining = rowsCount - rowIdx;
        uint32_t maxTake = (rowsRemaining >= coresLeft) ? (rowsRemaining - (coresLeft - 1)) : rowsRemaining;
        uint32_t chunkStartB = bN2;
        uint32_t chunkStartG = gS1;
        uint32_t acc = 0;
        uint32_t took = 0;
        uint32_t lastB = bN2;
        uint32_t lastG = gS1;
        uint32_t lastW = 0;
        while (took < maxTake) {
            uint32_t w, tailM;
            RowCursorInfo(bN2, gS1, w, tailM);
            if (took > 0 && acc + w > target) {
                break;
            }
            acc += w;
            took++;
            lastB = bN2;
            lastG = gS1;
            lastW = w;
            rowIdx++;
            if (!AdvanceRowCursor(bN2, gS1)) {
                break;
            }
        }
        blocksLeft -= acc;
        if (curCore == cubeCoreIdx) {
            splitCoreInfo.isCoreEnable = true;
            splitCoreInfo.bN2Start = chunkStartB;
            splitCoreInfo.gS1Start = chunkStartG;
            splitCoreInfo.s2Start = 0;
            splitCoreInfo.bN2End = lastB;
            splitCoreInfo.gS1End = lastG;
            splitCoreInfo.s2End = lastW - 1;
            ldInfo.saveWorkSpaceIdx = 0;
            if ASCEND_IS_AIV {
                ldInfo.isLdCoreEnable = false;
            }
            return;
        }
    }
    splitCoreInfo.isCoreEnable = false;
    if ASCEND_IS_AIV {
        ldInfo.isLdCoreEnable = false;
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        if (constInfo.outputLayout == LI_LAYOUT::TND) {
            uint32_t tSizeIdx = (constInfo.batchSupperFlag) ? constInfo.batchSize : constInfo.batchSize - 1;
            uint32_t tBaseIdx = (constInfo.batchSupperFlag) ? bIdx : bIdx - 1;
            uint32_t tSize = actualSeqLengthsGmQ.GetValue(tSizeIdx);
            uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(tBaseIdx);
            uint32_t s1Count = tempLoopInfo.actS1Size;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                uint64_t indiceOutOffset =
                    (tBase + s1Idx) * constInfo.kHeadNum * constInfo.sparseCount +  // T轴、s1轴偏移
                    n2Idx * constInfo.sparseCount;                                  // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        } else if (constInfo.outputLayout == LI_LAYOUT::BSND) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset = bIdx * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount +
                                           s1Idx * constInfo.kHeadNum * constInfo.sparseCount +  // B轴、S1轴偏移
                                           n2Idx * constInfo.sparseCount;                        // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                              __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale,
                                              __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK,
                                              __gm__ uint8_t *blockTable,
                                              __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                              const QLITilingData *__restrict tiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx();  // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx();  // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengthsK);

    // 获取分核信息（on-device，无 metadata 输入）
    ComputeSplitInfo(aiCoreIdx, tmpBlockIdx);

    pipe = tPipe;

    uint64_t offset = 0;
    uint32_t topkCountAlign16_ = QLICommon::Align(constInfo.sparseCount, (uint64_t)16); // topkCount对齐到16
    // vec 把整个s2的score存储在GM，大小为s1BaseSize * 16K * 4
    GlobalTensor<SCORE_T> scoreGm; // 存放vec核写出的score
    // 每核 GM segment 步长必须与 host workspace 分配口径一致（s1BaseSizeWs = host 分配的
    // s1BaseSize = ceil(256/gSize)），不能用 Branch 2 对齐后的计算值，否则与 host 布局错位。
    uint64_t singleCoreScoreSize = constInfo.s1BaseSizeWs * QLICommon::Align((uint64_t)constInfo.kSeqSize, (uint64_t)constInfo.s2BaseSize)  * sizeof(SCORE_T);
    scoreGm.SetGlobalBuffer((__gm__ SCORE_T *)(workspace + aiCoreIdx * singleCoreScoreSize));
    offset += GetBlockNum() * singleCoreScoreSize;
    // vec 存储需要LD的s1对应的s2的score与index，大小为s1BaseSize * sparseCount * 2，一个核内最多有两个s1BaseSize需要LD
    GlobalTensor<SCORE_T> ldScoreGm; // 存放进行LD的s2 score
    ldScoreGm.SetGlobalBuffer((__gm__ SCORE_T *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSizeWs * topkCountAlign16_ * 2 * sizeof(SCORE_T);
    GlobalTensor<int32_t> ldIndexGm; // 存放进行LD的s2 Index
    ldIndexGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSizeWs * topkCountAlign16_ * 2 * sizeof(int32_t);

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, ldInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        weightsGm.SetGlobalBuffer((__gm__ float *)weights);
        qScaleGm.SetGlobalBuffer((__gm__ float *)queryScale);
        kScaleGm.SetGlobalBuffer((__gm__ float *)keyScale);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        vectorService.InitVecInputTensor(weightsGm, qScaleGm, kScaleGm, indiceOutGm, blockTableGm);
        vectorService.InitVecWorkspaceTensor(scoreGm, ldScoreGm, ldIndexGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm);
    }
    InitBuffers();
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    bool isEnd = (bN2LoopIdx == splitCoreInfo.bN2End) && (gS1LoopIdx == splitCoreInfo.gS1End);
    uint32_t s2BlockNum;
    if (constInfo.attenMaskFlag) {
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2SizeOrig);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    tempLoopInfo.s2LoopEnd = isEnd ? splitCoreInfo.s2End : s2BlockNum - 1;
    if (splitCoreInfo.s2Start > 0 || tempLoopInfo.s2LoopEnd < s2BlockNum - 1) {
        tempLoopInfo.isNeedLD = true;
    } else {
        tempLoopInfo.isNeedLD = false;
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size, tempLoopInfo.actS2SizeOrig);
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.actS2Size % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd = (bN2LoopIdx == splitCoreInfo.bN2End) ? splitCoreInfo.gS1End : gS1SplitNum - 1;
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum - 1 && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, QLICommon::RunInfo &runInfo,
                                                     uint32_t qScaleLoop, uint32_t kScaleLoop)
{
    runInfo.loop = loop;
    runInfo.qScaleLoop = qScaleLoop;
    runInfo.kScaleLoop = kScaleLoop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.isValid = s2LoopIdx <= tempLoopInfo.s2LoopEnd;
    runInfo.isNeedLD = tempLoopInfo.isNeedLD;
    if (runInfo.isNeedLD && s2LoopIdx == tempLoopInfo.s2LoopEnd) {
        runInfo.saveWorkSpaceIdx = ldInfo.saveWorkSpaceIdx;
        ldInfo.saveWorkSpaceIdx++;
    }

    if (!runInfo.isValid) {
        return;  // 需要验证， v1 时候需要runInfo
    }

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.actS2SizeOrig = tempLoopInfo.actS2SizeOrig;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        QLICommon::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, QLICommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
            uint32_t actualSeqLengthsGmQIdx = (constInfo.batchSupperFlag) ? runInfo.bIdx : runInfo.bIdx - 1;
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(actualSeqLengthsGmQIdx);
        } else {  // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        // B,S1,N2,k/T,N2,k
        indiceOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount + runInfo.n2Idx * constInfo.sparseCount;
    }
    uint64_t actualSeqKPrefixSum;
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) { // T N2 D
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx - 1);
        actualSeqKPrefixSum = actualSeqKPrefixSum / constInfo.cmpRatio;
    } else {
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
    }
    uint64_t tndBIdxOffsetForK = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
    keyCoreOffset = tndBIdxOffsetForK + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum * constInfo.headDim;
    keyScaleCoreOffset = (actualSeqKPrefixSum + runInfo.s2Idx * constInfo.s2BaseSize) * constInfo.kHeadNum;
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset = keyCoreOffset;
    runInfo.tensorKeyScaleOffset = keyScaleCoreOffset;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::Process()
{
    if (isUsedCoreEqZero) {
        // 没有计算任务，直接清理输出
        ProcessInvalid();
        return;
    }

    ProcessMain();

    ProcessDecode();
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2;  // 2 means c:v = 1:2
        uint64_t totalOutputSize =
            constInfo.batchSize * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount;
        uint64_t singleCoreSize =
            QLICommon::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::ProcessMain()
{
    if(!splitCoreInfo.isCoreEnable){
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + 1);
    } else {
        matmulService.AllocEventID();
    }

    QLICommon::RunInfo runInfo;
    uint32_t gloop = 0;
    uint32_t qScaleLoop = 0;  // 每行 +1：weight/qScale 双缓冲槽位（官方同式）
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);
            continue;
        }
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            runInfo.s2Start = splitCoreInfo.s2Start;
            runInfo.s2LoopEnd = tempLoopInfo.s2LoopEnd;
            uint32_t kScaleLoop = 0;  // 每 16 个 s2 块 +1：kScale 双缓冲槽位（官方同式）
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                if ((s2LoopIdx - (int)splitCoreInfo.s2Start) % 16 == 0) {
                    ++kScaleLoop;
                }
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo, qScaleLoop, kScaleLoop);
                ++gloop;
            }
            ++qScaleLoop;
            splitCoreInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        splitCoreInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLICommon::ConstInfo::CROSS_VC_EVENT + 0); 
        CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLICommon::ConstInfo::CROSS_VC_EVENT + 1); 
    }
}

template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                                          QLICommon::RunInfo runInfo, uint32_t qScaleLoop,
                                                          uint32_t kScaleLoop)
{
    CalcRunInfo(loop, s2LoopIdx, runInfo, qScaleLoop, kScaleLoop);
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(runInfo);
    } else {
        vectorService.ProcessVec1(runInfo);
        if (runInfo.isLastS2InnerLoop) {   //本核s2last
            vectorService.ProcessTopK(runInfo);   
        }
    }
}

 template <typename QLIT>
__aicore__ inline void QLIPreload<QLIT>::ProcessDecode()
{
    if ASCEND_IS_AIV {
        vectorService.InitLDBuffers(pipe, ldInfo);
        ICachePreLoad(LD_PREFETCH_LEN);
        SyncAll();
        if (ldInfo.isLdCoreEnable) {
            vectorService.ProcessLD();
        }
    }
}

}  // namespace QLIKernel
#endif  // quant_lightning_indexer_KERNEL_H