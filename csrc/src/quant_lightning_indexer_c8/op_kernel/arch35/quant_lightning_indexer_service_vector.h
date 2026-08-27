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
 * \file quant_lightning_indexer_service_vector.h
 * \brief
 */
#ifndef quant_lightning_indexer_SERVICE_VECTOR_H
#define quant_lightning_indexer_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_common.h"
#include "../arch35/vf/quant_lightning_indexer_vector1.h"
#include "../arch35/vf/quant_lightning_indexer_topk.h"

namespace QLIKernel {
using namespace QLICommon;
constexpr uint32_t TRUNK_LEN_16K = 16384;
template <typename QLIT>
class QLIVector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = QLIT::pageAttention;

    using QK_T = typename QLIT::queryKeyType;
    using SCORE_T = typename QLIT::scoreType;

    __aicore__ inline QLIVector(){};
    __aicore__ inline void ProcessVec1(const QLICommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const QLICommon::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct QLICommon::ConstInfo &constInfo,
                                      const struct QLICommon::LdSplitCoreInfo &ldInfo,
                                      const QLITilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm,
                                                    GlobalTensor<SCORE_T> ldScoreGm,
                                                    GlobalTensor<int32_t> ldIndexGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<float> weightsGm, GlobalTensor<float> qScaleGm,
                                              GlobalTensor<float> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe, const struct QLICommon::LdSplitCoreInfo &ldInfo);

protected:
    GlobalTensor<SCORE_T> scoreGm;
    GlobalTensor<float> weightsGm;
    GlobalTensor<SCORE_T> ldScoreGm;
    GlobalTensor<int32_t> ldIndexGm;
    GlobalTensor<float> qScaleGm;
    GlobalTensor<float> kScaleGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;
    // =================================常量区=================================
    // 官方 arch35 事件分派：weight/qScale（QSCALE）与 kScale（KSCALE）各用独立事件对，
    // 与 score 拷贝（VEC1_MTE3_V_EVENT+pingpong）解耦。跨 HardEvent 类的 ID 复用同官方。
    static constexpr uint32_t VEC1_V_MTE2_EVENT_KSCALE = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT_KSCALE = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;
    static constexpr uint32_t VEC1_V_MTE2_EVENT_QSCALE = EVENT_ID6;
    static constexpr uint32_t VEC1_MTE2_V_EVENT_QSCALE = EVENT_ID3;

    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t KSCALE_S_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;

private:
    __aicore__ inline void GetKeyScale(const QLICommon::RunInfo &runInfo, LocalTensor<float> &kScaleUB,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<QK_T> resMm1UB_;
    //tmp buff for weight
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<float> weightUB_;
    //tmp buff for kScale
    TBuf<TPosition::VECCALC> kScaleBuf_;
    LocalTensor<float> kScaleUB_;
    //tmp buff for qScale
    TBuf<TPosition::VECCALC> qScaleBuf_;
    LocalTensor<float> qScaleUB_;
    //tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;
    // tmp buff for LD
    TBuf<TPosition::VECCALC> ldValueBuf_;
    LocalTensor<uint32_t> ldValueLocal_;  // SCORE_T

    TBuf<TPosition::VECCALC> topkIndexBuf_;
    LocalTensor<uint32_t> topkIndexLocal_;

    TBuf<TPosition::VECCALC> topkValueBuf_;
    LocalTensor<uint32_t> topkValueLocal_;

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;

    TBuf<TPosition::VECCALC> outInvalidBuf_;
    LocalTensor<int32_t> outInvalidLocal_;

    LocalTensor<uint32_t> ldIndexLocal_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;
    uint32_t topkCount_ = 0;
    uint32_t topkCountAlign256_ = 0; // topkCount对齐到256(直方图需要)，支持topk泛化
    uint32_t topkCountAlign16_ = 0; // LD读取到UB，需要满足32B对齐
    uint32_t trunkLen_ = 0;
    bool isWholeRowGreedy_ = false; // Branch 2 走官方 ProcessVec1（load-once + kScale 16块批量）

    struct QLICommon::ConstInfo constInfo_;
    struct QLICommon::LdSplitCoreInfo ldInfo_;
    topk::LITopk<SCORE_T> topkOp_;
};

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(QK_T));   //大小：2(开dB) * 2 * 64 * 128 * 4 = 128KB
    resMm1UB_ = resMm1Buf_.Get<QK_T>();//qk
    // pingpong load 偏移固定 UB_BANK_STRIDE/4=256B，缓冲须≥256B+块内数据；2×CeilDiv×gSize×4 在 s1BaseSize=2 仅 256B 会写穿 kScale/out
    pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightUB_ = weightBuf_.Get<float>();//weight
    // 官方整行路径 kScale 每 16 块批量加载需 2*16*s2BaseSize 槽位（16KB）；Branch 1 逐块加载 1KB 即可（UB 预算不放大）
    pipe->InitBuffer(kScaleBuf_, isWholeRowGreedy_ ? (2 * s2BaseSize_ * 16 * sizeof(float)) : (2 * s2BaseSize_ * sizeof(float)));
    kScaleUB_ = kScaleBuf_.Get<float>();//kScale
    pipe->InitBuffer(qScaleBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    qScaleUB_ = qScaleBuf_.Get<float>();//qScale
    pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T));      // 大小：2(开dB) * 2 * 128 * 4 = 2KB
    vec1OutUB_ = outBuf_.Get<SCORE_T>();//out

    // Topk
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T));     // 大小：(topkCountAlign256_ + 每次排序长度) * sizeof(SCORE_T)
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    
    pipe->InitBuffer(indicesOutBuf_, (topkCountAlign256_ + 64) * sizeof(uint32_t));         // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    pipe->InitBuffer(scoreOutBuf_, (topkCountAlign256_ + 64) * sizeof(SCORE_T));            // (topkCountAlign256_ + 64) * sizeof(SCORE_T) 64:duplicate刷-1额外空间
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_);

    //刷-1
    pipe->InitBuffer(outInvalidBuf_, topkCount_ * sizeof(int32_t));
    outInvalidLocal_ = outInvalidBuf_.Get<int32_t>();
    Duplicate(kScaleUB_, float(0), isWholeRowGreedy_ ? (2 * s2BaseSize_ * 16) : (2 * s2BaseSize_));
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitLDBuffers(TPipe *pipe, const struct QLICommon::LdSplitCoreInfo &ldInfo)
{
    ldIndexLocal_ = resMm1UB_.template ReinterpretCast<uint32_t>();
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitParams(const struct QLICommon::ConstInfo &constInfo,
                                                   const struct QLICommon::LdSplitCoreInfo &ldInfo,
                                                   const QLITilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    this->ldInfo_ = ldInfo;
    blockS2StartIdx_ = 0;
    isWholeRowGreedy_ = constInfo.isWholeRowGreedy;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;  // 4
    s2BaseSize_ = constInfo.s2BaseSize;  // 128
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    blockId_ = GetBlockIdx();
    trunkLen_ = TRUNK_LEN_16K;
    topkCount_ = constInfo.sparseCount;
    topkCountAlign256_ = QLICommon::Align(constInfo.sparseCount, (uint64_t)256); // topkCount对齐到256
    topkCountAlign16_ = QLICommon::Align(constInfo.sparseCount, (uint64_t)16); // topkCount对齐到16
    topkOp_.Init(topkCount_, trunkLen_);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecInputTensor(GlobalTensor<float> weightsGm, GlobalTensor<float> qScaleGm,
                                                           GlobalTensor<float> kScaleGm,
                                                           GlobalTensor<int32_t> indiceOutGm,
                                                           GlobalTensor<int32_t> blockTableGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm,
                                                            GlobalTensor<SCORE_T> ldScoreGm,
                                                            GlobalTensor<int32_t> ldIndexGm)
{
    this->scoreGm = scoreGm;//resucesum*k
    this->ldScoreGm = ldScoreGm;
    this->ldIndexGm = ldIndexGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
    // QSCALE 事件仅 Branch 2（整行 load-once）使用。Branch 1 若预置则 QSCALE+1=EVENT_ID7
    // 会把陈旧置位泄漏进 ProcessTopK 的 V_MTE2_EVENT 自屏障（Set/Wait 立即通过，破坏 V→MTE2 排序）。
    if (isWholeRowGreedy_) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 1);
    }

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
    if (isWholeRowGreedy_) {
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 1);
    }

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // init -1 and copy to output
    Duplicate(outInvalidLocal_, constInfo_.INVALID_IDX, constInfo_.sparseCount);

    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

    AscendC::DataCopyParams dataCopyOutParams;
    dataCopyOutParams.blockCount = 1;
    dataCopyOutParams.blockLen = constInfo_.sparseCount * sizeof(int32_t);
    dataCopyOutParams.srcStride = 0;
    dataCopyOutParams.dstStride = 0;
    AscendC::DataCopyPad(indiceOutGm[invalidS1Offset], outInvalidLocal_, dataCopyOutParams);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::GetKeyScale(const QLICommon::RunInfo &runInfo, LocalTensor<float> &kScaleUB,
                                                    int64_t batchId, int64_t startS2, int64_t getLen)
{
    // startS2一定能整除kCacheBlockSize_
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    // 官方整行路径按 kScaleLoop（每16块+1）选双缓冲槽（每槽16*s2BaseSize 元素）；
    // Branch 1 逐块加载按 loop%2 选 1 块槽。
    int32_t kScaleDstBase = isWholeRowGreedy_
        ? (16 * (int32_t)(runInfo.kScaleLoop % 2) * s2BaseSize_)
        : ((int32_t)(runInfo.loop % 2) * s2BaseSize_);
    if constexpr (PAGE_ATTENTION) {
        int32_t startBlockTableIdx = startS2 / kCacheBlockSize_;
        int32_t startBlockTableOffset = startS2 % kCacheBlockSize_;
        int32_t blockTableBatchOffset = batchId * maxBlockNumPerBatch_;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        int32_t resUbBaseOffset = 0;
        if (startBlockTableOffset > 0) {
            int32_t firstPartLen =
                kCacheBlockSize_ - startBlockTableOffset > getLen ? getLen : kCacheBlockSize_ - startBlockTableOffset;
            copyInParams.blockLen = firstPartLen * sizeof(float);
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[kScaleDstBase], kScaleGm[blockId * constInfo_.keyDequantScaleStride0 + startBlockTableOffset],
                                 copyInParams, padParams);
            startBlockTableIdx++;
            getLen = getLen - firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        copyInParams.blockLen = kCacheBlockSize_ * sizeof(float);
        for (int32_t i = 0; i < getLoopNum; i++) {
            if (i == getLoopNum - 1) {
                copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(float);
            }
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[kScaleDstBase + resUbBaseOffset + i * kCacheBlockSize_],
                                 kScaleGm[blockId * constInfo_.keyDequantScaleStride0],
                                 copyInParams, padParams);
        }
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(kScaleUB[kScaleDstBase], kScaleGm[runInfo.tensorKeyScaleOffset], copyInParams, padParams);
    }
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessVec1(const QLICommon::RunInfo &info)
{
    auto pingpong = (info.loop % 2);
    auto qScalepingpong = (info.qScaleLoop % 2);  // 每行+1：weight/qScale 槽位（官方同式）
    auto kScalepingpong = (info.kScaleLoop % 2);  // 每16块+1：kScale 槽位（官方同式）
    auto s1BaseSizePerAIV = CeilDiv(s1BaseSize_, 2);
    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;	 
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;	 
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2; 

    if (curAivS1ProcNum == 0) {
        CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_CV_EVENT + pingpong);  // V核等C核计算完mm1，mm1Res已搬运到UB
        CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + pingpong);   // V核处理完，通知C核可以把mm1Res搬运到UB
        return;
    }
    int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * kHeadNum_ * gSize_;
    static_assert(std::is_same_v<SCORE_T, uint16_t>);
    auto outBase = vec1OutUB_[pingpong * (UB_BANK_STRIDE / sizeof(SCORE_T))];
    if (isWholeRowGreedy_) {
        // ===== 官方整行路径：weight/qScale 每行仅首块加载、kScale 每16块批量加载 =====
        if (info.isFirstS2InnerLoop) {
            WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qScalepingpong);
            //weightsGm --> weightUB_
            DataCopyPadExtParams<float> padWeightsParams{false, 0, 0, 0};
            DataCopyExtParams qwDataCopyExtParams;
            qwDataCopyExtParams.blockCount = curAivS1ProcNum;
            qwDataCopyExtParams.blockLen = gSize_ * sizeof(float);
            qwDataCopyExtParams.srcStride = 0;
            qwDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - UB_BANK_STRIDE) / 32;
            DataCopyPad(weightUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(float))],
                        weightsGm[weightGmOffset], qwDataCopyExtParams, padWeightsParams);
            //qScaleGm --> qScaleUB_
            DataCopyPadExtParams<float> padQScaleParams{false, 0, 0, 0};
            DataCopyPad(qScaleUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(float))],
                        qScaleGm[weightGmOffset], qwDataCopyExtParams, padQScaleParams);
            SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + qScalepingpong);
            WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + qScalepingpong);
        }
        if ((info.s2Idx - info.s2Start) % 16 == 0) {
            WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
            //kScaleGm --> kScaleUB_：一次取 16 块（尾组截断）
            uint32_t remainLen = (info.actS2Size > info.s2Idx * (uint32_t)s2BaseSize_)
                                     ? (uint32_t)(info.actS2Size - info.s2Idx * (uint32_t)s2BaseSize_)
                                     : 0;
            uint32_t getLen = (16 * (uint32_t)s2BaseSize_ > remainLen) ? remainLen : 16 * (uint32_t)s2BaseSize_;
            GetKeyScale(info, kScaleUB_, info.bIdx, curS2Idx, getLen);
            SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
            WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
        }
        WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);

        //CV同步
        CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_CV_EVENT + info.loop % 2);   //V核等C核计算完mm1，mm1Res已搬运到UB

        auto weightBase = weightUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(float))];
        auto qScaleBase = qScaleUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(float))];
        auto kScaleBase = kScaleUB_[kScalepingpong * 16 * s2BaseSize_ +
                                    ((info.s2Idx - info.s2Start) % 16) * s2BaseSize_];

        auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(QK_T))];
        auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(QK_T)) / 2 * constInfo_.mBaseSize;
        vector1::BatchMulWeightAndReduceSum(outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T),
                                            qkBase, qkVLstride, (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(QK_T)),
                                            weightBase, UB_BANK_DEPTH_STRIDE / sizeof(float),
                                            kScaleBase, (uint32_t)0,
                                            qScaleBase, UB_BANK_DEPTH_STRIDE / sizeof(float),
                                            gSize_, curAivS1ProcNum);
        if (info.isFirstS2InnerLoop) {
            SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qScalepingpong);
        }
        if ((info.s2Idx - info.s2Start) % 16 == 0) {
            SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
        }
    } else {
        // ===== 小 batch 拆 S2 + LD 路径：逐块加载 weight/qScale/kScale（保持现状）=====
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + pingpong);
        //weightsGm --> weightUB_
        DataCopyPadExtParams<float> padWeightsParams{false, 0, 0, 0};
        DataCopyExtParams qwDataCopyExtParams;
        qwDataCopyExtParams.blockCount = curAivS1ProcNum;
        qwDataCopyExtParams.blockLen = gSize_ * sizeof(float);
        qwDataCopyExtParams.srcStride = 0;
        qwDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - UB_BANK_STRIDE) / 32;
        DataCopyPad(weightUB_[pingpong * (UB_BANK_STRIDE / sizeof(float))],
                    weightsGm[weightGmOffset], qwDataCopyExtParams, padWeightsParams);
        //qScaleGm --> qScaleUB_
        DataCopyPadExtParams<float> padQScaleParams{false, 0, 0, 0};
        DataCopyPad(qScaleUB_[pingpong * (UB_BANK_STRIDE / sizeof(float))],
                    qScaleGm[weightGmOffset], qwDataCopyExtParams, padQScaleParams);
        //kScaleGm --> kScaleUB_
        GetKeyScale(info, kScaleUB_, info.bIdx, curS2Idx, info.actualSingleProcessSInnerSize);
        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + pingpong);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + pingpong);
        WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);

        //CV同步
        CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_CV_EVENT + info.loop % 2);   //V核等C核计算完mm1，mm1Res已搬运到UB

        auto weightBase = weightUB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
        auto qScaleBase = qScaleUB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
        auto kScaleBase = kScaleUB_[pingpong * s2BaseSize_];

        auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(QK_T))];
        auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(QK_T)) / 2 * constInfo_.mBaseSize;
        vector1::BatchMulWeightAndReduceSum(outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T),
                                            qkBase, qkVLstride, (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(QK_T)),
                                            weightBase, UB_BANK_DEPTH_STRIDE / sizeof(float),
                                            kScaleBase, (uint32_t)0,
                                            qScaleBase, UB_BANK_DEPTH_STRIDE / sizeof(float),
                                            gSize_, curAivS1ProcNum);
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + pingpong);
    }
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    //outUB_ --->  scoreGm
    int64_t vec1OutGmOffset = blockId_ % 2 == 0 ? curS2Idx : 
                            s1BaseSizePerAIV * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + curS2Idx;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(SCORE_T);
    copyOutParams.srcStride = (UB_BANK_DEPTH_STRIDE - UB_BANK_STRIDE) / 32;
    copyOutParams.dstStride = (QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) - s2BaseSize_) * sizeof(SCORE_T);
    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, copyOutParams);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + pingpong);   //V核处理完，通知C核可以把mm1Res搬运到UB
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessTopK(const QLICommon::RunInfo &info)
{
    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);

    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;	 
    int64_t curS2StartIdx = info.s2Start * s2BaseSize_;
    int64_t curS2EndIdx = (info.s2LoopEnd + 1) * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;	 
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2; 

    // LD 需要搬运到的起始位置 32字节
    int64_t ldDstOffset = (info.isNeedLD) ? curS2StartIdx : 0;
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    AscendC::DataCopyParams ldCopyOutParams; // ldIndicesCopyOutParams
    ldCopyOutParams.blockCount = 1;
    ldCopyOutParams.blockLen = topkCountAlign16_ * sizeof(uint32_t); // bytes
    ldCopyOutParams.srcStride = 0;
    ldCopyOutParams.dstStride = 0;

    AscendC::DataCopyParams ldCopyScoreOutParams;
    ldCopyScoreOutParams.blockCount = 1;
    ldCopyScoreOutParams.blockLen = topkCountAlign16_ * sizeof(SCORE_T); // bytes
    ldCopyScoreOutParams.srcStride = 0;
    ldCopyScoreOutParams.dstStride = 0;

    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    } 

    int32_t validAllS2Len = cuRealAcSeq;
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;
        uint32_t vecOffset = blockId_ % 2 * CeilDiv(s1BaseSize_, 2) + i;
        
        SCORE_T zero = 0;
        int32_t neg = -1;
        uint32_t scoreAlign = sizeof(SCORE_T) == 4 ? 8 : 16; // int32对应UB对齐数为8，int16需要16
        if (constInfo_.attenMaskFlag) {
            validAllS2Len = ((int32_t)i + cuRealAcSeq) / static_cast<int32_t>(constInfo_.cmpRatio);
        }
        int32_t validS2Len = validAllS2Len;
        if (info.isNeedLD) {
            // 当前核处理的s2长度validS2Len
            validS2Len = Min((info.s2LoopEnd + 1) * s2BaseSize_, validAllS2Len) - curS2StartIdx;
        }
        
        uint64_t offset = info.saveWorkSpaceIdx * s1BaseSize_ * topkCountAlign16_ + rowIdx * topkCountAlign16_;
        if (validS2Len <= 0 && !info.isNeedLD) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCount_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            continue;
        } else if (validS2Len <= 0 && info.isNeedLD) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCountAlign16_);
            Duplicate(scoreOutLocal_, zero, topkCountAlign16_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            // 将每一行S1的Topk结果存放在ldScoreGm和ldIndexGm中
            AscendC::DataCopyPad(ldScoreGm[offset], scoreOutLocal_, ldCopyScoreOutParams);
            AscendC::DataCopyPad(ldIndexGm[offset], indicesOutLocal_.ReinterpretCast<int32_t>(), ldCopyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
        if (validS2Len >= topkCount_) {
            uint32_t s2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
            if (s2LoopNum == 1) {
                uint32_t validS2LenAlign = QLICommon::Align(validS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[validS2Len / 256 * 256], zero, validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                copyInParams.blockLen = validS2Len * sizeof(SCORE_T); // byte
                AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(mrgValueLocal_, scoreGm[vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + ldDstOffset], copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                topkOp_.TopK(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, validS2LenAlign, 0, 1, info.isNeedLD);
            } else {
                for (uint32_t loopIdx = 0; loopIdx < s2LoopNum; loopIdx++) {
                    if (loopIdx == 0) {
                        copyInParams.blockLen = trunkLen_ * sizeof(SCORE_T); // byte
                        AscendC::DataCopyPad(mrgValueLocal_, scoreGm[vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + ldDstOffset], copyInParams, padParams);
                        SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        topkOp_.TopK(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, trunkLen_, loopIdx, s2LoopNum, info.isNeedLD);
                        continue;
                    }
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    uint32_t validTrunkLen = (loopIdx * trunkLen_ + trunkLen_) > validS2Len ? validS2Len % trunkLen_ : trunkLen_;
                    uint32_t offset = vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + loopIdx * trunkLen_ + ldDstOffset;
                    AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkCountAlign256_);
                    // topk如果没有对齐到256，则把topkCountAlign256_ - topkCount_部分刷0
                    if (topkCountAlign256_ != topkCount_) {
                        uint64_t mask[1];
                        mask[0] = ~0;
                        mask[0] = mask[0] << (topkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        // 把topkCount_对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                        PipeBarrier<PIPE_V>();
                        // 把topk剩余对齐到256的部分刷0
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64 + 64], zero, topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    copyInParams.blockLen = validTrunkLen * sizeof(SCORE_T); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if ((topkCountAlign256_ + validTrunkLen) % 256 != 0) {
                        Duplicate(mrgValueLocal_[topkCountAlign256_ + validTrunkLen / 256 * 256], zero, QLICommon::Align(validTrunkLen, (uint32_t)256) - validTrunkLen / 256 * 256);
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                    }
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[offset], copyInParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    topkOp_.TopK(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, QLICommon::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256), loopIdx, s2LoopNum, info.isNeedLD);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                }   
            }
        } else {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero, validS2Len);
            // 如果需要LD 则将需要保存Value值
            if (info.isNeedLD) {
                copyInParams.blockLen = validS2Len * sizeof(SCORE_T);
                AscendC::DataCopyPad(scoreOutLocal_, scoreGm[vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + ldDstOffset], copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            }
        }
        
        if (!info.isNeedLD) {
            if (validS2Len < topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (validS2Len % 8); // 将`mask[0]`左移`validS2Len % 8`位 对齐
                PipeBarrier<PIPE_V>();
                // repeatTime  每次读取连续的8个datablock（每个block32Bytes，共256Bytes）
                // dstBlockStride 单次迭代内，矢量目的操作数不同datablock间地址步长。
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, mask, 1, 1, 0);
            }
            
            if (validS2Len / 8 * 8 + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64], neg, topkCount_ - (validS2Len / 8 * 8 + 64));
            }
            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        } else {
            PipeBarrier<PIPE_V>();
            AscendC::Adds(indicesOutLocal_.ReinterpretCast<int32_t>(), indicesOutLocal_.ReinterpretCast<int32_t>(), static_cast<int32_t>(curS2StartIdx), topkCountAlign16_);
            if (validS2Len < topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (validS2Len % scoreAlign);
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[validS2Len / scoreAlign * scoreAlign], zero, mask, 1, 1, 0);
                uint64_t maskI[1];
                maskI[0] = ~0;
                maskI[0] = maskI[0] << (validS2Len % 8);
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, maskI, 1, 1, 0);
            }
            if (validS2Len / scoreAlign * scoreAlign + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[validS2Len / scoreAlign * scoreAlign + 64], zero, topkCount_ - (validS2Len / scoreAlign * scoreAlign + 64));
            }
            if (validS2Len / 8 * 8 + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64], neg, topkCount_ - (validS2Len / 8 * 8 + 64));
            }
            if (topkCountAlign16_ != topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (topkCount_ % scoreAlign);
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[topkCount_ / scoreAlign * scoreAlign], zero, mask, 1, 1, 0);
                uint64_t maskIndices[1];
                maskIndices[0] = ~0;
                maskIndices[0] = maskIndices[0] << (topkCount_ % 8);
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[topkCount_ / 8 * 8], neg, maskIndices, 1, 1, 0);
            }

            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            // 将每一行S1的Topk结果存放在ldScoreGm和ldIndexGm中
            AscendC::DataCopyPad(ldScoreGm[offset], scoreOutLocal_, ldCopyScoreOutParams);
            AscendC::DataCopyPad(ldIndexGm[offset], indicesOutLocal_.ReinterpretCast<int32_t>(), ldCopyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        } 
    }
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessLD()
{
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    uint64_t ldProcessLen = ldInfo_.workspaceNum * topkCountAlign16_;
    if (ldProcessLen <= 0) {
        return;
    }
    uint64_t mrgValueLen = trunkLen_ + topkCountAlign256_;
    uint32_t ldWorkspaceNum = (ldProcessLen > mrgValueLen) ? (trunkLen_ / topkCountAlign16_) : ldInfo_.workspaceNum;
    uint32_t ldProcessNum = CeilDiv(ldInfo_.workspaceNum, ldWorkspaceNum);   // 搬运次数 一次搬运ldworkspaceNum块
    uint32_t ldProcessOffset = 0;

    SCORE_T zero = 0;
    int32_t neg = -1;
    uint32_t zero32 = 0;
    uint32_t scoreAlign = sizeof(SCORE_T) == 4 ? 8 : 16; // int32对应UB对齐数为8，int16需要16
    uint32_t ldProWorkspaceNum = ldInfo_.workspaceNum;
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    for (uint32_t j = 0; j < ldInfo_.mNum; j++) {
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        for (uint32_t i = 0; i < ldProcessNum; i++) {
            // 读取数据段过长
            if (ldProcessNum > 1) {
                ldProWorkspaceNum = (i == ldProcessNum - 1) ? ldInfo_.workspaceNum - i * ldWorkspaceNum : ldWorkspaceNum; // 当前搬运块数
            }
            ldProcessOffset = (i != 0) ? topkCountAlign256_ : 0;
            PipeBarrier<PIPE_V>();
            // 索引全部刷-1 value全部刷0
            Duplicate(mrgValueLocal_[ldProcessOffset], zero, topkCountAlign256_ + trunkLen_ - ldProcessOffset);
            Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[ldProcessOffset], neg, topkCountAlign256_ + trunkLen_ - ldProcessOffset);

            AscendC::DataCopyPadExtParams<SCORE_T> scorePadParams{true, 0, 0, 0};
            AscendC::DataCopyExtParams ldScoreParams;
            ldScoreParams.blockCount = ldProWorkspaceNum;
            ldScoreParams.blockLen = topkCountAlign16_ * sizeof(SCORE_T); // bytes
            ldScoreParams.srcStride = (s1BaseSize_ - 1) * topkCountAlign16_ * sizeof(SCORE_T);  // s1BaseSize-1 两个基本块之间的距离（slot 内行间为 T，跨 slot 步长 = s1BaseSize*T）
            ldScoreParams.dstStride = 0;

            AscendC::DataCopyPadExtParams<uint32_t> indexPadParams{true, 0, 0, 0};
            AscendC::DataCopyExtParams ldIndexParams;
            ldIndexParams.blockCount = ldProWorkspaceNum;
            ldIndexParams.blockLen = topkCountAlign16_ * sizeof(uint32_t); // bytes
            ldIndexParams.srcStride = (s1BaseSize_ - 1) * topkCountAlign16_ * sizeof(uint32_t);
            ldIndexParams.dstStride = 0;

            int32_t s2Len = topkCountAlign16_ * ldProWorkspaceNum + ldProcessOffset;
            uint32_t s2LenAlign = QLICommon::Align(s2Len, (int32_t)256); // 寄存器需要256对齐
            uint64_t LDGmOffset = ldInfo_.workspaceIdx * s1BaseSize_ * topkCountAlign16_ +
                                  topkCountAlign16_ * (ldInfo_.mStart + j) + i * ldWorkspaceNum * s1BaseSize_ * topkCountAlign16_; // 加入LD处理偏移
            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            AscendC::DataCopyPad(mrgValueLocal_[ldProcessOffset], ldScoreGm[LDGmOffset], ldScoreParams, scorePadParams);
            AscendC::DataCopyPad(ldIndexLocal_[ldProcessOffset], ldIndexGm[LDGmOffset].ReinterpretCast<uint32_t>(),
                                 ldIndexParams, indexPadParams);
            
            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);

            // 对非对齐索引和值刷0 -1
            if (s2LenAlign != s2Len) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (s2Len % 64);
                Duplicate(mrgValueLocal_[s2Len / 64 * 64], zero, mask, 1, 1, 0);
                // 把s2Len对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[s2Len / 64 * 64], neg, mask, 1, 1, 0);
            }
            if (s2Len / 64 * 64 + 64 < s2LenAlign) {
                PipeBarrier<PIPE_V>();
                Duplicate(mrgValueLocal_[s2Len / 64 * 64 + 64], zero, s2LenAlign - (s2Len / 64 * 64 + 64));
                PipeBarrier<PIPE_V>();
                Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[s2Len / 64 * 64 + 64], neg, s2LenAlign - (s2Len / 64 * 64 + 64));
            }
            
            PipeBarrier<PIPE_V>();
            topkOp_.LdTopK(mrgValueLocal_, ldIndexLocal_, indicesOutLocal_, scoreOutLocal_, s2LenAlign, j, ldProcessNum);
            if (topkCountAlign256_ != topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (topkCount_ % 64);
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                uint64_t maskIndices[1];
                maskIndices[0] = ~0;
                maskIndices[0] = maskIndices[0] << (topkCount_ % 64);
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[topkCount_ / 64 * 64], neg, maskIndices, 1, 1, 0);
            }
            if (topkCount_ / 64 * 64 + 64 < topkCountAlign256_) {
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[topkCount_ / 64 * 64 + 64], zero, topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                PipeBarrier<PIPE_V>();
                Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[topkCount_ / 64 * 64 + 64], neg, topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
            }
            
            PipeBarrier<PIPE_V>();
            uint32_t copyBytes = topkCountAlign256_; // 32B对齐 topkCountAlign256_
            AscendC::DataCopy(ldIndexLocal_, indicesOutLocal_, copyBytes);
            AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, copyBytes);
        }
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        uint64_t indiceOutGmOffset = ldInfo_.indiceOutCoreOffset + (ldInfo_.mStart + j) * constInfo_.kHeadNum * topkCount_;
        AscendC::DataCopyPad(indiceOutGm[indiceOutGmOffset], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}
}  // namespace QLIKernel
#endif