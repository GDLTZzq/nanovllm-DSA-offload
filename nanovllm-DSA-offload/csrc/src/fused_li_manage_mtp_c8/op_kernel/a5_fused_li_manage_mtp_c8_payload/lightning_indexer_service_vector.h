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
 * \file lightning_indexer_service_vector.h
 * \brief C8 MTP QuantLI vector 服务：打分（fp32 QK + vf relu + kScale）+
 *        token-only topk（MODE 0 payload）+ 请求级并集一次性缓存更新。
 */
#ifndef QUANT_LIGHTNING_INDEXER_SERVICE_VECTOR_H
#define QUANT_LIGHTNING_INDEXER_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_common.h"
// C8 vf（与 A5FusedLiManageC8 同一份）：QK 为 fp32（Mmad fp8×fp8→fp32 累加），
// relu 在 WeightedAccum 中（与官方"Relu在cube随路做"数值等价），
// kScale 经 6 参 MulWeightAndReduceSum 在 Σ 之后乘，无 ×1/1024、无 fp16 往返。
#include "vf/lightning_indexer_vector1.h"
#include "payload/hist_topk_index_update_a5_topk.h"
#include "payload/hist_topk_index_update_a5_evict_vf.h"
// MTP classify vf：DecodeTokenIds + 三件 squeeze（本算子根目录，flatten 后同级）
#include "../a5_fused_li_manage_mtp_c8_classify_vf.h"

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
    using W_T = typename QLIT::weightType;
    using SCALE_T = typename QLIT::scaleType;   // 反量化 scale 精度（fp32）
    using QK_T = typename QLIT::qkType;         // Cube 输出类型（fp32 累加）
    using SCORE_T = typename QLIT::scoreType;   // score 存储类型（uint16，bf16 sortable）
    static_assert(std::is_same_v<SCORE_T, uint16_t>, "A5FusedLiManageMtpC8 score must be uint16_t");

    __aicore__ inline QLIVector(){};
    __aicore__ inline void ProcessVec1(const QLICommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const QLICommon::RunInfo &info,
                                       bool allowSlotPrefetch);
    __aicore__ inline void FinalizeMtpRequest(uint32_t bIdx,
                                              uint32_t queryBegin,
                                              uint32_t queryCount);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct QLICommon::ConstInfo &constInfo,
                                      const LIMtpC8TilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<SCALE_T> qScaleGm,
                                              GlobalTensor<SCALE_T> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm,
                                              GlobalTensor<int32_t> cacheSlotsGm,
                                              GlobalTensor<int32_t> topkSlotsGm,
                                              GlobalTensor<int32_t> missCountGm,
                                              GlobalTensor<int32_t> missSourceIdsGm,
                                              GlobalTensor<int32_t> missDestinationSlotsGm,
                                              GlobalTensor<int32_t> reqPoolEntriesGm,
                                              GlobalTensor<int32_t> cacheTokensGm,
                                              GlobalTensor<int32_t> candidateLensGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

protected:
    GlobalTensor<SCORE_T> scoreGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<SCALE_T> qScaleGm;
    GlobalTensor<SCALE_T> kScaleGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> topkSlotsGm;
    GlobalTensor<int32_t> missCountGm;
    GlobalTensor<int32_t> missSourceIdsGm;
    GlobalTensor<int32_t> missDestinationSlotsGm;
    GlobalTensor<int32_t> reqPoolEntriesGm;
    GlobalTensor<int32_t> cacheTokensGm;
    GlobalTensor<int32_t> candidateLensGm;
    // =================================常量区=================================
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
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;
    static constexpr uint32_t V_MTE2_EVENT = EVENT_ID7;

private:
    __aicore__ inline void GetKeyScale(const QLICommon::RunInfo &runInfo, LocalTensor<SCALE_T> &kScaleUB,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<QK_T> resMm1UB_;
    // tmp buff for weight
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<W_T> weightUB_;
    // tmp buff for weight cast float
    TBuf<TPosition::VECCALC> weightFloatBuf_;
    LocalTensor<float> weightFloatUB_;
    // tmp buff for kScale
    TBuf<TPosition::VECCALC> kScaleBuf_;
    LocalTensor<SCALE_T> kScaleUB_;
    // tmp buff for qScale
    TBuf<TPosition::VECCALC> qScaleBuf_;
    LocalTensor<SCALE_T> qScaleUB_;

    // tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;

    // cache_slots DMA 暂存（victim 扫描 + per-query slot 发布共用）
    TBuf<TPosition::VECCALC> slotStageBuf_;
    LocalTensor<int32_t> slotStageLocal_;

    TBuf<TPosition::VECCALC> candidatePayloadBuf_;
    LocalTensor<uint32_t> candidatePayloadLocal_;

    LocalTensor<int32_t> outInvalidLocal_;

    int32_t blockId_ = -1;
    // para for vector
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
    uint32_t trunkLen_ = 0;
    struct QLICommon::ConstInfo constInfo_;
    const LIMtpC8TilingData *tilingData_ = nullptr;
    static constexpr uint32_t EVICT_CANDIDATE_CAP = 2048;
    // MTP: 请求级并集相关常量
    static constexpr uint32_t MTP_CACHE_SIZE = 8192;
    static constexpr uint32_t MTP_UNION_HASH_CAPACITY = 16384;
    static constexpr uint32_t MTP_UNION_HASH_MASK = MTP_UNION_HASH_CAPACITY - 1;
    static constexpr uint32_t MTP_MIN_QUERY_COUNT = 2;
    static constexpr uint32_t MTP_MAX_QUERY_COUNT = 4;
    static constexpr uint32_t MTP_MAX_CACHE_TOKENS = 16256;
    hist_topk_index_update_a5_payload::LITopk<SCORE_T> topkOp_;
};

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(QK_T));
    resMm1UB_ = resMm1Buf_.Get<QK_T>();
    // weight/qScale 预取在 decode 下每 (bN2, gS1) 只发生一次（无 pingpong），
    // 尺寸沿用 2 * 布局以保持 UB 规划不变，实际只使用前一半。
    pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightUB_ = weightBuf_.Get<W_T>();
    pipe->InitBuffer(weightFloatBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightFloatUB_ = weightFloatBuf_.Get<float>();
    pipe->InitBuffer(kScaleBuf_, 2 * s2BaseSize_ * 16 * sizeof(SCALE_T));
    kScaleUB_ = kScaleBuf_.Get<SCALE_T>();
    pipe->InitBuffer(qScaleBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    qScaleUB_ = qScaleBuf_.Get<SCALE_T>();

    pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T));
    vec1OutUB_ = outBuf_.Get<SCORE_T>();

    // Topk
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T));
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    outInvalidLocal_ = mrgValueBuf_.Get<int32_t>();

    pipe->InitBuffer(indicesOutBuf_, (topkCountAlign256_ + 64) * sizeof(uint32_t));         // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    pipe->InitBuffer(scoreOutBuf_, topkCountAlign256_ * sizeof(SCORE_T));
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_);

    pipe->InitBuffer(slotStageBuf_, topkCount_ * sizeof(int32_t));
    slotStageLocal_ = slotStageBuf_.Get<int32_t>();

    pipe->InitBuffer(candidatePayloadBuf_, EVICT_CANDIDATE_CAP * sizeof(uint32_t));
    candidatePayloadLocal_ = candidatePayloadBuf_.Get<uint32_t>();

    //刷-1
    Duplicate(kScaleUB_, static_cast<SCALE_T>(0), 2 * s2BaseSize_ * 16);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitParams(const struct QLICommon::ConstInfo &constInfo,
                                                   const LIMtpC8TilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    this->tilingData_ = tilingData;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;  // 2 (tSize<=64)
    s2BaseSize_ = constInfo.s2BaseSize;  // 128
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    blockId_ = GetBlockIdx();
    trunkLen_ = TRUNK_LEN_16K;
    topkCount_ = constInfo.sparseCount;
    topkOp_.Init(topkCount_, topkCount_, trunkLen_);
    topkCountAlign256_ = QLICommon::Align(constInfo.sparseCount, (uint64_t)256); // topkCount对齐到256
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<SCALE_T> qScaleGm,
                                                           GlobalTensor<SCALE_T> kScaleGm,
                                                           GlobalTensor<int32_t> indiceOutGm,
                                                           GlobalTensor<int32_t> blockTableGm,
                                                           GlobalTensor<int32_t> cacheSlotsGm,
                                                           GlobalTensor<int32_t> topkSlotsGm,
                                                           GlobalTensor<int32_t> missCountGm,
                                                           GlobalTensor<int32_t> missSourceIdsGm,
                                                           GlobalTensor<int32_t> missDestinationSlotsGm,
                                                           GlobalTensor<int32_t> reqPoolEntriesGm,
                                                           GlobalTensor<int32_t> cacheTokensGm,
                                                           GlobalTensor<int32_t> candidateLensGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
    this->cacheSlotsGm = cacheSlotsGm;
    this->topkSlotsGm = topkSlotsGm;
    this->missCountGm = missCountGm;
    this->missSourceIdsGm = missSourceIdsGm;
    this->missDestinationSlotsGm = missDestinationSlotsGm;
    this->reqPoolEntriesGm = reqPoolEntriesGm;
    this->cacheTokensGm = cacheTokensGm;
    this->candidateLensGm = candidateLensGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm)
{
    this->scoreGm = scoreGm; // resucesum*k
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);

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
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // init -1 and copy to output（topk workspace 行）
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
__aicore__ inline void QLIVector<QLIT>::GetKeyScale(const QLICommon::RunInfo &runInfo, LocalTensor<SCALE_T> &kScaleUB,
                                                    int64_t batchId, int64_t startS2, int64_t getLen)
{
    // startS2一定能整除kCacheBlockSize_
    AscendC::DataCopyPadExtParams<SCALE_T> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
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
            copyInParams.blockLen = firstPartLen * sizeof(SCALE_T);
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_], kScaleGm[blockId *
                                constInfo_.keyDequantScaleStride0 + startBlockTableOffset], copyInParams, padParams);
            startBlockTableIdx++;
            getLen = getLen - firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        copyInParams.blockLen = kCacheBlockSize_ * sizeof(SCALE_T);
        for (int32_t i = 0; i < getLoopNum; i++) {
            if (i == getLoopNum - 1) {
                copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(SCALE_T);
            }
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_ +
                                               resUbBaseOffset + i * kCacheBlockSize_],
                                 kScaleGm[blockId * constInfo_.keyDequantScaleStride0],
                                 copyInParams, padParams);
        }
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(SCALE_T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_],
                                            kScaleGm[runInfo.tensorKeyScaleOffset],
                                            copyInParams, padParams);
    }
}

// =====================================================================================
// ProcessVec1 — 与 A5FusedLiManageC8 同构：fp32 QK（Fixpipe 直通）+ vf relu/乘权归约
//   MTP 差异：S1 = packed query 数，每 (bN2, gS1) 预取一次 weight/qScale。
// =====================================================================================
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessVec1(const QLICommon::RunInfo &info)
{
    auto pingpong = (info.loop % 2);
    auto kScalepingpong = (info.kScaleLoop % 2);
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

    if (info.isFirstS2InnerLoop) {
        // weight / qScale 预取（每个 (bN2, gS1) 一次；mBaseSize=4*gSize 对齐
        // BF16 版后每请求只有一个 gS1 组，此处即每请求一次）
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
        //weightsGm --> weightUB_
        int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * kHeadNum_ * gSize_;
        DataCopyPadExtParams<W_T> padWeightsParams{false, 0, 0, 0};
        DataCopyExtParams wDataCopyExtParams;
        wDataCopyExtParams.blockCount = curAivS1ProcNum;
        wDataCopyExtParams.blockLen = gSize_ * sizeof(W_T);
        wDataCopyExtParams.srcStride = 0;
        wDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - wDataCopyExtParams.blockLen) / 32;
        DataCopyPad(weightUB_, weightsGm[weightGmOffset], wDataCopyExtParams, padWeightsParams);

        //qScaleGm  -->  qScaleUB_
        DataCopyPadExtParams<SCALE_T> padQScaleParams{false, 0, 0, 0};
        DataCopyExtParams qDataCopyExtParams;
        qDataCopyExtParams.blockCount = curAivS1ProcNum;
        qDataCopyExtParams.blockLen = gSize_ * sizeof(SCALE_T);
        qDataCopyExtParams.srcStride = 0;
        qDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - qDataCopyExtParams.blockLen) / 32;
        DataCopyPad(qScaleUB_, qScaleGm[weightGmOffset], qDataCopyExtParams, padQScaleParams);
        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + 0);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + 0);

        // weightFloat = float(weight) * qScale（fp32 域）。行以 bank 深度
        // (512B) 为 stride 摆放（DataCopyPad dstStride + blockLen = 512B），
        // 与 BF16 版的行布局约定一致；逐行 Cast/Mul 避免跨行连续访问。
        for (int64_t row = 0; row < curAivS1ProcNum; ++row) {
            Cast(weightFloatUB_[row * (UB_BANK_DEPTH_STRIDE / sizeof(float))],
                 weightUB_[row * (UB_BANK_DEPTH_STRIDE / sizeof(W_T))],
                 RoundMode::CAST_NONE, gSize_);
            Mul(weightFloatUB_[row * (UB_BANK_DEPTH_STRIDE / sizeof(float))],
                weightFloatUB_[row * (UB_BANK_DEPTH_STRIDE / sizeof(float))],
                qScaleUB_[row * (UB_BANK_DEPTH_STRIDE / sizeof(SCALE_T))],
                gSize_);
        }
    }

    if ((info.s2Idx - info.s2Start) % 16 == 0) {
        // kScale 预取: 一次 16 个 s2 块, pingpong 覆盖大 source 长度的连续窗口
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
        uint32_t getLen = 16 * s2BaseSize_ > (info.validS2Len - info.s2Idx * s2BaseSize_)
                                           ? info.validS2Len - info.s2Idx * s2BaseSize_
                                           : 16 * s2BaseSize_;
        //kScaleGm  -->  kScaleUB_（含 PA blkTable 跳转）
        GetKeyScale(info, kScaleUB_, info.bIdx, curS2Idx, getLen);
        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
    }
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);

    //CV同步
    CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_CV_EVENT + info.loop % 2);   //V核等C核计算完mm1，mm1Res已搬运到UB

    auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(float)) / 2 * constInfo_.mBaseSize;
    // QK 已是 fp32（Mmad fp8×fp8→fp32 累加 + Fixpipe 原样搬运），
    // 无 ×1/1024、无 fp16 往返；relu 由 vf 的 WeightedAccum 完成
    // （与官方"Relu在cube随路做"数值等价）。

    LocalTensor<SCORE_T> outBase = vec1OutUB_[pingpong * (UB_BANK_STRIDE / sizeof(uint16_t))];
    auto kScaleBase = kScaleUB_[kScalepingpong * 16 * s2BaseSize_ + ((info.s2Idx - info.s2Start) % 16) * s2BaseSize_];

    // 复用 lidu 原版 vf（QK 已是 fp32）: Relu(QK)×weight 归约 + bf16 sortable,
    // kScale 经 6 参重载（必填）在 Σ 之后乘（官方 QuantLI 语义, 逐位一致）。
    // mBaseSize=4*gSize 后每 AIV 最多 2 行（2 个 query），逐行调用单行 vf；
    // 行 stride 与 BF16 版约定一致（out 256 个 uint16 / weight 128 个 float）。
    auto kScale = (__ubuf__ SCALE_T *)kScaleBase.GetPhyAddr();
    for (int64_t row = 0; row < curAivS1ProcNum; ++row) {
        auto out = (__ubuf__ SCORE_T *)outBase.GetPhyAddr() +
                   row * (UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T));
        auto qk = (__ubuf__ float *)qkBase.GetPhyAddr() + row * gSize_ * 128;
        auto weightFloat = (__ubuf__ float *)weightFloatUB_.GetPhyAddr() +
                           row * (UB_BANK_DEPTH_STRIDE / sizeof(float));
        vector1::MulWeightAndReduceSum(out, qk, qkVLstride, weightFloat, gSize_, kScale);
    }
    if (info.isFirstS2InnerLoop) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
    }
    if ((info.s2Idx - info.s2Start) % 16 == 0) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
    }
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    //outUB_ --->  scoreGm
    int64_t vec1OutGmOffset = blockId_ % 2 == 0 ? curS2Idx :
                            s1BaseSizePerAIV * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + curS2Idx;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(SCORE_T);
    copyOutParams.srcStride = (UB_BANK_DEPTH_STRIDE - copyOutParams.blockLen) / 32;
    copyOutParams.dstStride = (QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) - s2BaseSize_) * sizeof(SCORE_T);
    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, copyOutParams);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + pingpong);   //V核处理完，通知C核可以把mm1Res搬运到UB
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessTopK(
    const QLICommon::RunInfo &info, bool allowSlotPrefetch)
{
    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);

    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

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
    // MODE 0 payload：topk 只带 token id，slot 在并集更新后统一解析，
    // 无需 slot 预取/侧车（RunAblationStage 内部忽略 slot 参数）。
    LocalTensor<int32_t> slotPrefetchLocal = slotStageLocal_;

    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    }

    int32_t validS2Len = cuRealAcSeq;
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;
        uint32_t vecOffset = blockId_ % 2 * CeilDiv(s1BaseSize_, 2) + i;

        uint16_t zero = 0;
        int32_t neg = -1;
        if (constInfo_.attenMaskFlag) {
            validS2Len = (int32_t)i + cuRealAcSeq;
        }
        if (validS2Len <= 0) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCount_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                                                          indicesOutLocal_.ReinterpretCast<int32_t>(),
                                                          copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        AscendC::DataCopyPadExtParams<uint16_t> padParams{true, 0, 0, 0};
        if (validS2Len >= topkCount_) {
            uint32_t s2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
            if (s2LoopNum == 1) {
                uint32_t validS2LenAlign = QLICommon::Align(validS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[validS2Len / 256 * 256], zero, validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                copyInParams.blockLen = validS2Len * sizeof(uint16_t); // byte
                AscendC::DataCopyPadExtParams<uint16_t> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(
                    mrgValueLocal_,
                    scoreGm[vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)],
                    copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                topkOp_.RunAblationStage(
                    mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                    slotPrefetchLocal, slotStageLocal_, cacheSlotsGm,
                    static_cast<uint64_t>(info.cacheRowIdx) * constInfo_.cacheSlotsSize,
                    0, topkCount_, validS2LenAlign,
                    static_cast<uint32_t>(validS2Len), 0, 1,
                    allowSlotPrefetch);
            } else {
                for (uint32_t loopIdx = 0; loopIdx < s2LoopNum; loopIdx++) {
                    if (loopIdx == 0) {
                        copyInParams.blockLen = trunkLen_ * sizeof(uint16_t); // byte
                        AscendC::DataCopyPad(
                            mrgValueLocal_,
                            scoreGm[vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)],
                            copyInParams, padParams);
                        SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        topkOp_.RunAblationStage(
                            mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                            slotPrefetchLocal, slotStageLocal_, cacheSlotsGm,
                            static_cast<uint64_t>(info.cacheRowIdx) * constInfo_.cacheSlotsSize,
                            0, topkCount_, trunkLen_, trunkLen_,
                            loopIdx, s2LoopNum, allowSlotPrefetch);
                        continue;
                    }
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    uint32_t validTrunkLen = (loopIdx * trunkLen_ + trunkLen_) > validS2Len
                                                                               ? validS2Len % trunkLen_
                                                                               :trunkLen_;
                    uint32_t offset = vecOffset *
                                 QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                                 loopIdx * trunkLen_;
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
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64 + 64], zero,
                                             topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    copyInParams.blockLen = validTrunkLen * sizeof(uint16_t); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if ((topkCountAlign256_ + validTrunkLen) % 256 != 0) {
                        Duplicate(mrgValueLocal_[topkCountAlign256_ + validTrunkLen / 256 * 256],
                                        zero, QLICommon::Align(validTrunkLen,
                                        (uint32_t)256) - validTrunkLen / 256 * 256);
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                    }
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[offset], copyInParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    topkOp_.RunAblationStage(
                        mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                        slotPrefetchLocal, slotStageLocal_, cacheSlotsGm,
                        static_cast<uint64_t>(info.cacheRowIdx) * constInfo_.cacheSlotsSize,
                        loopIdx * trunkLen_, topkCount_,
                        QLICommon::Align(topkCountAlign256_ + validTrunkLen,
                                        static_cast<uint32_t>(256)),
                        validTrunkLen, loopIdx, s2LoopNum,
                        allowSlotPrefetch);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                }
            }
        } else {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero, validS2Len);
        }

        if (validS2Len < topkCount_) {
            uint64_t mask[1];
            mask[0] = ~0;
            mask[0] = mask[0] << (validS2Len % 8);
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, mask, 1, 1, 0);
        }

        if (validS2Len / 8 * 8 + 64 < topkCount_) {
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64],
                                        neg, topkCount_ - (validS2Len / 8 * 8 + 64));
        }

        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        uint32_t outputRow = static_cast<uint32_t>(info.indiceOutOffset / topkCount_) +
                             curS1Idx + rowIdx;
        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        if (validS2Len >= static_cast<int32_t>(topkCount_)) {
            // MODE 0 幸存者只带 token id；mask 为幂等保护（与 BF16 版一致）
            TopkIndexerClassifyVF::DecodeTokenIds(
                (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(),
                topkCount_ / TopkIndexerClassifyVF::CHUNK_SIZE);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyPad(
            indiceOutGm[static_cast<uint64_t>(outputRow) * topkCount_],
            indicesOutLocal_.template ReinterpretCast<int32_t>(),
            copyOutParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    }
}

// =====================================================================================
// FinalizeMtpRequest — 请求级并集 + 一次性缓存更新 + 逐 query 发布 slot
// =====================================================================================
// 与 BF16 版 FinalizeMtpRequest 同构，C8 差异：
//   1. cache 行寻址：poolRow(req_pool_entries[bIdx]) × cacheSlotsSize
//      （BF16 固定 262144 行 stride）
//   2. 输入校验与错误语义沿用旧版 RequestPoolManager（C=0 no-op、越界 -1 等）
//   3. topk 行在 workspace 尾部（indiceOutGm），公共输出只有 slot 行
// =====================================================================================
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::FinalizeMtpRequest(
    uint32_t bIdx, uint32_t queryBegin, uint32_t queryCount)
{
    constexpr uint32_t TOKEN_MASK =
        hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_TOKEN_MASK;
    constexpr uint32_t SLOT_SHIFT =
        hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_SLOT_SHIFT;
    constexpr uint32_t SLOT_MASK =
        hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_SLOT_MASK;
    constexpr uint32_t SCAN_CHUNK = 2048;

    // TopK 已完成，resMm1 成为死区。复用为
    // [16K union hash | 8K miss tokens | 8K victim payloads]。
    // 注意：该区域（32KB/64KB 缓冲 + 溢出部分）会覆盖后续 weight/kScale/
    // mrgValue 等缓冲，但 finalize 阶段它们全部不再使用；slotStage 与
    // candidatePayload 分配在最后，不会被覆盖（与 BF16 版同构的布局假设）。
    // 并集区布局（与 BF16 版同构：16K union hash | 8K miss tokens |
    // 8K victim payloads；C8 的 resMm1 只有 32KB，溢出进已死的
    // weight/kScale/mrgValue 等缓冲，扫描期临时缓冲已移到安全区）
    LocalTensor<uint32_t> unionHash =
        resMm1UB_.template ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> missTokens = unionHash[MTP_UNION_HASH_CAPACITY];
    LocalTensor<uint32_t> victimPayloads = missTokens[MTP_CACHE_SIZE];
    // 并集区 [0, 128KB) 覆盖 resMm1 之外的 weight/kScale/mrgValue/indicesOut/
    // scoreOut 等缓冲（C8 的 resMm1 只有 32KB，溢出比 BF16 更广），扫描期的
    // 临时缓冲必须放在并集区之外：topkSharedTmp 的尾部 [131072, 158720) 是
    // 安全区。packedSlots（victim 扫描的 int16 侧车）与逐 query topk 装载
    // 都移到这里，避免 Cast/DMA 覆盖并集 hash 条目（曾导致 ~1-3% 并集 token
    // 被误驱逐）。
    LocalTensor<int16_t> packedSlots =
        topkSharedTmpLocal_[7232].template ReinterpretCast<int16_t>();
    LocalTensor<int32_t> finalizeTopkLocal =
        topkSharedTmpLocal_[7424].template ReinterpretCast<int32_t>();

    const int32_t budgetValue = cacheTokensGm.GetValue(bIdx);
    const int32_t candidateValue = candidateLensGm.GetValue(bIdx);
    const int32_t poolRowValue = reqPoolEntriesGm.GetValue(bIdx);

    AscendC::DataCopyParams slotCopyOut{1, static_cast<uint16_t>(topkCount_ * sizeof(int32_t)), 0, 0};
    AscendC::DataCopyParams scalarCopyOut{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};

    // ---- 输入校验与错误语义（与旧版 RequestPoolManager 一致）----
    if (queryCount < MTP_MIN_QUERY_COUNT || queryCount > MTP_MAX_QUERY_COUNT ||
        queryBegin + queryCount > tilingData_->packedQueryCount) {
        Duplicate(slotStageLocal_, static_cast<int32_t>(-1), topkCount_);
        // Duplicate 是 VECTOR 管线的 UB 写，MTE3 读之前必须用 V_MTE3 旗标
        // 排序（S_MTE3 只排序 scalar，会读到 Duplicate 完成前的旧数据）。
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        for (uint32_t q = 0; q < queryCount; ++q) {
            uint32_t row = queryBegin + q;
            if (row >= tilingData_->packedQueryCount) {
                break;
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
            DataCopyPad(topkSlotsGm[static_cast<uint64_t>(row) * topkCount_],
                        slotStageLocal_, slotCopyOut);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        }
        LocalTensor<int32_t> missCountLocal =
            scoreOutLocal_.template ReinterpretCast<int32_t>();
        missCountLocal.SetValue(0, -1);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopyPad(missCountGm[bIdx], missCountLocal, scalarCopyOut);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        return;
    }
    if (budgetValue == 0) {
        Duplicate(slotStageLocal_, static_cast<int32_t>(-1), topkCount_);
        // Duplicate 是 VECTOR 管线的 UB 写，MTE3 读之前必须用 V_MTE3 旗标
        // 排序（S_MTE3 只排序 scalar，会读到 Duplicate 完成前的旧数据）。
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        for (uint32_t q = 0; q < queryCount; ++q) {
            uint32_t row = queryBegin + q;
            SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
            DataCopyPad(topkSlotsGm[static_cast<uint64_t>(row) * topkCount_],
                        slotStageLocal_, slotCopyOut);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        }
        LocalTensor<int32_t> missCountLocal =
            scoreOutLocal_.template ReinterpretCast<int32_t>();
        missCountLocal.SetValue(0, 0);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopyPad(missCountGm[bIdx], missCountLocal, scalarCopyOut);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        return;
    }
    if (budgetValue < static_cast<int32_t>(topkCount_) ||
        budgetValue > static_cast<int32_t>(MTP_MAX_CACHE_TOKENS) ||
        candidateValue < static_cast<int32_t>(topkCount_) ||
        candidateValue > static_cast<int32_t>(tilingData_->cacheSlotsSize) ||
        poolRowValue < 0 ||
        poolRowValue >= static_cast<int32_t>(tilingData_->poolSize)) {
        Duplicate(slotStageLocal_, static_cast<int32_t>(-1), topkCount_);
        // Duplicate 是 VECTOR 管线的 UB 写，MTE3 读之前必须用 V_MTE3 旗标
        // 排序（S_MTE3 只排序 scalar，会读到 Duplicate 完成前的旧数据）。
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        for (uint32_t q = 0; q < queryCount; ++q) {
            uint32_t row = queryBegin + q;
            SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
            DataCopyPad(topkSlotsGm[static_cast<uint64_t>(row) * topkCount_],
                        slotStageLocal_, slotCopyOut);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        }
        LocalTensor<int32_t> missCountLocal =
            scoreOutLocal_.template ReinterpretCast<int32_t>();
        missCountLocal.SetValue(0, -1);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopyPad(missCountGm[bIdx], missCountLocal, scalarCopyOut);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        return;
    }

    const uint32_t budget = static_cast<uint32_t>(budgetValue);
    const uint64_t cacheBase =
        static_cast<uint64_t>(poolRowValue) * tilingData_->cacheSlotsSize;

    Duplicate(unionHash.template ReinterpretCast<int32_t>(),
              static_cast<int32_t>(-1), MTP_UNION_HASH_CAPACITY);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);

    AscendC::DataCopyExtParams copyIn{1, 0, 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> intPad{true, 0, 0, 0};
    uint32_t missCount = 0;

    // 每个 query 的 TopK 插入请求级并集（与 BF16 版同构的 hash 判重）；
    // 跨 query 重复 token 只分类一次。
    for (uint32_t q = 0; q < queryCount; ++q) {
        uint64_t rowOffset =
            static_cast<uint64_t>(queryBegin + q) * topkCount_;
        copyIn.blockLen = topkCount_ * sizeof(int32_t);
        DataCopyPad(finalizeTopkLocal,
                    indiceOutGm[rowOffset], copyIn, intPad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        for (uint32_t i = 0; i < topkCount_; ++i) {
            int32_t signedToken = finalizeTopkLocal.GetValue(i);
            if (signedToken < 0) {
                continue;
            }
            uint32_t token = static_cast<uint32_t>(signedToken) & TOKEN_MASK;
            uint32_t hashPos = (token * 2654435761U) & MTP_UNION_HASH_MASK;
            uint32_t stored = unionHash.GetValue(hashPos);
            while (stored != 0xffffffffU && stored != token) {
                hashPos = (hashPos + 1U) & MTP_UNION_HASH_MASK;
                stored = unionHash.GetValue(hashPos);
            }
            if (stored == token) {
                continue;
            }
            unionHash.SetValue(hashPos, token);
            if (cacheSlotsGm.GetValue(cacheBase + token) < 0) {
                missTokens.SetValue(missCount++, token);
            }
        }
    }

    if (missCount > MTP_CACHE_SIZE) {
        // union 容量保护（理论上 <= 4*2048，此处防越界）
        Duplicate(slotStageLocal_, static_cast<int32_t>(-1), topkCount_);
        // Duplicate 是 VECTOR 管线的 UB 写，MTE3 读之前必须用 V_MTE3 旗标
        // 排序（S_MTE3 只排序 scalar，会读到 Duplicate 完成前的旧数据）。
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        for (uint32_t q = 0; q < queryCount; ++q) {
            uint32_t row = queryBegin + q;
            SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
            DataCopyPad(topkSlotsGm[static_cast<uint64_t>(row) * topkCount_],
                        slotStageLocal_, slotCopyOut);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        }
        LocalTensor<int32_t> missCountLocal =
            scoreOutLocal_.template ReinterpretCast<int32_t>();
        missCountLocal.SetValue(0, -1);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopyPad(missCountGm[bIdx], missCountLocal, scalarCopyOut);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        return;
    }

    // SIMD 压实已缓存 token，再用 UB hash 排除并集保护 token（与 BF16 版
    // 同构）；victim 找满 missCount 个即可早停。
    uint32_t victimCount = 0;
    for (uint32_t chunkBase = 0;
         chunkBase < static_cast<uint32_t>(candidateValue) && victimCount < missCount;
         chunkBase += SCAN_CHUNK) {
        uint32_t chunkLen = Min(SCAN_CHUNK, static_cast<uint32_t>(candidateValue) - chunkBase);
        uint32_t alignedLen = QLICommon::Align(chunkLen, static_cast<uint32_t>(64));
        copyIn.blockLen = chunkLen * sizeof(int32_t);
        DataCopyPad(slotStageLocal_, cacheSlotsGm[cacheBase + chunkBase],
                    copyIn, intPad);
        SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
        Cast(packedSlots, slotStageLocal_, RoundMode::CAST_NONE, chunkLen);
        if (alignedLen > chunkLen) {
            Duplicate(packedSlots[chunkLen], static_cast<int16_t>(-1),
                      alignedLen - chunkLen);
        }
        PipeBarrier<PIPE_V>();
        LightningIndexerPayloadEvictVF::CompactPayloads(
            (__ubuf__ uint32_t *)candidatePayloadLocal_.GetPhyAddr(),
            (__ubuf__ uint16_t *)packedSlots.GetPhyAddr(), chunkBase,
            alignedLen / 64);
        uint32_t compactCount = static_cast<uint32_t>(
            AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>() /
            sizeof(uint32_t));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
        WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);

        for (uint32_t i = 0;
             i < compactCount && victimCount < missCount; ++i) {
            uint32_t payload = candidatePayloadLocal_.GetValue(i);
            uint32_t token = payload & TOKEN_MASK;
            uint32_t hashPos = (token * 2654435761U) & MTP_UNION_HASH_MASK;
            uint32_t stored = unionHash.GetValue(hashPos);
            while (stored != 0xffffffffU && stored != token) {
                hashPos = (hashPos + 1U) & MTP_UNION_HASH_MASK;
                stored = unionHash.GetValue(hashPos);
            }
            if (stored != token) {
                victimPayloads.SetValue(victimCount++, payload);
            }
        }
    }
    if (victimCount != missCount) {
        Duplicate(slotStageLocal_, static_cast<int32_t>(-1), topkCount_);
        // Duplicate 是 VECTOR 管线的 UB 写，MTE3 读之前必须用 V_MTE3 旗标
        // 排序（S_MTE3 只排序 scalar，会读到 Duplicate 完成前的旧数据）。
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        for (uint32_t q = 0; q < queryCount; ++q) {
            uint32_t row = queryBegin + q;
            SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
            DataCopyPad(topkSlotsGm[static_cast<uint64_t>(row) * topkCount_],
                        slotStageLocal_, slotCopyOut);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        }
        LocalTensor<int32_t> missCountLocal =
            scoreOutLocal_.template ReinterpretCast<int32_t>();
        missCountLocal.SetValue(0, -1);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopyPad(missCountGm[bIdx], missCountLocal, scalarCopyOut);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        return;
    }

    // 一次性提交：驱逐 victim、分配 miss（同一 slot 复用）
    for (uint32_t i = 0; i < missCount; ++i) {
        uint32_t payload = victimPayloads.GetValue(i);
        uint32_t evictToken = payload & TOKEN_MASK;
        uint32_t slot = (payload >> SLOT_SHIFT) & SLOT_MASK;
        uint32_t missToken = missTokens.GetValue(i);
        cacheSlotsGm.SetValue(cacheBase + evictToken, -1);
        cacheSlotsGm.SetValue(cacheBase + missToken, static_cast<int32_t>(slot));
        victimPayloads.SetValue(i, slot);
    }
    PipeBarrier<PIPE_ALL>();

    AscendC::DataCopyParams copyOut{1, 0, 0, 0};
    if (missCount > 0) {
        copyOut.blockLen =
            static_cast<uint16_t>(missCount * sizeof(int32_t));
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        uint64_t missOffset = static_cast<uint64_t>(bIdx) * MTP_CACHE_SIZE;
        DataCopyPad(missSourceIdsGm[missOffset],
                    missTokens.template ReinterpretCast<int32_t>(), copyOut);
        DataCopyPad(missDestinationSlotsGm[missOffset],
                    victimPayloads.template ReinterpretCast<int32_t>(), copyOut);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
    }

    LocalTensor<int32_t> missCountLocal =
        scoreOutLocal_.template ReinterpretCast<int32_t>();
    missCountLocal.SetValue(0, static_cast<int32_t>(missCount));
    copyOut.blockLen = static_cast<uint16_t>(sizeof(int32_t));
    SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
    DataCopyPad(missCountGm[bIdx], missCountLocal, copyOut);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);

    // 每个 query 的 top-2048 必须指向一次性更新之后的缓存；
    // 跨 query 重复的并集 token 在每个 query 里得到同一个 slot。
    for (uint32_t q = 0; q < queryCount; ++q) {
        uint64_t rowOffset =
            static_cast<uint64_t>(queryBegin + q) * topkCount_;
        copyIn.blockLen = topkCount_ * sizeof(int32_t);
        DataCopyPad(finalizeTopkLocal,
                    indiceOutGm[rowOffset], copyIn, intPad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t i = 0; i < topkCount_; ++i) {
            int32_t token = finalizeTopkLocal.GetValue(i);
            int32_t slot = token < 0
                               ? -1
                               : cacheSlotsGm.GetValue(
                                     cacheBase + static_cast<uint32_t>(token));
            slotStageLocal_.SetValue(i, slot);
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopyPad(topkSlotsGm[rowOffset], slotStageLocal_, slotCopyOut);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
    }
}

}  // namespace QLIKernel
#endif
