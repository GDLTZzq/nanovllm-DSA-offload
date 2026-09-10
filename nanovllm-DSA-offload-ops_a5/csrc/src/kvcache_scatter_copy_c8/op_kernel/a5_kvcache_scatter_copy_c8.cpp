/**
 * Ascend 950 token-granular copy for GLM packed-C8 MLA cache rows.
 *
 * Each opaque 656-byte row is copied byte-for-byte so FP8 latent values,
 * BF16 RoPE values, and FP32 per-tile scales remain together.
 */

#include "kernel_operator.h"
#include "a5_kvcache_scatter_copy_c8_tiling.h"

namespace {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 128;
constexpr uint32_t BLOCK_SHIFT = 7;
constexpr uint32_t BLOCK_MASK = BLOCK_SIZE - 1;

class A5KvcacheScatterCopyC8Kernel {
public:
    __aicore__ inline A5KvcacheScatterCopyC8Kernel(
        TPipe *pipe,
        const A5KvcacheScatterCopyC8TilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR sourceTokenIds,
        GM_ADDR destinationSlots,
        GM_ADDR copyCounts,
        GM_ADDR hbmBlockTable,
        GM_ADDR dramBlockTable,
        GM_ADDR hbmKv,
        GM_ADDR dramKv)
    {
        coreIdx_ = GetBlockIdx();
        const uint32_t packedBufferBytes =
            (tiling_->packedRowBytes + 31U) & ~31U;
        pipe_->InitBuffer(copyQueue_, 2, packedBufferBytes);
        sourceTokenIdsGm_.SetGlobalBuffer((__gm__ int32_t *)sourceTokenIds);
        destinationSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)destinationSlots);
        copyCountsGm_.SetGlobalBuffer((__gm__ int32_t *)copyCounts);
        hbmBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)hbmBlockTable);
        dramBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)dramBlockTable);
        hbmKvGm_.SetGlobalBuffer((__gm__ uint8_t *)hbmKv);
        dramKvGm_.SetGlobalBuffer((__gm__ uint8_t *)dramKv);
    }

    __aicore__ inline void Process()
    {
        cachedBatch_ = static_cast<uint32_t>(-1);
        cachedCount_ = 0;
        uint64_t current = FindNextValid(coreIdx_);
        CopyAddress currentAddress;
        while (current < tiling_->totalPairSlots &&
               !Resolve(current, currentAddress)) {
            current = FindNextValid(current + tiling_->usedCoreNum);
        }
        if (current >= tiling_->totalPairSlots) {
            return;
        }

        CopyIn(currentAddress);
        while (true) {
            uint64_t next = FindNextValid(current + tiling_->usedCoreNum);
            CopyAddress nextAddress;
            while (next < tiling_->totalPairSlots &&
                   !Resolve(next, nextAddress)) {
                next = FindNextValid(next + tiling_->usedCoreNum);
            }
            const bool hasNext = next < tiling_->totalPairSlots;
            if (hasNext) {
                CopyIn(nextAddress);
            }
            CopyOut(currentAddress);
            if (!hasNext) {
                break;
            }
            current = next;
            currentAddress = nextAddress;
        }
    }

private:
    struct CopyAddress {
        uint64_t sourceByteOffset = 0;
        uint64_t destinationByteOffset = 0;
    };

    __aicore__ inline uint64_t FirstOwnedAtOrAfter(uint64_t start) const
    {
        if (start <= coreIdx_) {
            return coreIdx_;
        }
        const uint64_t distance = start - coreIdx_;
        const uint64_t steps =
            (distance + tiling_->usedCoreNum - 1) / tiling_->usedCoreNum;
        return coreIdx_ + steps * tiling_->usedCoreNum;
    }

    __aicore__ inline uint64_t FindNextValid(uint64_t flatPair)
    {
        while (flatPair < tiling_->totalPairSlots) {
            const uint32_t batch = static_cast<uint32_t>(
                flatPair / tiling_->copyCap);
            const uint32_t copyIndex = static_cast<uint32_t>(
                flatPair - static_cast<uint64_t>(batch) * tiling_->copyCap);
            if (batch != cachedBatch_) {
                cachedCount_ = copyCountsGm_.GetValue(batch);
                if (cachedCount_ < 0) {
                    cachedCount_ = 0;
                } else if (cachedCount_ >
                           static_cast<int32_t>(tiling_->copyCap)) {
                    cachedCount_ = static_cast<int32_t>(tiling_->copyCap);
                }
                cachedBatch_ = batch;
            }
            if (copyIndex < static_cast<uint32_t>(cachedCount_)) {
                return flatPair;
            }
            flatPair = FirstOwnedAtOrAfter(
                (static_cast<uint64_t>(batch) + 1) * tiling_->copyCap);
        }
        return tiling_->totalPairSlots;
    }

    __aicore__ inline bool Resolve(uint64_t flatPair, CopyAddress &address)
    {
        const uint32_t batch = static_cast<uint32_t>(
            flatPair / tiling_->copyCap);
        const uint32_t copyIndex = static_cast<uint32_t>(
            flatPair - static_cast<uint64_t>(batch) * tiling_->copyCap);
        const uint64_t metadataOffset =
            static_cast<uint64_t>(batch) * tiling_->copyCap + copyIndex;
        const int32_t sourceToken =
            sourceTokenIdsGm_.GetValue(metadataOffset);
        const int32_t destinationSlot =
            destinationSlotsGm_.GetValue(metadataOffset);
        if (sourceToken < 0 || destinationSlot < 0) {
            return false;
        }

        const uint32_t sourceBlockColumn =
            static_cast<uint32_t>(sourceToken) >> BLOCK_SHIFT;
        const uint32_t destinationBlockColumn =
            static_cast<uint32_t>(destinationSlot) >> BLOCK_SHIFT;
        if (sourceBlockColumn >= tiling_->dramMaxBlockNum ||
            destinationBlockColumn >= tiling_->hbmMaxBlockNum) {
            return false;
        }
        const int32_t sourceBlock = dramBlockTableGm_.GetValue(
            static_cast<uint64_t>(batch) * tiling_->dramMaxBlockNum +
            sourceBlockColumn);
        const int32_t destinationBlock = hbmBlockTableGm_.GetValue(
            static_cast<uint64_t>(batch) * tiling_->hbmMaxBlockNum +
            destinationBlockColumn);
        if (sourceBlock < 0 || destinationBlock < 0) {
            return false;
        }

        const uint64_t sourceRow =
            static_cast<uint64_t>(sourceBlock) * BLOCK_SIZE +
            (static_cast<uint32_t>(sourceToken) & BLOCK_MASK);
        const uint64_t destinationRow =
            static_cast<uint64_t>(destinationBlock) * BLOCK_SIZE +
            (static_cast<uint32_t>(destinationSlot) & BLOCK_MASK);
        address.sourceByteOffset = sourceRow * tiling_->packedRowBytes;
        address.destinationByteOffset =
            destinationRow * tiling_->packedRowBytes;
        return true;
    }

    __aicore__ inline void CopyIn(const CopyAddress &address)
    {
        LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
        DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
        DataCopyExtParams params{1, tiling_->packedRowBytes, 0, 0, 0};
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            local, dramKvGm_[address.sourceByteOffset], params, pad);
        copyQueue_.EnQue<uint8_t>(local);
    }

    __aicore__ inline void CopyOut(const CopyAddress &address)
    {
        LocalTensor<uint8_t> local = copyQueue_.DeQue<uint8_t>();
        DataCopyExtParams params{1, tiling_->packedRowBytes, 0, 0, 0};
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            hbmKvGm_[address.destinationByteOffset], local, params);
        copyQueue_.FreeTensor(local);
    }

private:
    TPipe *pipe_;
    const A5KvcacheScatterCopyC8TilingData *tiling_;
    uint32_t coreIdx_ = 0;
    uint32_t cachedBatch_ = static_cast<uint32_t>(-1);
    int32_t cachedCount_ = 0;
    GlobalTensor<int32_t> sourceTokenIdsGm_;
    GlobalTensor<int32_t> destinationSlotsGm_;
    GlobalTensor<int32_t> copyCountsGm_;
    GlobalTensor<int32_t> hbmBlockTableGm_;
    GlobalTensor<int32_t> dramBlockTableGm_;
    GlobalTensor<uint8_t> hbmKvGm_;
    GlobalTensor<uint8_t> dramKvGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
};
} // namespace

extern "C" __global__ __aicore__ void a5_kvcache_scatter_copy_c8(
    GM_ADDR sourceTokenIds,
    GM_ADDR destinationSlots,
    GM_ADDR copyCounts,
    GM_ADDR hbmBlockTable,
    GM_ADDR dramBlockTable,
    GM_ADDR hbmKv,
    GM_ADDR dramKv,
    GM_ADDR hbmKvOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)hbmKvOut;
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5KvcacheScatterCopyC8TilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    A5KvcacheScatterCopyC8Kernel op(&pipe, &tilingData);
    op.Init(
        sourceTokenIds, destinationSlots, copyCounts,
        hbmBlockTable, dramBlockTable, hbmKv, dramKv);
    op.Process();
}
