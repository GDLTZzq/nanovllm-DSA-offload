#ifndef FUSED_LI_MANAGE_MTP_UNION_H
#define FUSED_LI_MANAGE_MTP_UNION_H

#include "kernel_operator.h"
#include "lightning_indexer_vector.h"

namespace MtpUnion {
using namespace AscendC;

constexpr uint32_t ROUTES = 4U;
constexpr uint32_t TOPK = 2048U;
constexpr uint32_t PAIR_WORDS = TOPK * 2U;
constexpr uint32_t CAPACITY = ROUTES * TOPK;
constexpr uint32_t SOURCE_BITS = 18U;
constexpr uint32_t SOURCE_MASK = (1U << SOURCE_BITS) - 1U;
constexpr int32_t INVALID_SLOT14 = (1U << (32U - SOURCE_BITS)) - 1U;
constexpr uint32_t MISS_KEY_BASE_BITS = 0x40000000U;
constexpr uint32_t EVICT_CHUNK = 512U;
constexpr uint32_t EVICT_PAIR_WORDS = EVICT_CHUNK * 2U;
constexpr uint32_t EVICT_SORT_REPEATS = EVICT_CHUNK / 32U;
// AscendC compare masks must own an aligned vector-work region.  Keep the
// same full-chunk spacing as the mature single-query LIM implementation;
// packing the uint8 view down to its logical byte count aliases the following
// invalid-key and Sort32 work tensors on device.
constexpr uint32_t EVICT_MASK_WORK_FLOATS = EVICT_CHUNK;
constexpr uint32_t EVICT_SCRATCH_FLOATS = EVICT_CHUNK * 12U;
constexpr uint32_t THRESHOLD_STRIDE = 8U;
// Sort32/MrgSort do not provide a useful ordering guarantee for infinities.
// Keep a wide finite gap between the invalid key and the scan stop threshold.
constexpr float INVALID_EVICT_KEY = -1.0e20F;
constexpr float EVICT_STOP_KEY = -5.0e19F;

template <HardEvent event>
__aicore__ inline void Sync(HardEvent e)
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(e));
    AscendC::SetFlag<event>(id);
    AscendC::WaitFlag<event>(id);
}

class MtpMissUnion {
public:
    __aicore__ inline void Init(GM_ADDR pair0, GM_ADDR pair1,
                                GM_ADDR candidateLengths, GM_ADDR cacheSlots,
                                GM_ADDR cacheTokens, GM_ADDR reqEntries,
                                GM_ADDR scoreScratch, GM_ADDR thresholdScratch,
                                GM_ADDR missSources, GM_ADDR missDestinations,
                                GM_ADDR counts, GM_ADDR topkSources,
                                GM_ADDR topkDestinations,
                                uint32_t batch,
                                uint32_t scoreCapacity, uint32_t cacheCapacity,
                                TPipe *pipe)
    {
        pair0Gm.SetGlobalBuffer((__gm__ float *)pair0);
        pair1Gm.SetGlobalBuffer((__gm__ float *)pair1);
        pair0BitsGm.SetGlobalBuffer((__gm__ uint32_t *)pair0);
        pair1BitsGm.SetGlobalBuffer((__gm__ uint32_t *)pair1);
        candidateLengthsGm.SetGlobalBuffer((__gm__ int32_t *)candidateLengths);
        cacheSlotsGm.SetGlobalBuffer((__gm__ int32_t *)cacheSlots);
        cacheTokensGm.SetGlobalBuffer((__gm__ int32_t *)cacheTokens);
        reqEntriesGm.SetGlobalBuffer((__gm__ int32_t *)reqEntries);
        scoreScratchGm.SetGlobalBuffer((__gm__ float *)scoreScratch);
        thresholdScratchGm.SetGlobalBuffer((__gm__ float *)thresholdScratch);
        missSourcesGm.SetGlobalBuffer((__gm__ int32_t *)missSources);
        missDestinationsGm.SetGlobalBuffer((__gm__ int32_t *)missDestinations);
        countsGm.SetGlobalBuffer((__gm__ int32_t *)counts);
        topkSourcesGm.SetGlobalBuffer((__gm__ int32_t *)topkSources);
        topkDestinationsGm.SetGlobalBuffer((__gm__ int32_t *)topkDestinations);
        batchSize = batch;
        scoreStride = ((scoreCapacity + EVICT_CHUNK - 1U) / EVICT_CHUNK) * EVICT_CHUNK;
        sourceCapacity = cacheCapacity;
        pipe->InitBuffer(pairInBuf, CAPACITY * 2U * sizeof(float));
        pipe->InitBuffer(pairOutBuf, CAPACITY * 2U * sizeof(float));
        pipe->InitBuffer(unionSourceBuf, CAPACITY * sizeof(int32_t));
        pipe->InitBuffer(countBuf, 32U);
    }

    __aicore__ inline void Process(uint32_t first, uint32_t stride)
    {
        for (uint32_t batch = first; batch < batchSize; batch += stride) {
            ProcessBatch(batch);
        }
    }

private:
    __aicore__ inline uint32_t HashEvictScanSeed(uint32_t actual, uint32_t cacheRow)
    {
        uint32_t value = actual ^ ((cacheRow + 1U) * 0x9e3779b9U);
        value ^= value >> 16U;
        value *= 0x7feb352dU;
        value ^= value >> 15U;
        value *= 0x846ca68bU;
        value ^= value >> 16U;
        return value;
    }

    __aicore__ inline void SortEvictChunk(LocalTensor<float> dst,
                                          LocalTensor<float> key,
                                          LocalTensor<uint32_t> payload,
                                          LocalTensor<float> tmp)
    {
        Sort32(tmp, key, payload, EVICT_SORT_REPEATS);
        PipeBarrier<PIPE_V>();
        MrgSort4Info params;
        params.elementLengths[0] = 32U;
        params.elementLengths[1] = 32U;
        params.elementLengths[2] = 32U;
        params.elementLengths[3] = 32U;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b1111;
        params.repeatTimes = 1;
        for (uint32_t group = 0U; group < 4U; ++group) {
            const uint32_t offset = group * 256U;
            MrgSortSrcList<float> sources;
            sources.src1 = tmp[offset];
            sources.src2 = tmp[offset + 64U];
            sources.src3 = tmp[offset + 128U];
            sources.src4 = tmp[offset + 192U];
            MrgSort<float>(dst[offset], sources, params);
        }
        PipeBarrier<PIPE_V>();
        params.elementLengths[0] = 128U;
        params.elementLengths[1] = 128U;
        params.elementLengths[2] = 128U;
        params.elementLengths[3] = 128U;
        MrgSortSrcList<float> sources;
        sources.src1 = dst;
        sources.src2 = dst[256U];
        sources.src3 = dst[512U];
        sources.src4 = dst[768U];
        MrgSort<float>(tmp, sources, params);
        PipeBarrier<PIPE_V>();
        DataCopy(dst, tmp, EVICT_PAIR_WORDS);
        PipeBarrier<PIPE_V>();
    }

    // Build one 512-token victim block.  A valid key is the negated sum of
    // the four route scores, so a descending sort places the lowest aggregate
    // score first.  Tokens in any route TopK and uncached tokens are masked.
    __aicore__ inline void BuildEvictCandidateChunk(
        uint32_t batch, uint32_t cacheRow, uint32_t start, uint32_t valid,
        const float thresholds[ROUTES], LocalTensor<float> chunkPair,
        LocalTensor<float> scratch)
    {
        LocalTensor<float> score0 = scratch;
        LocalTensor<float> score1 = scratch[EVICT_CHUNK];
        LocalTensor<float> score2 = scratch[EVICT_CHUNK * 2U];
        LocalTensor<float> score3 = scratch[EVICT_CHUNK * 3U];
        LocalTensor<float> key = scratch[EVICT_CHUNK * 4U];
        LocalTensor<float> temp = scratch[EVICT_CHUNK * 5U];
        LocalTensor<int32_t> slots = scratch[EVICT_CHUNK * 6U].ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> payload = scratch[EVICT_CHUNK * 7U].ReinterpretCast<uint32_t>();
        LocalTensor<uint8_t> invalidMask = scratch[EVICT_CHUNK * 8U].ReinterpretCast<uint8_t>();
        LocalTensor<float> invalidKey =
            scratch[EVICT_CHUNK * 8U + EVICT_MASK_WORK_FLOATS];
        LocalTensor<float> sortTmp = scratch[EVICT_CHUNK * 10U];

        const uint64_t requestRoute = static_cast<uint64_t>(batch) * ROUTES;
        DataCopyPad(score0, scoreScratchGm[requestRoute * scoreStride + start],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0},
                    AscendC::DataCopyPadExtParams<float>{
                        true, 0, static_cast<uint8_t>((8U - valid % 8U) % 8U), 0.0F});
        DataCopyPad(score1, scoreScratchGm[(requestRoute + 1U) * scoreStride + start],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0},
                    AscendC::DataCopyPadExtParams<float>{
                        true, 0, static_cast<uint8_t>((8U - valid % 8U) % 8U), 0.0F});
        DataCopyPad(score2, scoreScratchGm[(requestRoute + 2U) * scoreStride + start],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0},
                    AscendC::DataCopyPadExtParams<float>{
                        true, 0, static_cast<uint8_t>((8U - valid % 8U) % 8U), 0.0F});
        DataCopyPad(score3, scoreScratchGm[(requestRoute + 3U) * scoreStride + start],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0},
                    AscendC::DataCopyPadExtParams<float>{
                        true, 0, static_cast<uint8_t>((8U - valid % 8U) % 8U), 0.0F});
        DataCopyPad(slots,
                    cacheSlotsGm[static_cast<uint64_t>(cacheRow) * sourceCapacity + start],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(valid * sizeof(int32_t)), 0, 0, 0},
                    AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        Sync<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        // Preserve the requested eviction ordering: lower aggregate score is
        // better, hence a descending sort over the negated four-route sum.
        Add(key, score0, score1, valid);
        PipeBarrier<PIPE_V>();
        Add(temp, score2, score3, valid);
        PipeBarrier<PIPE_V>();
        Add(key, key, temp, valid);
        PipeBarrier<PIPE_V>();
        Muls(key, key, -1.0F, valid);
        PipeBarrier<PIPE_V>();

        // Pack before testing cache validity.  The persistent INVALID_IDX=-1
        // becomes the positive 14-bit value 0x3fff after decoding, matching
        // the proven single-query LIM codec and avoiding signed-compare
        // ambiguity on the vector path.
        ArithProgression<int32_t>(payload.ReinterpretCast<int32_t>(),
                                  static_cast<int32_t>(start), 1, EVICT_CHUNK);
        PipeBarrier<PIPE_V>();
        ShiftLeft(slots.ReinterpretCast<uint32_t>(), slots.ReinterpretCast<uint32_t>(),
                  SOURCE_BITS, valid);
        PipeBarrier<PIPE_V>();
        Add(payload.ReinterpretCast<int32_t>(), payload.ReinterpretCast<int32_t>(),
            slots, valid);
        PipeBarrier<PIPE_V>();

        LocalTensor<int32_t> decodedSlots = temp.ReinterpretCast<int32_t>();
        ShiftRight(decodedSlots.ReinterpretCast<uint32_t>(), payload,
                   SOURCE_BITS, valid);
        PipeBarrier<PIPE_V>();

        // Build one protection margin instead of repeatedly selecting into
        // key.  margin >= 0 means the token belongs to at least one route's
        // TopK (ties included) and therefore cannot be evicted.
        Adds(score0, score0, -thresholds[0], valid);
        PipeBarrier<PIPE_V>();
        Adds(score1, score1, -thresholds[1], valid);
        PipeBarrier<PIPE_V>();
        Max(score0, score0, score1, valid);
        PipeBarrier<PIPE_V>();
        Adds(score2, score2, -thresholds[2], valid);
        PipeBarrier<PIPE_V>();
        Max(score0, score0, score2, valid);
        PipeBarrier<PIPE_V>();
        Adds(score3, score3, -thresholds[3], valid);
        PipeBarrier<PIPE_V>();
        Max(score0, score0, score3, valid);
        PipeBarrier<PIPE_V>();

        // AscendC's proven protection path compares the negated margin with
        // zero using LE: margin >= 0 iff -margin <= 0.  This protects every
        // token selected by at least one of the four TopK routes, including
        // threshold ties.
        Muls(score0, score0, -1.0F, valid);
        PipeBarrier<PIPE_V>();
        Duplicate(invalidKey, INVALID_EVICT_KEY, EVICT_CHUNK);
        PipeBarrier<PIPE_V>();
        CompareScalar(invalidMask, score0, 0.0F, CMPMODE::LE, valid);
        PipeBarrier<PIPE_V>();
        Select(key, invalidMask, invalidKey, key,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, valid);
        PipeBarrier<PIPE_V>();
        CompareScalar(invalidMask, decodedSlots, INVALID_SLOT14, CMPMODE::EQ, valid);
        PipeBarrier<PIPE_V>();
        Select(key, invalidMask, invalidKey, key,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, valid);
        PipeBarrier<PIPE_V>();
        if (valid < EVICT_CHUNK) {
            Duplicate(key[valid], INVALID_EVICT_KEY, EVICT_CHUNK - valid);
            PipeBarrier<PIPE_V>();
        }
        SortEvictChunk(chunkPair, key, payload, sortTmp);
    }

    // This scalar prefix walk is reserved for correctness/fallback paths.  The
    // steady union-miss<=512 path checks its exact kth key in O(1) instead.
    __aicore__ inline uint32_t ValidCandidatePrefix(LocalTensor<float> pairs,
                                                    uint32_t capacity)
    {
        uint32_t count = 0U;
        while (count < capacity &&
               pairs.GetValue(count * 2U) > EVICT_STOP_KEY) {
            ++count;
        }
        return count;
    }

    __aicore__ inline uint32_t ApplyCandidateUpdates(
        uint32_t actual, uint32_t cacheRow,
        uint32_t cacheTokenCount, uint32_t outputOffset,
        LocalTensor<float> pairs, uint32_t count,
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations)
    {
        LocalTensor<uint32_t> pairBits = pairs.ReinterpretCast<uint32_t>();
        const uint64_t cacheBase =
            static_cast<uint64_t>(cacheRow) * sourceCapacity;
        uint32_t updated = 0U;
        for (uint32_t index = 0U; index < count; ++index) {
            const uint32_t payload = pairBits.GetValue(index * 2U + 1U);
            const uint32_t victimSource = payload & SOURCE_MASK;
            const int32_t victimSlot =
                static_cast<int32_t>(payload >> SOURCE_BITS);
            const int32_t missSourceValue =
                unionSources.GetValue(outputOffset + index);
            if (victimSource >= actual || missSourceValue < 0 ||
                static_cast<uint32_t>(missSourceValue) >= actual ||
                victimSlot < 0 ||
                static_cast<uint32_t>(victimSlot) >= cacheTokenCount) {
                break;
            }
            unionDestinations.SetValue(outputOffset + updated, victimSlot);
            cacheSlotsGm.SetValue(cacheBase + victimSource, -1);
            cacheSlotsGm.SetValue(cacheBase +
                                      static_cast<uint32_t>(missSourceValue),
                                  victimSlot);
            ++updated;
        }
        return updated;
    }

    __aicore__ inline uint32_t PairPayloadSource(uint32_t batch, uint32_t route,
                                                 uint32_t index)
    {
        const uint64_t base = static_cast<uint64_t>(batch) * CAPACITY;
        if (route < 2U) {
            return pair0BitsGm.GetValue(base + route * PAIR_WORDS + index * 2U + 1U) &
                   SOURCE_MASK;
        }
        return pair1BitsGm.GetValue(base + (route - 2U) * PAIR_WORDS +
                                    index * 2U + 1U) & SOURCE_MASK;
    }

    __aicore__ inline bool SegmentContains(uint32_t batch, uint32_t route,
                                            uint32_t begin, uint32_t end,
                                            uint32_t source)
    {
        const uint32_t segmentEnd = end;
        while (begin < end) {
            const uint32_t middle = (begin + end) >> 1U;
            const uint32_t current = PairPayloadSource(batch, route, middle);
            if (current < source) {
                begin = middle + 1U;
            } else {
                end = middle;
            }
        }
        return begin < segmentEnd && PairPayloadSource(batch, route, begin) == source;
    }

    __aicore__ inline bool IsProtectedTopk(uint32_t batch, uint32_t source,
                                           const uint32_t lengths[ROUTES])
    {
        for (uint32_t route = 0U; route < ROUTES; ++route) {
            if (SegmentContains(batch, route, 0U, lengths[route], source) ||
                SegmentContains(batch, route, lengths[route], TOPK, source)) {
                return true;
            }
        }
        return false;
    }

    // Threshold ties are rare on the target BF16 workload.  The vector fast
    // path masks every tie to guarantee that no selected TopK token can be
    // evicted.  If that conservative rule leaves too few victims, resolve
    // only tied tokens against the exact four sorted TopK rows.
    __aicore__ inline uint32_t AppendThresholdTieFallback(
        uint32_t batch, uint32_t actual, uint32_t cacheRow,
        uint32_t cacheTokenCount, const uint32_t lengths[ROUTES],
        const float thresholds[ROUTES], uint32_t written, uint32_t required,
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations)
    {
        const uint64_t cacheBase = static_cast<uint64_t>(cacheRow) * sourceCapacity;
        const uint64_t requestRoute = static_cast<uint64_t>(batch) * ROUTES;
        for (uint32_t source = 0U; source < actual && written < required; ++source) {
            const int32_t slot = cacheSlotsGm.GetValue(cacheBase + source);
            if (slot < 0 || static_cast<uint32_t>(slot) >= cacheTokenCount) {
                continue;
            }
            bool above = false;
            bool tied = false;
            for (uint32_t route = 0U; route < ROUTES; ++route) {
                const float score = scoreScratchGm.GetValue(
                    (requestRoute + route) * scoreStride + source);
                if (score > thresholds[route]) {
                    above = true;
                    break;
                }
                tied = tied || score == thresholds[route];
            }
            if (above || !tied || IsProtectedTopk(batch, source, lengths)) {
                continue;
            }
            const int32_t missSource = unionSources.GetValue(written);
            if (missSource < 0 || static_cast<uint32_t>(missSource) >= actual) {
                continue;
            }
            cacheSlotsGm.SetValue(cacheBase + source, -1);
            cacheSlotsGm.SetValue(cacheBase +
                                      static_cast<uint32_t>(missSource),
                                  slot);
            unionDestinations.SetValue(written, slot);
            ++written;
        }
        return written;
    }

    __aicore__ inline uint32_t FindEvictSlotsAndUpdateCache(
        uint32_t batch, uint32_t count, const uint32_t lengths[ROUTES],
        LocalTensor<float> input, LocalTensor<float> accumulator,
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations)
    {
        if (count == 0U) {
            return 0U;
        }
        const uint32_t actual = static_cast<uint32_t>(candidateLengthsGm.GetValue(batch));
        const uint32_t cacheTokenCount = static_cast<uint32_t>(cacheTokensGm.GetValue(batch));
        const uint32_t cacheRow = static_cast<uint32_t>(reqEntriesGm.GetValue(batch));
        const uint32_t chunks = (actual + EVICT_CHUNK - 1U) / EVICT_CHUNK;
        if (chunks == 0U) {
            return 0U;
        }
        const uint32_t startChunk = HashEvictScanSeed(actual, cacheRow) % chunks;
        LocalTensor<float> chunkPair = input;
        LocalTensor<float> scratch = input[EVICT_PAIR_WORDS];
        // BuildEvictCandidateChunk uses 6144 floats from scratch.  Keep the
        // two-list merge output disjoint from both the chunk and its scratch.
        LocalTensor<float> mergeTmp = input[EVICT_PAIR_WORDS + EVICT_SCRATCH_FLOATS];

        // Thresholds are produced by whichever AIV owns each final TopK row,
        // while this request's eviction scan always runs on one even AIV.
        // Fetch the four strided values through MTE2 after the kernel-wide
        // barrier.  This avoids stale scalar-cache lines across invocations
        // and makes the producer-MTE3 -> SyncAll -> consumer-MTE2 dependency
        // explicit.
        LocalTensor<float> thresholdLocal = mergeTmp;
        const uint64_t requestRoute = static_cast<uint64_t>(batch) * ROUTES;
        for (uint32_t route = 0U; route < ROUTES; ++route) {
            DataCopyPad(
                thresholdLocal[route * 8U],
                thresholdScratchGm[(requestRoute + route) * THRESHOLD_STRIDE],
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<float>{true, 0, 7U, 0.0F});
        }
        Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        float thresholds[ROUTES];
        for (uint32_t route = 0U; route < ROUTES; ++route) {
            thresholds[route] = thresholdLocal.GetValue(route * 8U);
        }

        if (count <= EVICT_CHUNK) {
            Duplicate(accumulator, INVALID_EVICT_KEY, EVICT_PAIR_WORDS);
            PipeBarrier<PIPE_V>();
            bool found = false;
            for (uint32_t scan = 0U; scan < chunks; ++scan) {
                const uint32_t chunk = (startChunk + scan) % chunks;
                const uint32_t start = chunk * EVICT_CHUNK;
                const uint32_t valid = start + EVICT_CHUNK > actual
                                           ? actual - start
                                           : EVICT_CHUNK;
                BuildEvictCandidateChunk(batch, cacheRow, start, valid,
                                         thresholds, chunkPair, scratch);
                LIServiceVec::MergeSort(accumulator, EVICT_CHUNK, chunkPair,
                                        EVICT_CHUNK, mergeTmp);
                Sync<HardEvent::V_S>(HardEvent::V_S);
                if (accumulator.GetValue((count - 1U) * 2U) >
                    EVICT_STOP_KEY) {
                    found = true;
                    break;
                }
            }
            const uint32_t vectorCount = found
                                             ? count
                                             : ValidCandidatePrefix(accumulator, count);
            const uint32_t updated = ApplyCandidateUpdates(
                actual, cacheRow, cacheTokenCount, 0U, accumulator,
                vectorCount, unionSources, unionDestinations);
            const uint32_t finalCount = AppendThresholdTieFallback(
                batch, actual, cacheRow, cacheTokenCount, lengths, thresholds,
                updated, count, unionSources, unionDestinations);
            return finalCount;
        }

        // Large union misses are correctness boundaries rather than the
        // steady-decode target.  Keep each 512-token block vectorized and
        // sorted, and append its legal unique slots to the local union map.
        // Source chunks are disjoint and a valid cache row owns each slot once.
        uint32_t written = 0U;
        for (uint32_t scan = 0U; scan < chunks && written < count; ++scan) {
            const uint32_t chunk = (startChunk + scan) % chunks;
            const uint32_t start = chunk * EVICT_CHUNK;
            const uint32_t valid = start + EVICT_CHUNK > actual
                                       ? actual - start
                                       : EVICT_CHUNK;
            BuildEvictCandidateChunk(batch, cacheRow, start, valid,
                                     thresholds, chunkPair, scratch);
            Sync<HardEvent::V_S>(HardEvent::V_S);
            const uint32_t remaining = count - written;
            const uint32_t capacity = remaining < EVICT_CHUNK
                                          ? remaining
                                          : EVICT_CHUNK;
            const uint32_t take = ValidCandidatePrefix(chunkPair, capacity);
            const uint32_t updated = ApplyCandidateUpdates(
                actual, cacheRow, cacheTokenCount, written, chunkPair,
                take, unionSources, unionDestinations);
            if (updated != 0U) {
                written += updated;
            }
        }
        const uint32_t finalCount = AppendThresholdTieFallback(
            batch, actual, cacheRow, cacheTokenCount, lengths, thresholds,
            written, count, unionSources, unionDestinations);
        return finalCount;
    }

    // The LI owner already published sorted miss prefixes in topk_src_ids.
    // Pull all four prefixes into UB in parallel, join them against the local
    // sorted union source/destination mapping, then publish the four patched
    // destination prefixes under one MTE3 fence.  This avoids scalar GM reads
    // from both the pair workspace and the mutable cache map.
    __aicore__ inline void PrepareTopkRows(
        uint32_t batch, const uint32_t lengths[ROUTES], uint32_t unionCount,
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations,
        LocalTensor<float> sourceStorage,
        LocalTensor<float> destinationStorage)
    {
        if (unionCount == 0U) {
            return;
        }
        LocalTensor<int32_t> allSources =
            sourceStorage.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> allDestinations =
            destinationStorage.ReinterpretCast<int32_t>();
        bool hasMiss = false;
        for (uint32_t route = 0U; route < ROUTES; ++route) {
            if (lengths[route] == 0U) {
                continue;
            }
            const uint64_t rowOffset =
                (static_cast<uint64_t>(batch) * ROUTES + route) * TOPK;
            LocalTensor<int32_t> rowSources = allSources[route * TOPK];
            DataCopyPad(
                rowSources, topkSourcesGm[rowOffset],
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(lengths[route] * sizeof(int32_t)),
                    0, 0, 0},
                AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            hasMiss = true;
        }
        if (!hasMiss) {
            return;
        }
        Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        for (uint32_t route = 0U; route < ROUTES; ++route) {
            LocalTensor<int32_t> rowSources = allSources[route * TOPK];
            LocalTensor<int32_t> rowDestinations =
                allDestinations[route * TOPK];
            uint32_t unionCursor = 0U;
            for (uint32_t miss = 0U; miss < lengths[route]; ++miss) {
                const int32_t source = rowSources.GetValue(miss);
                while (unionCursor < unionCount &&
                       unionSources.GetValue(unionCursor) < source) {
                    ++unionCursor;
                }
                const int32_t destination =
                    unionCursor < unionCount &&
                            unionSources.GetValue(unionCursor) == source
                        ? unionDestinations.GetValue(unionCursor)
                        : -1;
                rowDestinations.SetValue(miss, destination);
            }
        }
    }

    // All caller-visible union/update outputs are independent until the
    // request-local computation is complete.  Publish them under one MTE3
    // fence instead of serializing miss sources, destinations, count, and
    // four TopK prefixes separately.
    __aicore__ inline void PublishFinalOutputs(
        uint32_t batch, const uint32_t lengths[ROUTES], uint32_t count,
        LocalTensor<int32_t> countLocal,
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations,
        LocalTensor<float> topkDestinationStorage)
    {
        countLocal.SetValue(0, static_cast<int32_t>(count));
        Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        if (count != 0U) {
            const uint64_t unionOffset =
                static_cast<uint64_t>(batch) * CAPACITY;
            const uint16_t unionBytes =
                static_cast<uint16_t>(count * sizeof(int32_t));
            DataCopyPad(missSourcesGm[unionOffset], unionSources,
                        {1, unionBytes, 0, 0});
            DataCopyPad(missDestinationsGm[unionOffset], unionDestinations,
                        {1, unionBytes, 0, 0});

            LocalTensor<int32_t> allDestinations =
                topkDestinationStorage.ReinterpretCast<int32_t>();
            for (uint32_t route = 0U; route < ROUTES; ++route) {
                if (lengths[route] == 0U) {
                    continue;
                }
                const uint64_t rowOffset =
                    (static_cast<uint64_t>(batch) * ROUTES + route) * TOPK;
                DataCopyPad(
                    topkDestinationsGm[rowOffset],
                    allDestinations[route * TOPK],
                    {1, static_cast<uint16_t>(lengths[route] * sizeof(int32_t)),
                     0, 0});
            }
        }
        DataCopyPad(countsGm[batch], countLocal,
                    {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline void ProcessBatch(uint32_t batch)
    {
        LocalTensor<float> input = pairInBuf.Get<float>();
        LocalTensor<float> merged = pairOutBuf.Get<float>();
        // The eviction merge temporary extends beyond input's lower half, so
        // union sources use a dedicated 32-KiB UB buffer.  The accumulator
        // only uses merged's lower half; its upper half safely holds union
        // destinations across eviction and final TopK patching.
        LocalTensor<int32_t> unionSources = unionSourceBuf.Get<int32_t>();
        LocalTensor<int32_t> unionDestinations =
            merged.ReinterpretCast<int32_t>()[CAPACITY];
        LocalTensor<int32_t> countLocal = countBuf.Get<int32_t>();
        const uint64_t pairBase = static_cast<uint64_t>(batch) * CAPACITY;
        DataCopy(input, pair0Gm[pairBase], CAPACITY);
        DataCopy(input[CAPACITY], pair1Gm[pairBase], CAPACITY);
        Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        uint32_t lengths[ROUTES] = {0U, 0U, 0U, 0U};
        LocalTensor<uint32_t> pairBits = input.ReinterpretCast<uint32_t>();
        for (uint32_t route = 0U; route < ROUTES; ++route) {
            const uint32_t routeOffset = route * PAIR_WORDS;
            uint32_t low = 0U;
            uint32_t high = TOPK;
            while (low < high) {
                const uint32_t middle = (low + high) >> 1U;
                const uint32_t key = pairBits.GetValue(routeOffset + middle * 2U);
                if (key >= MISS_KEY_BASE_BITS) {
                    low = middle + 1U;
                } else {
                    high = middle;
                }
            }
            lengths[route] = low;
        }

        const uint32_t total = lengths[0] + lengths[1] + lengths[2] + lengths[3];
        if (total == 0U) {
            PublishFinalOutputs(batch, lengths, 0U, countLocal, unionSources,
                                unionDestinations, merged);
            return;
        }

        Sync<HardEvent::S_V>(HardEvent::S_V);
        MrgSort4Info params;
        params.elementLengths[0] = lengths[0];
        params.elementLengths[1] = lengths[1];
        params.elementLengths[2] = lengths[2];
        params.elementLengths[3] = lengths[3];
        params.ifExhaustedSuspension = false;
        params.validBit = (lengths[0] > 0U ? 0b0001 : 0U) |
                          (lengths[1] > 0U ? 0b0010 : 0U) |
                          (lengths[2] > 0U ? 0b0100 : 0U) |
                          (lengths[3] > 0U ? 0b1000 : 0U);
        params.repeatTimes = 1;
        MrgSortSrcList<float> sources;
        sources.src1 = input;
        sources.src2 = input[PAIR_WORDS];
        sources.src3 = input[PAIR_WORDS * 2U];
        sources.src4 = input[PAIR_WORDS * 3U];
        MrgSort<float>(merged, sources, params);
        Sync<HardEvent::V_S>(HardEvent::V_S);

        uint32_t count = 0U;
        int32_t last = -1;
        LocalTensor<uint32_t> mergedBits = merged.ReinterpretCast<uint32_t>();
        for (uint32_t index = 0U; index < total; ++index) {
            const uint32_t key = mergedBits.GetValue(index * 2U);
            const int32_t sourceId =
                static_cast<int32_t>(SOURCE_MASK - (key - MISS_KEY_BASE_BITS));
            if (sourceId != last) {
                unionSources.SetValue(count++, sourceId);
                last = sourceId;
            }
        }

        const uint32_t updated = FindEvictSlotsAndUpdateCache(
            batch, count, lengths, input, merged, unionSources,
            unionDestinations);
        PrepareTopkRows(batch, lengths, updated, unionSources,
                        unionDestinations, input, merged);
        PublishFinalOutputs(batch, lengths, updated, countLocal,
                            unionSources, unionDestinations, merged);
    }

    GlobalTensor<float> pair0Gm;
    GlobalTensor<float> pair1Gm;
    GlobalTensor<uint32_t> pair0BitsGm;
    GlobalTensor<uint32_t> pair1BitsGm;
    GlobalTensor<int32_t> candidateLengthsGm;
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> cacheTokensGm;
    GlobalTensor<int32_t> reqEntriesGm;
    GlobalTensor<float> scoreScratchGm;
    GlobalTensor<float> thresholdScratchGm;
    GlobalTensor<int32_t> missSourcesGm;
    GlobalTensor<int32_t> missDestinationsGm;
    GlobalTensor<int32_t> countsGm;
    GlobalTensor<int32_t> topkSourcesGm;
    GlobalTensor<int32_t> topkDestinationsGm;
    TBuf<TPosition::VECCALC> pairInBuf;
    TBuf<TPosition::VECCALC> pairOutBuf;
    TBuf<TPosition::VECCALC> unionSourceBuf;
    TBuf<TPosition::VECCALC> countBuf;
    uint32_t batchSize = 0U;
    uint32_t scoreStride = 0U;
    uint32_t sourceCapacity = 0U;
};
} // namespace MtpUnion

#endif
