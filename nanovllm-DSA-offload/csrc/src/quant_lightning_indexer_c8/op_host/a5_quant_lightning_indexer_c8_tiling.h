/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef a5_quant_lightning_indexer_c8_TILING_H_
#define a5_quant_lightning_indexer_c8_TILING_H_

#include "a5_sfa_shared/ops_log_compat.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct QLI8RequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct QLI8OptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

constexpr uint32_t QLI8_QUERY_INDEX = 0;
constexpr uint32_t QLI8_KEY_INDEX = 1;
constexpr uint32_t QLI8_WEIGHTS_INDEX = 2;
constexpr uint32_t QLI8_QUERY_DEQUANT_SCALE_INDEX = 3;
constexpr uint32_t QLI8_KEY_DEQUANT_SCALE_INDEX = 4;
constexpr uint32_t QLI8_ACTUAL_SEQ_Q_INDEX = 5;
constexpr uint32_t QLI8_ACTUAL_SEQ_K_INDEX = 6;
constexpr uint32_t QLI8_BLOCK_TABLE_INDEX = 7;
constexpr uint32_t QLI8_SPARSE_INDICES_INDEX = 0;

constexpr uint32_t QLI8_DIM_IDX_ONE = 1;
constexpr uint32_t QLI8_DIM_IDX_TWO = 2;
constexpr uint32_t QLI8_DIM_IDX_THREE = 3;
constexpr uint32_t QLI8_DIM_NUM_ONE = 1;
constexpr uint32_t QLI8_DIM_NUM_TWO = 2;
constexpr uint32_t QLI8_DIM_NUM_THREE = 3;
constexpr uint32_t QLI8_DIM_NUM_FOUR = 4;

constexpr uint32_t QLI8_DECODE_N2 = 1;
constexpr uint32_t QLI8_HEAD_DIM = 128;
constexpr uint32_t QLI8_SPARSE_COUNT = 2048;
constexpr uint32_t QLI8_BLOCK_SIZE = 128;
constexpr uint32_t QLI8_KEY_STRIDE0 = QLI8_BLOCK_SIZE * QLI8_HEAD_DIM; // 16384
constexpr uint32_t QLI8_KEY_DEQUANT_SCALE_STRIDE0 = QLI8_BLOCK_SIZE;   // 128
constexpr uint32_t QLI8_M_BASE_SIZE = 256;
constexpr uint32_t QLI8_S2_BASE_SIZE = 128;
constexpr uint32_t QLI8_TOP_K_COUNT_ALIGN_16 = 2048; // Align(sparseCount=2048, 16)

// 融合单算子：无 metadata 输入、无 sparse_values 输出，on-device 分核。
// 字段集合与 ops-transformer 新版 QLITilingData（kernel InitTilingData
// 逐名读取）完全一致：bSize/gSize/s1Size/s2Size/sparseCount/keyStride0/
// keyDequantScaleStride0/usedCoreNum/blockSize/maxBlockNumPerBatch/
// sparseMode/cmpRatio/batchSupperFlag。
BEGIN_TILING_DATA_DEF(QLITilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, keyStride0)
TILING_DATA_FIELD_DEF(uint32_t, keyDequantScaleStride0)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(uint32_t, cmpRatio)
TILING_DATA_FIELD_DEF(uint32_t, batchSupperFlag)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(A5QuantLightningIndexerC8, QLITilingData)

struct QLI8CompileInfo {};

struct QLI8ParaInfo {
    QLI8RequiredParaInfo query = {nullptr, nullptr};
    QLI8RequiredParaInfo key = {nullptr, nullptr};
    QLI8RequiredParaInfo weights = {nullptr, nullptr};
    QLI8RequiredParaInfo queryDequantScale = {nullptr, nullptr};
    QLI8RequiredParaInfo keyDequantScale = {nullptr, nullptr};
    QLI8OptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    QLI8OptionalParaInfo actualSeqLengthsK = {nullptr, nullptr};
    QLI8OptionalParaInfo blockTable = {nullptr, nullptr};
    QLI8RequiredParaInfo sparseIndices = {nullptr, nullptr};
};

class QLI8TilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    QLI8ParaInfo opParamInfo;

    uint32_t bSize = 0;
    uint32_t gSize = 0;
    uint32_t s2Size = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
};

class A5QuantLightningIndexerC8Tiling {
public:
    explicit A5QuantLightningIndexerC8Tiling(gert::TilingContext *context) : context_(context) {};
    ge::graphStatus ParseAndCheck(QLI8TilingInfo &tilingInfo);
    ge::graphStatus DoTiling(QLI8TilingInfo *tilingInfo);

private:
    ge::graphStatus GetNpuInfo(QLI8TilingInfo &tilingInfo) const;
    ge::graphStatus GetTensorInfo(QLI8TilingInfo &tilingInfo) const;
    ge::graphStatus CheckDtype(const QLI8TilingInfo &tilingInfo) const;
    ge::graphStatus CheckShape(QLI8TilingInfo &tilingInfo) const;

    gert::TilingContext *context_ = nullptr;
    QLITilingData tilingData_;
};

} // namespace optiling
#endif // a5_quant_lightning_indexer_c8_TILING_H_
