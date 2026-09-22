#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {

void CheckQLI8Common(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& query_dequant_scale,
    const at::Tensor& key_dequant_scale,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_key,
    const at::Tensor& block_table) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 &&
          (query.size(1) == 8 || query.size(1) == 16 ||
           query.size(1) == 24 || query.size(1) == 32 ||
           query.size(1) == 64) &&
          query.size(2) == kIndexerDim &&
          query.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "QLI C8 query must be float8_e4m3fn [B,8|16|24|32|64,128].");
  const int64_t batch = query.size(0);
  const int64_t heads = query.size(1);
  TORCH_CHECK(
      key.dim() == 4 && key.size(0) > 0 && key.size(1) == kBlockSize &&
          key.size(2) == 1 && key.size(3) == kIndexerDim &&
          key.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "QLI C8 key must be float8_e4m3fn [blocks,128,1,128].");
  TORCH_CHECK(
      weights.dim() == 2 && weights.size(0) == batch &&
          weights.size(1) == heads &&
          weights.scalar_type() == at::kFloat &&
          query_dequant_scale.dim() == 2 &&
          query_dequant_scale.size(0) == batch &&
          query_dequant_scale.size(1) == heads &&
          query_dequant_scale.scalar_type() == at::kFloat,
      "QLI C8 weights/query_dequant_scale must be FP32 [B,N].");
  TORCH_CHECK(
      key_dequant_scale.dim() == 3 &&
          key_dequant_scale.size(0) == key.size(0) &&
          key_dequant_scale.size(1) == kBlockSize &&
          key_dequant_scale.size(2) == 1 &&
          key_dequant_scale.scalar_type() == at::kFloat,
      "QLI C8 key_dequant_scale must be FP32 [blocks,128,1].");
  TORCH_CHECK(
      actual_seq_lengths_query.dim() == 1 &&
          actual_seq_lengths_query.size(0) == batch &&
          actual_seq_lengths_key.dim() == 1 &&
          actual_seq_lengths_key.size(0) == batch &&
          block_table.dim() == 2 && block_table.size(0) == batch &&
          block_table.size(1) > 0,
      "QLI C8 metadata must be int32[B], block_table int32[B,maxblocks].");
  for (const at::Tensor* tensor : {&actual_seq_lengths_query,
                                   &actual_seq_lengths_key, &block_table}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt,
        "QLI C8 metadata/block_table tensors must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &key, &weights, &query_dequant_scale,
       &key_dequant_scale, &actual_seq_lengths_query,
       &actual_seq_lengths_key, &block_table},
      "QLI C8");
}

at::Tensor QuantLightningIndexerC8Npu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& query_dequant_scale,
    const at::Tensor& key_dequant_scale,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_key,
    const at::Tensor& block_table) {
  CheckQLI8Common(
      query, key, weights, query_dequant_scale, key_dequant_scale,
      actual_seq_lengths_query, actual_seq_lengths_key, block_table);
  auto options = query.options().dtype(at::kInt);
  auto sparseIndices = at::empty({query.size(0), 1, kSparseCount}, options);
  auto keepalive = std::make_tuple(
      query, key, weights, query_dequant_scale, key_dequant_scale,
      actual_seq_lengths_query, actual_seq_lengths_key, block_table,
      sparseIndices);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5QuantLightningIndexerC8,
      keepalive,
      query,
      key,
      weights,
      query_dequant_scale,
      key_dequant_scale,
      actual_seq_lengths_query,
      actual_seq_lengths_key,
      block_table,
      sparseIndices);
  return sparseIndices;
}

at::Tensor QuantLightningIndexerC8Meta(
    const at::Tensor& query,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  auto options = query.options().dtype(at::kInt);
  return at::empty({query.size(0), 1, kSparseCount}, options);
}
}  // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl("quant_lightning_indexer_c8",
         &nanovllm_dsa_a5_impl::QuantLightningIndexerC8Npu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl("quant_lightning_indexer_c8",
         &nanovllm_dsa_a5_impl::QuantLightningIndexerC8Meta);
}
