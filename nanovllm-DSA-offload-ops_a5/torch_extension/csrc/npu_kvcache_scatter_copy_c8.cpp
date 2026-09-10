#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {
namespace {

constexpr int64_t kMaxC8CopyCapacity = 8192;

int64_t CopyMetadataCapacity(const at::Tensor& tensor) {
  if (tensor.dim() == 2) {
    return tensor.size(1);
  }
  TORCH_CHECK(
      tensor.dim() == 3 && tensor.size(1) == 1,
      "C8 SCATTER source/destination metadata must be [B,K] or [B,1,K].");
  return tensor.size(2);
}

void CheckPackedScatterInputs(
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& hbm_kv_bytes,
    const at::Tensor& dram_kv_bytes) {
  TORCH_CHECK(
      hbm_kv_bytes.dim() == 4 && hbm_kv_bytes.size(0) > 0 &&
          hbm_kv_bytes.size(1) == kBlockSize &&
          hbm_kv_bytes.size(2) == 1 &&
          hbm_kv_bytes.size(3) == kPackedKvDim &&
          dram_kv_bytes.dim() == 4 && dram_kv_bytes.size(0) > 0 &&
          dram_kv_bytes.size(1) == kBlockSize &&
          dram_kv_bytes.size(2) == 1 &&
          dram_kv_bytes.size(3) == kPackedKvDim,
      "C8 SCATTER KV byte views must be [blocks,128,1,656].");
  TORCH_CHECK(
      hbm_kv_bytes.scalar_type() == at::kChar &&
          dram_kv_bytes.scalar_type() == at::kChar,
      "C8 SCATTER expects int8 byte views of packed C8 KV caches.");
  TORCH_CHECK(
      copy_counts.dim() == 1 && copy_counts.size(0) > 0,
      "C8 SCATTER copy_counts must be int32[B].");
  const int64_t batch = copy_counts.size(0);
  const int64_t copy_capacity = CopyMetadataCapacity(source_token_ids);
  TORCH_CHECK(
      source_token_ids.size(0) == batch &&
          destination_slots.size(0) == batch &&
          source_token_ids.sizes() == destination_slots.sizes() &&
          CopyMetadataCapacity(destination_slots) == copy_capacity &&
          copy_capacity > 0 && copy_capacity <= kMaxC8CopyCapacity &&
          hbm_block_table.dim() == 2 &&
          hbm_block_table.size(0) == batch &&
          hbm_block_table.size(1) > 0 &&
          dram_block_table.dim() == 2 &&
          dram_block_table.size(0) == batch &&
          dram_block_table.size(1) > 0 &&
          dram_block_table.size(1) * kBlockSize <= kMaxSourceCapacity,
      "C8 SCATTER metadata shapes are inconsistent.");
  for (const at::Tensor* tensor : {
           &source_token_ids, &destination_slots, &copy_counts,
           &hbm_block_table, &dram_block_table}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt,
        "C8 SCATTER metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      hbm_kv_bytes,
      {&source_token_ids, &destination_slots, &copy_counts,
       &hbm_block_table, &dram_block_table, &hbm_kv_bytes,
       &dram_kv_bytes},
      "C8 SCATTER");
}

}  // namespace

void KvcacheScatterCopyC8Npu(
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    at::Tensor hbm_kv_bytes,
    const at::Tensor& dram_kv_bytes) {
  CheckPackedScatterInputs(
      source_token_ids, destination_slots, copy_counts,
      hbm_block_table, dram_block_table, hbm_kv_bytes, dram_kv_bytes);
  auto keepalive = std::make_tuple(
      source_token_ids, destination_slots, copy_counts,
      hbm_block_table, dram_block_table, hbm_kv_bytes, dram_kv_bytes);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5KvcacheScatterCopyC8,
      keepalive,
      source_token_ids,
      destination_slots,
      copy_counts,
      hbm_block_table,
      dram_block_table,
      hbm_kv_bytes,
      dram_kv_bytes,
      hbm_kv_bytes);
}

void KvcacheScatterCopyC8Meta(
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor,
    const at::Tensor&) {
  TORCH_CHECK(
      copy_counts.dim() == 1 && copy_counts.size(0) > 0 &&
          source_token_ids.size(0) == copy_counts.size(0) &&
          source_token_ids.sizes() == destination_slots.sizes(),
      "C8 SCATTER metadata shapes are inconsistent.");
  const int64_t capacity = CopyMetadataCapacity(source_token_ids);
  TORCH_CHECK(
      capacity > 0 && capacity <= kMaxC8CopyCapacity,
      "C8 SCATTER copy capacity must be in [1,8192].");
}

}  // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl(
      "kvcache_scatter_copy_c8",
      &nanovllm_dsa_a5_impl::KvcacheScatterCopyC8Npu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl(
      "kvcache_scatter_copy_c8",
      &nanovllm_dsa_a5_impl::KvcacheScatterCopyC8Meta);
}
