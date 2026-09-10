#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {
namespace {

void CheckFusedLiManageC8Inputs(
    const at::Tensor& query,
    const at::Tensor& index_weights,
    const at::Tensor& index_key_cache,
    const at::Tensor& query_dequant_scale,
    const at::Tensor& key_dequant_scale,
    const at::Tensor& index_block_table,
    const at::Tensor& num_candidate_tokens,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& req_pool_entries,
    const at::Tensor& cache_slots_pool,
    const at::Tensor& topk_src_ids,
    const at::Tensor& topk_dst_slots,
    const at::Tensor& miss_counts) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 &&
          (query.size(1) == 32 || query.size(1) == 64) &&
          query.size(2) == kIndexerDim &&
          query.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "C8 LIM query must be float8_e4m3fn [B,32|64,128].");
  const int64_t batch = query.size(0);
  const int64_t heads = query.size(1);
  TORCH_CHECK(
      index_key_cache.dim() == 4 && index_key_cache.size(0) > 0 &&
          index_key_cache.size(1) == kBlockSize &&
          index_key_cache.size(2) == 1 &&
          index_key_cache.size(3) == kIndexerDim &&
          index_key_cache.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "C8 LIM index_key_cache must be float8_e4m3fn [blocks,128,1,128].");
  TORCH_CHECK(
      index_weights.dim() == 2 && index_weights.size(0) == batch &&
          index_weights.size(1) == heads &&
          index_weights.scalar_type() == at::kBFloat16 &&
          query_dequant_scale.dim() == 2 &&
          query_dequant_scale.size(0) == batch &&
          query_dequant_scale.size(1) == heads &&
          query_dequant_scale.scalar_type() == at::kFloat,
      "C8 LIM index_weights/query_dequant_scale must be BF16/FP32 [B,N].");
  TORCH_CHECK(
      key_dequant_scale.dim() == 3 &&
          key_dequant_scale.size(0) == index_key_cache.size(0) &&
          key_dequant_scale.size(1) == kBlockSize &&
          key_dequant_scale.size(2) == 1 &&
          key_dequant_scale.scalar_type() == at::kFloat,
      "C8 LIM key_dequant_scale must be FP32 [blocks,128,1].");
  TORCH_CHECK(
      num_candidate_tokens.dim() == 1 &&
          num_candidate_tokens.size(0) == batch &&
          num_cache_tokens.dim() == 1 &&
          num_cache_tokens.size(0) == batch &&
          req_pool_entries.dim() == 1 && req_pool_entries.size(0) == batch,
      "C8 LIM request metadata must be int32[B].");
  TORCH_CHECK(
      cache_slots_pool.dim() == 2 && cache_slots_pool.size(0) > 0 &&
          cache_slots_pool.size(1) > 0 &&
          cache_slots_pool.size(1) <= kMaxSourceCapacity,
      "C8 LIM cache_slots_pool must be [pool_size,capacity], capacity <= 2^18.");
  TORCH_CHECK(
      index_block_table.dim() == 2 &&
          index_block_table.size(0) == batch &&
          index_block_table.size(1) > 0 &&
          index_block_table.size(1) * kBlockSize == cache_slots_pool.size(1),
      "C8 LIM index_block_table capacity must equal cache_slots_pool.shape[1].");
  CheckLiduOutputs(query, topk_src_ids, topk_dst_slots, miss_counts);
  for (const at::Tensor* tensor : {
           &index_block_table, &num_candidate_tokens, &num_cache_tokens,
           &req_pool_entries, &cache_slots_pool, &topk_src_ids,
           &topk_dst_slots, &miss_counts}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt,
        "C8 LIM metadata/state/output tensors must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &index_weights, &index_key_cache, &query_dequant_scale,
       &key_dequant_scale, &index_block_table, &num_candidate_tokens,
       &num_cache_tokens, &req_pool_entries, &cache_slots_pool,
       &topk_src_ids, &topk_dst_slots, &miss_counts},
      "C8 LIM");
}

}  // namespace

void FusedLiManageC8Npu(
    const at::Tensor& query,
    const at::Tensor& index_weights,
    const at::Tensor& index_key_cache,
    const at::Tensor& query_dequant_scale,
    const at::Tensor& key_dequant_scale,
    const at::Tensor& index_block_table,
    const at::Tensor& num_candidate_tokens,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    at::Tensor topk_src_ids,
    at::Tensor topk_dst_slots,
    at::Tensor miss_counts) {
  CheckFusedLiManageC8Inputs(
      query, index_weights, index_key_cache, query_dequant_scale,
      key_dequant_scale, index_block_table, num_candidate_tokens,
      num_cache_tokens, req_pool_entries, cache_slots_pool,
      topk_src_ids, topk_dst_slots, miss_counts);
  auto keepalive = std::make_tuple(
      query, index_weights, index_key_cache, query_dequant_scale,
      key_dequant_scale, index_block_table, num_candidate_tokens,
      num_cache_tokens, req_pool_entries, cache_slots_pool,
      topk_src_ids, topk_dst_slots, miss_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5FusedLiManageC8,
      keepalive,
      query,
      index_key_cache,
      index_weights,
      query_dequant_scale,
      key_dequant_scale,
      req_pool_entries,
      cache_slots_pool,
      num_cache_tokens,
      num_candidate_tokens,
      index_block_table,
      topk_src_ids,
      topk_dst_slots,
      miss_counts,
      cache_slots_pool);
}

void FusedLiManageC8Meta(
    const at::Tensor& query,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor,
    at::Tensor topk_src_ids,
    at::Tensor topk_dst_slots,
    at::Tensor miss_counts) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0,
      "C8 LIM query must be rank 3 with a positive batch.");
  CheckLiduOutputs(query, topk_src_ids, topk_dst_slots, miss_counts);
}

}  // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl("fused_li_manage_c8", &nanovllm_dsa_a5_impl::FusedLiManageC8Npu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl("fused_li_manage_c8", &nanovllm_dsa_a5_impl::FusedLiManageC8Meta);
}
