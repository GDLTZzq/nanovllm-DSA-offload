#include <limits>
#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {
namespace {

constexpr int64_t kC8QueryDim = kCkvDim + kKpeDim;

void CheckSparseTailAttentionC8Inputs(
    const at::Tensor& query,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& topk_dst_slots,
    const at::Tensor& block_table,
    const at::Tensor& packed_kv,
    const at::Tensor& attention_out) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 &&
          query.size(1) >= 1 && query.size(1) <= 64 &&
          query.size(2) == kC8QueryDim,
      "C8 sparse+tail attention query must be [T,Q_HEAD,576] "
      "with 1 <= Q_HEAD <= 64.");
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16 ||
          query.scalar_type() == at::kHalf,
      "C8 sparse+tail attention query must be BF16 or FP16.");
  TORCH_CHECK(
      packed_kv.dim() == 4 && packed_kv.size(0) > 0 &&
          packed_kv.size(1) == kBlockSize && packed_kv.size(2) == 1 &&
          packed_kv.size(3) == kPackedKvDim && packed_kv.element_size() == 1 &&
          packed_kv.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "C8 sparse+tail attention packed KV must be float8_e4m3fn "
      "[blocks,128,1,656].");
  const int64_t packed_queries = query.size(0);
  TORCH_CHECK(
      actual_seq_lengths_query.dim() == 1 &&
          actual_seq_lengths_query.numel() > 0,
      "C8 sparse+tail attention actual_seq_lengths_query must be "
      "cumulative int32[B].");
  const int64_t batch = actual_seq_lengths_query.numel();
  // The same TND kernel handles non-MTP (T == B) and packed MTP
  // (B <= T <= 4B). Its causal mask exposes only the tail tokens up to the
  // current packed query, while each query row uses its own top-2048 slots.
  TORCH_CHECK(
      packed_queries >= batch && packed_queries <= batch * 4 &&
          actual_seq_lengths_kv.dim() == 1 &&
          actual_seq_lengths_kv.numel() == batch &&
          num_cache_tokens.dim() == 1 &&
          num_cache_tokens.numel() == batch &&
          topk_dst_slots.dim() == 3 &&
          topk_dst_slots.size(0) == packed_queries &&
          topk_dst_slots.size(1) == 1 &&
          topk_dst_slots.size(2) == kSparseCount &&
          block_table.dim() == 2 && block_table.size(0) == batch &&
          block_table.size(1) > 0,
      "C8 sparse+tail attention metadata shapes are inconsistent.");
  TORCH_CHECK(
      attention_out.dim() == 3 &&
          attention_out.size(0) == packed_queries &&
          attention_out.size(1) == query.size(1) &&
          attention_out.size(2) == kCkvDim &&
          attention_out.scalar_type() == query.scalar_type(),
      "C8 sparse+tail attention output must be [T,Q_HEAD,512] "
      "with the query dtype.");
  for (const at::Tensor* tensor : {
           &actual_seq_lengths_query, &actual_seq_lengths_kv,
           &num_cache_tokens, &topk_dst_slots, &block_table}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt,
        "C8 sparse+tail attention metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &actual_seq_lengths_query, &actual_seq_lengths_kv,
       &num_cache_tokens, &topk_dst_slots, &block_table, &packed_kv,
       &attention_out},
      "C8 sparse+tail attention");
}

}  // namespace

void SparseTailAttentionC8Npu(
    const at::Tensor& query,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& topk_dst_slots,
    const at::Tensor& block_table,
    const at::Tensor& packed_kv,
    double scale_value,
    at::Tensor attention_out) {
  CheckSparseTailAttentionC8Inputs(
      query, actual_seq_lengths_query, actual_seq_lengths_kv,
      num_cache_tokens, topk_dst_slots, block_table, packed_kv,
      attention_out);

  // The local op never returns LSE. Stable one-element placeholders avoid
  // CANN-version-dependent zero-size output descriptor behavior.
  auto softmax_max = at::empty({1}, query.options().dtype(at::kFloat));
  auto softmax_sum = at::empty({1}, query.options().dtype(at::kFloat));
  const c10::optional<at::Tensor> no_external_scale = c10::nullopt;
  constexpr int64_t key_quant_mode = 2;
  constexpr int64_t value_quant_mode = 2;
  constexpr int64_t sparse_block_size = 1;
  std::string layout_query = "TND";
  std::string layout_kv = "PA_BSND";
  char* layout_query_ptr = layout_query.data();
  char* layout_kv_ptr = layout_kv.data();
  constexpr int64_t sparse_mode = 3;
  constexpr int64_t all_tokens = std::numeric_limits<int64_t>::max();
  constexpr int64_t attention_mode = 2;
  constexpr int64_t quant_scale_repo_mode = 1;
  constexpr int64_t tile_size = 128;
  constexpr int64_t rope_head_dim = kKpeDim;
  constexpr bool return_softmax_lse = false;

  auto keepalive = std::make_tuple(
      query, actual_seq_lengths_query, actual_seq_lengths_kv,
      num_cache_tokens, topk_dst_slots, block_table, packed_kv,
      attention_out, softmax_max, softmax_sum);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5SparseTailAttentionC8,
      keepalive,
      query,
      packed_kv,
      packed_kv,
      topk_dst_slots,
      no_external_scale,
      no_external_scale,
      block_table,
      actual_seq_lengths_query,
      actual_seq_lengths_kv,
      num_cache_tokens,
      scale_value,
      key_quant_mode,
      value_quant_mode,
      sparse_block_size,
      layout_query_ptr,
      layout_kv_ptr,
      sparse_mode,
      all_tokens,
      all_tokens,
      attention_mode,
      quant_scale_repo_mode,
      tile_size,
      rope_head_dim,
      return_softmax_lse,
      attention_out,
      softmax_max,
      softmax_sum);
}

void SparseTailAttentionC8Meta(
    const at::Tensor& query,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double,
    at::Tensor attention_out) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 &&
          query.size(1) >= 1 && query.size(1) <= 64 &&
          query.size(2) == kC8QueryDim,
      "C8 sparse+tail attention query must be [T,Q_HEAD,576].");
  TORCH_CHECK(
      attention_out.dim() == 3 && attention_out.size(0) == query.size(0) &&
          attention_out.size(1) == query.size(1) &&
          attention_out.size(2) == kCkvDim,
      "C8 sparse+tail attention output must be [B,Q_HEAD,512].");
}

}  // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl(
      "sparse_tail_attention_c8",
      &nanovllm_dsa_a5_impl::SparseTailAttentionC8Npu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl(
      "sparse_tail_attention_c8",
      &nanovllm_dsa_a5_impl::SparseTailAttentionC8Meta);
}
