#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY(nanovllm_dsa, m) {
  m.def(
      "fused_li_manage(Tensor query, Tensor key, Tensor weights, "
      "Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
      "Tensor cache_tokens, Tensor candidate_lens, Tensor block_table) "
      "-> (Tensor, Tensor, Tensor, Tensor(a!))");
  m.def(
      "fused_li_manage_out(Tensor query, Tensor key, Tensor weights, "
      "Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
      "Tensor cache_tokens, Tensor candidate_lens, Tensor block_table, "
      "Tensor(b!) source_ids, Tensor(c!) destination_slots, "
      "Tensor(d!) miss_counts) "
      "-> (Tensor(b!), Tensor(c!), Tensor(d!), Tensor(a!))");
  m.def(
      "fused_li_manage_mtp(Tensor query, Tensor key, Tensor weights, "
      "Tensor(a!) cache_slots, Tensor actual_seq_lengths_query, "
      "Tensor actual_seq_lengths_key, Tensor block_table) "
      "-> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "fused_li_manage_c8("
      "Tensor query, Tensor index_weights, Tensor index_key_cache, "
      "Tensor query_dequant_scale, Tensor key_dequant_scale, "
      "Tensor index_block_table, Tensor num_candidate_tokens, "
      "Tensor num_cache_tokens, Tensor req_pool_entries, "
      "Tensor(a!) cache_slots_pool, Tensor(b!) topk_src_ids, "
      "Tensor(c!) topk_dst_slots, Tensor(d!) miss_counts) -> ()");
  m.def(
      "fused_li_manage_mtp_c8("
      "Tensor query, Tensor index_weights, Tensor index_key_cache, "
      "Tensor query_dequant_scale, Tensor key_dequant_scale, "
      "Tensor actual_seq_lengths_query, Tensor index_block_table, "
      "Tensor num_candidate_tokens, Tensor num_cache_tokens, "
      "Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
      "Tensor(b!) topk_src_ids, Tensor(c!) topk_dst_slots, "
      "Tensor(d!) topk_miss_count, Tensor(e!) copy_src_ids, "
      "Tensor(f!) copy_dst_slots, Tensor(g!) copy_counts) -> ()");
  m.def(
      "kvcache_scatter_copy(Tensor(a!) hbm_kpe, Tensor(b!) hbm_ckv, "
      "Tensor dram_kpe, Tensor dram_ckv, Tensor hbm_block_table, "
      "Tensor dram_block_table, Tensor source_token_ids, "
      "Tensor destination_slots, Tensor copy_counts) "
      "-> (Tensor(a!), Tensor(b!))");
  m.def(
      "kvcache_scatter_copy_c8("
      "Tensor source_token_ids, Tensor destination_slots, "
      "Tensor copy_counts, Tensor hbm_block_table, "
      "Tensor dram_block_table, Tensor(a!) hbm_kv_bytes, "
      "Tensor dram_kv_bytes) -> ()");
  m.def(
      "sparse_tail_attention(Tensor query, Tensor key, Tensor value, "
      "Tensor sparse_slots, Tensor cache_tokens, Tensor block_table, "
      "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv, "
      "Tensor query_rope, Tensor key_rope, float scale_value) -> Tensor");
  m.def(
      "sparse_tail_attention_c8("
      "Tensor query, Tensor actual_seq_lengths_query, "
      "Tensor actual_seq_lengths_kv, Tensor num_cache_tokens, "
      "Tensor topk_dst_slots, Tensor block_table, Tensor packed_kv, "
      "float scale_value, Tensor(a!) attention_out) -> ()");
  m.def(
      "fused_copy_sparse_tail_attention("
      "Tensor query, Tensor(a!) hbm_ckv, Tensor sparse_slots, "
      "Tensor cache_tokens, Tensor hbm_block_table, "
      "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv, "
      "Tensor query_rope, Tensor(b!) hbm_kpe, Tensor dram_kpe, "
      "Tensor dram_ckv, Tensor dram_block_table, "
      "Tensor source_token_ids, Tensor copy_counts, float scale_value, "
      "int prefetch_rows_per_step=5) "
      "-> (Tensor, Tensor(b!), Tensor(a!))");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
