# Ascend 950 C8 Decode Offload 算子

面向 GLM-5.1/5.2 W4A4C8，Torch namespace 为 `nanovllm_dsa`。Index cache 与 KV cache 均为 C8；block size 固定为 128，packed KV 每 token 为 656 bytes，`Q_HEAD<=64`，source capacity 不超过 `262144`。

## 非 MTP 算子

| Torch 入口 | CANN 算子 | 功能 |
| --- | --- | --- |
| `fused_li_manage_c8` | `A5FusedLiManageC8` | C8 LightningIndexer top-2048 与 request-pool 索引管理 |
| `kvcache_scatter_copy_c8` | `A5KvcacheScatterCopyC8` | packed C8 KV 的 swapped-memory DRAM → HBM 搬运 |
| `sparse_tail_attention_c8` | `A5SparseTailAttentionC8` | packed C8 KV 上的 top-2048 sparse + dense tail MLA |

```python
# B为请求数，N为index head数，Q_HEAD为本rank的attention head数，C为每请求的HBM缓存预算。
# 每个packed C8 KV token占656 bytes：512 bytes FP8 nope + 128 bytes BF16 rope（64维）+ 16 bytes FP32 scale（4个）。

torch.ops.nanovllm_dsa.fused_li_manage_c8(
    query,                   # float8_e4m3fn[B,N,128]                    , 只读, 当前decode step的index query，N=32或64
    index_weights,           # bf16[B,N]                                 , 只读, 各index head的top-k分数聚合权重
    index_key_cache,         # float8_e4m3fn[INDEX_BLOCKS,128,1,128]     , 只读, C8 index key cache，block size固定为128
    query_dequant_scale,     # fp32[B,N]                                 , 只读, query各index head的反量化scale
    key_dequant_scale,       # fp32[INDEX_BLOCKS,128,1]                  , 只读, index key每个token的反量化scale
    index_block_table,       # int32[B,INDEX_MAX_BLOCKS]                 , 只读, source逻辑token到index cache物理block的映射
    num_candidate_tokens,    # int32[B]                                  , 只读, 每个请求参与top-2048选择的prefill满块token数
    num_cache_tokens,        # int32[B]                                  , 只读, 每个请求的HBM缓存预算C；C=0表示该请求不卸载
    req_pool_entries,        # int32[B]                                  , 只读, batch行到request-pool行的映射；活跃请求之间必须唯一
    cache_slots_pool,        # int32[POOL_SIZE,SOURCE_CAPACITY]          , 读写, source token到HBM逻辑slot的持久映射，-1表示未缓存
    topk_src_ids,            # int32[B,1,2048]                           , 只写, 完整top-2048 source ID；前miss_counts个为miss，之后为hit
    topk_dst_slots,          # int32[B,1,2048]                           , 只写, top-2048对应的HBM逻辑slot；miss前缀同时作为SCATTER的dest
    miss_counts,             # int32[B]                                  , 只写, 每个请求本step需要从DRAM搬入HBM的token数
) -> None

torch.ops.nanovllm_dsa.kvcache_scatter_copy_c8(
    source_token_ids,        # int32[B,K]或int32[B,1,K]                  , 只读, DRAM source逻辑token ID；每行仅前copy_counts个有效
    destination_slots,       # int32[B,K]或int32[B,1,K]                  , 只读, HBM目标逻辑slot；与source_token_ids有效前缀一一对应
    copy_counts,             # int32[B]                                  , 只读, 每个请求的有效copy数量；0<=copy_counts[b]<=K<=8192
    hbm_block_table,         # int32[B,HBM_MAX_BLOCKS]                   , 只读, HBM逻辑slot到物理block的映射
    dram_block_table,        # int32[B,DRAM_MAX_BLOCKS]                  , 只读, DRAM source逻辑token到物理block的映射
    hbm_kv_bytes,            # int8[HBM_BLOCKS,128,1,656]                , 读写, packed C8 HBM KV byte view，搬移目标
    dram_kv_bytes,           # int8[DRAM_BLOCKS,128,1,656]               , 只读, empty_with_swapped_memory分配的packed C8 DRAM KV byte view
) -> None

torch.ops.nanovllm_dsa.sparse_tail_attention_c8(
    query,                   # bf16/fp16[B,Q_HEAD,576]                   , 只读, MLA query，最后一维为512维nope+64维rope，1<=Q_HEAD<=64
    actual_seq_lengths_query,# int32[B]                                  , 只读, TND累计query长度；非MTP decode通常为[1,2,...,B]
    actual_seq_lengths_kv,   # int32[B]                                  , 只读, 每个请求的HBM resident长度L；卸载时L=C+tail
    num_cache_tokens,        # int32[B]                                  , 只读, 每个请求的HBM缓存预算C，与L共同确定dense tail区间
    topk_dst_slots,          # int32[B,1,2048]                           , 只读, LIM输出的top-2048 HBM逻辑slot；C=0时忽略
    block_table,             # int32[B,HBM_MAX_BLOCKS]                   , 只读, HBM逻辑slot到packed KV物理block的映射
    packed_kv,               # float8_e4m3fn[HBM_BLOCKS,128,1,656]      , 只读, SCATTER更新后的packed C8 HBM KV cache
    scale_value,             # float                                     , 只读, attention score缩放系数
    attention_out,           # bf16/fp16[B,Q_HEAD,512]                   , 只写, caller-owned MLA attention结果，dtype与query相同
) -> None
```

`C=0` 时 LIM no-op，SFA 计算 `[0, actual_seq_lengths_kv)`；`C>0` 时 SFA 计算 `topk_dst_slots` 指向的 2048 个 token 与 `[C, actual_seq_lengths_kv)` tail。SCATTER 只复制 `copy_counts[b]` 个 token，不生成 attention metadata。

## MTP 算子

完整链路为 `fused_li_manage_mtp_c8 → kvcache_scatter_copy_c8 → sparse_tail_attention_c8`。MTP 只新增 LIM 算子，SCATTER 和 SFA 与非 MTP 复用。当前仅支持纯 MTP3：每个请求固定四路 query，`query=[4B,N,128]`，`actual_seq_lengths_query=[4,8,...,4B]`；不支持 MTP1/MTP2 或混合路数。

```python
# B为请求数，T=4B；同一请求的四路query共享一套cache_slots_pool映射。
# 每个packed C8 KV token仍为656 bytes；SCATTER通过int8 byte view搬移，SFA通过float8_e4m3fn view读取同一块内存。

torch.ops.nanovllm_dsa.fused_li_manage_mtp_c8(
    query,                   # float8_e4m3fn[4B,N,128]                   , 只读, packed MTP3 index query，N=32或64
    index_weights,           # bf16[T,N]                                 , 只读, 每一路query各index head的top-k分数聚合权重
    index_key_cache,         # float8_e4m3fn[INDEX_BLOCKS,128,1,128]     , 只读, 所有MTP路径共享的C8 index key cache
    query_dequant_scale,     # fp32[T,N]                                 , 只读, 每一路query各index head的反量化scale
    key_dequant_scale,       # fp32[INDEX_BLOCKS,128,1]                  , 只读, index key每个token的反量化scale
    actual_seq_lengths_query,# int32[B]                                  , 只读, 固定为[4,8,...,4B]
    index_block_table,       # int32[B,INDEX_MAX_BLOCKS]                 , 只读, source逻辑token到index cache物理block的映射
    num_candidate_tokens,    # int32[B]                                  , 只读, 每个请求参与各路top-2048选择的prefill满块token数
    num_cache_tokens,        # int32[B]                                  , 只读, 每个请求共享的HBM缓存预算C；C=0表示该请求不卸载
    req_pool_entries,        # int32[B]                                  , 只读, batch行到request-pool行的映射；活跃请求之间必须唯一
    cache_slots_pool,        # int32[POOL_SIZE,SOURCE_CAPACITY]          , 读写, 所有MTP路径共享的source token到HBM逻辑slot映射
    topk_src_ids,            # int32[T,1,2048]                           , 只写, 每路完整top-2048 source ID；miss前缀升序，之后为hit
    topk_dst_slots,          # int32[T,1,2048]                           , 只写, 与topk_src_ids逐元素对应；miss前缀为新slot，hit后缀为旧slot
    topk_miss_count,         # int32[T]                                  , 只写, 每一路query在cache更新前的miss数，也是该路miss前缀长度
    copy_src_ids,            # int32[B,8192]                             , 只写, 各路top-2048并集中的升序unique miss source ID，仅有效前缀有定义
    copy_dst_slots,          # int32[B,8192]                             , 只写, unique union miss对应的HBM逻辑slot，仅有效前缀有定义
    copy_counts,             # int32[B]                                  , 只写, 每个请求union miss有效前缀长度，最大为4*2048=8192
) -> None

torch.ops.nanovllm_dsa.kvcache_scatter_copy_c8(
    copy_src_ids,            # int32[B,8192]                             , 只读, LIM输出的unique union miss source ID
    copy_dst_slots,          # int32[B,8192]                             , 只读, LIM输出的unique union miss HBM逻辑slot
    copy_counts,             # int32[B]                                  , 只读, 每行copy metadata的有效前缀长度
    hbm_block_table,         # int32[B,HBM_MAX_BLOCKS]                   , 只读, HBM逻辑slot到物理block的映射
    dram_block_table,        # int32[B,DRAM_MAX_BLOCKS]                  , 只读, DRAM source逻辑token到物理block的映射
    hbm_kv_bytes,            # int8[HBM_BLOCKS,128,1,656]                , 读写, packed C8 HBM KV byte view；完成union miss回填
    dram_kv_bytes,           # int8[DRAM_BLOCKS,128,1,656]               , 只读, empty_with_swapped_memory分配的packed C8 DRAM KV byte view
) -> None

torch.ops.nanovllm_dsa.sparse_tail_attention_c8(
    query,                   # bf16/fp16[T,Q_HEAD,576]                   , 只读, packed MTP MLA query，最后一维为512维nope+64维rope
    actual_seq_lengths_query,# int32[B]                                  , 只读, 与LIM相同的TND累计query长度，末元素必须等于T
    actual_seq_lengths_kv,   # int32[B]                                  , 只读, 每个请求最后一路query对应的最终HBM resident长度L_b
    num_cache_tokens,        # int32[B]                                  , 只读, 每个请求共享的HBM缓存预算C_b
    topk_dst_slots,          # int32[T,1,2048]                           , 只读, LIM输出的每一路top-2048 HBM逻辑slot
    hbm_block_table,         # int32[B,HBM_MAX_BLOCKS]                   , 只读, 与SCATTER相同的HBM逻辑slot到物理block映射
    hbm_kv,                  # float8_e4m3fn[HBM_BLOCKS,128,1,656]      , 只读, SCATTER完成union miss回填后的packed C8 HBM KV
    scale_value,             # float                                     , 只读, attention score缩放系数
    attention_out,           # bf16/fp16[T,Q_HEAD,512]                   , 只写, caller-owned各路causal MLA结果，dtype与query相同
) -> None
```

MTP 直接把 `copy_src_ids/copy_dst_slots/copy_counts` 交给同一个 `kvcache_scatter_copy_c8`。对每个卸载请求，`topk_src_ids[t,0,0:2048]` 全部有效，完整表示该路从原始序列中选出的 top-2048 source token；`topk_src_ids[t,0,j]` 与 `topk_dst_slots[t,0,j]` 逐元素对应，`[0,topk_miss_count[t])` 为cache更新前的miss，后缀为hit。`copy_counts[b]` 是请求b四路miss去重并集的数量，通常小于四个 `topk_miss_count` 之和。调用 `sparse_tail_attention_c8` 时，`query/topk_dst_slots/attention_out` 第一维为 `4B`，其他 metadata 第一维为 B。算子令第 i 路计算自己的 top-2048，并只看到本步 tail 的 `0..i` 路。四路共享 slot 映射，卸载请求必须 `8192<=C<=16256` 且四路 top-2048 去重并集不超过 C。

`C=0` 时，LIM 不更新cache，并写出 `topk_src_ids/topk_dst_slots=-1`、`topk_miss_count=copy_counts=0`；SCATTER不搬运；SFA忽略top-2048 metadata，仍按每一路的因果边界计算dense attention。

## 编译

```bash
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_A5_INSTALL_OPP_PATH
unset NANOVLLM_CUST_OPAPI_LIB
unset A5_SOC_VERSION
unset SOC_VERSION
unset CANN_INSTALL_PATH
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export CANN_INSTALL_PATH=$ASCEND_HOME_PATH
source "$ASCEND_HOME_PATH/set_env.sh"
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH
export SOC_VERSION=ascend950
export NANOVLLM_A5_OPS_PYTHON=python3
export NANOVLLM_A5_OPS_BUILD_JOBS=64
bash build_c8.sh
export ASCEND_CUSTOM_OPP_PATH=$PWD/_custom_opp_c8/vendors/customize
export NANOVLLM_A5_INSTALL_OPP_PATH=$PWD/_custom_opp_c8
export NANOVLLM_CUST_OPAPI_LIB=$PWD/_custom_opp_c8/vendors/customize/op_api/lib/libcust_opapi.so
```

## 测试

每个算子脚本同时执行行为检查和 NPU Event 时延测试；时延单位为 `us`，不设置性能门槛。

```bash
python3 tests/test_fused_li_manage_c8.py --device npu:0 --heads 32,64 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --miss-ranges 0:300 --warmup 3 --iters 20 --seed 7
python3 tests/test_fused_li_manage_mtp_c8.py --device npu:0 --batch-size 24 --heads 32 --source-len 65536 --cache-tokens 12288 --per-query-miss-count 100 --union-miss-count 300 --query-noise 0.25 --pool-extra 7 --warmup 10 --iters 300 --seed 7
python3 tests/test_kvcache_scatter_copy_c8.py --device npu:0 --batch-size 24 --source-len 20096 --hbm-slots 6144 --copy-min 0 --copy-max 300 --warmup 3 --iters 20 --seed 7
python3 tests/test_sparse_tail_attention_c8.py --device npu:0 --heads 8 --batch-sizes 24 --cache-tokens 6144 --tail-tokens 64 --warmup 3 --iters 20 --seed 7
python3 tests/test_offload_split_c8_graph.py --device npu:0 --replays 4 --seed 7
```

## 官方 C8 LightningIndexer 基线

Ascend950DT_9581，`N=32`，`query 数=MTP+1`，TND sparse3 causal，warmup=10、iters=300；表中为平均时延，单位为 `μs`。

| MTP | query 数 | 序列长度 | B=1 | B=4 | B=8 | B=16 | B=32 | B=48 | B=64 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 65536 | 85.199 | 85.746 | 105.440 | 129.904 | 149.288 | 274.471 | 292.197 |
| 0 | 1 | 131072 | 165.409 | 230.587 | 245.357 | 257.045 | 291.268 | 543.662 | 586.258 |
| 1 | 2 | 65536 | 91.088 | 95.229 | 111.730 | 131.382 | 153.127 | 279.188 | 299.951 |
| 1 | 2 | 131072 | 184.377 | 227.989 | 246.875 | 259.770 | 299.286 | 553.373 | 624.786 |
| 2 | 3 | 65536 | 95.183 | 95.343 | 115.257 | 130.794 | 170.281 | 313.361 | 336.395 |
| 2 | 3 | 131072 | 184.382 | 220.806 | 248.706 | 265.877 | 337.552 | 637.909 | 767.496 |
| 3 | 4 | 65536 | 94.654 | 95.411 | 111.171 | 130.942 | 172.377 | 318.060 | 346.879 |
| 3 | 4 | 131072 | 184.696 | 224.129 | 252.878 | 267.331 | 348.823 | 652.652 | 785.738 |
| 4 | 5 | 65536 | 94.967 | 95.396 | 103.042 | 152.967 | 313.950 | 493.445 | 674.760 |
| 4 | 5 | 131072 | 184.177 | 234.873 | 257.411 | 299.599 | 665.931 | 1085.260 | 1427.117 |

## LIM-MTP3 时延

Ascend950DT_9581，`N=32`，**序列长度 65536**，`C=12288`，每路 miss=100，union miss=300，warmup=10、iters=300；单位为 `μs`。

| Batch size | 官方 LI C8（MTP3） | LIM-MTP3 | 额外时延 |
| ---: | ---: | ---: | ---: |
| 1 | 99.855 | 176.355 | 76.500 |
| 4 | 97.196 | 186.975 | 89.779 |
| 8 | 115.018 | 184.493 | 69.475 |
| 16 | 130.612 | 224.184 | 93.572 |
| 24 | 157.219 | 245.880 | 88.661 |
| 32 | 172.632 | 260.648 | 88.016 |

Ascend950DT_9581，`N=32`，**序列长度 131072**，`C=12288`，每路 miss=100，union miss=300，warmup=10、iters=300；单位为 `μs`。

| Batch size | 官方 LI C8（MTP3） | LIM-MTP3 | 额外时延 |
| ---: | ---: | ---: | ---: |
| 1 | 184.164 | 272.481 | 88.317 |
| 4 | 217.230 | 283.793 | 66.563 |
| 8 | 252.850 | 354.852 | 102.002 |
| 16 | 267.630 | 370.602 | 102.972 |
| 24 | 312.805 | 408.434 | 95.629 |
| 32 | 349.555 | 479.857 | 130.302 |
