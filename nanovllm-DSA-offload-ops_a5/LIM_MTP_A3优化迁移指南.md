# A3 LIM-MTP 优化手段与 A5 C8 迁移指南

本文面向后续负责优化 Ascend 950 C8 `fused_li_manage_mtp_c8` 的 code agent。目标不是复制 A3 源码，而是复用 A3 已验证的算法分解、数据组织、同步方法和测试方法，再结合 A5 的 FP8 C8 QLI、Arch35 MicroAPI 与现有非 MTP C8 LIM 实现重新落地。

## 1. 参考代码与边界

### A3 已验证实现

- 仓库：`D:\vLLM-ascend\ops_lim_mtp`
- 核心功能完成后的生产版本：commit `36cd61e`
- 主要文件：
  - `csrc/nanovllm_ascend_ops/ops/fused_li_manage_mtp/op_kernel/lightning_indexer_service_vector.h`
  - `csrc/nanovllm_ascend_ops/ops/fused_li_manage_mtp/op_kernel/fused_li_manage_mtp_union.h`
  - `csrc/nanovllm_ascend_ops/ops/fused_li_manage_mtp/op_kernel/fused_li_manage_mtp_workspace.h`
  - `csrc/nanovllm_ascend_ops/ops/fused_li_manage_mtp/op_kernel/nanovllm_fused_li_manage_mtp.cpp`
  - `ut_ops/test_fused_li_manage_mtp.py`

### A5 C8 当前初版

- 仓库：`D:\vLLM-ascend\ops_a5`
- 目标目录：`csrc/src/fused_li_manage_mtp_c8`
- 当前实现只应视为 ABI 与行为参考，不应继续在其慢速算法上小修小补。
- A5 已有的成熟非 MTP C8 LIM 也应作为直接参考：
  - `csrc/src/fused_li_manage_c8`
  - 特别是 `arch35_payload_c8` 下的 payload TopK、VF 与 eviction 实现。

### 不可直接照搬的差异

| 项目 | A3 LIM-MTP | A5 C8 LIM-MTP |
| --- | --- | --- |
| Q/K | BF16/FP16 | FP8 E4M3FN |
| scale | 无独立反量化 scale | query/key FP32 dequant scale |
| TopK 实现 | A3 LightningIndexer，float score | A5 Quant LightningIndexer，原生 `uint16` sortable score/key 路径 |
| query 数 | 固定每请求 4 路 | 当前 ABI 支持每请求 2～4 路，TND packed query |
| query 语义 | 四路搜索同一 immutable prefill source，性能基线用 `sparse_mode=0` | 必须保持当前 A5 official C8 golden 与 causal/TND 语义，不得擅自改成 A3 语义 |
| Vector API | A3/C220 AscendC API | A5/Arch35 MicroAPI，已有 `Squeeze` 等更适合的原语 |
| 公开输出 | 还包含 `topk_src_ids` | 当前 C8 MTP ABI 只输出 `topk_dst_slots` 和 union copy metadata |

因此，迁移原则是：复制算法和数据流，不复制 BF16 Cube/Vector 内核，不强行复制 A3 的固定四路布局，不改变当前 A5 算子 ABI 与语义。

## 2. A3 最终算法的核心思路

A3 最终实现可概括为：

```text
批量并行 LI
  └─ TopK 过程中携带 {source_id, old_cache_slot}
      └─ 每路 Top2048 重排为 [miss prefix | hit suffix]
          └─ 四路 miss prefix 做 MrgSort + 去重，得到 union miss
              └─ 分块扫描完整 source，向量化筛选并排序 eviction candidates
                  └─ O(union miss) 更新 cache mapping
                      └─ 只修补四路 miss prefix 的 dst slots
                          └─ 批量发布最终输出
```

最重要的性能原则有五条：

1. 不把 MTP3 拆成四次串行 LI，保留官方 LI 的批量 Q×K、Cube/Vector 和 S2 并行策略。
2. cache hit/miss 信息在 TopK payload 中一路携带，避免 TopK 结束后再对 2048 项做随机 GM 查询。
3. 四路 miss 先排序，再用 merge 处理；不用 scalar hash 表遍历全部 8192 个 TopK 项。
4. steady workload 的 scalar 工作量控制在 `O(union_miss + 四路 miss 总数)`，典型约几百项，而不是 `O(4×2048)` 或 `O(source_len)`。
5. 跨 AIV 的 score、threshold 和 pair 数据必须通过 DMA 发布与显式同步，不依赖 scalar GM cache 的偶然可见性。

## 3. 阶段一：LI 内携带 cache payload，并重排每路 TopK

### 3.1 保留官方 LI 并行骨架

A3 重构成功的第一步不是优化 cache manager，而是先替换旧的慢 LI 主干：

- 将 `[B*4,H,128]` 的 query 作为一个批量任务进入官方 LI 派生的调度。
- Q×K 仍由 Cube 批量执行；Vector 仍按 S2 chunk 归约和做 TopK。
- 四路 query 不在 Host 或 kernel 内按请求串行调用四次 LI。
- 后处理阶段再令一个 request owner 管理同一请求的四路结果。

A3 在 `B=24, source=64K` 时，阶段一相对官方 LI-MTP3 只增加约 `34.4μs`。这说明“在 LI 内携带 cache 信息”本身可以很便宜；如果 A5 阶段一就比 official C8 LI 慢很多，优先检查是否破坏了原生 QLI 的并行调度，而不是继续优化 union。

### 3.2 payload 编码

A3 source capacity 不超过 `2^18`，cache slot 使用 14 bit，因此一个 `uint32` 可编码：

```text
payload = (slot14 << 18) | source_id18
```

- `source_id`：原始 source token ID。
- `slot`：更新前的 HBM logical slot。
- 未缓存的 `slot=-1` 在无符号打包后解码为 `0x3fff`，可作为 invalid slot。
- A3 用同一个 payload 同时支持 TopK survivor 携带 source 和 hit/miss 状态。

A5 C8 当前同样限制 source capacity `<=2^18`，且 cache budget 小于 14-bit sentinel，因此该编码可以继续使用。优先复用 A5 非 MTP `arch35_payload_c8` 已有的 payload codec，不要另写一套未经验证的编码。

### 3.3 连续读取 cache state

在每个 S2 chunk 进入 TopK 前：

- 根据 `req_pool_entries[b]` 找到持久化 pool row。
- 对 `cache_slots_pool[row, source_base:source_base+valid]` 做连续 DMA。
- 将 slot 与连续 source ID 合成 payload。
- score 与 payload 一起进入 Sort/TopK survivor 流。

这样 TopK 结束时，2048 个 survivor 已经知道自己的旧 slot，不需要再做 2048 次 scalar GM lookup。

### 3.4 每路 TopK 的确定性重排

每路 Top2048 在最终输出前按复合 key 重排：

1. miss 在前，hit 在后；
2. miss prefix 按 source ID 升序；
3. hit 保留旧 slot。

这个重排同时产生三类内部信息：

- miss prefix 长度；
- 按 source ID 有序的 miss rows，供后续四路 merge；
- hit 的最终 destination slot，不必由 request owner 重新读取 cache map。

A3 将四路 `{sort_key,payload}` pair 分成 `pair0/pair1` 两块 workspace，每块容纳两路，后续 request owner 一次搬入 UB。

A5 公开 ABI 没有 `topk_src_ids`，但仍建议在私有 workspace 中保留每路有序 source/payload row。不能因为 ABI 不输出 source IDs，就让后处理重新遍历公开 TopK 或随机查询 cache state。

### 3.5 score 与 threshold 的发布

淘汰阶段需要四路完整 score 和四个 Top2048 threshold。A3 的做法是：

- 在 reduction result 尚未被 TopK 复用前，将每路每个 S2 chunk 的 score DMA 到 workspace。
- score row stride 按 512 token 对齐。
- 每路 threshold 独占 8 个 float 的对齐区域，只使用第一个值。
- threshold 使用 MTE3 写出；request owner 在全局 barrier 后使用 MTE2 搬入 UB，再由 scalar 读取。

不要用 producer AIV 的 scalar `SetValue` 写 threshold、另一个 AIV 的 scalar `GetValue` 读 threshold。A3 实测这种写法会遇到 scalar cache 可见性问题，即使中间有 `SyncAll`。

## 4. 阶段二：四路有序 merge 与 union miss 去重

### 4.1 不使用 hash union

A3 没有为四路 TopK 建 8192 项 scalar hash 表。因为阶段一已经把每路 miss prefix 按 source ID 排好序，union 可直接做：

```text
merged = MrgSort(route0_misses, route1_misses,
                 route2_misses, route3_misses)
union_misses = unique(merged)
```

每路 miss 长度通过对重排后的 pair row 做二分查找得到，无需扫描完整 2048 项。

### 4.2 A3 当前的向量去重

A3 最初用 scalar 单指针遍历 merged row，功能正确但在 B=24 时仍有明显开销。当前实现改成：

1. 向量提取 merged pair 的 source sort key；
2. 用 `Gather` 构造右移一项的 predecessor row；
3. `Compare(NE)` 得到 unique mask；
4. `GatherMask` 压紧 first occurrences；
5. 向量解码成升序 source IDs；
6. 只有异常 count 才进入 scalar fallback。

A5 不应机械使用 A3 的 `GatherMask` 写法。Arch35 已有 `MicroAPI::Squeeze`，优先用 register mask + Squeeze 完成压紧。但需要保留以下已验证的数据流：

```text
sorted merged keys
  -> compare with predecessor
  -> compact first occurrence
  -> decode source IDs
```

### 4.3 向量去重的三个同步陷阱

A3 为这个看似简单的去重路径付出了较多调试成本，A5 实现时必须主动规避：

1. Vector 指令按 64 个 B32 lane 执行。尾部对齐 lane 必须让 `key == predecessor`，否则脏 UB 会被错误计为 unique。
2. `GatherMask/Squeeze` 产生的 count 在 Vector pipe 中完成。scalar 读取 count 前必须有明确的 `V→S` 同步；`PipeBarrier<PIPE_V>` 不能替代跨 pipe 同步。
3. predecessor 的 lane 0 必须单独改成与 key0 不同。A3 曾用向量构造 `[1,0,0,...]`，在 C220 上暴露 lane 1 数据依赖问题；最终只用 scalar 修改 predecessor[0]，并做 `S→V` 同步。

A5 MicroAPI 的具体同步原语可能不同，但语义必须逐项满足。

### 4.4 2～4 路适配

A3 固定四路。A5 需要根据 `actual_seq_lengths_query` 的累计差分得到每请求 `Q_b=2..4`：

- `MrgSort` 的 valid bits 和 element lengths 只启用实际路数。
- workspace 的 route offset 必须按 packed query row，而不是 `batch*4+route` 写死。
- union capacity 仍为 8192，但实际上界为 `Q_b*2048`。
- 缺失 route 不参与 score aggregate、threshold protection 或 output publication。

## 5. 阶段三：向量化寻找 eviction candidates

### 5.1 为什么要保存四路 score

只知道 union TopK token 并不足以高效选择 victim。A3 保存四路完整 score，使每个 source chunk 可以在向量路径上同时完成：

- 判断 token 是否已缓存；
- 判断 token 是否属于任一路 TopK；
- 计算 eviction priority；
- 将合法 candidate 排序。

这避免了对每个缓存 token 做 scalar hash lookup 或四路二分查询。

### 5.2 512-token chunk

A3 每次扫描 512 个 source token：

- 连续 DMA 四路 score；
- 连续 DMA cache slots；
- 向量构造 protection mask 和 eviction key；
- `Sort32` 后分层 `MrgSort`，得到一个完整的 512 项有序 candidate list。

扫描起点使用 `(candidate_len, pool_row)` 的确定性 hash 选择 chunk，并循环覆盖整个 source。这样不会长期只淘汰低 source ID 区域，也便于 graph replay 保持确定性。

### 5.3 protection 与排序 key

A3 已确认的规则是：

```text
invalid_key = -1e20
stop_key    = -5e19

if token 未缓存，或属于任一路 Top2048:
    evict_key = invalid_key
else:
    evict_key = -(score0 + score1 + score2 + score3)
```

排序按 key 降序，因此四路聚合分数越低，越优先淘汰。

TopK protection 不应只做粗糙的 source 范围判断。A3 使用：

```text
margin = max(score_route - threshold_route)
margin >= 0  => protected
```

threshold tie 也先保守地保护，保证绝不淘汰真实 TopK token。如果保守 tie 保护导致候选不足，再走低频 exact fallback，对 tie token 查询四路有序 TopK row。

### 5.4 停止条件

典型 steady workload 的 union miss 约 300，小于一个 512-token chunk 的 candidate capacity。A3 的快路径维护跨 chunk 的 Top512 accumulator：

```text
for chunk in deterministic_scan_order:
    sorted_chunk = build_and_sort_512_candidates(chunk)
    accumulator = merge_top512(accumulator, sorted_chunk)
    if accumulator[union_miss_count - 1].key > -5e19:
        stop
```

不再额外做 `ValidCandidateCount()` 的完整 scalar 扫描。第 K 个 key 已有效，就说明已经找到至少 K 个 candidate。

### 5.5 union miss 大于 512

A3 对 `union_miss_count > 512` 保留正确性优先路径：

- 每个 source chunk 仍使用向量筛选与排序；
- 按 chunk 追加合法且唯一的 victim；
- source chunks 不重叠，合法 cache row 中 slot 唯一，因此无需全局 victim 去重 hash；
- 直到写满 union miss count，最后再处理 threshold tie fallback。

这是边界路径，不应为了让它与典型路径共用一种复杂数据结构而拖慢 `U≈300` 的稳定 decode。

### 5.6 A5 C8 的 score 域

A3 的 `-(score0+...+score3)` 使用 FP32 reduction score。A5 QLI 内部主要维护 `uint16` sortable score/key，因此必须先做一个明确选择：

- 若要完全复现 A3 的 sum-score 淘汰策略，应在反量化、乘权和归约后保存可加的 FP32 score；
- 若使用 A5 原生 uint16 sortable key 做近似 aggregate，需要证明保护判断正确，并承认 eviction ordering 只是近似策略；
- 无论采用哪种 ranking，真实 TopK protection 都必须正确，不能因为量化 score tie 而淘汰 union token。

建议优先复用 A5 非 MTP `hist_topk_index_update_a5_evict_vf.h` 的 score-key、slot compaction 和 MicroAPI mask 逻辑，再扩展到实际 `Q_b` 路，而不是把 A3 float 实现直接移植到 Arch35。

C8 `uint16` score 可能比 BF16/FP32 更容易产生 threshold tie，必须专门统计 tie fallback 命中率；若 fallback 频繁，应考虑对 TopK union 建轻量 exact membership 辅助结构，而不是接受大范围 scalar fallback。

## 6. 阶段四：缓存更新与输出回填

### 6.1 只做 O(union miss) cache update

对有序 union miss 与 victim 一一配对：

```text
cache_slots_pool[pool_row, victim_source] = -1
cache_slots_pool[pool_row, miss_source]   = victim_slot
miss_src_ids[i] = miss_source
miss_dst_slots[i] = victim_slot
```

A3 当前 `ApplyCandidateUpdates` 仍是每个 union miss 做两次随机 GM 写，典型约 300 项。这是当前 A3 剩余的主要可优化点之一，但在功能打通前不应过早复杂化。A5 第一版高性能重构可以先保留 O(U) scalar update，再单独评估 grouped/scatter 写回。

### 6.2 只修补每路 miss prefix

阶段一已经为所有 hit 写好了旧 destination slot。因此 request owner 不需要重建完整四路 Top2048：

- 只搬入每路长度约 N 的 miss source prefix；
- 维护 union source/destination 的有序映射；
- 用 merge join 为每个 miss source 找到新 slot；
- 只覆写每路 `topk_dst_slots` 的 miss prefix；
- hit suffix 完全不动。

A3 对应优化 commit `e0e0307`，它删除了完整 2048-row 的重复修补。后续 commit `52822a5` 又把 union source→destination mapping 留在 UB，避免重新从 GM 读取 mutable cache map。

A5 虽然不公开 `topk_src_ids`，也应在内部保留有序 miss source prefix，不能在 update 后重新读取整路 TopK 并对 2048 项随机查 cache mapping。

### 6.3 合并输出发布

A3 最终将以下输出放在同一个 MTE3 publication 区间：

- union miss source prefix；
- union miss destination prefix；
- 各路 TopK destination 的 miss prefix；
- miss count。

只在所有 request-local 计算完成后统一发布，减少多次 `S↔MTE3` fence。对应优化 commit `b443e72`。

## 7. Workspace 与 UB 组织

A3 production ABI 不暴露任何调试 tensor。私有 workspace 依次包含：

```text
official LI workspace
pair0: B × 8192 float words
pair1: B × 8192 float words
scores: B × 4 × aligned_source_len float
thresholds: B × 4 × 8 float
```

其中：

- `aligned_source_len = ceil(source_len / 512) * 512`；
- pair workspace 保存四路有序 `{key,payload}`；
- score 必须在 TopK UB 被复用前发布；
- threshold stride 取 8 是为了 32-byte 对齐和可靠 DMA；
- union source、union destination、pair input/output 尽量留在同一 request owner 的 UB 中跨阶段复用。

A5 不必照搬 float 布局，应根据 C8 score 类型与 UB 容量重新计算；但必须保留“LI 私有 workspace”和“request owner UB 生命周期”的分层。不要把临时中间结果写到公开输出后再读回来。

## 8. A3 调试中确认过的错误模式

这些问题在 A5 上很可能以不同形式再次出现：

### 8.1 破坏官方 LI 并行

症状：功能正确，但 LIM-MTP 比 official LI 多出毫秒级时间。

根因：把每请求四路 query 变成四次串行 LI，或在每一路后等待整个 score workspace 消费完成。

### 8.2 compare mask 与 UB scratch 重叠

A3 曾只按逻辑 uint8 mask 大小预留空间，实际 AscendC compare 工作区覆盖了后续 `invalidKey/Sort32` scratch，最终把 invalid slot 解码成 `16383`。

A5 使用 mask register 后布局不同，但任何 `Squeeze/StoreUnAlign` 临时区都必须按 Arch35 API 的真实要求隔离，不能只按逻辑输出字节数估算。

### 8.3 score row stride 错误

A3 曾因泛型 `CeilDiv` 与整数类型组合产生错误 stride，表现为只有 route 0 score 有效，其余 route 写到同一行。最终改成显式 `uint32` 对齐公式。

A5 所有 packed query 的 score offset、request offset、route offset都应在 Host 与 kernel 各自独立校验。

### 8.4 score 被 TopK UB 覆盖后才写出

score 必须在 TopK 排序复用 reduction buffer 前发布。TopK 集合正确并不能证明 score workspace 正确；两者要分别验证。

### 8.5 threshold 跨核可见性

producer scalar store + `SyncAll` + consumer scalar load 不可靠。使用 MTE3 发布、全局 barrier、MTE2 消费。

### 8.6 只用 `PipeBarrier` 等待 scalar count

`PipeBarrier<PIPE_V>` 只约束 Vector pipe 内顺序。Vector 产生的 compact count 交给 scalar 前必须显式同步。

### 8.7 未初始化的对齐尾部

向量 compare/compact 的最后一个 repeat 必须显式 padding。否则 union count 会随 batch 或运行次序出现 `±1/±4` 的非确定性。

### 8.8 性能测试复用了已更新 cache

第一次调用后 cache 已包含 union miss；第二次同样输入应是零 miss。如果计时循环不重置 cache，测到的是零 miss 热状态，会严重乐观。

A3 UT 在每次计时前执行：

```python
perf_cache.copy_(perf_cache_seed)
```

该 reset 不应计入算子 event 时间。

## 9. A5 C8 当前初版中应替换的慢路径

当前 `fused_li_manage_mtp_c8` 初版有以下结构性问题：

1. `QuantLiMtpPhase` 将每请求 2～4 路 query 逐路串行执行，并在每路后用 `REQUEST_DONE_EVENT` 等待 workspace 可复用。
2. 每路原生 TopK token ID 先完整写入 workspace，manager 再重新搬入。
3. request manager 用 16384-entry open-addressing hash 做 scalar union。
4. manager 对每路 2048 token 做 scalar `GetValue` 和 hash probe。
5. victim 搜索按 2048-token chunk 搬入 slot，但仍逐 token scalar `ContainsUnion`。
6. cache update 后再次逐路搬入完整 TopK，并对 2048 项逐项随机查询更新后的 cache mapping。
7. 管理阶段大量 `SetValue/GetValue`、逐输出 fence 和 `pipe.Reset()`，没有利用 A5 VF/MicroAPI 的并行压紧能力。

这些问题叠加后，局部微调 hash 或 chunk 大小不会解决根本时延。应从“批量 QLI + payload survivor + sorted union + vector eviction + prefix patch”重新实现。

## 10. 推荐给 A5 code agent 的分阶段执行顺序

每个阶段独立提交、独立编译、独立验收。测试打印时延，不设置性能 assert。

### A5 阶段 0：先建立可信基准

- 保留当前算子作为 correctness reference，另建可替换的 optimized path。
- official baseline 使用仓库已有 `official_c8_lightning_indexer`，输入、TND query lengths、causal/sparse 语义必须与 fused op 完全一致。
- semantic golden 使用每路隔离的 official C8 LI，集合比较而非依赖 tie 顺序。
- 记录：
  - `official_c8_li_mtp_us`
  - `fused_lim_mtp_c8_us`
  - `management_overhead_us = fused - official`
- 计时前重置 cache state。

### A5 阶段 1：批量 QLI + payload TopK

- 以 A5 非 MTP `arch35_payload_c8` 为主干，扩展到 packed TND 2～4 路 query。
- 删除逐请求逐 query 的串行 QLI loop。
- cache slot 在 histogram TopK survivor 中携带。
- 每路形成有序 miss prefix 与 hit suffix。
- 暂不做 union、eviction 和 cache update。
- 验收 TopK 集合、hit/miss、slot 及每路输出；重点看相对 official C8 LI 的阶段一增量。

### A5 阶段 2：有序 union

- 根据实际 `Q_b` 使用 2～4 路 MrgSort。
- 用 Arch35 mask + Squeeze 去重，不使用 scalar hash。
- 输出 union miss source 与 count；暂不更新 cache。
- 覆盖 union 0、1、300、512、513、>2048、8192。

### A5 阶段 3：vector eviction

- 为每个 request 指定唯一 owner。
- 512-token chunk 连续加载 score 和 slots。
- 用实际 `Q_b` 路构造 protection 与 aggregate key。
- 典型 `U<=512` 走 kth-key stop 快路径；大 U 走正确性路径。
- 暂不更新 cache，先验证 victim 合法、唯一、已缓存且不属于任一路 TopK。

### A5 阶段 4：cache update + prefix patch

- O(U) 更新 cache mapping。
- union mapping 保持在 UB。
- 只修补各路 miss prefix 的 `topk_dst_slots`。
- 合并 MTE3 输出发布。
- 验证第二次相同输入 `copy_counts=0` 且 cache 稳定。

### A5 阶段 5：完成后再做较大性能优化

优先级建议：

1. 优化 O(U) scalar random GM cache update；
2. 减少或压缩四路 full-score workspace；
3. 将 union merge、eviction 和 output publication 做更深流水；
4. 针对常见 `Q_b=4、U≈300` 做 specialization，但必须保留 2～4 路 ABI 语义；
5. 最后才考虑改变 eviction ranking 或引入近似策略。

## 11. 测试负载要求

A5 当前测试只控制每请求 union miss 范围还不够。建议移植 A3 UT 的 exact workload 思路，并推广到 `Q_b=2..4`：

- `--query-miss-count=N`：每一路恰好 N 个 miss；
- `--union-miss-count=U`：每请求 union 恰好 U 个 miss；
- 合法范围：`N <= U <= Q_b*N`；
- `query_noise` 只控制各路 TopK overlap，不直接等于 miss rate；
- 若 TopK membership buckets 无法构造指定 N/U，应明确报错并提示调整 query noise。

测试数据必须满足：

- initial cache 的 C 个 token 随机分散在完整 source，不聚集在低 source ID；
- miss source 覆盖尽可能多的 512-token chunk；
- slot 是 `[0,C)` 的随机排列，并显式覆盖 slot 0；
- block table 随机；
- `req_pool_entries` 乱序且非连续；
- 未参与 batch 的 pool row 完全不变。

正确性至少覆盖：

- query count 2、3、4，以及同 batch 混合 2～4 路；
- B=1、2、5、16、24，以及大于物理 AI Core 数；
- H=32、64；
- C=0、8192、12288、16256 与混合 budget；
- source 2K、8K、64K、`2^18`；
- per-query miss 0、1、64、200、512、513、2048；
- union miss 0、典型约300、512、513、>2048、8192；
- threshold ties；
- 相同输入重复执行后零 miss；
- graph replay 时刷新 query、scale、query lengths、candidate lengths、pool entries 和 block table。

性能主场景建议先统一为：

```text
Q_b=4
H=32
source_len=65536
C=12288（另测8192）
per-query miss≈200
union miss≈300
TopK union≈3K～4K
warmup>=10
iters>=300
```

不要用四路完全独立的随机 query 作为唯一性能负载，那会把 TopK union 和 union miss 构造成远高于真实 MTP3 的极端情况。

## 12. A3 当前性能参考

A3 commit `36cd61e`，H=32、source=64K、C=12288、每路 miss=200、union miss=300 的配对结果：

| Batch size | Official LI-MTP3 | Fused LIM-MTP3 | 管理额外时延 |
| ---: | ---: | ---: | ---: |
| 1 | 120.045μs | 205.409μs | 85.364μs |
| 2 | 138.420μs | 228.501μs | 90.081μs |
| 5 | 153.629μs | 262.661μs | 109.032μs |
| 16 | 371.358μs | 518.776μs | 147.419μs |
| 24 | 524.968μs | 688.536μs | 163.568μs |

这些数字只用于判断数量级，不能作为 A5 的硬门槛。A5 必须使用同机、同输入、同 QLI 语义下的 paired baseline。

## 13. A3 关键提交索引

| Commit | 内容 |
| --- | --- |
| `a2bcdfc` | 使用官方 LI 派生主干，实现 TopK 携带 cache slot 与每路 hit/miss 重排 |
| `2d8680a` | 四路 miss prefix 的 MrgSort 与 union miss |
| `2a097d1` | 512-token vector eviction scan |
| `190baa1` | 完成 cache update、union outputs 与四路 TopK slot 回填 |
| `e0e0307` | 只修补 TopK miss prefix，不重写完整 2048 row |
| `52822a5` | union source→destination mapping 保持在 UB |
| `b443e72` | 合并最终输出 publication |
| `15f101e` | union dedup 从 scalar 改成 vector compare/compact |
| `afcd494` | 修复 predecessor lane-0 sentinel 与重复首元素问题 |
| `0eb0a63` | 删除临时调试 ABI，中间状态回归私有 workspace |
| `36cd61e` | 清理 union 调试输出，形成当前 production ABI |

## 14. 给后续 code agent 的硬性约束

1. 只修改 A5 仓库，不修改 A3 参考仓库。
2. 当前 A5 C8 MTP ABI 与行为被视为正确，优化阶段默认不改接口语义。
3. 新实现必须基于 A5 C8 official QLI/非 MTP payload 内核，不能引入仓库外算子依赖。
4. 不允许把 packed MTP query 拆成 2～4 次串行 official QLI 作为最终方案。
5. 每个阶段先保证正确，再测相对 official C8 LI 的增量；不设置时延 assert。
6. 不得用已更新 cache 重复计时。
7. 跨核 GM 中间结果必须有明确 producer DMA、barrier 和 consumer DMA；不要依赖 scalar cache。
8. 任何性能优化都必须覆盖 zero miss、典型 U≈300、U=512/513 与 U>2048 边界。
9. 功能未全部打通前，允许临时增加 write-only debug tensor；每次改 ABI 前必须列出新增参数。最终合入前删除调试 ABI，并将中间状态放回 workspace。
10. 不要在阶段一到四之间混入近似 eviction、TopK 语义变化或外部依赖。

按照这套方法，A5 C8 LIM-MTP 的第一目标不是立刻达到某个绝对微秒数，而是先把当前“串行 QLI + scalar hash/scan”重构为可分阶段验收的高性能骨架。只要阶段一相对 official C8 LI 的增量足够小，后续 union、eviction 和 update 才有继续优化的价值。
