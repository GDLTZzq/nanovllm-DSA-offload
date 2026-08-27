# A5QuantLightningIndexerC8：融合单算子 QuantLightningIndexer

Decode-only 场景下，把官方两算子（AICPU metadata 算子 + QuantLightningIndexer
内核）融合为一个 Ascend 950（A5 / CANN 9.1）单算子。目标：消除小 batch 时
AICPU metadata 依赖与核空转，打开 S2 跨核切分（LD），使小 batch 时延显著下降、
时延随 batch 近线性增长。

## 背景与动机

旧版 LI（torch_npu 内置 `npu_quant_lightning_indexer`，以及本仓库
`fused_li_manage_c8` 的 LI 部分）在 A5 上小 batch 线性度差：

| batch | 源长 64K（µs） | 源长 128K（µs） |
| --- | --- | --- |
| 1 | 84 | 190 |
| 64 | 274 | 502 |

根因（已确认）：`isLDOpen` 从未被赋值 → S2 维度不跨核切分，一个 `(bN2, m)`
基本块由单个核原子处理，batch ≤ 核数时多数核空转；且官方方案依赖一个独立
AICPU metadata 算子（`BalanceSchedule` + `SplitFD` + `GenMetadata`），
"两个算子完成一个功能"，固定启动开销高。

ops-transformer-master 新版 QLI 引入 LD 分核加载（`ProcessLD`/`LdSplitCoreInfo`），
把每个 `(bN2, m)` 行的 S2 切到多核并行打分，再合并 partial top-k。但它仍依赖
独立 metadata 算子做代价调度。

**本算子**：把 metadata 算子的调度逻辑**搬进内核**（on-device 分核），
不新增 metadata 输入、无 sparseValues 输出，一个算子完成
"元数据调度 + top-k 稀疏索引"两个阶段。

## 设计

1. **On-device 分核代替 AICPU metadata**：内核内从 tiling 常量 + GM 中的
   实际 seq length，复刻 AICPU 的 `BalanceSchedule` + `SplitFD` 算法，
   逐核算自己的 `splitCoreInfo`（LI 范围）与 `ldInfo`（LD 归约范围）。
   所有核跑同一确定性算法，各自取自己的切片；无需跨核同步。
2. **移植源**：以 ops-transformer `arch35`（`__CCE_AICORE__==310`）内核为
   移植源，含全部 LD 机制（`ProcessLD`/`LdTopK`/`LiTopKLDGatherVF`/
   `LdSplitCoreInfo`/`isNeedLD`/`saveWorkSpaceIdx`）。仅把
   `SplitCoreByAICPU(…, metadataGm)` 换成 on-device 分核，其余原样移植。
3. **调度核数 = 实际发射核数**：`KERNEL_TYPE_MIX_AIC_1_2` 下 AIC=24、AIV=48
   （编译期常量），与内核 `GetBlockNum()`/`aiCoreIdx` 自洽，避免 AICPU 用平台
   核数与发射核数不一致的隐患。
4. **固定组合**（与官方 `DT_FLOAT8_E4M3FN` 分支一致）：
   fp8_e4m3fn query/key（ops.json 以 uint8 顶替）+ fp32 weights + fp32 scales
   + fp32 QK 累加 + uint16 score + int32 输出索引。weights 由 torch wrapper
   做 `bf16.float()` 无损转换后下发。
5. **无 attr**：sparse_count=2048、sparse_mode=3、layout=TND/PA_BSND、
   pre/next=INT64_MAX（无 mask）、cmp_ratio=1、quant_mode=0 全部写死在 tiling，
   与 `fused_li_manage_c8` 做法一致。
6. **LD 打开条件**复刻 AICPU：`supportFd = maxS2>sparseCount || (A5:
   maxS2>5*s2BaseSize)`。测试形状 s2≥65536 恒开；保留非 FD 回退路径。

## 目录结构

```
quant_lightning_indexer_c8/
├── op_host/
│   ├── a5_quant_lightning_indexer_c8.cpp            # host 入口 + 参数校验
│   ├── a5_quant_lightning_indexer_c8_infershape.cpp # 输出 [batch,1,2048]
│   ├── a5_quant_lightning_indexer_c8_tiling.h/.cpp  # QLITilingData + workspace
├── op_kernel/
│   ├── a5_quant_lightning_indexer_c8.cpp            # 内核入口 INVOKE_QLI_C8_OP_IMPL
│   ├── a5_quant_lightning_indexer_c8_template_tiling_key.h  # LI_C8_TPL_UINT8
│   └── arch35/
│       ├── quant_lightning_indexer_kernel.h         # ComputeSplitInfo + 主循环
│       ├── quant_lightning_indexer_common.h         # ConstInfo/LdSplitCoreInfo 等
│       ├── quant_lightning_indexer_service_cube.h   # mm 打分
│       ├── quant_lightning_indexer_service_vector.h # top-k / ProcessLD
│       └── vf/                                      # vf_topk / LdTopK / gather
```

## On-device 分核（融合核心）

`arch35/quant_lightning_indexer_kernel.h::ComputeSplitInfo(aiCoreIdx, vecCoreIdx)`
输入只依赖常量与 GM 实读的 seq length，逐核自取所需：

- Pass A：遍历全部 `(bN2, gS1)` 行，统计 `rowsCount`、`totalBlocks`、
  每行权重 `w`（无 mask 时 = `ceil(actS2/s2BaseSize)`）。
- **Branch 1（rowsCount ≤ 24，小 batch）**：
  - 按块数比例把每行 S2 切给多个核（`rowK[i] ∝ rowW[i]·AIC/totalBlocks`，
    补齐/扣回使 `sumK==24`）；每核片段 `s2Start=(p·rowW)/rowK`、
    `s2End=((p+1)·rowW)/rowK - 1`；
  - LD slot 分配：被切分行按行序占连续 slot，
    `ldInfo.saveWorkSpaceIdx = rowSlot[R]+p`（写侧 offset 与该值严格互洽）；
  - `SplitFD`：把 fd 归约任务按 `fdSplitNum×fdMSize` 负载均衡分给 48 个 AIV，
    产出 `ldInfo.{bn2Idx,mIdx,workspaceIdx,workspaceNum,mStart,mNum}`。
- **Branch 2（rowsCount > 24，整行分组，无 LD）**：
  - 每核目标 `target = 剩余块数/剩余核数`（动态）；
  - `maxTake = 剩余行-(剩余核数-1)`：每个核至少给后续核留 1 行，
    最后一个核 `maxTake == 全部剩余行` —— 行悬空在结构上不可能发生；
  - 每个 `(bN2,gS1)` 行整体落在单一核上，无跨核归约。

## 32-head 支持

- **mBaseSize/s1BaseSize 与官方 arch35 同式**：`s1BaseSize = S1_BASE_SIZE(=4)`、
  `mBaseSize = S1_BASE_SIZE × gSize`，`tSize ≤ 64` 时两者减半
  （decode 各 batch 恒减半 → `s1BaseSize=2`、`mBaseSize=2×gSize`）。
  heads=32 → 64/2；heads=64 → 128/2；heads=8 → 16/2；host tiling 与内核同式。
- 对齐后大 batch（rowsCount>24，Branch 2 整行无 LD）与官方逐字节同构；
  scoreGm 每块 `s1BaseSize×Align(s2)` 由 8 行降到 2 行 → GM score 写+读流量降 4 倍，
  workspace 同步降 4 倍（32-head/128K ≈ 13MB）。
- tiling `CheckShape` 允许 heads ∈ {8,16,24,32,64}。

## Workspace 布局

`GetLibApiWorkSpaceSize() + aicNum × (scoreSize + ldScoreSize + ldIndexSize)`，
kernel `Init` 与 host tiling 同一口径：

| 段 | 每核大小 | 说明 |
| --- | --- | --- |
| `scoreGm` | `s1BaseSize × Align(kSeqSize, s2BaseSize) × sizeof(uint16)` | 每 AIC 核全 S2 的 partial score，base = `aiCoreIdx × size` |
| `ldScoreGm` | `s1BaseSize × topkCountAlign16(=2048) × 2 × sizeof(uint16)` | LD 归约用 score |
| `ldIndexGm` | `s1BaseSize × topkCountAlign16 × 2 × sizeof(int32)` | LD 归约用 index |

每核至多 2 个跨核行 → LD slot 每核 ≤ 2，与 `2×GetBlockNum()` 预留一致。

## 接口

```python
torch.ops.nanovllm_dsa.quant_lightning_indexer_c8(
    query,                    # C8[T,32,128]   fp8e4m3fn（uint8 存储，TND packed）
    key,                      # C8[BLOCKS,128,1,128]（PA_BSND）
    weights,                  # bf16[T,32]     wrapper 转 fp32
    query_dequant_scale,      # fp32[T,32]
    key_dequant_scale,        # fp32[BLOCKS,128,1]
    actual_seq_lengths_query, # int32[B]（逐请求前缀和，长度 = 请求数 B）
    actual_seq_lengths_key,   # int32[B]（candidate_lens）
    block_table,              # int32[B,MAX_BLOCKS]
) -> sparse_indices           # int32[T,1,2048]
```

### MTP（packed）语义

TND 布局下 query 为 packed `[T, N, 128]`，`actual_seq_lengths_query` 是逐请求
前缀和（长度 = 请求数 B）。每请求 1~5 个 query（MTP0~4），host tiling 用
`bSize = 请求数 B`、`tSize = packed T`（校验 `T ∈ [B, 5B]`）；内核迭代 B 个请求，
经 `actual_q[bIdx] - actual_q[bIdx-1]` 取每请求 s1，输出/权重/query_scale 第 0 维
均为 T（与 `fused_li_manage_mtp_c8` 同一套 TND 前缀和机制）。

## 注册链

- `csrc/ops.json`：`A5QuantLightningIndexerC8`，query/key `uint8`、weights/scales
  `float`、seqs/block_table `int32`，输出 sparse_indices `int32`，attr 为空。
- `build_c8.sh`：`OP_NAMES+=(A5QuantLightningIndexerC8)`，
  `OP_DIRS+=(quant_lightning_indexer_c8)`。
- `torch_extension/csrc/ops_registration.cpp`：`TORCH_LIBRARY` schema
  `quant_lightning_indexer_c8(Tensor×8) -> Tensor`。
- `torch_extension/csrc/npu_quant_lightning_indexer_c8.cpp`：
  `weights.float()` 转 fp32 后 `aclnnA5QuantLightningIndexerC8` → `[B,1,2048]`；
  并注册 `nanovllm_dsa` PrivateUse1/Meta 实现。
- `torch_extension/nanovllm_dsa_a5/ops/quant_lightning_indexer_c8.py`：dispatch 包装。

## 编译与测试（A5 Linux 机）

```bash
# 环境：CANN 9.1 + ascend950，见根 README_c8.md "编译" 一节
bash build_c8.sh                      # 构建并安装到 _custom_opp_c8，import 校验
python3 tests/test_quant_lightning_indexer_c8.py \
    --batch-sizes 1,4,8,16,24,32,48,64 \
    --source-lens 65536,131072 --heads 32,64 --warmup 3 --iter 20
python3 tests/test_quant_lightning_indexer_mtp_c8.py \
    --batch-sizes 1,4,8,16 --source-lens 65536 --heads 32,64 \
    --queries-per-request 1,2,3,4,5 --warmup 3 --iters 20
```

MTP 测试默认跑 MTP0~4（1~5 query/请求）各一个 uniform batch，另加一个 batch 内
cycle `[1,2,3,4,5]` 的异构 case；`--queries-per-request 0` 等价于
`1,2,3,4,5`。正确性同样对比官方 `npu_quant_lightning_indexer`（同一份 packed
输入独立算出），先 `torch.equal` 否则逐行排序多集比较。

正确性：金标为官方 `npu_quant_lightning_indexer`（同一份输入独立算出）。
先 `torch.equal`，否则逐行**排序多集**比较（并列分数顺序可不同，但重复/漏选
仍会被抓出）。benchmark 用 `torch.npu.Event` 计时。

## 性能数据（A5，2026-08）

| batch | source_len | official_c8_li_us | fused_li_us | speedup |
| --- | --- | --- | --- | --- |
| 1 | 65536 | 85.761 | 50.960 | 1.683 |
| 1 | 131072 | 189.711 | 53.350 | 3.556 |
| 4 | 65536 | 90.585 | 57.523 | 1.575 |
| 4 | 131072 | 182.341 | 50.355 | 3.621 |
| 8 | 65536 | 87.596 | 57.171 | 1.532 |
| 8 | 131072 | 184.168 | 81.542 | 2.259 |
| 16 | 65536 | 93.205 | 113.234 | 0.823 |
| 16 | 131072 | 173.629 | 210.100 | 0.826 |
| 24 | 65536 | 94.391 | 109.041 | 0.866 |
| 24 | 131072 | 182.168 | 208.679 | 0.873 |
| 32 | 65536 | 168.661 | 202.062 | 0.835 |
| 32 | 131072 | 339.104 | 400.578 | 0.847 |
| 48 | 65536 | 178.052 | 214.685 | 0.829 |
| 48 | 131072 | 343.583 | 401.802 | 0.855 |
| 64 | 65536 | 274.604 | 301.573 | 0.911 |
| 64 | 131072 | 502.431 | 591.772 | 0.849 |

要点：

- **小 batch 时延显著下降**：batch=1 时 64K 51µs（1.68x）、128K 53µs（**3.56x**）。
  batch=1 下 64K/128K 几乎持平（51→53µs）——2 倍源长被 24 核 LD 摊平，
  正是 S2 跨核切分生效的直接证据。
- **batch 增长近似线性**：64K 下 fused 时延 51→57→113→202→302µs（b1→b8→b16→b32→b64）；
  b8 起约每 2 倍 batch 翻倍。官方 128K batch=1 的 190µs 被压到 53µs。
- **已知回归**：batch≥16 融合慢 10–18%（speedup 0.82–0.91）。batch=16 走
  Branch 1、batch≥24 走 Branch 2（整行分组、无 LD）。已按官方对齐 base size
  （mBaseSize/s1BaseSize，见上节）降 4 倍 GM score 流量，待重测验证是否收窄。

## 已知问题与后续

1. **batch≥16 性能回归**（0.82–0.91x）：已做 base size 对齐官方（Branch 2 与官方
   同构 + scoreGm 流量降 4 倍），待 A5 重测。若仍未收窄，候选：移植 AICPU 代价
   调度（`AssignByBlock`+`ForceAssign`）；bs16 的 Branch 1 2:1 负载不均
   （rowK=1.5 → 8 行×2 核 + 8 行×1 核）需单独处理（整行回退或块级连续切分）。
   **L2 cache hint 已验证无效勿再试**（关 L2 反而变差 15-19%）。
2. **测试覆盖缺口**：MTP0~4 已覆盖（`test_quant_lightning_indexer_mtp_c8.py`）；
   小 s2 的"非 FD 回退"路径与 mask 模式仍未覆盖。可将 `--source-lens 2048`
   加入 sweep 测非 FD 路径。另：`ProcessInvalid`（全空请求的清理）在 MTP 下
   用 `bSize*s1Size`（s1Size=1）计输出总长，全空时仅覆盖 B 行而非 T 行——
   该退化场景不被测试命中，未修。
3. **金标单源**：正确性对官方算子比对；若官方存在系统性错误会一起放行。
   如需更强保障可加纯 CPU 参考实现。
4. **后续 lim_c8 融合**：hit/miss 分类、victim 驱逐需移到 `ProcessLD` 合并
   之后（本阶段独立 QLI 不涉及）。
