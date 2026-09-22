# 小 batch 线性度怎么解：S2 分核 + LD 归并（Branch 1）

> 本文讲融合算子 `A5QuantLightningIndexerC8` 里"小 batch 下 S2 跨核切分、结果再归并"
> 的完整机制：S2 按什么规则切到不同 core，每核算什么，写到哪里，最后怎么聚合回全局 top-k。
> 对应代码行号以当前磁盘版本（base size 对齐已回退、`M_BASE_SIZE=256`）为准。

## 0. 前后对比图（总览）

```
┌──────────── 对比：一个 (bN2,gS1) 行的 S2 打分流程 ──────────────┐
│                                                                │
│  BEFORE ─ 官方/旧版 LI：isLDOpen=false，S2 不切分             │
│                                                                │
│  核 i 独占整行：                                               │
│  ┌──────────┬────────────┬───────────┬────────────┬─────────┐ │
│  │ 读 key   │ mm 整行     │ score 整行 │ top-k 整行  │ 写输出  │ │
│  │ 全 S2    │ QK^T       │ 全 S2     │ 全局 2048  │         │ │
│  └──────────┴────────────┴───────────┴────────────┴─────────┘ │
│        时延 ∝ S2（单核串行）        batch<核数时其他核空转      │
│                                                                │
│  AFTER ─ 融合算子 Branch 1：S2 切 24 段 + LD 归并              │
│                                                                │
│  ① S2 分核（Pass A）：行切成 k 份，段 p 的块区间               │
│     [ (p·w)/k , ((p+1)·w)/k-1 ]   （w=ceil(S2/128)）           │
│                                                                │
│  ② 24 个 AIC 并行打分（各自只取一段）：                        │
│  ┌──────────┬────────────┬───────────┬────────────┐            │
│  │ 读段 key │ mm 段内     │ score 段内 │ 局部 top-k  │           │
│  │ 只自己段 │ QK^T       │ S2/k      │ 段内 2048   │           │
│  └──────────┴────────────┴───────────┴────────────┘            │
│       核0..23 并行，每核只做 S2/k     │ 索引 +curS2StartIdx     │
│                                      ↓ 写 LD slot（连续）      │
│                       ┌─────────────┐                        │
│                       │  SyncAll()  │  ← 全局屏障             │
│                       └─────────────┘                        │
│  ③ 归并 AIV（SplitFD 分配）：                                 │
│     ┌───────────────────────────────┐                        │
│     │ 读回同行的 k 段局部 top-2048   │                        │
│     │ → LdTopK k 路归并             │ → 全部数据 全局 top-2048│
│     └───────────────────────────────┘                        │
│        时延 ≈ S2/24 + 固定屏障/归并开销（batch=1 时 64K/128K  │
│        都≈51-53µs，2×源长被 24 核摊平）                       │
└──────────────────────────────────────────────────────────────┘
```

## 1. 为什么切 S2 能解决线性度

Decode 一个 (bN2, gS1) 行的打分是 `QK^T`：M 轴 = 该行 s1-query 行数（bs=1 时就是 1），
N 轴 = S2 = 源长（64K/128K）。**M 只有 1 行、S2 有几百上千个 128 块** —— 单核顺序算的话，
时延 ∝ S2，batch 每加 1 就串一串，batch=64 时核全占满但每个请求都慢，线性度差。

Branch 1 的做法：把**每一行的 S2 切成多段、分给多个 AIC 核并行打分**（时延 ∝ S2/核数），
各核只算自己那一段的局部 top-k，最后再用一批专用 AIV 核把各段的 top-k **归并**回全局 top-k。
结果：bs=1 时 24 个 AIC 全在打同一行的不同 S2 段，64K→51µs、128K→53µs（2 倍源长被摊平）。

切换阈值：`ComputeSplitInfo` 里 `rowsCount <= AIC_CORE_NUM(24)` → Branch 1（本文，带 LD）；
`rowsCount > 24` → Branch 2（整行贪心分组、无 LD）。decode 下 `rowsCount == batch`，阈值 = 24。

## 2. 两阶段总览

```
阶段A  Pass A + Apportionment      —— 每核算自己该切哪几段（on-device，无 metadata 算子）
阶段B  打分 & 局部 top-k            —— 每核只处理自己的 S2 段（AIC mm + AIV 归约/topk）
阶段C  ProcessLD 归并              —— 专用 AIV 把同一行各段的局部 top-k 合并成全局 top-k
```

阶段 C 由 `ProcessDecode` 在**所有核把阶段 B 做完、`SyncAll()` 之后**再跑
（kernel.h:923-933），保证各段局部结果都已落盘再归并。

## 3. 阶段A：S2 怎么切（ComputeSplitInfo，kernel.h:343-606）

### 2.1 Pass A：统计每行的 S2 块数

遍历所有 (bN2, gS1) 行（decode 下 = 逐请求，每行 `w = ceil(actS2 / s2BaseSize)` 块，
s2BaseSize=128），累计 `rowsCount` 与 `totalBlocks = Σw`，并记下每行 `rowW[i]=w`、
`rowB[i]=bN2`、`rowG[i]=gS1`、`rowTailM[i] = 该行 s1 行数`（kernel.h:346-377）。

### 2.2 每行分几份：rowK（Apportionment）

- 初值：`rowK[i] = rowW[i] * 24 / totalBlocks`，钳到 `[1, rowW[i]]`（每行至少 1 核、
  且份数不超过块数）（kernel.h:388-400）。
- 补齐到 24 核：`sumK < 24` 时，把剩余核加给"分数余数最大且未饱和"的行
  （`frac = rowW[i]*24 % totalBlocks`）（kernel.h:402-419）。
- 超出则从 `rowK` 最大的行扣回（保底 1）（kernel.h:421-435）。

### 2.3 每核的具体片段边界

把第 R 行切成 `rowK[R]` 份，第 p 份（p ∈ [0, rowK[R])）负责的 S2 块区间（**块号**）：

```
s2Start = (p * rowW[R]) / rowK[R]
s2End   = ((p+1) * rowW[R]) / rowK[R] - 1      // kernel.h:484-485
```

→ 该核负责 S2 元素 `[s2Start*128, (s2End+1)*128)`。核到 (R, p) 的映射：按行序累加 rowK，
`cubeCoreIdx` 落在第几行的第几份就是 (R, p)（kernel.h:462-472）。`cubeCoreIdx >= sumK` 的核闲置。

### 2.4 LD slot 分配（写/读侧互洽的关键）

被切分的行（`rowK[i] >= 2`）按行序占**连续 slot**：`rowSlot[i] = slotBase; slotBase += rowK[i]`
（kernel.h:438-457）。第 R 行第 p 份写入的 slot = `saveWorkSpaceIdx = rowSlot[R] + p`
（kernel.h:487）。所以同一行的所有份 → 连续 slot 区间，归并时一次性顺序读即可。
`rowK[i] == 1` 的行（整行单核、不需要归并）`saveWorkSpaceIdx = 0`，直接直出最终输出。

### 2.5 SplitFD：把归并任务分给 48 个 AIV

对每个"被切分行"（fd 任务），负载均衡分给 AIV（kernel.h:494-547）：

```
totalFDLoad = Σ fdSplitNum[i] * fdMSize[i]     // 份数 × 该行 s1 行数
averageLoad = ceil(totalFDLoad / 48)
每 fd 任务分到的 AIV 数 curFDVectorNum ≈ fdSplitNum*fdMSize/averageLoad（≥1）
```

每个 AIV 拿到 `ldInfo = { bn2Idx, mIdx(=gS1), workspaceIdx(=slot 基址), workspaceNum(=份数),
mStart, mNum }`：负责归并该行第 `mStart..mStart+mNum` 个 s1 行。

## 4. 阶段B：每核独立打分 + 局部 top-k（service_vector.h）

### 3.1 打分（ProcessVec1，service_vector.h:344-413）

AIC 核 `ComputeMm1` 算自己 S2 段的 `QK^T`（M=该块 s1g 行数，N=段内 s2），AIV 核
`ProcessVec1` 做 weight×qScale 加权归约 + kScale，得到 `score[该行 s1 行, 段内 S2]`，
写到自己核的 `scoreGm`（每 AIC 核独享一整段 score 缓冲，kernel.h:665-666）。

### 3.2 局部 top-k（ProcessTopK，service_vector.h:416-636）

对块内每个 s1 行（`rowIdx`），从 `scoreGm` 取自己的段，算**段内 top-2048**（sparseCount）。

- 若该核只占行的**一部分**（`isNeedLD=true`，即 `s2Start>0` 或没占到行尾，
  CalcS2LoopParams kernel.h:723-727）：
  - 把索引**加上段起点** `curS2StartIdx`（service_vector.h:593），使索引变回"行内全局"坐标；
  - 不足 2048 的部分 value 刷 0、index 刷 -1；
  - 写入 LD workspace：`offset = saveWorkSpaceIdx*s1BaseSize*topkCountAlign16 + rowIdx*topkCountAlign16`
    （service_vector.h:476），即 **value→ldScoreGm、index→ldIndexGm 同一 offset**。
- 若该核占了**整行**（`isNeedLD=false`）：不需要归并，直接写最终输出
  `indiceOutGm[indiceOutOffset + (curS1Idx+rowIdx)*topkCount]`（service_vector.h:589）。

## 5. 阶段C：归并（ProcessLD，service_vector.h:639-751）

### 4.1 读回所有段的局部 top-k

对归并负责的每个 s1 行 j，从该行的连续 slot 读回全部局部结果（service_vector.h:691-697）：

```
LDGmOffset = workspaceIdx*s1BaseSize*topkCountAlign16       // slot 基址
           + topkCountAlign16*(mStart+j)                    // 行内 s1 行
           + i*ldWorkspaceNum*s1BaseSize*topkCountAlign16   // 分批读时的批偏移
一次 DataCopy：blockCount = ldProWorkspaceNum（份数）, blockLen = topkCountAlign16,
              srcStride = (s1BaseSize-1)*topkCountAlign16   // 相邻 slot 间距 = s1BaseSize*T
```

`srcStride` 的设计正是 §2.4 的 slot 布局：槽内 s1 行间距 = `topkCountAlign16`，
槽间距 = `s1BaseSize*topkCountAlign16`。写侧 offset（§3.2）与读侧 offset 同式，天然互洽。

### 4.2 k 路归并（LdTopK）→ 写最终输出

把 `workspaceNum` 个"段内 top-2048"列表做 k 路归并（`topkOp_.LdTopK`），得到行内全局 top-2048，
写 `indiceOutGm[indiceOutCoreOffset + (mStart+j)*kHeadNum*topkCount]`
（service_vector.h:746-747）。`indiceOutCoreOffset` 在 TND 下 = 该请求的 s1 前缀和 × 输出步长。

### 4.3 归并为什么是正确的

每个段只产生段内 top-2048。只要某元素不在全局 top-2048 里，就一定不在它所在段的
top-2048 里（否则该段里至少有 2048 个比它大的元素，全局 ≥2049 个更大，矛盾）。
因此"各段 top-2048 的并集"⊇ 全局 top-2048，k 路归并不丢候选。段不足 2048 时整段都在，
无遗漏；刷 -1/0 的 padding 不会超过真实 top-k。

## 6. 同步

- 阶段 B 内：AIV 等 AIC 的 mm1（`CROSS_CV_EVENT`），AIC 等 AIV 打分完（`CROSS_VC_EVENT`），
  逐 loop 乒乓（kernel.h:899-905，ProcessVec1:383/412）。
- 阶段 B → 阶段 C：`ProcessDecode` 里 `SyncAll()`（kernel.h:928），所有核到齐后
  归并 AIV 才开始读他人写的局部结果。
- LD 数据段每核预留 2 个 slot（`2*GetBlockNum()`，kernel.h:668-674），因每核至多负责
  2 个跨核行。

## 7. 效果与代价

- **收益**：单行 S2 摊到 24 个 AIC（打分）+ 48 个 AIV（归约/topk），bs≤8 反超官方 1.5-3.6x，
  bs1 的 64K/128K 几乎持平。
- **代价**：被切分行的局部 top-k 要走一遍 LD workspace（多一轮 GM 写 + 读 + 归并），
  以及 `SyncAll()` 全局屏障。batch 越大、单行 S2 越小，切分收益越弱——这是 bs16 之后
  Branch 1/2 交界处性能回落的原因之一；bs>24 直接走 Branch 2 整行分组（无 LD）。

## 8. 关键位置速查

| 逻辑 | 位置 |
| --- | --- |
| 分支选择（rowsCount≤24 → LD） | `op_kernel/arch35/quant_lightning_indexer_kernel.h:385` |
| rowK 切分 + 补齐/扣回 | kernel.h:388-435 |
| 片段边界 `s2Start/s2End` | kernel.h:484-485 |
| LD slot 分配 | kernel.h:438-457 |
| SplitFD（AIV 归并任务分配） | kernel.h:494-547 |
| 局部 top-k 写 LD workspace | `service_vector.h:476`（offset）、`:593`（+curS2StartIdx） |
| 归并读回 + LdTopK | `service_vector.h:691-719` |
| 最终输出 | service_vector.h:746-747 |
| SyncAll 后归并 | kernel.h:923-933 |
