# A5QuantLightningIndexerC8 交接文档：小 batch 收益 / 大 batch 劣化（2026-08-26）

> 目的：本会话上下文已长，此为给新会话的完整交接。**核心未决问题：base size 对齐
> 改动破坏了 heads=64 batch=1 的正确性（尚未回退），下一步需先判"是否本次改动引入"再决定去留。**

## 1. 性能现状（A5，heads=32，base size 对齐**之前**的实测）

| batch | 64K source | 128K source | 结论 |
| --- | --- | --- | --- |
| 1–8 | speedup 1.5–3.6x | 1.5–3.6x | LD 分核反超官方 |
| ≥16 | speedup 0.82–0.91x | 0.82–0.87x | 慢官方 10–18% |

- 分核阈值：`kernel.h:378` `if (rowsCount <= AIC_CORE_NUM(24))` → Branch 1（按块数比例切 S2 + LD，小 batch）；
  else → Branch 2（整行贪心分组、无 LD，大 batch）。decode 下 `rowsCount == batch`，阈值 = 24。
- 注意：bs16 落在 Branch 1，rowK=1.5 → 8 行×2核 + 8 行×1核，2:1 负载不均，也是劣化点之一。
- bs≥24 走 Branch 2，结构上与官方逐字节同构，唯一实质差异是 base block 大小。

## 2. 在途改动：base size 对齐官方（**当前未回退，正因此 heads=64 bs1 出错**）

**目标**：Branch 2（大 batch）与官方同构 + scoreGm 流量降 4 倍。

- 官方 arch35 公式：`s1BaseSize = S1_BASE_SIZE(4)`；`mBaseSize = S1_BASE_SIZE × gSize`；
  `tSize ≤ 64` 时两者**同时减半**。
- 改动前 fused：`mBaseSize = 256`、`s1BaseSize = ceil(256/gSize)`（heads=32→8，heads=64→4）。
- 改动后 fused（decode 各 batch 恒减半）：`s1BaseSize = 2`、`mBaseSize = 2×gSize`。
  - heads=32 → mBaseSize=64；heads=64 → mBaseSize=128。
- 对齐后 scoreGm 每块 `s1BaseSize×Align(s2,128)` 由 8 行降到 2 行 → GM score 写+读流量降 4 倍。

**改动文件清单（回退/复查时对照）：**

| 文件 | 改动 | 位置 |
| --- | --- | --- |
| `op_kernel/arch35/quant_lightning_indexer_kernel.h` | 常量 `S1_BASE_SIZE=4`（原 `M_BASE_SIZE=256`）；`InitTilingData` 改官方同式（`mBaseSize=s1BaseSize*gSize`，`tSize≤64` 减半） | L82, L189-198 |
| `op_kernel/arch35/quant_lightning_indexer_common.h` | `ConstInfo` 加回 `tSize` 字段 | L111 |
| `op_host/a5_quant_lightning_indexer_c8_tiling.h` | `QLI8_S1_BASE_SIZE=4`（原 `QLI8_M_BASE_SIZE=256`）；tiling 加 `tSize` 字段 | L54, L65 |
| `op_host/a5_quant_lightning_indexer_c8_tiling.cpp` | `set_tSize`；workspace 的 `s1BaseSize` 同式（`tSize≤64` 减半） | L218-221, L232 |
| `README.md` | "32-head 支持"与"已知问题"更新 | — |

host tiling workspace 与 kernel `Init` 口径已核对一致（`scoreGm= s1BaseSize×Align(s2,128)×2B`，`ldScore/ldIndex= s1BaseSize×2048×2×sizeof`）。分核逻辑（`rowW=ceil(actS2/s2BaseSize)`）不依赖 s1BaseSize，不受影响。

## 3. 已验证结论（勿重做）

- **L2 cache hint 无效**：对 key/key_scale 加 `SetL2CacheHint(CACHE_MODE_DISABLE)`（对标官方
  `setL2DisableFlag=tSize/batchSize<=s1BaseSize`）**反而变差**：bs4-8 64K 也变慢（57→63/65µs），
  bs≥32 128K 恶化 15-19%。融合算子的 key DMA 吃 L2 红利，与官方"key 零复用关 L2 防污染"前提不同。已完整回退 6 处。
- **Python 层 batch dispatch 已被用户否决**：用户明确要"算子里加 if-else 分支"（小 batch 走融合，
  大 batch 走官方逻辑），不要 Python 分派。现有分支即 `kernel.h:378`。

## 4. 未决问题：heads=64 batch=1 正确性失败（当前阻塞点）

A5 上 `build_c8.sh` 成功，但：

```
python3 tests/test_quant_lightning_indexer_c8.py --device npu:0 --heads 64 --warmup 10 --iters 300
→ AssertionError: QLI C8 row sets differ in 1/1 rows   （首个 case：heads=64, batch=1, source_len=65536）
→ ERR99999 UNKNOWN applicaiton exception
```

**关键未知：**
- base size 对齐后 **heads=32 是否也坏，未测**（用户只跑了 `--heads 64`）。
- heads=64 在**改动前是否本来就有 latent bug，未验证**（README 性能表只有 heads=32，
  测试默认也是 heads=32；heads=64 从未进过验证集）。
- 注意：`project_qli_crash.md` 另有"QLI 32 头大 S2 崩溃(507015)"历史问题，与本次是不同现象（崩溃 vs 错结果）。

**排查中已确认的机制线索（mBaseSize 相关 UB 布局契约）：**
- cube 侧 Fixp（`service_cube.h`）：
  - `Fixp` 每个 pingpong 的 dst 基址 = `mm1ResUB_[(loop%2)*s2BaseSize/2]`（L373，**不依赖 mBaseSize**）
  - `dstNdStride = s2BaseSize*mBaseSize/2`（L371，**依赖 mBaseSize**）
  - UB 总大小 = `2*CeilDiv(mBaseSize,2)*s2BaseSize*sizeof(QK_T)`（InitBuffers L124，依赖 mBaseSize）
- vector 侧 ProcessVec1（`service_vector.h`）：
  - qk 读基址 = `resMm1UB_[pingpong*(UB_BANK_STRIDE/sizeof(QK_T))]`（L391，**不依赖 mBaseSize**）
  - `qkVLstride = (UB_BANK_DEPTH_STRIDE/sizeof(QK_T))/2 * mBaseSize`（L392，**依赖 mBaseSize**）
  - weight/qScale 缓冲 = `2*CeilDiv(s1BaseSize,2)*gSize`（L163/167，依赖 s1BaseSize）
  - odd-AIV scoreGm 写偏移基址 = `CeilDiv(s1BaseSize,2)*Align(kSeqSize,s2BaseSize)`（L403-404，依赖 s1BaseSize）
  - `MulWeightAndReduceSum`（vector1.h L206+）：qk 布局 [G, 2×S2]？每 head 读两段 128-float
    （`qk+128*i` 与 `qk+128*i+qkVLStride`），经 DeInterleave/交错 cast 合并成 S2。**qkVLStride 变了，
    这段布局对 mBaseSize 的依赖需要逐行核对。**
- cube 的 pingpong 基址（`loop%2*64`）与 vector 的 pingpong 基址（`pingpong*UB_BANK_STRIDE/4`）
  是两个**不同公式**，只有在 `UB_BANK_STRIDE==256B` 时相等；改动 mBaseSize 后需确认二者仍一致。
- `BatchMulWeightAndReduceSum`（vector1.h L597）**只支持 batch==1/2**；heads=64 bs1 时
  `curAivS1ProcNum∈{1,0}`（even/odd AIV），OK，但确认过该分支仍成立。

## 5. 新会话建议的第一步（诊断顺序）

1. **回退 base size 对齐**（5 个文件见 §2），A5 重测 `--heads 64 --batch-sizes 1 --source-lens 65536`：
   - 若仍 `row sets differ` → heads=64 是**改动前就存在**的 latent bug，与本次改动无关，
     需单独立项查 heads=64（s1BaseSize=4/mBaseSize=256 老配置下已出错）。
   - 若通过 → 本次改动引入，需在 §4 的 UB 布局契约里定位。
2. 回退后再测 `--heads 32` 全 sweep，确认恢复此前已知正确行为（bs≤8 反超、bs≥16 0.82-0.91x）。
3. 若确认改动引入，可选择：
   - 修 UB 布局（对齐 qkVLstride / dstNdStride / pingpong 基址公式）；
   - 或**放弃全量对齐，改为只对齐 Branch 2**（大 batch 整行无 LD，走与官方相同的 base size，
     Branch 1 保持 mBaseSize=256/s1BaseSize=ceil(256/gSize)）——Branch 1/2 在 `ComputeSplitInfo`
     处天然可各自取 base size，代价是两分支 scoreGm 布局不同、workspace 需按最大者分配。
4. 大 batch 劣化若对齐后仍收窄不足，候选：移植 AICPU 代价调度（`AssignByBlock`+`ForceAssign`）；
   bs16 的 Branch 1 2:1 不均需单独处理（整行回退或块级连续切分）。

## 6. 环境与验证命令

- 本机（Windows）只写码；A5 Linux 机编译验证：`bash build_c8.sh` +
  `python3 tests/test_quant_lightning_indexer_c8.py --batch-sizes 1,4,8,16,24,32,48,64 --source-lens 65536,131072 --heads 32`。
- MTP：`python3 tests/test_quant_lightning_indexer_mtp_c8.py --batch-sizes 1,4,8,16 --source-lens 65536 --heads 32,64 --queries-per-request 1,2,3,4,5`。
