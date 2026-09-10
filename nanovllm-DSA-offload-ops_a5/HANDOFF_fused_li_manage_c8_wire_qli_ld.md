# fused_li_manage_c8（非 MTP）接入 QLI LD 引擎（arch35_ld）——交接文档

> 写给**另一个 Claude 会话**的实现说明（一个算子一个会话的分工）。
> 写于 2026-09-08。本文只描述怎么做 + 必须避开的坑；代码由执行会话完成。
> MTP 版双路径合并（本方案的参照实现）决策细节见 auto-memory 项目文件
> `project_fused_li_mtp_c8_debug.md` 与既有会话摘要，读到不理解处先去看那份。

## 0. 目标与红线

**目标**：把**非 MTP** 算子 `fused_li_manage_c8` 的打分/归并/发布引擎，从当前的
`arch35_payload_c8` 换成/接上「此前做的 QLI 线性度优化算子」＝`arch35_ld` LD 引擎
（块池切分 + S2 跨核打分并行 + LD 归并），输出语义不变，官方套件照过。

**红线（违反=污染别的现场）**：
1. **不要动** `fused_li_manage_mtp_c8/`（尤其 `op_kernel/arch35_ld/` 与
   `a5_fused_li_manage_mtp_c8_qli_gl.h`）——那是另一个会话正等远端验证的现场，
   你对 arch35_ld 的任何修改都会连带污染它。
2. **绝对不要动** `csrc/src/backup/fused_li_manage_mtp_c8`（历史参照版，bs>16 崩 / ≤16 对）。
3. 判据 = 远端编译 + 官方测试套件 **printf-free 整机跑通**（末行 `ok=1` 干净回 prompt）。

## 1. 现状速览（建立心智模型）

### 1.1 要改的目标 op：fused_li_manage_c8（非 MTP）

目录 `csrc/src/fused_li_manage_c8/`（以下相对此目录）：

- 入口 `op_kernel/a5_fused_li_manage_c8.cpp`：单 kernel `a5_fused_li_manage_c8`
  （`KERNEL_TYPE_MIX_AIC_1_2`），宏 `INVOKE_LI_NO_KFC_OP_IMPL` 实例化
  `QuantLightningIndexerKernel<QLIType<fp8,fp8,int32_t,true,TND,PA_BSND,bf16,fp32,fp32,uint16_t>>`
  （命名空间 `QLIKernel`），`Init`+`Process`。include 链：
  `a5_fused_li_manage_c8_kernel.h`（仅 10 行）→ `arch35_payload_c8/lightning_indexer_kernel.h`。
- 引擎 = `op_kernel/arch35_payload_c8/`（lightning_indexer_common/service_cube/service_vector +
  `payload/` hist_topk 五件套）。**组长 payload 服务族，命名空间 `QLICommon/QLIKernel/vector1/topkb16gather`**，
  物理上自含在本 op 的 op_kernel 下。
- **单查询绑定**（与 MTP 的本质差异）：host 强制 `T==B`、`s1Size=1`、`gSize=n1Size`
  （`op_host/a5_fused_li_manage_c8_tiling.cpp` :246-251 强制 T==B、:283 gSize=n1Size、:284 s1Size=1）；
  kernel 侧 payload 驱动 `qSeqSize = tiling->s1Size = 1`
  （`arch35_payload_c8/lightning_indexer_kernel.h:190`）。即**每 request 只有 1 条 query**。
- tiling：`LIC8TilingData` **只定义在 `op_host/a5_fused_li_manage_c8_tiling.h:62-78`**
  （BEGIN/END_TILING_DATA_DEF + REGISTER）。字段顺序＝kernel 侧内存布局契约：
  `bSize,tSize,n2Size,gSize,s1Size,s2Size,sparseCount,keyStride0,keyDequantScaleStride0,
   usedCoreNum,blockSize,maxBlockNumPerBatch,sparseMode,poolSize,cacheSlotsSize`。
  ⚠️ 本 op **没有 op_kernel/tiling.h，也没有 op_kernel/workspace.h**（要新建）。
- host 核数/workspace：`blockDim = CalcTschBlockDim(aivNum, aicNum, aivNum)`＝满 24 AIC
  （tiling.cpp:266-270）；`usedCoreNum = aicNum`（:289）；
  workspace = `libApi + 4*ceil(s2Size/128)*128*2B*aicNum`（仅 score 区，:272-278）。
- **无 stage2 OrderedMissUnion**：victim 驱逐 + cache 提交**内联**在 payload service 里——
  `QLIVector::ProcessTopK` → `FindVictimsOnDemand`（service_vector.h:572）/`FinalizePayloadUpdate`（:803），
  `cacheSlotsGm.SetValue(...)` 在 :964-965。
- classify 用 `op_kernel` 顶层 `a5_fused_li_manage_classify_vf.h`（guard
  `LIGHTNING_INDEXER_DECODE_UPDATE_A5_CLASSIFY_VF_H`，ns `TopkIndexerClassifyVF`）。
- `op_kernel/arch35/` 是**孤儿遗留**（无引用），别碰。

### 1.2 要接入的 LD 引擎：`fused_li_manage_mtp_c8/op_kernel/arch35_ld/`

quant_lightning_indexer_c8 的 fork（这就是「qli 线性度优化」的成果）。**物理上只存在于
mtp op 的 op_kernel 下**（含 `vf/` 子目录）：common.h / kernel.h（驱动 `QLIPreload`）/
service_cube.h / service_vector.h + `vf/`（topk、vector1、vf_topk、vf_topk_16_gather、classify）。

在 MTP 双路径合并时已 sed 去冲突，命名空间为 **`QLILdCommon / QLILdKernel / vec1ld / LdTopkB16Gather`**
（原 QLICommon/QLIKernel/vector1/topkb16gather）。**所以它已可在同一 TU 与 payload 服务族共存而不撞名。**

关键绑定事实（MTP 参考实现已验证调通）：
- 驱动 `QLIPreload<QLIT>::Init` 签名：`arch35_ld/quant_lightning_indexer_kernel.h:58-66`，
  18 个 GM 参数（query…workspace, **ldWorkspace**, tiling, pipe）。
- `InitTilingData`（:184-224）**直接按 MTP tiling 字段名**读：`usedCoreNum`、`batchSize`、
  `indexHeads`→qHeadNum/gSize、`maxCandidateLen`→kSeqSize、`maxBlockNumPerBatch`、
  `keyStride`、`scaleStride`、`poolSize`、`sourceCapacity`→cacheSlotsSize。
- **硬编码 MTP-4 绑定**：`constInfo.qSeqSize = 4`（:192，注释「MTP-4 绑定」）；
  `IsActiveRequest` 期望 `queryEnd==(bIdx+1)*4`（:303）。→ 非 MTP 单查询必须改成 1。
- `constInfo.isWholeRowGreedy = batchSize>16`（:223）是 host 级开关。
  注意与 `arch35_ld/quant_lightning_indexer_service_vector.h:274` 的
  `isWholeRowGreedy_ = false`（Branch1 强制、勿还原）是两处，别混。
- classify：`arch35_ld/vf/a5_fused_li_manage_classify_vf.h` 与 fused_li_manage_c8 顶层那份
  **代码逐字节相同、共享同一 include guard**，仅 LD 份多一段注释（含 ClearSpr 说明）；
  ClearSpr<AR> 代码两侧都在（diff 只有注释差）。

### 1.3 include 解析机制（反直觉，务必先懂）

`build_c8.sh:58-61` 把 4 个 op 的 `op_host`、`op_kernel` **全量 `cp -a` 合并**到
`GENERATED/op_kernel` 再统一编译。因此**跨 op include 合法**：MTP 的 qli_gl.h 就
`#include "arch35_payload_c8/..."`，实际命中 fused_li_manage_c8 自己那份；
同理非 MTP 若 `#include "arch35_ld/..."` 也能命中 mtp 那份。

**陷阱＝cp 覆盖**：op_dir 循环按 fused_li_manage_c8 → fused_li_manage_mtp_c8 顺序，
**同名相对路径后者覆盖前者**。推论：
- 你若在 fused_li_manage_c8/op_kernel/arch35_ld **复制一份就地改**，会被 mtp 的 arch35_ld
  在合并时覆盖 → 改动编译期丢失。**严禁同名目录复制。**
- 正确做法＝复制到**异名目录**（如 `arch35_ld_c8/`），并把 LD 引擎内部 self-include 的相对路径
  `../arch35_ld/` sed 成 `../arch35_ld_c8/`（先 grep，只有
  `quant_lightning_indexer_service_vector.h:40-42` 三行 `#include "../arch35_ld/vf/..."`；
  vf 内部若有互相引用也一起核）。这样自含、不碰 mtp、无覆盖。

## 2. 参照实现（精读对象）：MTP 单算子双路径

架构/改名/字段坑全部在此跑通过（本地已合并、待远端验证）。文件（mtp op 内）：
- 入口 `op_kernel/a5_fused_li_manage_mtp_c8.cpp:105-139`：kernel 里
  `if (tilingData.batchSize > 16U)` → gl(组长 payload)；else → impl(我方 LD)。
  分叉键 host/kernel 一致：host 侧 `MAX_BATCH_OUR_PATH=16`
  （`op_host/a5_fused_li_manage_mtp_c8.cpp:158-160`）。
- `op_kernel/a5_fused_li_manage_mtp_c8_qli.h`（我方 LD 包装）：顶部
  `using QLITilingData = A5FusedLiManageMtpC8TilingData;`(:24) 后
  `#include "arch35_ld/quant_lightning_indexer_kernel.h"`(:26)；包装类 `QuantLiMtpPhase::Init`
  = 18 参（含 ldWorkspace）。
- `op_kernel/a5_fused_li_manage_mtp_c8_qli_gl.h`（组长 payload 包装）：
  `using LIC8TilingData = A5FusedLiManageMtpC8TilingData;`(:19) 后 include arch35_payload_c8 三件套；
  只改 include guard 与外层命名空间 `..._impl`→`..._gl`。
- 同一 tiling 结构被两套引擎共用：前 11 字段是 QLI 布局，mtp 追加
  `queryTileSize/splitEnable/scoringCoreNum` 3 字段（组长用）。
- 同一 TU 两套服务族的命名冲突面只有 4 个全局命名空间名 → 已把 LD 族改名 QLILd*；
  classify 因同 guard+代码同 → 单实例化、无需改名。
- workspace 统一用我方 superset `TotalBytes(scoreStride64, batch, heads)` 3 参版
  （`a5_fused_li_manage_mtp_c8_workspace.h:72-77`）；组长路径只用阈值区之前。
- stage2 union 两版同语义 → 统一用我方 union.h（PIPE_ALL 加固）。

## 3. 需要执行会话先拍板的 4 个设计决策

MTP 方案不能照抄，因为引擎形态、tiling 契约、stage2 模型都不同。

### 决策 A：全替换 or 双路径
- **全替换**：stage1 引擎整体换成 LD 引擎，不再 include payload。TU 干净、改动小。
  风险：LD 引擎在 MTP 上 bs>16 + 大 source（65536）曾确定性崩（507015，结构性上限、未根因）；
  非 MTP 是单查询，崩形是否存在**未验证**——官方套件很可能含 bs>16 + source 2^18，若崩则失败。
- **双路径**（MTP 同款）：payload 作 >16（或某阈值）fallback，≤阈值走 LD。安全但复杂（见决策 C）。
- **建议**：先全替换做通 + 远端验官方套件；若某 bs 崩，再上阈值 fallback。
  因此先把 payload 整树**保留不删**（先注释掉 include，不物理删除），方便回退。

### 决策 B：LD 引擎放哪
- **推荐**：复制到 `fused_li_manage_c8/op_kernel/arch35_ld_c8/`（异名目录）+ sed 相对 include
  （§1.3）。**不要**用 `arch35_ld` 同名复制；**不要**跨-op include mtp 的 arch35_ld
  （耦合且会污染在途的 mtp 验证）。

### 决策 C：tiling 契约（工作量最大的部分）
- LD 引擎按 **QLI/MTP 布局字段名**读（§1.2）；现有 `LIC8TilingData` 字段名/含义是另一套
  （§1.1），两者不是字段超集关系。
- **单引擎路径**：把 fused_li_manage_c8 的 tiling 结构换成/扩展成 LD 引擎要读的字段名并填对应值
  （batchSize=b、indexHeads=n1Size、maxCandidateLen=cacheSlotsSize、keyStride=BLOCK_SIZE*HEAD_DIM、
  scaleStride=BLOCK_SIZE、poolSize=…、sourceCapacity=cacheSlotsSize、maxBlockNumPerBatch 沿用、
  usedCoreNum=24）。kernel 入口 `LI_COPY_TILING_DATA` 的 struct 跟着换。
  ⚠️ 需**新建 op_kernel/tiling.h**（参照 mtp `a5_fused_li_manage_mtp_c8_tiling.h`）。
- **双引擎路径**：需要装两套字段的 superset tiling 结构（MTP 只加 3 字段，这里差距大，须手工并集；
  host 单点填、kernel 各读各的）。工作量显著更大。
- host workspace 需**追加 LD 归并区**：抄 mtp `op_kernel/a5_fused_li_manage_mtp_c8_workspace.h`
  的 `LdScoreBytes/LdIndexBytes/TotalBytes`（heads=indexHeads）。

### 决策 D：单查询绑定 qSeqSize=1（LD 引擎现在硬编码 4）
- 非 MTP：每 request 1 query（T==B，无 actualSeqLengthsQuery 输入；payload 驱动默认 1）。
- 必须把 arch35_ld 的 `qSeqSize=4` 改成 1（或从 tiling 读 s1Size=1），并核对
  `IsActiveRequest` 期望 queryEnd、`actualSeqLengthsGmQ` 为空时的默认长度、
  以及打分/归并/classify 的逐 token 发布语义在单查询下是否成立。**最易翻车，改前精读。**
- kernel 入参对齐：非 MTP 无 actualSeqLengthsQuery → LD 的 actualSeqLengthsQ 传 null（走默认）；
  K 侧 payload 把 candidateLens 当 actualSeqLengthsK 传（payload `Init` 第 7 参），LD Init 第 7 参含义一致，对齐即可。

## 4. 建议实施步骤
1. 建备份/记录基线（可选：先把未改的 fused_li_manage_c8 远端编一次确认绿）。
2. 复制 arch35_ld → `fused_li_manage_c8/op_kernel/arch35_ld_c8/`（含 vf/）；
   grep `../arch35_ld/` 全量改 `../arch35_ld_c8/`。
3. 新建 op_kernel/tiling.h（QLI/MTP 布局）或扩 LIC8TilingData；host DoTiling 填齐 LD 字段（决策 C）。
4. host workspace 补 LD 归并区（抄 mtp workspace.h 公式）。
5. kernel 入口换驱动：保留非 MTP 输出张量绑定（sourceIds/destinationSlots/missCounts/cacheSlotsPool…），
   把 `INVOKE_LI_NO_KFC_OP_IMPL` 换成对 QLIPreload（或自写薄包装）的 Init+Process；
   对齐 LD 的发布目标（routePairRows/thresholds/missCount/topkSlots）与非 MTP 输出口径。
6. 决策 D：qSeqSize=1 + 相关校验。
7. **提交模型**：非 MTP 无 OrderedMissUnion。victim 驱逐 + cacheSlotsPool 提交目前内联在 payload
   （service_vector.h:572/:803/:964-965）。接 LD 后谁来做？两条路：
   (i) LD 只出 topk/miss 集合，victim+提交复用 payload 的内联逻辑（意味着 payload 部分仍在 TU 内）；
   (ii) 把 victim+提交逻辑仿写/移植到 LD 后处理。
   定决策时精读 §1.1 指出的 payload 三处。
8. 本地自查（编译错三连查：include guard 撞、命名空间撞、tiling 字段名），再交用户远端编 + 官方套件。

## 5. 验收 checklist（执行会话自检）
- [ ] 未动 fused_li_manage_mtp_c8/op_kernel/arch35_ld 与 qli_gl.h（可 md5 对比）
- [ ] 未动 csrc/src/backup/
- [ ] arch35_ld_c8 内 `../arch35_ld/` 相对 include 已全改（grep 为空）
- [ ] tiling 字段名/顺序与 LD 引擎所读一致
- [ ] qSeqSize=1 逻辑核对（含 IsActiveRequest / 空 actualSeqLengthsQ 默认）
- [ ] victim/cache 提交模型已定且实现
- [ ] 远端编译通过
- [ ] 官方套件 printf-free、末行 ok=1

## 6. 坑位汇总
- **cp -a 同名目录覆盖**（§1.3）→ 必须异名目录。
- **classify 双份同 guard**：若同时 include LD classify 与 payload classify，只有先 include 的生效；
  代码相同无碍。若只改 LD classify 会与 payload classify 行为分裂 → 不要两边都改。
- **isWholeRowGreedy 两处**：LD `service_vector.h:274` 的 `isWholeRowGreedy_=false`（Branch1，勿还原）
  与 `kernel.h:223` 的 host 级开关是两回事。
- **qSeqSize=4 是 LD 引擎深绑**：改 1 时把 actS1Size/IsActiveRequest/发布逐 token 语义全过一遍。
- **别照抄 MTP 入口结构**：MTP 有 stage2 union 与 bs>16 分叉；非 MTP 没有这些，抄了只引入死代码。
