# QuantLightningIndexer (QLI) 在 Ascend 950 (A5) 上的调通指南

> 目标：在 A5 机器上跑通 ops-transformer `experimental/attention/quant_lightning_indexer` 的官方 pytest，
> 并对比新版 QLI（两阶段：metadata + 主算子）与 torch_npu 内置旧版在小 batch 下的线性度。
>
> 本文档在以下环境验证通过：Ascend 950PR (A5/950PR_9579)、Python 3.11.10、torch 2.10.0+cpu、
> torch_npu 2.10.0.post1.dev20260528、CANN 9.1、Debian 容器。

## 1. 背景（30 秒版）

新版 QLI 为两阶段架构：

```
npu_quant_lightning_indexer_metadata（AICPU 算子）
  └─ 输出 1024 个 int32 的 QliMetadata：
     LI 区间（每个 AIC core 负责的 S2 段）、LD 区间（每个 AIV core 的合并任务）
  └─ 输入：actual_seq_lengths、head 数、layout、sparse 参数等（不含 query/key 数据）

npu_quant_lightning_indexer（kernel 算子，第 9 个输入是上面的 metadata）
  └─ 每个 AIC core 在分到的 S2 段上做 partial topk，AIV core 做 ProcessLD 合并
```

S1 = query 侧实际 token 数，S2 = key 侧实际 token 数。当 maxS2 超过阈值
（A5 上 > 5×128=640，或 > sparse_count）时，metadata 按 128-block 粒度把 S2
拆给多个 AIC core 并行，解决大 S2 单核串行导致的 batch 线性度差问题。

**metadata 只依赖 shape/长度/参数，不依赖 tensor 数值**——actual_seq_lengths
不变时可缓存复用，不必每次重算。

## 2. npu-smi info  检查npu状态

## 3. 构建并安装 custom vendor 包


`torch_ops_extension` 的 aclnn 调用层通过 `EXEC_NPU_CMD_V1` 宏在运行时
`dlopen` 自 `ASCEND_CUSTOM_OPP_PATH` 指向的 `op_api/lib/libcust_opapi.so`。
因此必须先构建/安装 QLI 的 vendor 包，使该路径生效。

```bash
# 从gitcode仓库上 clone master分支
https://gitcode.com/cann/ops-transformer.git

#编译安装
bash build.sh --pkg --experimental --soc=ascend950 --ops=quant_lightning_indexer,quant_lightning_indexer_metadata
./build_out/cann-ops-transformer-custom_linux-x86_64.run --install-path={your_install_path}

#可以先不着急source,在步骤5时再source也行
source {your_install_path}/vendors/custom_transformer/bin/set_env.bash

# 验证环境变量
echo "$ASCEND_CUSTOM_OPP_PATH"   # 必须指向 custom_transformer
ls "$ASCEND_CUSTOM_OPP_PATH"/op_api/lib/libcust_opapi.so   # 必须存在
```

> 注意：`set_env.bash` 可能重复追加/污染 `ASCEND_CUSTOM_OPP_PATH`
> （本环境出现过 4 份重复路径），不影响使用。
> `$HOME` 在 root 容器下是 `/root`，找文件时用绝对路径。

## 4. 补齐 torch_ops_extension（upstream 缺失，必须手动添加）

QLI 是实验目录，**官方仓库没有带 torch_ops_extension**（gitcode master 确认缺失）。
torch 接口的注册（`torch.ops.custom.*`）完全依赖这个扩展——vendor 包的
`libcust_opapi.so` 只提供 aclnn C 接口，**不会**注册 `torch.ops.custom`。

在 `<repo>/experimental/attention/quant_lightning_indexer/torch_ops_extension/`
下补齐以下文件：

### 4.1 从 stem_indexer 复制 4 个公共文件

```bash
cd <repo>/experimental/attention/quant_lightning_indexer
mkdir -p torch_ops_extension/custom_ops/csrc
cp ../stem_indexer/torch_ops_extension/setup.py               torch_ops_extension/
cp ../stem_indexer/torch_ops_extension/build_and_install.sh   torch_ops_extension/
cp ../stem_indexer/torch_ops_extension/custom_ops/csrc/ops_common.h    torch_ops_extension/custom_ops/csrc/
cp ../stem_indexer/torch_ops_extension/custom_ops/csrc/ops_common.cpp  torch_ops_extension/custom_ops/csrc/
```

### 4.2 新建 4 个 QLI 专用文件，同路径下已经提供，复制相同的绝对路径即可

| 文件 | 作用 |
|---|---|
| `custom_ops/csrc/ops_def_registration.cpp` | `TORCH_LIBRARY(custom)` 注册两个算子的 schema |
| `custom_ops/csrc/npu_quant_lightning_indexer.cpp` | 主算子 NPU/Meta 实现，`EXEC_NPU_CMD_V1` 调 aclnn |
| `custom_ops/csrc/npu_quant_lightning_indexer_metadata.cpp` | metadata 算子 NPU/Meta 实现 |
| `custom_ops/__init__.py` | `from . import custom_ops_lib` 触发 .so 加载 + 挂载到 torch_npu |


### 4.3 编译安装

```bash
cd <repo>/experimental/attention/quant_lightning_indexer/torch_ops_extension
bash build_and_install.sh    # setup.py build bdist_wheel + pip install -I
```

### 4.4 验证注册

> 不要在 `torch_ops_extension/` 目录内执行！本地源码目录会遮蔽 site-packages，
> 导致 `import custom_ops` 报 "cannot import name 'custom_ops_lib' (circular import)"。

```bash
cd /tmp
python3 -c "
import torch, torch_npu
import custom_ops
print(torch.ops.custom.npu_quant_lightning_indexer)
print(torch.ops.custom.npu_quant_lightning_indexer_metadata)
"
```

两个 op 能打印出对象即注册成功。
**`dir(torch.ops.custom)` 只显示 `['name']` 是正常现象**（torch 2.10 的 namespace
`__dir__` 不枚举动态算子），用属性访问判断。

## 5. 官方 pytest

```bash
cd <repo>/experimental/attention/quant_lightning_indexer/tests/pytest
# 修正 golden.py 的两处 dead import（官方文件带但仓库没有对应模块）：
sed -i '13s/^/#/' quant_lightning_indexer_golden.py   # import test
sed -i '22s/^/#/' quant_lightning_indexer_golden.py   # import custom_ops
bash test_run.sh single
```

预期：`1 passed`，精度对比 `PctRlt ~97.8% > 95%`（fp8 精度下正常）。
用例参数由 `test_quant_lightning_indexer_paramset.py` 按设备自动选择：
Ascend950 → fp8 + fp32 weights + BSND/PA_BSND + block_size=512 + sparse_mode=3。

## 6. 线性度 benchmark，这个benchmark在fuse_li_managed_c8的tests路径下，模仿之前对office_LI算子的测试，不过通过一个参数区分

bench 脚本在 nanovllm 测试目录（依赖其同目录工具文件，整体拷贝）：
`bench_custom_qli_c8.py`。**旧版（torch_npu 内置）与新版（custom 包）共享同名
CANN op，不能在同一个进程共存（会 segfault），必须两个独立进程分别测。**

进程 A —— 旧版（干净 shell，先不 source set_env.bash）：

```bash
cd <bench_dir>
python3 bench_custom_qli_c8.py --op official --batch-sizes 1,4,8,16,24,32,48,64 --source-lens 65536,131072
```

进程 B —— 新版（source 后的 shell）：

```bash
source <WORK>/qli_custom/vendors/custom_transformer/bin/set_env.bash
cd <bench_dir>
python3 bench_custom_qli_c8.py --op custom --batch-sizes 1,4,8,16,24,32,48,64 --source-lens 65536,131072
```

- `custom` 侧 `custom_qli_us` 不含 metadata 开销（每 case 只算一次 metadata，
  计时只包主算子）；两版对比的都是 QLI 主算子的耗时。
- 若跑 `--op both` 只为验证共存，不能用于数据对比。
- 跑之前 `ps aux | grep bench_custom | grep -v grep` 检查残留进程，有则 `kill -9`。

## 7. 已知坑速查表

| 现象 | 原因 | 处理 |
|---|---|---|
| 编译错 `cannot bind non-const lvalue reference ... to rvalue` | EXEC_NPU_CMD_V1 参数必须是具名左值 | 把 `static_cast<int64_t>(0)` 等换成局部变量 |
| `import custom_ops` 报 circular import | cwd 在 torch_ops_extension/ 下，本地源码遮蔽 site-packages | `cd /tmp` 再跑 |
| `dir(torch.ops.custom)` 只有 `['name']` | torch 2.10 namespace `__dir__` 行为 | 属性访问验证，正常 |
| `AttributeError: _OpNamespace 'custom' object has no attribute ...` | 没 import custom_ops，.so 未加载 | 调用前 `import custom_ops` |
| `call aclnnQuantLightningIndexerMetadata failed` + 日志 `num_heads_q should only be 64, but got 32` | A5 metadata 硬性要求 num_heads_q=64 | 用 `--heads 64` |
| `call aclnnQuantLightningIndexer failed`（int8/fp16 输入） | A5 只接受 fp8_e4m3fn query/key + fp32 weights/scale | 对齐 op_def 的 ascend950 配置 |
| pytest collection 报 `call aclnnQuantLightningIndexer failed` | golden.py `import test` 拖入旧调试脚本 | 注释 dead import（第 5 节） |
| 终端日志刷屏 | `ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1` 残留 | `unset` 两个变量 |
| official + custom 同进程跑崩 | 同名 CANN op 加载旧/新 host+kernel 库冲突 | 独立进程 |
| aclnn 失败但看不到原因 | V1 宏报错不带 errmsg | 开日志：`export ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1`，输出落文件后 `grep -E "\[ERROR\]"` |

## 8. 涉及文件清单

```
<repo>/experimental/attention/quant_lightning_indexer/
├── op_host/quant_lightning_indexer_def.cpp     # 950 配置：fp8 + fp32 weights
├── op_host/quant_lightning_indexer_tiling.cpp  # 主算子 tiling（读 metadata 分核）
├── op_host/quant_lightning_indexer_infershape.cpp
├── op_kernel/quant_lightning_indexer_metadata.h  # LI/LD metadata 布局、QLI_META_SIZE=1024
├── op_kernel/...arch35/...                     # A5 kernel（cube partial topk + vector ProcessLD）
├── torch_ops_extension/                        # 【手动补齐】见第 4 节
│   ├── setup.py / build_and_install.sh         # 复制自 stem_indexer
│   └── custom_ops/
│       ├── __init__.py
│       └── csrc/
│           ├── ops_common.h / ops_common.cpp   # 复制自 stem_indexer
│           ├── ops_def_registration.cpp
│           ├── npu_quant_lightning_indexer.cpp
│           └── npu_quant_lightning_indexer_metadata.cpp
└── tests/pytest/                               # golden.py 两处 dead import 需注释
```
