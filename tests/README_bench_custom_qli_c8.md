# ops-transformer 新版 QuantLightningIndexer 线性度复测

## 背景

`fused_li_manage_c8` 的 LI 部分参考旧版 `quant_lightning_indexer`。旧版（torch_npu
内置 `npu_quant_lightning_indexer`）在 A5 上小 batch 线性度差：

| batch | source_len | official_c8_li_us |
| --- | --- | --- |
| 1 | 65536 | 84.334 |
| 1 | 131072 | 163.732 |
| 4 | 65536 | 84.986 |
| 4 | 131072 | 164.630 |
| 8 | 65536 | 86.851 |
| 8 | 131072 | 179.414 |
| 16 | 65536 | 90.607 |
| 16 | 131072 | 171.977 |
| 24 | 65536 | 89.638 |
| 24 | 131072 | 177.826 |
| 32 | 65536 | 168.008 |
| 32 | 131072 | 326.661 |
| 48 | 65536 | 171.722 |
| 48 | 131072 | 336.086 |
| 64 | 65536 | 273.994 |
| 64 | 131072 | 494.131 |

ops-transformer-master 下的 `experimental/attention/quant_lightning_indexer`
为最新版本，内核新增 LD 分核加载阶段（`ProcessLD`/`LdSplitCoreInfo`），
疑似针对该问题。本复测用同口径用例对比新旧两版时延。

## 前置条件

- **A5（Ascend 950）Linux 机器**：CANN 编译与算子运行不支持 Windows，
  编译测试全部在 A5 机器上进行；
- 已 source CANN 9.1 环境、已安装 torch/torch_npu（即能跑
  `test_fused_li_manage_c8.py` 的同一套 python 环境）；
- `ops-transformer-master` 源码已拷贝到 A5 机器。

## 步骤

### 1. 编译 custom 算子包（A5 机器上，ops-transformer-master 根目录）

```bash
bash build.sh --pkg --experimental --soc=ascend950 \
    --ops=quant_lightning_indexer,quant_lightning_indexer_metadata
```

出现 `Self-extractable archive ... successfully created.` 即成功，
run 包位于 `build_out/`（文件名按 CPU 架构为
`cann-ops-transformer-custom_linux_x86_64.run` 或
`cann-ops-transformer-custom_linux_aarch64.run`）。

### 2. 安装并激活（不影响系统默认 CANN 环境）

```bash
./build_out/cann-ops-transformer-custom_linux_$(uname -m).run \
    --install-path=$HOME/qli_custom
source $HOME/qli_custom/vendors/custom_transformer/bin/set_env.bash
```

激活后 `torch.ops.custom.npu_quant_lightning_indexer` 即可用。
只想跑旧版对比时，开一个新 shell 不 source 即可（安装不改动系统环境）。

### 3. 运行复测

把 `bench_custom_qli_c8.py` 拷贝到本仓库 `tests/` 目录下（在 A5 机器上），
用跑 `test_fused_li_manage_c8.py` 的同一 python 环境执行。

**注意：新旧算子必须在两个独立进程里跑。** source set_env.bash 后，
torch_npu 内置的 `npu_quant_lightning_indexer` 与 custom 包算子同名，
旧入口会解析到 custom 包的 host/kernel 库并按旧接口调用，直接段错误
（`--op both` 实测崩溃即此原因）。因此：

```bash
# 旧版基线：干净的 shell（不 source set_env.bash）
cd nanovllm-DSA-offload-ops_a5/tests
python bench_custom_qli_c8.py --op official --iters 20

# 新版：source 后的 shell
source $HOME/qli_custom/vendors/custom_transformer/bin/set_env.bash
python bench_custom_qli_c8.py --op custom --iters 20
```

参数与原有 sweep 一致（默认值）：
`--batch-sizes 1,4,8,16,24,32,48,64`、`--source-lens 65536,131072`、
`--heads 32`、`--budget 6144`（budget 仅保留参数，不影响 LI 计时）。

输入生成与 `_c8_lidu_case.py` 的 `make_case` 完全同口径（仅去掉
pool/native_topk 部分）。custom 侧 weights 默认转 **fp32**（950 上
custom 算子要求 fp32 weights，官方 pytest 即此配置；`--custom-weights-dtype
bf16` 可切回对比）。metadata 分核信息每个 shape 只算一次，不计入时延。

两列结果合并对比；如需同一进程内做 topk 一致性校验，可试
`--op both --cross-check`（若段错误即回到双进程方案，平手排序不同仅
打印 mismatch 计数，不报错）。

### 3.5 确认调用的是 ops-transformer 编译的算子

按可信度从低到高，四层验证：

**a. 命名空间来源**：`torch.ops.custom` 命名空间只由 custom 算子包注册，
torch_npu 内置版注册在 `torch_npu.*` / `torch.ops._C_ascend.*`，不会注册
`custom`。source 前后对比：

```bash
# source set_env.bash 之前
python -c "import torch, torch_npu; print(hasattr(torch.ops, 'custom'))"   # False
# source 之后
python -c "import torch, torch_npu; print(hasattr(torch.ops, 'custom'))"   # True
```

**b. 搜索路径**：source 后 custom 包的 vendor 路径应在最前：

```bash
echo "$ASCEND_CUSTOM_OPP_PATH" | tr ':' '\n' | head -3   # 应含 qli_custom/vendors/custom_transformer
echo "$LD_LIBRARY_PATH"     | tr ':' '\n' | grep -i custom_transformer
```

**c. 进程已加载库**：首次调用算子后，检查进程实际加载了哪些 .so：

```python
import torch, torch_npu, os
# 先随便跑一次 custom 算子，触发 dlopen
with open("/proc/self/maps") as f:
    hits = sorted({l.split()[-1] for l in f if "custom_transformer" in l})
print(hits)   # 应列出 qli_custom 下的 libopapi / libop_kernel 等
```

**d. 决定性冒烟测试（改目录法）**：把安装目录改名再跑，调用失败即证明
走的就是这个包，改回来恢复：

```bash
mv $HOME/qli_custom $HOME/qli_custom.bak
python -c "import torch, torch_npu; print(hasattr(torch.ops, 'custom'))"   # 应 False/报错
mv $HOME/qli_custom.bak $HOME/qli_custom
```

另外注意：custom 版**必须传 metadata** 输入（旧版无此参数），
`bench_custom_qli_c8.py` 能正常出结果本身就说明走的是新算子链路。

### 4. 结果解读

输出格式：

```
batch source_len official_c8_li_us custom_qli_us speedup (device=...)
```

- `official_c8_li_us`：torch_npu 内置旧版（应复现上表）；
- `custom_qli_us`：ops-transformer 新版；
- `speedup = official / custom`。

线性度判据：`custom_qli_us` 随 batch 增长应近似线性（batch 翻倍、时延约翻倍），
且小 batch（1/4/8）相对旧版有明显下降；若 batch=1→64 区间仍出现
batch 增大而时延不增（核用不满）或增速陡增的拐点，说明新版未解决该问题。

## 回滚 / 恢复出问题前的环境

若发现"干净 shell"也受影响（原测试报错/段错误），按顺序排查，全程用
`mv` 改名隔离而非删除，可随时改回：

```bash
# 1. 新开 shell，确认环境变量没被持久化
env | grep -iE "ascend.*(path|vendor)|ld_library"
grep -n "set_env\|qli_custom\|custom_transformer" ~/.bashrc ~/.profile 2>/dev/null
#   有 source 行则注释掉，重开 shell

# 2. 检查是否装进了系统 CANN vendors（未加 --install-path 会装到这里，
#    导致所有 shell 都加载 custom 包）
ls ${ASCEND_HOME_PATH}/opp/vendors/
#   若存在 custom_transformer：
mv ${ASCEND_HOME_PATH}/opp/vendors/custom_transformer \
   ${ASCEND_HOME_PATH}/opp/vendors/custom_transformer.bak

# 3. 自定义安装目录（若用了 --install-path 只影响这里）
mv $HOME/qli_custom $HOME/qli_custom.bak 2>/dev/null

# 4. 验证恢复：干净 shell 跑原测试
python test_fused_li_manage_c8.py --batch-sizes 1 --source-lens 65536 \
    --heads 32 --iters 2

# 5. 仍异常：查 NPU 与残留进程（段错误可能弄脏驱动状态）
npu-smi info
ps aux | grep -E "python|acl" | grep -v grep
#    必要时复位设备或重启服务器
```

注意：run 包"不支持卸载"，所以安装时必须带 `--install-path`，
保证系统 CANN 目录零改动；回滚只需改名/删除自定义目录。

### 高危：installer 会装进你自己的 ASCEND_CUSTOM_OPP_PATH

run 包 installer 的选目录逻辑：`--install-path` > 环境变量
`ASCEND_CUSTOM_OPP_PATH` > `${ASCEND_OPP_PATH}`。本仓库正常跑测试的
shell 必然已 `export ASCEND_CUSTOM_OPP_PATH=_custom_opp_c8/vendors/customize`
（README_c8.md 要求），**此时不带 `--install-path` 执行 run 包，
custom_transformer 会被装进你自己算子的 vendor 目录**，导致你自己的
算子和 torch_npu 内置算子全部加载到 custom 包文件 → 处处段错误。

排查（在仓库根目录）：

```bash
ls _custom_opp_c8/vendors/ _custom_opp_c8/vendors/customize/ 2>/dev/null
find _custom_opp_c8 -maxdepth 4 -name "*custom_transformer*" 2>/dev/null
ls ${ASCEND_HOME_PATH}/opp/vendors/ 2>/dev/null
```

恢复：把找到的 `custom_transformer` 目录 `mv` 成 `.bak` 隔离，
再按 README_c8.md 的编译章节重新 export 一遍环境并验证原测试。

教训：**run 包必须带 `--install-path`**，且最好在未设置
`ASCEND_CUSTOM_OPP_PATH` 的干净 shell 里安装；set_env.bash 会把
custom_transformer 前插到 `ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH`，
因此新旧算子必须分 shell 跑（见第 3 节）。

### Docker 环境回滚

容器内污染（默认路径装进 `${ASCEND_HOME_PATH}/opp/vendors/`、bashrc 被改、
驱动状态异常）最省事的恢复是**弃用当前容器，从原镜像起新容器**：

```bash
# 宿主机上
docker ps -a | grep <你的容器名>          # 记下原容器启动参数（-v 挂载、--device 等）
docker rm -f <旧容器名>                   # 代码若在宿主机挂载目录则不丢
# 用与原来完全相同的参数起新容器（关键是 -v 挂载路径和 --device /dev/davinciX）
```

前提：镜像未被 `docker commit` 污染（`docker history <镜像名>` 确认；
若被 commit 过，回退到原始镜像 tag 再起）。

不想弃容器则按上文第 1~5 步在容器内做 `mv` 隔离 + `docker restart`
（清掉会话级 env）。后续复测建议：build/install 全在容器内做，且
`--install-path` 只指到容器内路径；污染了就整个容器重来，宿主机零影响。

## 复测结论记录

（跑完后把输出表贴到此处/发回分析，若新版仍不满足线性度，
再按改造构想在 fused_li_manage_c8 上实施。）
