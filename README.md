
本仓库提供 A5 下 quant_lightning_indexer_c8算子，为解决原qli算子小batch下线性度不足的问题。

其中ops-transfomer下的cann官方实验性算子，编译与测试参考文件夹内的md文件，该算子被拆分为了2个算子来完成，并且不支持heads = 32。

csrc下的quant_lightning_indexer_c8为当前实现的算子，实现了单算子逻辑与对heads = 32的支持。



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
bash build_bf16.sh
export ASCEND_CUSTOM_OPP_PATH=$PWD/_custom_opp_bf16/vendors/customize
export NANOVLLM_A5_INSTALL_OPP_PATH=$PWD/_custom_opp_bf16
export NANOVLLM_CUST_OPAPI_LIB=$PWD/_custom_opp_bf16/vendors/customize/op_api/lib/libcust_opapi.so
```

## 测试

每个脚本固定执行行为检查和 NPU Event 时延测试；时延统一打印为 `us`，不设置性能门槛。

```bash
python tests/test_quant_lightning_indexer_c8.py --device npu:0 --batch-size 1,4,8,16,32,48,64 --source-lens 65536,131072 --heads 32 --warmup 10 --iters 300
```
## 当前测试情况

| SeqLen | batch | official_c8(us) | fused_li(us) | speedup |
| -------- | ------- | ------------------ | --------------- | --------- |
| 64K    | 1     | 82.406           | 37.284        | 2.210   |
| 64K    | 4     | 83.677           | 46.683        | 1.792   |
| 64K    | 8     | 84.500           | 70.350        | 1.201   |
| 64K    | 16    | 88.835           | 81.783        | 1.086   |
| 64K    | 32    | 165.507          | 125.851       | 1.315   |
| 64K    | 48    | 171.374          | 178.076       | 0.962   |
| 64K    | 64    | 250.529          | 239.213       | 1.047   |
| 128K   | 1     | 161.389          | 38.461        | 4.196   |
| 128K   | 4     | 165.241          | 47.292        | 3.494   |
| 128K   | 8     | 164.870          | 79.922        | 2.063   |
| 128K   | 16    | 166.068          | 149.102       | 1.114   |
| 128K   | 32    | 325.538          | 232.835       | 1.398   |
| 128K   | 48    | 341.316          | 338.038       | 1.010   |
| 128K   | 64    | 497.184          | 453.943       | 1.095   |
