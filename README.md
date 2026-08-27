
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


  | heads | batch | source_len | official_c8_li_us | fused_li_us(Our) | speedup |
| ------- | ------- | ------------- | ---------------------- | -------------------- | --------- |
| 32    | 1     | 65536       | 82.458               | 52.538             | 1.569   |
| 32    | 1     | 131072      | 161.433              | 51.147             | 3.156   |
| 32    | 4     | 65536       | 83.831               | 69.111             | 1.213   |
| 32    | 4     | 131072      | 173.396              | 49.934             | 3.472   |
| 32    | 8     | 65536       | 84.849               | 60.076             | 1.412   |
| 32    | 8     | 131072      | 166.815              | 76.912             | 2.169   |
| 32    | 16    | 65536       | 89.532               | 91.560             | 0.978   |
| 32    | 16    | 131072      | 168.552              | 187.427            | 0.899   |
| 32    | 32    | 65536       | 166.142              | 176.587            | 0.941   |
| 32    | 32    | 131072      | 324.618              | 336.319            | 0.965   |
| 32    | 48    | 65536       | 170.645              | 180.212            | 0.947   |
| 32    | 48    | 131072      | 345.219              | 351.964            | 0.981   |
| 32    | 48    | 65536       | 170.645              | 180.212            | 0.947   |
| 32    | 64    | 131072      | 494.895              | 504.797            | 0.980   |
