/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

// 在custom命名空间里注册quant_lightning_indexer算子
//   - schema 入参顺序：必选张量在前，'*' 之后为可选张量与带默认值的属性
//   - 张量 dtype / 属性默认值对齐 op_host/quant_lightning_indexer_def.cpp 与
//     op_graph/quant_lightning_indexer_metadata_proto.h
//   - return_value / device 命名对齐官方 pytest 的调用方式
TORCH_LIBRARY(custom, m)
{
    m.def("npu_quant_lightning_indexer(Tensor query, Tensor key, Tensor weights, "
          "Tensor query_dequant_scale, Tensor key_dequant_scale, "
          "Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None, "
          "Tensor? block_table=None, Tensor? metadata=None, *, "
          "int query_quant_mode=0, int key_quant_mode=0, "
          "str layout_query='BSND', str layout_key='PA_BSND', "
          "int sparse_count=2048, int sparse_mode=3, "
          "int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, "
          "int cmp_ratio=1, bool return_value=False) -> (Tensor, Tensor)");

    m.def("npu_quant_lightning_indexer_metadata(Tensor? actual_seq_lengths_query=None, "
          "Tensor? actual_seq_lengths_key=None, *, "
          "int num_heads_q, int num_heads_k, int head_dim, "
          "int query_quant_mode=0, int key_quant_mode=0, "
          "int batch_size=0, int max_seqlen_q=0, int max_seqlen_k=0, "
          "str layout_query='BSND', str layout_key='BSND', "
          "int sparse_count=2048, int sparse_mode=3, "
          "int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, "
          "int cmp_ratio=1, str device='npu:0') -> Tensor");
}

// 通过pybind将c++接口和python接口绑定，这里绑定的是接口不是算子
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
}
