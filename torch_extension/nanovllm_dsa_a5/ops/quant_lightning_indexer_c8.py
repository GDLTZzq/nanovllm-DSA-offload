from __future__ import annotations

import torch

# Fused single-op QuantLightningIndexer (on-device split, no AICPU metadata).
# Registered by the C++ extension; each call launches one repository-local
# MIX kernel. weights 必须以 fp32 下发（kernel GlobalTensor<float> 无转置），
# bf16 权重由调用方 .float() 无损转换（与 bench_custom_qli_c8.py --custom-weights-dtype fp32 一致）。
quant_lightning_indexer_c8 = torch.ops.nanovllm_dsa.quant_lightning_indexer_c8


def fused_quant_lightning_indexer_c8(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    block_table,
):
    """Dispatch wrapper: converts bf16 weights to fp32 for the 950 kernel."""
    if weights.dtype != torch.float32:
        weights = weights.float()
    return quant_lightning_indexer_c8.default(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
    )


__all__ = ["quant_lightning_indexer_c8", "fused_quant_lightning_indexer_c8"]
