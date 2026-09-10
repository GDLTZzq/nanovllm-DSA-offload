"""Public binding for the one-kernel non-MTP C8 LIM path."""

import torch


fused_li_manage_c8 = torch.ops.nanovllm_dsa.fused_li_manage_c8


__all__ = ["fused_li_manage_c8"]
