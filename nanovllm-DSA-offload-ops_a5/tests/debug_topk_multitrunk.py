#!/usr/bin/env python3
"""Diagnose c8 topk divergence: sweep batch x candidate to isolate the trigger."""
from __future__ import annotations

import torch
import torch_npu  # type: ignore  # noqa: F401
import nanovllm_dsa_a5  # noqa: F401

from _c8_lidu_case import make_case
from _lidu_utils import TOPK


def run_case(device: torch.device, batch: int, source_len: int, candidate: int, seed: int):
    case = make_case(device, batch, source_len, 32, [3072] * batch, (0, 10), 7, seed,
                     [candidate] * batch)
    pool = case.initial_pool.clone()
    src, slots, miss, alias = torch.ops.nanovllm_dsa.fused_li_manage_c8.default(
        case.query, case.key, case.weights, case.query_scale, case.key_scale,
        case.actual_q, case.req_entries, pool, case.cache_tokens,
        case.candidate_lens, case.block_table,
    )
    torch.npu.synchronize()
    ours = src.reshape(batch, TOPK).cpu().numpy()
    ref = case.native_topk.cpu().numpy()
    row = 0
    overlap = len(set(ours[row].tolist()) & set(ref[row].tolist()))
    only_ours = sorted(set(ours[row].tolist()) - set(ref[row].tolist()))
    print(
        f"batch={batch} candidate={candidate} trunks={(candidate + 16383) // 16384} "
        f"overlap={overlap}/{TOPK} miss={int(miss.cpu()[0])}",
        flush=True,
    )
    print(f"  ours_first50={ours[row][:50].tolist()}", flush=True)
    print(f"  only_ours_first20={only_ours[:20]}", flush=True)


def main() -> None:
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    # cross-sweep: batch x candidate (12288 = 96 blocks single-trunk; 16384 = 128 blocks)
    for batch in (1, 6):
        for candidate in (12288, 16384):
            run_case(device, batch, 32768, candidate, 42 + batch * 10 + candidate // 1000)


if __name__ == "__main__":
    main()
