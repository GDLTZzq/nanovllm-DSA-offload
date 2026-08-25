#!/usr/bin/env python3
"""Benchmark the ops-transformer custom QuantLightningIndexer against the
installed torch_npu version, with the same input generation as
test_fused_li_manage_c8.py.

IMPORTANT: sourcing the custom package's set_env.bash changes op resolution
process-wide. The torch_npu built-in npu_quant_lightning_indexer and the
custom torch.ops.custom.npu_quant_lightning_indexer share the same CANN op
name; calling the built-in entry in a sourced shell can load the custom
host/kernel libraries with the old interface and segfault. Therefore the two
sides are benchmarked in SEPARATE processes:

    # clean shell (no set_env.bash sourced)
    python bench_custom_qli_c8.py --op official

    # shell with: source $HOME/qli_custom/vendors/custom_transformer/bin/set_env.bash
    python bench_custom_qli_c8.py --op custom

Run --op both (same process) only to test whether the two coexist.
"""

from __future__ import annotations

import argparse
import statistics

import torch

import torch_npu  # type: ignore  # noqa: E402,F401

from _c8_lidu_case import (
    normalized_hadamard_128,
    official_c8_lightning_indexer,
    quantize_fp8,
)
from _lidu_utils import TOPK
from _utils import csv_ints, require_a5


BLOCK_SIZE = 128
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--batch-sizes", type=csv_ints, default=csv_ints("1,4,8,16,24,32,48,64")
    )
    parser.add_argument(
        "--source-lens", type=csv_ints, default=csv_ints("65536,131072")
    )
    parser.add_argument("--heads", type=csv_ints, default=csv_ints("64"))
    parser.add_argument("--budget", type=int, default=6144)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    parser.add_argument(
        "--op", choices=["official", "custom", "both"], default="both"
    )
    parser.add_argument(
        "--custom-weights-dtype",
        choices=["bf16", "fp32"],
        default="fp32",
        help="weights dtype for the custom op; 950 requires fp32 per its tests",
    )
    parser.add_argument(
        "--cross-check",
        action="store_true",
        help="compare custom vs native topk (requires --op both)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="print per-phase wall-clock timing to localize hangs",
    )
    return parser.parse_args()


class Inputs:
    pass


def build_inputs(
    device: torch.device,
    batch: int,
    source_len: int,
    heads: int,
    seed: int,
) -> Inputs:
    """Same generation as make_case in _c8_lidu_case.py, minus pool/native_topk."""
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    blocks = source_len // BLOCK_SIZE
    block_table_cpu = torch.stack(
        [
            torch.randperm(blocks, dtype=torch.int64).to(torch.int32)
            for _ in range(batch)
        ]
    )
    query_fp = torch.empty(
        (batch, heads, HEAD_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-1, 1)
    key_fp = torch.empty(
        (blocks, BLOCK_SIZE, 1, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-1, 1)
    hadamard = normalized_hadamard_128(dtype=query_fp.dtype, device=device)
    query_fp = torch.matmul(query_fp, hadamard)
    key_fp = torch.matmul(key_fp, hadamard)
    query, query_scale = quantize_fp8(query_fp)
    key, key_scale = quantize_fp8(key_fp)
    weights = torch.empty(
        (batch, heads), dtype=torch.bfloat16, device=device
    ).uniform_(0.01, 1.0).contiguous()

    inputs = Inputs()
    inputs.query = query
    inputs.key = key
    inputs.weights = weights
    inputs.query_scale = query_scale
    inputs.key_scale = key_scale
    inputs.actual_q = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    inputs.candidate_lens = torch.full(
        (batch,), source_len, dtype=torch.int32, device=device
    )
    inputs.block_table = block_table_cpu.to(device)
    inputs.batch = batch
    inputs.heads = heads
    return inputs


def custom_namespace() -> object:
    # import custom_ops 触发 .so 加载与 TORCH_LIBRARY(custom) 注册；
    # 不 import 时 torch.ops.custom 是空 namespace（torch_npu 创建）
    try:
        import custom_ops  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "custom_ops 包未安装；请先在 torch_ops_extension 目录执行 "
            "bash build_and_install.sh"
        ) from exc
    namespace = getattr(torch.ops, "custom", None)
    if namespace is None:
        raise RuntimeError(
            "torch.ops.custom is not registered; source the custom package "
            "set_env.bash before running --op custom/both"
        )
    for op_name in (
        "npu_quant_lightning_indexer",
        "npu_quant_lightning_indexer_metadata",
    ):
        if not hasattr(namespace, op_name):
            raise RuntimeError(
                f"torch.ops.custom.{op_name} 未注册；请确认 custom_ops 扩展 "
                "已安装且 set_env.bash 已 source"
            )
    return namespace


def custom_metadata(inputs: Inputs, device: torch.device) -> torch.Tensor:
    if inputs.heads != 64:
        raise ValueError(
            f"custom QLI metadata requires num_heads_q=64, got {inputs.heads}; "
            "run with --heads 64"
        )
    namespace = custom_namespace()
    metadata = namespace.npu_quant_lightning_indexer_metadata(
        num_heads_q=inputs.heads,
        num_heads_k=1,
        head_dim=HEAD_DIM,
        query_quant_mode=0,
        key_quant_mode=0,
        actual_seq_lengths_query=inputs.actual_q,
        actual_seq_lengths_key=inputs.candidate_lens,
        batch_size=inputs.batch,
        max_seqlen_q=int(inputs.actual_q.max().item()),
        max_seqlen_k=int(inputs.candidate_lens.max().item()),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        cmp_ratio=1,
        device=str(device),
    )
    return metadata.npu()


def custom_launch(inputs: Inputs, metadata: torch.Tensor, weights_dtype: str):
    weights = (
        inputs.weights.float()
        if weights_dtype == "fp32"
        else inputs.weights
    )
    return custom_namespace().npu_quant_lightning_indexer(
        inputs.query,
        inputs.key,
        weights,
        inputs.query_scale,
        inputs.key_scale,
        actual_seq_lengths_query=inputs.actual_q,
        actual_seq_lengths_key=inputs.candidate_lens,
        block_table=inputs.block_table,
        metadata=metadata,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        cmp_ratio=1,
    )


def official_launch(inputs: Inputs):
    return official_c8_lightning_indexer(
        inputs.query,
        inputs.key,
        inputs.weights,
        inputs.query_scale,
        inputs.key_scale,
        inputs.actual_q,
        inputs.candidate_lens,
        inputs.block_table,
    )


def timed_mean_us(fn, warmup: int, iters: int) -> float:
    retained = []
    for _ in range(warmup):
        retained.append(fn())
    torch.npu.synchronize()

    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        retained.append(fn())
        end.record()
    ends[-1].synchronize()
    if not retained:
        raise AssertionError("timed outputs were not retained")
    return statistics.mean(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    ) * 1000


def check_custom_matches_native(
    inputs: Inputs, metadata: torch.Tensor, weights_dtype: str
) -> None:
    output = custom_launch(inputs, metadata, weights_dtype)
    torch.npu.synchronize()
    topk = output[0] if isinstance(output, tuple) else output
    if topk.numel() != inputs.batch * TOPK:
        raise RuntimeError(
            f"custom QLI returned {tuple(topk.shape)}, "
            f"expected {inputs.batch * TOPK} elements"
        )
    custom_topk = topk.reshape(inputs.batch, TOPK).cpu()
    native_topk = official_launch(inputs).cpu()
    torch.npu.synchronize()
    match = torch.equal(custom_topk, native_topk)
    mismatches = int((custom_topk != native_topk).sum().item()) if not match else 0
    print(
        f"custom_vs_native_match={int(match)} "
        f"mismatches={mismatches}/{inputs.batch * TOPK}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)
    if args.op in ("custom", "both"):
        custom_namespace()
    if args.cross_check and args.op != "both":
        raise ValueError("--cross-check requires --op both")

    print(
        f"batch source_len official_c8_li_us custom_qli_us speedup "
        f"(device={device_name!r} op={args.op})",
        flush=True,
    )
    case_index = 0
    for heads in args.heads:
        for batch in args.batch_sizes:
            for source_len in args.source_lens:
                inputs = build_inputs(
                    device, batch, source_len, heads, args.seed + case_index
                )
                case_index += 1
                metadata = None
                if args.op in ("custom", "both"):
                    if args.trace:
                        import time
                        t0 = time.monotonic()
                        print(
                            f"[trace] case batch={batch} source_len={source_len} "
                            "computing metadata ...",
                            flush=True,
                        )
                    metadata = custom_metadata(inputs, device)
                    torch.npu.synchronize()
                    if args.trace:
                        print(
                            f"[trace] metadata done in "
                            f"{time.monotonic() - t0:.3f}s",
                            flush=True,
                        )
                        t0 = time.monotonic()
                        custom_launch(inputs, metadata, args.custom_weights_dtype)
                        torch.npu.synchronize()
                        print(
                            f"[trace] first custom launch done in "
                            f"{time.monotonic() - t0:.3f}s",
                            flush=True,
                        )
                if args.cross_check:
                    check_custom_matches_native(
                        inputs, metadata, args.custom_weights_dtype
                    )
                native_us = custom_us = float("nan")
                if args.op in ("official", "both"):
                    native_us = timed_mean_us(
                        lambda: official_launch(inputs), args.warmup, args.iters
                    )
                if args.op in ("custom", "both"):
                    custom_us = timed_mean_us(
                        lambda: custom_launch(
                            inputs, metadata, args.custom_weights_dtype
                        ),
                        args.warmup,
                        args.iters,
                    )
                speedup = (
                    f"{native_us / custom_us:.3f}"
                    if native_us == native_us and custom_us == custom_us
                    else "nan"
                )
                print(
                    f"{batch} {source_len} {native_us:.3f} {custom_us:.3f} "
                    f"{speedup}",
                    flush=True,
                )
    print("BENCH_CUSTOM_QLI_C8_DONE", flush=True)


if __name__ == "__main__":
    main()
