#!/usr/bin/env python3
"""Validate the fused on-device-split A5QuantLightningIndexerC8 single op.

正确性对比对象为官方 A5 C8 LightningIndexer（torch_npu
npu_quant_lightning_indexer）：并列分数顺序可能不同，故先 torch.equal，
否则逐行断言排序后的集合相等。性能判据：小 batch 时延显著低于 official，
且随 batch 近线性增长。
"""

from __future__ import annotations

import argparse
import statistics

import torch

import nanovllm_dsa_a5  # noqa: F401
import torch_npu  # type: ignore  # noqa: E402,F401

from _c8_lidu_case import make_case, official_c8_lightning_indexer
from _lidu_utils import MAX_SOURCE_CAPACITY, TOPK
from _utils import csv_ints, require_a5


BLOCK_SIZE = 128
DEFAULT_BUDGET = 6144


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--batch-sizes", type=csv_ints, default=csv_ints("1,4,8,16,24,32,48,64")
    )
    parser.add_argument(
        "--source-lens", type=csv_ints, default=csv_ints("65536,131072")
    )
    parser.add_argument("--heads", type=csv_ints, default=csv_ints("32"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("all batch sizes must be positive")
    if any(length <= 0 or length % BLOCK_SIZE for length in args.source_lens):
        raise ValueError("source lengths must be positive multiples of 128")
    if any(length > MAX_SOURCE_CAPACITY for length in args.source_lens):
        raise ValueError("source length exceeds the 18-bit token-ID capacity")
    if any(heads not in (8, 16, 24, 32, 64) for heads in args.heads):
        raise ValueError("QLI C8 query heads must be 8/16/24/32/64")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters positive")


def launch(case) -> torch.Tensor:
    return torch.ops.nanovllm_dsa.quant_lightning_indexer_c8.default(
        case.query,
        case.key,
        case.weights.float(),
        case.query_scale,
        case.key_scale,
        case.actual_q,
        case.candidate_lens,
        case.block_table,
    )


def check_case(case) -> None:
    output = launch(case)
    torch.npu.synchronize()
    topk = output.reshape(case.query.size(0), TOPK).cpu()
    native = case.native_topk.reshape(case.query.size(0), TOPK).cpu()
    exact = torch.equal(topk, native)
    row_mismatch = 0
    if not exact:
        for row in range(case.query.size(0)):
            got = torch.sort(topk[row]).values
            ref = torch.sort(native[row]).values
            if not torch.equal(got, ref):
                row_mismatch += 1
    if row_mismatch != 0:
        raise AssertionError(
            f"QLI C8 row sets differ in {row_mismatch}/{case.query.size(0)} rows"
        )
    print(
        "A5_QUANT_LIGHTNING_INDEXER_C8_CHECK "
        f"heads={case.query.size(1)} batch={case.query.size(0)} "
        f"source_capacity={case.source_capacity} "
        f"candidate_lens={case.candidate_lens.cpu().tolist()} "
        f"exact_match={int(exact)} unordered_row_mismatch={row_mismatch} ok=1",
        flush=True,
    )


def check_meta() -> None:
    query = torch.empty((3, 32, 128), dtype=torch.float8_e4m3fn, device="meta")
    key = torch.empty(
        (96, BLOCK_SIZE, 1, 128), dtype=torch.float8_e4m3fn, device="meta"
    )
    weights = torch.empty((3, 32), dtype=torch.float32, device="meta")
    query_scale = torch.empty((3, 32), dtype=torch.float32, device="meta")
    key_scale = torch.empty(
        (96, BLOCK_SIZE, 1), dtype=torch.float32, device="meta"
    )
    ints = torch.empty((3,), dtype=torch.int32, device="meta")
    table = torch.empty((3, 96), dtype=torch.int32, device="meta")
    output = torch.ops.nanovllm_dsa.quant_lightning_indexer_c8.default(
        query, key, weights, query_scale, key_scale, ints, ints, table
    )
    if tuple(output.shape) != (3, 1, TOPK) or output.dtype != torch.int32:
        raise AssertionError("QLI C8 Meta implementation returned wrong shape/dtype")
    print("A5_QUANT_LIGHTNING_INDEXER_C8_META_CHECK ok=1", flush=True)


def event_benchmark(case, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        launch(case)
    torch.npu.synchronize()

    fused_starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    fused_ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    retained = []
    for start, end in zip(fused_starts, fused_ends):
        start.record()
        retained.append(launch(case))
        end.record()
    fused_ends[-1].synchronize()
    fused_us = statistics.mean(
        start.elapsed_time(end) for start, end in zip(fused_starts, fused_ends)
    ) * 1000

    native_starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    native_ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(native_starts, native_ends):
        start.record()
        official_c8_lightning_indexer(
            case.query,
            case.key,
            case.weights,
            case.query_scale,
            case.key_scale,
            case.actual_q,
            case.candidate_lens,
            case.block_table,
        )
        end.record()
    native_ends[-1].synchronize()
    native_us = statistics.mean(
        start.elapsed_time(end) for start, end in zip(native_starts, native_ends)
    ) * 1000
    if not retained:
        raise AssertionError("timed QLI C8 outputs were not retained")
    return native_us, fused_us


def main() -> None:
    args = parse_args()
    check_args(args)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)
    check_meta()

    case_index = 0
    for heads in args.heads:
        for batch in args.batch_sizes:
            for source_len in args.source_lens:
                case = make_case(
                    device,
                    batch,
                    source_len,
                    heads,
                    [DEFAULT_BUDGET] * batch,
                    (0, 300),
                    7,
                    args.seed + case_index,
                )
                case_index += 1
                check_case(case)
                native_us, fused_us = event_benchmark(
                    case, args.warmup, args.iters
                )
                speedup = (
                    native_us / fused_us if fused_us > 0 else float("nan")
                )
                print(
                    "A5_QUANT_LIGHTNING_INDEXER_C8_RESULT "
                    f"device_name={device_name!r} heads={heads} "
                    f"batch={batch} source_len={source_len} "
                    f"official_c8_li_us={native_us:.3f} "
                    f"fused_li_us={fused_us:.3f} "
                    f"speedup={speedup:.3f} "
                    f"warmup={args.warmup} iters={args.iters}",
                    flush=True,
                )
    print("A5_QUANT_LIGHTNING_INDEXER_C8_UT_OK", flush=True)


if __name__ == "__main__":
    main()
