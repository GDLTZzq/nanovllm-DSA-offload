#!/usr/bin/env python3
"""Validate the fused on-device-split A5QuantLightningIndexerC8 single op with
packed MTP queries (1-5 queries per request, MTP0-4).

TND 布局下 query 为 packed [T, N, 128]，actual_seq_lengths_query 是逐请求前缀和
（长度 = 请求数 B），weights/query_dequant_scale 与输出 sparse_indices 的第 0 维
都是 T。正确性对比对象为官方 A5 C8 LightningIndexer（torch_npu
npu_quant_lightning_indexer，同一份 packed 输入独立算出）：并列分数顺序可能不同，
故先 torch.equal，否则逐行断言排序后的集合相等（重复/漏选仍会被抓出）。
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch
import torch_npu  # type: ignore  # noqa: E402,F401

import nanovllm_dsa_a5  # noqa: F401

from _c8_lidu_case import (
    normalized_hadamard_128,
    official_c8_lightning_indexer,
    quantize_fp8,
)
from _lidu_utils import MAX_SOURCE_CAPACITY, TOPK
from _utils import csv_ints, require_a5


BLOCK_SIZE = 128
ALL_MTP_QUERY_COUNTS = [1, 2, 3, 4, 5]  # MTP0-4


@dataclass
class MtpQliCase:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    query_scale: torch.Tensor
    key_scale: torch.Tensor
    actual_q: torch.Tensor
    candidate_lens: torch.Tensor
    block_table: torch.Tensor
    native_topk: torch.Tensor
    query_counts: list[int]
    source_capacity: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--batch-sizes", type=csv_ints, default=csv_ints("1,4,8,16")
    )
    parser.add_argument("--source-lens", type=csv_ints, default=csv_ints("65536"))
    parser.add_argument("--heads", type=csv_ints, default=csv_ints("32,64"))
    parser.add_argument(
        "--queries-per-request",
        type=csv_ints,
        default=csv_ints("1,2,3,4,5"),
        help="per-request query count = MTP+1 (1..5), or 0 to sweep all MTP0-4",
    )
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
    if any(
        qpr != 0 and qpr not in ALL_MTP_QUERY_COUNTS
        for qpr in args.queries_per_request
    ):
        raise ValueError("queries-per-request must be in 1..5, or 0 for all MTP0-4")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters positive")


def make_mtp_case(
    *,
    device: torch.device,
    batch: int,
    heads: int,
    source_len: int,
    query_counts: list[int],
    seed: int,
) -> MtpQliCase:
    if len(query_counts) != batch:
        raise ValueError("query-count list must match batch")
    if any(qc not in ALL_MTP_QUERY_COUNTS for qc in query_counts):
        raise ValueError("every query count must be in 1..5 (MTP0-4)")
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    packed_queries = sum(query_counts)
    actual_q_cpu: list[int] = []
    total = 0
    for qc in query_counts:
        total += qc
        actual_q_cpu.append(total)

    blocks = source_len // BLOCK_SIZE
    block_table_cpu = torch.stack(
        [
            torch.randperm(blocks, generator=generator).to(torch.int32)
            for _ in range(batch)
        ]
    )
    query_fp = torch.empty(
        (packed_queries, heads, 128), dtype=torch.bfloat16, device=device
    ).uniform_(-1, 1)
    key_fp = torch.empty(
        (blocks, BLOCK_SIZE, 1, 128), dtype=torch.bfloat16, device=device
    ).uniform_(-1, 1)
    hadamard = normalized_hadamard_128(dtype=torch.bfloat16, device=device)
    query, query_scale = quantize_fp8(torch.matmul(query_fp, hadamard))
    key, key_scale = quantize_fp8(torch.matmul(key_fp, hadamard))
    weights = torch.empty(
        (packed_queries, heads), dtype=torch.bfloat16, device=device
    ).uniform_(0.01, 1.0).contiguous()
    actual_q = torch.tensor(actual_q_cpu, dtype=torch.int32, device=device)
    candidate_lens = torch.tensor(
        [source_len] * batch, dtype=torch.int32, device=device
    )
    block_table = block_table_cpu.to(device)
    native_topk = official_c8_lightning_indexer(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        actual_q,
        candidate_lens,
        block_table,
    )
    torch.npu.synchronize()
    return MtpQliCase(
        query=query,
        key=key,
        weights=weights,
        query_scale=query_scale,
        key_scale=key_scale,
        actual_q=actual_q,
        candidate_lens=candidate_lens,
        block_table=block_table,
        native_topk=native_topk,
        query_counts=query_counts,
        source_capacity=source_len,
    )


def launch(case: MtpQliCase) -> torch.Tensor:
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


def check_case(case: MtpQliCase) -> None:
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
            f"QLI C8 MTP row sets differ in {row_mismatch}/{case.query.size(0)} rows"
        )
    print(
        "A5_QUANT_LIGHTNING_INDEXER_C8_MTP_CHECK "
        f"heads={case.query.size(1)} batch={len(case.query_counts)} "
        f"packed_t={case.query.size(0)} query_counts={case.query_counts} "
        f"source_capacity={case.source_capacity} "
        f"exact_match={int(exact)} unordered_row_mismatch={row_mismatch} ok=1",
        flush=True,
    )


def check_meta() -> None:
    query = torch.empty((9, 32, 128), dtype=torch.float8_e4m3fn, device="meta")
    key = torch.empty(
        (96, BLOCK_SIZE, 1, 128), dtype=torch.float8_e4m3fn, device="meta"
    )
    weights = torch.empty((9, 32), dtype=torch.float32, device="meta")
    query_scale = torch.empty((9, 32), dtype=torch.float32, device="meta")
    key_scale = torch.empty(
        (96, BLOCK_SIZE, 1), dtype=torch.float32, device="meta"
    )
    ints = torch.empty((3,), dtype=torch.int32, device="meta")
    table = torch.empty((3, 96), dtype=torch.int32, device="meta")
    output = torch.ops.nanovllm_dsa.quant_lightning_indexer_c8.default(
        query, key, weights, query_scale, key_scale, ints, ints, table
    )
    if tuple(output.shape) != (9, 1, TOPK) or output.dtype != torch.int32:
        raise AssertionError("QLI C8 MTP Meta implementation returned wrong shape/dtype")
    print("A5_QUANT_LIGHTNING_INDEXER_C8_MTP_META_CHECK ok=1", flush=True)


def event_benchmark(case: MtpQliCase, warmup: int, iters: int) -> tuple[float, float]:
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
        raise AssertionError("timed QLI C8 MTP outputs were not retained")
    return native_us, fused_us


def report_result(tag: str, case: MtpQliCase, args, native_us: float, fused_us: float) -> None:
    speedup = native_us / fused_us if fused_us > 0 else float("nan")
    print(
        f"A5_QUANT_LIGHTNING_INDEXER_C8_MTP_{tag} "
        f"heads={case.query.size(1)} batch={len(case.query_counts)} "
        f"packed_t={case.query.size(0)} query_counts={case.query_counts} "
        f"source_len={case.source_capacity} "
        f"official_c8_li_us={native_us:.3f} fused_li_us={fused_us:.3f} "
        f"speedup={speedup:.3f} warmup={args.warmup} iters={args.iters}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    check_args(args)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    require_a5(device, args.allow_non_a5)
    check_meta()

    qpr_values = list(args.queries_per_request)
    if 0 in qpr_values:
        qpr_values = list(ALL_MTP_QUERY_COUNTS)

    case_index = 0
    for heads in args.heads:
        for qpr in qpr_values:
            for batch in args.batch_sizes:
                for source_len in args.source_lens:
                    query_counts = [qpr] * batch
                    case = make_mtp_case(
                        device=device,
                        batch=batch,
                        heads=heads,
                        source_len=source_len,
                        query_counts=query_counts,
                        seed=args.seed + case_index,
                    )
                    case_index += 1
                    check_case(case)
                    native_us, fused_us = event_benchmark(
                        case, args.warmup, args.iters
                    )
                    report_result("RESULT", case, args, native_us, fused_us)

    # 混合 case：同一 batch 内 cycle MTP0-4（异构 s1），打散前缀和/输出偏移
    for heads in args.heads:
        for batch in args.batch_sizes:
            if batch < 2:
                continue
            for source_len in args.source_lens:
                query_counts = [
                    ALL_MTP_QUERY_COUNTS[row % 5] for row in range(batch)
                ]
                case = make_mtp_case(
                    device=device,
                    batch=batch,
                    heads=heads,
                    source_len=source_len,
                    query_counts=query_counts,
                    seed=args.seed + case_index,
                )
                case_index += 1
                check_case(case)
                native_us, fused_us = event_benchmark(case, args.warmup, args.iters)
                report_result("MIXED_RESULT", case, args, native_us, fused_us)
    print("A5_QUANT_LIGHTNING_INDEXER_C8_MTP_UT_OK", flush=True)


if __name__ == "__main__":
    main()
