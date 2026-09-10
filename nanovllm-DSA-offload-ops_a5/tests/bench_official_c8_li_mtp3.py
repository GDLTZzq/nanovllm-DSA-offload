#!/usr/bin/env python3
"""Latency scan for the official Ascend 950 C8 LightningIndexer."""

from __future__ import annotations

import argparse
import gc
import statistics
from dataclasses import dataclass

import torch
import torch_npu  # type: ignore

from _c8_lidu_case import quantize_fp8
from _utils import csv_ints, require_a5


BLOCK_SIZE = 128
HEAD_DIM = 128
TOPK = 2048


@dataclass
class Case:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    query_scale: torch.Tensor
    key_scale: torch.Tensor
    actual_q: torch.Tensor
    candidate_lens: torch.Tensor
    block_table: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--batch-sizes",
        type=csv_ints,
        default=csv_ints("1,4,8,16,32,48,64"),
    )
    parser.add_argument(
        "--source-lens",
        type=csv_ints,
        default=csv_ints("65536,131072"),
    )
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument(
        "--mtp",
        type=int,
        default=3,
        help="MTP depth: 0 means one query per request, 3 means four",
    )
    parser.add_argument("--query-noise", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    if any(length <= 0 or length % BLOCK_SIZE for length in args.source_lens):
        raise ValueError("source lengths must be positive multiples of 128")
    if args.heads not in (32, 64):
        raise ValueError("C8 LightningIndexer heads must be 32 or 64")
    if args.mtp < 0:
        raise ValueError("mtp must be non-negative")
    if args.query_noise <= 0:
        raise ValueError("query-noise must be positive")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters positive")


def make_case(
    device: torch.device,
    *,
    batch: int,
    source_len: int,
    heads: int,
    queries_per_request: int,
    query_noise: float,
    seed: int,
) -> Case:
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)

    packed_t = batch * queries_per_request
    blocks_per_request = source_len // BLOCK_SIZE
    total_blocks = batch * blocks_per_request

    base_query = torch.randn(
        (batch, 1, heads, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    route_noise = torch.randn(
        (batch, queries_per_request, heads, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    query_fp = (
        base_query + query_noise * route_noise
    ).reshape(packed_t, heads, HEAD_DIM)
    query, query_scale = quantize_fp8(query_fp)

    key_fp = torch.randn(
        (total_blocks, BLOCK_SIZE, 1, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    key, key_scale = quantize_fp8(key_fp)

    base_weights = torch.empty(
        (batch, 1, heads), dtype=torch.bfloat16, device=device
    ).uniform_(0.01, 1.0)
    weights = (
        base_weights.expand(-1, queries_per_request, -1)
        .reshape(packed_t, heads)
        .contiguous()
    )
    actual_q = torch.arange(
        queries_per_request,
        packed_t + 1,
        queries_per_request,
        dtype=torch.int32,
        device=device,
    )
    candidate_lens = torch.full(
        (batch,), source_len, dtype=torch.int32, device=device
    )

    # Each request owns disjoint physical blocks.  Randomizing the global
    # physical order avoids measuring cross-request L2 reuse of one key pool.
    block_table = (
        torch.randperm(total_blocks, generator=generator)
        .reshape(batch, blocks_per_request)
        .to(torch.int32)
        .to(device)
    )

    torch.npu.synchronize()
    del base_query, route_noise, query_fp, key_fp, base_weights
    torch.npu.empty_cache()
    return Case(
        query=query,
        key=key,
        weights=weights,
        query_scale=query_scale,
        key_scale=key_scale,
        actual_q=actual_q,
        candidate_lens=candidate_lens,
        block_table=block_table,
    )


def launch(case: Case) -> torch.Tensor:
    result = torch_npu.npu_quant_lightning_indexer(
        query=case.query,
        key=case.key,
        weights=case.weights,
        query_dequant_scale=case.query_scale,
        key_dequant_scale=case.key_scale,
        actual_seq_lengths_query=case.actual_q,
        actual_seq_lengths_key=case.candidate_lens,
        block_table=case.block_table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    expected_numel = case.query.size(0) * TOPK
    if isinstance(result, torch.Tensor):
        output = result
    else:
        output = next(
            (
                tensor
                for tensor in result
                if isinstance(tensor, torch.Tensor)
                and tensor.dtype == torch.int32
                and tensor.numel() == expected_numel
            ),
            None,
        )
    if output is None or output.dtype != torch.int32:
        raise RuntimeError("official C8 LightningIndexer returned no int32 TopK")
    if output.numel() != expected_numel:
        raise RuntimeError(
            "official C8 LightningIndexer returned an unexpected output: "
            f"shape={tuple(output.shape)}, expected_numel={expected_numel}"
        )
    return output


def benchmark(case: Case, *, warmup: int, iters: int) -> list[float]:
    output = launch(case)
    torch.npu.synchronize()
    output_view = output.reshape(case.query.size(0), TOPK)
    if bool((output_view < 0).any()) or bool(
        (output_view >= int(case.candidate_lens.max())).any()
    ):
        raise AssertionError("official C8 LightningIndexer returned invalid IDs")

    event_pairs = [
        (
            torch.npu.Event(enable_timing=True),
            torch.npu.Event(enable_timing=True),
        )
        for _ in range(iters)
    ]
    for _ in range(warmup):
        launch(case)
    for start, end in event_pairs:
        start.record()
        launch(case)
        end.record()
    torch.npu.synchronize()
    return [
        float(start.elapsed_time(end)) * 1000.0
        for start, end in event_pairs
    ]


def main() -> None:
    args = parse_args()
    check_args(args)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)

    print(
        "A5_OFFICIAL_C8_LI_MTP_CONFIG "
        f"device={device} device_name={device_name!r} heads={args.heads} "
        f"batch_sizes={args.batch_sizes} source_lens={args.source_lens} "
        f"mtp={args.mtp} queries_per_request={args.mtp + 1} "
        f"query_noise={args.query_noise} "
        "layout_query=TND layout_key=PA_BSND sparse_mode=3 "
        "physical_key_pool=disjoint_random timer=queued_npu_event "
        f"warmup={args.warmup} iters={args.iters}",
        flush=True,
    )

    case_index = 0
    for source_len in args.source_lens:
        for batch in args.batch_sizes:
            case = make_case(
                device,
                batch=batch,
                source_len=source_len,
                heads=args.heads,
                queries_per_request=args.mtp + 1,
                query_noise=args.query_noise,
                seed=args.seed + case_index,
            )
            case_index += 1
            samples = benchmark(case, warmup=args.warmup, iters=args.iters)
            key_gib = case.key.numel() * case.key.element_size() / (1024**3)
            print(
                "A5_OFFICIAL_C8_LI_MTP_RESULT "
                f"mtp={args.mtp} queries_per_request={args.mtp + 1} "
                f"batch={batch} packed_t={batch * (args.mtp + 1)} "
                f"heads={args.heads} source_len={source_len} "
                f"physical_blocks={case.key.size(0)} key_gib={key_gib:.3f} "
                f"avg_us={statistics.mean(samples):.3f} "
                f"p50_us={statistics.median(samples):.3f} "
                f"min_us={min(samples):.3f} max_us={max(samples):.3f} "
                "timer=queued_npu_event",
                flush=True,
            )
            del case, samples
            gc.collect()
            torch.npu.empty_cache()

    print("A5_OFFICIAL_C8_LI_MTP_BENCH_OK", flush=True)


if __name__ == "__main__":
    main()
