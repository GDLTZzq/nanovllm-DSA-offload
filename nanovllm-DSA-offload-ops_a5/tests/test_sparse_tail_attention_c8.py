#!/usr/bin/env python3
"""Correctness/performance test for local packed-C8 sparse+tail attention."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

import nanovllm_dsa_a5
import torch_npu  # type: ignore  # noqa: E402,F401

from _utils import csv_ints, require_a5


BLOCK_SIZE = 128
NOPE_DIM = 512
ROPE_DIM = 64
QUERY_DIM = NOPE_DIM + ROPE_DIM
SCALE_COUNT = NOPE_DIM // BLOCK_SIZE
PACKED_DIM = 656
TOPK = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-sizes", type=csv_ints, default=csv_ints("24"))
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--cache-tokens", type=csv_ints, default=csv_ints("6144"))
    parser.add_argument("--tail-tokens", type=csv_ints, default=csv_ints("64"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    if not 1 <= args.heads <= 64:
        raise ValueError("this project intentionally supports Q_HEAD <= 64")
    if any(
        tokens != 0 and (tokens < TOPK or tokens % BLOCK_SIZE)
        for tokens in args.cache_tokens
    ):
        raise ValueError("cache tokens must be 0 or block-aligned >= 2048")
    if any(tail < 0 for tail in args.tail_tokens):
        raise ValueError("tail tokens must be non-negative")
    if any(
        cache == 0 and tail == 0
        for cache in args.cache_tokens
        for tail in args.tail_tokens
    ):
        raise ValueError("dense C=0 test requires at least one resident token")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters positive")


def pack_cache(
    nope: torch.Tensor, rope: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    packed_bytes = torch.cat(
        (
            nope.contiguous().view(torch.uint8),
            rope.contiguous().view(torch.uint8),
            scales.contiguous().view(torch.uint8),
        ),
        dim=-1,
    )
    if packed_bytes.size(-1) != PACKED_DIM:
        raise AssertionError(f"packed row has {packed_bytes.size(-1)} bytes")
    return packed_bytes.view(torch.float8_e4m3fn)


def make_inputs(
    *,
    device: torch.device,
    batch: int,
    heads: int,
    cache_tokens: int,
    tail_tokens: int,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    resident_len = cache_tokens + tail_tokens
    blocks_per_row = (resident_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    physical_blocks = batch * blocks_per_row
    block_table_cpu = torch.empty((batch, blocks_per_row), dtype=torch.int32)
    for row in range(batch):
        base = row * blocks_per_row
        block_table_cpu[row] = base + torch.randperm(
            blocks_per_row, generator=generator
        ).to(torch.int32)
    nope_cpu = torch.randint(
        -3,
        4,
        (physical_blocks, BLOCK_SIZE, 1, NOPE_DIM),
        generator=generator,
        dtype=torch.int16,
    ).float().to(torch.float8_e4m3fn)
    rope_cpu = torch.empty(
        (physical_blocks, BLOCK_SIZE, 1, ROPE_DIM), dtype=torch.float32
    ).uniform_(-0.5, 0.5, generator=generator).to(torch.bfloat16)
    scales_cpu = torch.empty(
        (physical_blocks, BLOCK_SIZE, 1, SCALE_COUNT), dtype=torch.float32
    ).uniform_(0.02, 0.08, generator=generator)
    packed_cpu = pack_cache(nope_cpu, rope_cpu, scales_cpu)
    query_cpu = torch.cat(
        (
            torch.empty((batch, heads, NOPE_DIM), dtype=torch.float32)
            .uniform_(-0.5, 0.5, generator=generator)
            .to(torch.bfloat16),
            torch.empty((batch, heads, ROPE_DIM), dtype=torch.float32)
            .uniform_(-0.5, 0.5, generator=generator)
            .to(torch.bfloat16),
        ),
        dim=-1,
    ).contiguous()
    topk_slots_cpu = torch.full((batch, 1, TOPK), -1, dtype=torch.int32)
    if cache_tokens:
        for row in range(batch):
            topk_slots_cpu[row, 0] = torch.randperm(
                cache_tokens, generator=generator
            )[:TOPK].to(torch.int32)
    output = torch.empty(
        (batch, heads, NOPE_DIM), dtype=query_cpu.dtype, device=device
    )
    return {
        "device": device,
        "query_cpu": query_cpu,
        "nope_cpu": nope_cpu,
        "rope_cpu": rope_cpu,
        "scales_cpu": scales_cpu,
        "packed": packed_cpu.to(device),
        "query": query_cpu.to(device),
        "topk_slots_cpu": topk_slots_cpu,
        "topk_slots": topk_slots_cpu.to(device),
        "block_table_cpu": block_table_cpu,
        "block_table": block_table_cpu.to(device),
        "actual_q": torch.arange(1, batch + 1, dtype=torch.int32, device=device),
        "actual_kv": torch.full(
            (batch,), resident_len, dtype=torch.int32, device=device
        ),
        "cache_tokens": torch.full(
            (batch,), cache_tokens, dtype=torch.int32, device=device
        ),
        "output": output,
        "scale": QUERY_DIM**-0.5,
    }


def make_mtp_inputs(
    *,
    device: torch.device,
    heads: int,
    query_counts: tuple[int, ...],
    cache_tokens: int,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    batch = len(query_counts)
    packed_queries = sum(query_counts)
    max_resident = cache_tokens + max(query_counts)
    blocks_per_row = (max_resident + BLOCK_SIZE - 1) // BLOCK_SIZE
    physical_blocks = batch * blocks_per_row
    block_table_cpu = torch.empty((batch, blocks_per_row), dtype=torch.int32)
    for row in range(batch):
        base = row * blocks_per_row
        block_table_cpu[row] = base + torch.randperm(
            blocks_per_row, generator=generator
        ).to(torch.int32)
    nope_cpu = torch.randint(
        -3,
        4,
        (physical_blocks, BLOCK_SIZE, 1, NOPE_DIM),
        generator=generator,
        dtype=torch.int16,
    ).float().to(torch.float8_e4m3fn)
    rope_cpu = torch.empty(
        (physical_blocks, BLOCK_SIZE, 1, ROPE_DIM), dtype=torch.float32
    ).uniform_(-0.5, 0.5, generator=generator).to(torch.bfloat16)
    scales_cpu = torch.empty(
        (physical_blocks, BLOCK_SIZE, 1, SCALE_COUNT), dtype=torch.float32
    ).uniform_(0.02, 0.08, generator=generator)
    query_cpu = torch.cat(
        (
            torch.empty(
                (packed_queries, heads, NOPE_DIM), dtype=torch.float32
            ).uniform_(-0.5, 0.5, generator=generator),
            torch.empty(
                (packed_queries, heads, ROPE_DIM), dtype=torch.float32
            ).uniform_(-0.5, 0.5, generator=generator),
        ),
        dim=-1,
    ).to(torch.bfloat16).contiguous()
    topk_slots_cpu = torch.full(
        (packed_queries, 1, TOPK), -1, dtype=torch.int32
    )
    if cache_tokens:
        for row in range(packed_queries):
            topk_slots_cpu[row, 0] = torch.randperm(
                cache_tokens, generator=generator
            )[:TOPK].to(torch.int32)
    cumulative = []
    total = 0
    for count in query_counts:
        total += count
        cumulative.append(total)
    return {
        "device": device,
        "query_counts": query_counts,
        "query_cpu": query_cpu,
        "nope_cpu": nope_cpu,
        "rope_cpu": rope_cpu,
        "scales_cpu": scales_cpu,
        "packed": pack_cache(nope_cpu, rope_cpu, scales_cpu).to(device),
        "query": query_cpu.to(device),
        "topk_slots_cpu": topk_slots_cpu,
        "topk_slots": topk_slots_cpu.to(device),
        "block_table_cpu": block_table_cpu,
        "block_table": block_table_cpu.to(device),
        "actual_q": torch.tensor(
            cumulative, dtype=torch.int32, device=device
        ),
        "actual_kv": torch.tensor(
            [cache_tokens + count for count in query_counts],
            dtype=torch.int32,
            device=device,
        ),
        "cache_tokens": torch.full(
            (batch,), cache_tokens, dtype=torch.int32, device=device
        ),
        "output": torch.empty(
            (packed_queries, heads, NOPE_DIM),
            dtype=query_cpu.dtype,
            device=device,
        ),
        "scale": QUERY_DIM**-0.5,
    }


def active_slots(
    inputs: dict[str, object], cache_tokens: int, tail_tokens: int, row: int
) -> torch.Tensor:
    if cache_tokens == 0:
        return torch.arange(tail_tokens, dtype=torch.int64)
    return torch.cat(
        (
            inputs["topk_slots_cpu"][row, 0].to(torch.int64),
            torch.arange(
                cache_tokens, cache_tokens + tail_tokens, dtype=torch.int64
            ),
        )
    )


def cpu_reference(
    inputs: dict[str, object], cache_tokens: int, tail_tokens: int
) -> torch.Tensor:
    query = inputs["query_cpu"].float()
    scales = inputs["scales_cpu"].repeat_interleave(BLOCK_SIZE, dim=-1)
    nope = inputs["nope_cpu"].float() * scales
    value = nope.to(torch.bfloat16).float()
    key = torch.cat(
        (nope.to(torch.bfloat16), inputs["rope_cpu"]), dim=-1
    ).float()
    table = inputs["block_table_cpu"].to(torch.int64)
    output = torch.empty(
        (query.size(0), query.size(1), NOPE_DIM), dtype=torch.float32
    )
    for row in range(query.size(0)):
        slots = active_slots(inputs, cache_tokens, tail_tokens, row)
        physical = table[row, slots // BLOCK_SIZE] * BLOCK_SIZE
        physical += slots.remainder(BLOCK_SIZE)
        selected_key = key.view(-1, QUERY_DIM)[physical]
        selected_value = value.view(-1, NOPE_DIM)[physical]
        scores = query[row] @ selected_key.T * inputs["scale"]
        probabilities = torch.softmax(scores, dim=-1)
        output[row] = probabilities.to(torch.bfloat16).float() @ selected_value
    return output


def cpu_reference_mtp(
    inputs: dict[str, object], cache_tokens: int
) -> torch.Tensor:
    query = inputs["query_cpu"].float()
    scales = inputs["scales_cpu"].repeat_interleave(BLOCK_SIZE, dim=-1)
    nope = inputs["nope_cpu"].float() * scales
    value = nope.to(torch.bfloat16).float()
    key = torch.cat(
        (nope.to(torch.bfloat16), inputs["rope_cpu"]), dim=-1
    ).float()
    table = inputs["block_table_cpu"].to(torch.int64)
    output = torch.empty(
        (query.size(0), query.size(1), NOPE_DIM), dtype=torch.float32
    )
    row = 0
    for request, query_count in enumerate(inputs["query_counts"]):
        for path in range(query_count):
            if cache_tokens == 0:
                slots = torch.arange(path + 1, dtype=torch.int64)
            else:
                slots = torch.cat(
                    (
                        inputs["topk_slots_cpu"][row, 0].to(torch.int64),
                        torch.arange(
                            cache_tokens,
                            cache_tokens + path + 1,
                            dtype=torch.int64,
                        ),
                    )
                )
            physical = table[request, slots // BLOCK_SIZE] * BLOCK_SIZE
            physical += slots.remainder(BLOCK_SIZE)
            selected_key = key.view(-1, QUERY_DIM)[physical]
            selected_value = value.view(-1, NOPE_DIM)[physical]
            scores = query[row] @ selected_key.T * inputs["scale"]
            probabilities = torch.softmax(scores, dim=-1)
            output[row] = (
                probabilities.to(torch.bfloat16).float() @ selected_value
            )
            row += 1
    return output


def launch(inputs: dict[str, object]) -> None:
    result = nanovllm_dsa_a5.sparse_tail_attention_c8(
        inputs["query"],
        inputs["actual_q"],
        inputs["actual_kv"],
        inputs["cache_tokens"],
        inputs["topk_slots"],
        inputs["block_table"],
        inputs["packed"],
        inputs["scale"],
        inputs["output"],
    )
    if result is not None:
        raise AssertionError("C8 SFA must write the caller-owned output")


def check(
    inputs: dict[str, object], cache_tokens: int, tail_tokens: int
) -> None:
    expected = cpu_reference(inputs, cache_tokens, tail_tokens)
    launch(inputs)
    torch.npu.synchronize()
    actual = inputs["output"].cpu().float()
    if not bool(torch.isfinite(actual).all()):
        raise AssertionError("C8 SFA produced NaN or Inf")
    torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.03)
    max_abs = float((actual - expected).abs().max())
    attended = tail_tokens if cache_tokens == 0 else TOPK + tail_tokens
    print(
        "A5_SPARSE_TAIL_ATTENTION_C8_CHECK "
        f"batch={actual.size(0)} heads={actual.size(1)} "
        f"cache_tokens={cache_tokens} tail_tokens={tail_tokens} "
        f"attended_tokens={attended} max_abs={max_abs:.9f} "
        "finite=1 caller_owned_output=1 ok=1",
        flush=True,
    )


def check_mtp(inputs: dict[str, object], cache_tokens: int) -> None:
    expected = cpu_reference_mtp(inputs, cache_tokens)
    launch(inputs)
    torch.npu.synchronize()
    actual = inputs["output"].cpu().float()
    if not bool(torch.isfinite(actual).all()):
        raise AssertionError("C8 MTP SFA produced NaN or Inf")
    torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.03)
    max_abs = float((actual - expected).abs().max())
    print(
        "A5_SPARSE_TAIL_ATTENTION_C8_MTP_CHECK "
        f"batch={len(inputs['query_counts'])} "
        f"packed_t={actual.size(0)} query_counts={inputs['query_counts']} "
        f"heads={actual.size(1)} cache_tokens={cache_tokens} "
        f"max_abs={max_abs:.9f} per_query_topk=1 causal_tail=1 "
        "finite=1 caller_owned_output=1 ok=1",
        flush=True,
    )


def benchmark(inputs: dict[str, object], warmup: int, iters: int) -> None:
    for _ in range(warmup):
        launch(inputs)
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        launch(inputs)
        end.record()
    ends[-1].synchronize()
    avg_us = statistics.mean(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    ) * 1000.0
    print(
        "A5_SPARSE_TAIL_ATTENTION_C8_RESULT "
        f"mode={'mtp' if 'query_counts' in inputs else 'non_mtp'} "
        f"batch={inputs['actual_q'].numel()} "
        f"packed_t={inputs['query'].size(0)} "
        f"heads={inputs['query'].size(1)} "
        f"avg_us={avg_us:.3f} warmup={warmup} iters={iters}",
        flush=True,
    )


def check_meta(heads: int) -> None:
    query = torch.empty((3, heads, QUERY_DIM), dtype=torch.bfloat16, device="meta")
    actual_q = torch.empty((3,), dtype=torch.int32, device="meta")
    actual_kv = torch.empty((3,), dtype=torch.int32, device="meta")
    cache_tokens = torch.empty((3,), dtype=torch.int32, device="meta")
    slots = torch.empty((3, 1, TOPK), dtype=torch.int32, device="meta")
    table = torch.empty((3, 96), dtype=torch.int32, device="meta")
    packed = torch.empty(
        (288, BLOCK_SIZE, 1, PACKED_DIM),
        dtype=torch.float8_e4m3fn,
        device="meta",
    )
    output = torch.empty((3, heads, NOPE_DIM), dtype=query.dtype, device="meta")
    result = nanovllm_dsa_a5.sparse_tail_attention_c8(
        query,
        actual_q,
        actual_kv,
        cache_tokens,
        slots,
        table,
        packed,
        1.0,
        output,
    )
    if result is not None:
        raise AssertionError("C8 SFA Meta must return None")
    mtp_query = torch.empty(
        (9, heads, QUERY_DIM), dtype=torch.bfloat16, device="meta"
    )
    mtp_slots = torch.empty((9, 1, TOPK), dtype=torch.int32, device="meta")
    mtp_output = torch.empty(
        (9, heads, NOPE_DIM), dtype=mtp_query.dtype, device="meta"
    )
    result = nanovllm_dsa_a5.sparse_tail_attention_c8(
        mtp_query,
        actual_q,
        actual_kv,
        cache_tokens,
        mtp_slots,
        table,
        packed,
        1.0,
        mtp_output,
    )
    if result is not None:
        raise AssertionError("C8 MTP SFA Meta must return None")
    print(
        f"A5_SPARSE_TAIL_ATTENTION_C8_META_CHECK heads={heads} "
        "non_mtp=1 mtp=1 ok=1",
        flush=True,
    )


def check_local_kernel_registration() -> None:
    vendor = Path(nanovllm_dsa_a5.local_opapi_path()).resolve().parents[2]
    metadata = tuple(
        vendor.glob("op_impl/ai_core/tbe/kernel/config/**/binary_info_config.json")
    )
    if not metadata or not any(
        "A5SparseTailAttentionC8" in path.read_text(
            encoding="utf-8", errors="ignore"
        )
        for path in metadata
    ):
        raise AssertionError(
            "local A5SparseTailAttentionC8 kernel metadata is missing; "
            "run bash build_c8.sh"
        )
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "nanovllm_dsa::sparse_tail_attention_c8", "PrivateUse1"
    ):
        raise AssertionError("sparse_tail_attention_c8 has no PrivateUse1 kernel")
    print(
        "A5_SPARSE_TAIL_ATTENTION_C8_LOCAL_KERNEL_CHECK "
        "cann_op=A5SparseTailAttentionC8 privateuse1=1 python_custom_op=0 ok=1",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)
    check_meta(args.heads)
    check_local_kernel_registration()
    print(
        "A5_SPARSE_TAIL_ATTENTION_C8_CONFIG "
        f"device={device} device_name={device_name!r} heads={args.heads} "
        f"batch_sizes={args.batch_sizes} cache_tokens={args.cache_tokens} "
        f"tail_tokens={args.tail_tokens}",
        flush=True,
    )
    mandatory = ((1, 0, 2048), (1, 2048, 0), (1, 6144, 64), (1, 12288, 257))
    for index, (batch, cache_tokens, tail_tokens) in enumerate(mandatory):
        inputs = make_inputs(
            device=device,
            batch=batch,
            heads=args.heads,
            cache_tokens=cache_tokens,
            tail_tokens=tail_tokens,
            seed=args.seed + 10 + index,
        )
        check(inputs, cache_tokens, tail_tokens)
    for index, cache_tokens in enumerate((0, 8192)):
        mtp_inputs = make_mtp_inputs(
            device=device,
            heads=args.heads,
            query_counts=(2, 3, 4),
            cache_tokens=cache_tokens,
            seed=args.seed + 100 + index,
        )
        check_mtp(mtp_inputs, cache_tokens)
        if cache_tokens:
            benchmark(mtp_inputs, args.warmup, args.iters)
    case_index = 0
    for batch in args.batch_sizes:
        for cache_tokens in args.cache_tokens:
            for tail_tokens in args.tail_tokens:
                inputs = make_inputs(
                    device=device,
                    batch=batch,
                    heads=args.heads,
                    cache_tokens=cache_tokens,
                    tail_tokens=tail_tokens,
                    seed=args.seed + 1000 + case_index,
                )
                check(inputs, cache_tokens, tail_tokens)
                benchmark(inputs, args.warmup, args.iters)
                case_index += 1
    print("A5_SPARSE_TAIL_ATTENTION_C8_UT_OK", flush=True)


if __name__ == "__main__":
    main()
