#!/usr/bin/env python3
"""Correctness/performance test for the pure packed-C8 DRAM -> HBM copy op."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch

import nanovllm_dsa_a5
import torch_npu  # type: ignore  # noqa: E402,F401

from _utils import physical_token_rows, require_a5, swapped_from_cpu


BLOCK_SIZE = 128
PACKED_DIM = 656
COPY_CAP = 2048
POISON = -91


@dataclass
class Case:
    device: torch.device
    device_name: str
    dram_cpu: torch.Tensor
    dram: torch.Tensor
    hbm: torch.Tensor
    dram_table_cpu: torch.Tensor
    hbm_table_cpu: torch.Tensor
    dram_table: torch.Tensor
    hbm_table: torch.Tensor
    sources_cpu: torch.Tensor
    destinations_cpu: torch.Tensor
    counts_cpu: torch.Tensor
    sources: torch.Tensor
    destinations: torch.Tensor
    counts: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--source-len", type=int, default=20096)
    parser.add_argument("--hbm-slots", type=int, default=6144)
    parser.add_argument("--copy-min", type=int, default=0)
    parser.add_argument("--copy-max", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.source_len <= 0 or args.source_len % BLOCK_SIZE:
        raise ValueError("source length must be positive and block aligned")
    if args.hbm_slots <= 0 or args.hbm_slots % BLOCK_SIZE:
        raise ValueError("HBM slots must be positive and block aligned")
    if not 0 <= args.copy_min <= args.copy_max <= COPY_CAP:
        raise ValueError("copy range must be within [0,2048]")
    if args.copy_max > min(args.source_len, args.hbm_slots):
        raise ValueError("copy count exceeds source or destination capacity")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters positive")


def private_table(
    batch: int, blocks_per_row: int, generator: torch.Generator
) -> tuple[torch.Tensor, int]:
    table = torch.empty((batch, blocks_per_row), dtype=torch.int32)
    for row in range(batch):
        base = row * blocks_per_row
        table[row] = base + torch.randperm(
            blocks_per_row, generator=generator
        ).to(torch.int32)
    return table, batch * blocks_per_row


def make_case(args: argparse.Namespace) -> Case:
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)
    generator = torch.Generator().manual_seed(args.seed)
    dram_table_cpu, dram_blocks = private_table(
        args.batch_size, args.source_len // BLOCK_SIZE, generator
    )
    hbm_table_cpu, hbm_blocks = private_table(
        args.batch_size, args.hbm_slots // BLOCK_SIZE, generator
    )
    dram_cpu = torch.randint(
        -128,
        128,
        (dram_blocks, BLOCK_SIZE, 1, PACKED_DIM),
        generator=generator,
        dtype=torch.int16,
    ).to(torch.int8)
    sources_cpu = torch.full(
        (args.batch_size, 1, COPY_CAP), -1, dtype=torch.int32
    )
    destinations_cpu = torch.full_like(sources_cpu, -1)
    counts = torch.randint(
        args.copy_min,
        args.copy_max + 1,
        (args.batch_size,),
        generator=generator,
        dtype=torch.int64,
    )
    if args.batch_size > 1:
        counts[0] = args.copy_min
        counts[1] = args.copy_max
    for row, count in enumerate(counts.tolist()):
        sources_cpu[row, 0, :count] = torch.randperm(
            args.source_len, generator=generator
        )[:count].to(torch.int32)
        destinations_cpu[row, 0, :count] = torch.randperm(
            args.hbm_slots, generator=generator
        )[:count].to(torch.int32)
    counts_cpu = counts.to(torch.int32)
    return Case(
        device=device,
        device_name=device_name,
        dram_cpu=dram_cpu,
        dram=swapped_from_cpu(dram_cpu, device),
        hbm=torch.empty(
            (hbm_blocks, BLOCK_SIZE, 1, PACKED_DIM),
            dtype=torch.int8,
            device=device,
        ),
        dram_table_cpu=dram_table_cpu,
        hbm_table_cpu=hbm_table_cpu,
        dram_table=dram_table_cpu.to(device),
        hbm_table=hbm_table_cpu.to(device),
        sources_cpu=sources_cpu,
        destinations_cpu=destinations_cpu,
        counts_cpu=counts_cpu,
        sources=sources_cpu.to(device),
        destinations=destinations_cpu.to(device),
        counts=counts_cpu.to(device),
    )


def launch(case: Case, counts: torch.Tensor | None = None) -> None:
    result = nanovllm_dsa_a5.kvcache_scatter_copy_c8(
        case.sources,
        case.destinations,
        case.counts if counts is None else counts,
        case.hbm_table,
        case.dram_table,
        case.hbm,
        case.dram,
    )
    if result is not None:
        raise AssertionError("C8 SCATTER must use caller-owned mutation only")


def active_physical_rows(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    source_rows: list[torch.Tensor] = []
    destination_rows: list[torch.Tensor] = []
    for row, count in enumerate(case.counts_cpu.tolist()):
        sources = case.sources_cpu[row, 0, :count].to(torch.int64)
        destinations = case.destinations_cpu[row, 0, :count].to(torch.int64)
        source_rows.append(
            physical_token_rows(case.dram_table_cpu, row, sources, BLOCK_SIZE)
        )
        destination_rows.append(
            physical_token_rows(case.hbm_table_cpu, row, destinations, BLOCK_SIZE)
        )
    return torch.cat(source_rows), torch.cat(destination_rows)


def check_meta() -> None:
    sources = torch.empty((3, 1, COPY_CAP), dtype=torch.int32, device="meta")
    destinations = torch.empty_like(sources)
    counts = torch.empty((3,), dtype=torch.int32, device="meta")
    hbm_table = torch.empty((3, 48), dtype=torch.int32, device="meta")
    dram_table = torch.empty((3, 157), dtype=torch.int32, device="meta")
    hbm = torch.empty(
        (144, BLOCK_SIZE, 1, PACKED_DIM), dtype=torch.int8, device="meta"
    )
    dram = torch.empty(
        (157, BLOCK_SIZE, 1, PACKED_DIM), dtype=torch.int8, device="meta"
    )
    result = nanovllm_dsa_a5.kvcache_scatter_copy_c8(
        sources, destinations, counts, hbm_table, dram_table, hbm, dram
    )
    if result is not None:
        raise AssertionError("C8 SCATTER Meta must return None")
    print("A5_KVCACHE_SCATTER_COPY_C8_META_CHECK ok=1", flush=True)


def check_copy(case: Case) -> None:
    source_rows, destination_rows = active_physical_rows(case)
    case.hbm.fill_(POISON)
    torch.npu.synchronize()
    launch(case)
    torch.npu.synchronize()
    if destination_rows.numel():
        expected = case.dram_cpu.view(-1, PACKED_DIM)[source_rows]
        actual = case.hbm.view(-1, PACKED_DIM)[
            destination_rows.to(case.device)
        ].cpu()
        if not torch.equal(actual, expected):
            raise AssertionError("packed C8 DRAM->HBM bytes differ")
    active = set(destination_rows.tolist())
    guard = next(
        row for row in range(case.hbm.numel() // PACKED_DIM) if row not in active
    )
    if not bool(torch.all(case.hbm.view(-1, PACKED_DIM)[guard] == POISON).item()):
        raise AssertionError("inactive packed-HBM guard row was modified")
    print(
        "A5_KVCACHE_SCATTER_COPY_C8_CHECK "
        f"copied_tokens={int(case.counts_cpu.sum())} row_bytes={PACKED_DIM} "
        "allocator=empty_with_swapped_memory byte_exact=1 "
        "guard_unchanged=1 caller_owned_hbm=1 ok=1",
        flush=True,
    )


def check_zero_copy(case: Case) -> None:
    case.hbm.fill_(POISON)
    launch(case, torch.zeros_like(case.counts))
    torch.npu.synchronize()
    if not bool(torch.all(case.hbm == POISON).item()):
        raise AssertionError("copy_count=0 modified packed HBM")
    print(
        "A5_KVCACHE_SCATTER_COPY_C8_ZERO_COPY_CHECK hbm_unchanged=1 ok=1",
        flush=True,
    )


def benchmark(case: Case, warmup: int, iters: int) -> None:
    for _ in range(warmup):
        launch(case)
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        launch(case)
        end.record()
    ends[-1].synchronize()
    avg_us = statistics.mean(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    ) * 1000.0
    payload_bytes = int(case.counts_cpu.sum()) * PACKED_DIM
    payload_gbps = payload_bytes / (avg_us * 1000.0) if avg_us else 0.0
    print(
        "A5_KVCACHE_SCATTER_COPY_C8_RESULT "
        f"batch={case.counts.size(0)} copied_tokens={int(case.counts_cpu.sum())} "
        f"avg_us={avg_us:.3f} payload_gbps={payload_gbps:.3f} "
        f"warmup={warmup} iters={iters}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    check_meta()
    case = make_case(args)
    print(
        "A5_KVCACHE_SCATTER_COPY_C8_CONFIG "
        f"device={case.device} device_name={case.device_name!r} "
        f"batch={args.batch_size} source_len={args.source_len} "
        f"hbm_slots={args.hbm_slots} copy_range=[{args.copy_min},{args.copy_max}] "
        f"opapi={nanovllm_dsa_a5.local_opapi_path()}",
        flush=True,
    )
    check_copy(case)
    check_zero_copy(case)
    benchmark(case, args.warmup, args.iters)
    print("A5_KVCACHE_SCATTER_COPY_C8_UT_OK", flush=True)


if __name__ == "__main__":
    main()
