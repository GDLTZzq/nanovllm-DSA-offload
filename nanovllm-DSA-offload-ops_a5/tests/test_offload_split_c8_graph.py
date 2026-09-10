#!/usr/bin/env python3
"""C8 non-MTP/MTP3 LIM -> SCATTER -> sparse+tail ACLGraph test.

Covers mixed offload, heterogeneous budgets/candidates, non-full prefill tails,
and decode tokens that remain in the dense HBM tail.
"""

from __future__ import annotations

import argparse
import math
import statistics

import torch

import nanovllm_dsa_a5  # noqa: F401
import torch_npu  # type: ignore  # noqa: F401

from _c8_lidu_case import make_case
from _lidu_utils import assert_pool_row
from _utils import physical_token_rows, require_a5, swapped_from_cpu
from test_fused_li_manage_mtp_c8 import (
    make_case as make_mtp_case,
    output_buffers as mtp_output_buffers,
)


BLOCK_SIZE = 128
NOPE_DIM = 512
ROPE_DIM = 64
QUERY_DIM = NOPE_DIM + ROPE_DIM
PACKED_DIM = 656
TOPK = 2048
SOURCE_LEN = 4096
CACHE_TOKENS = 3072
MTP_SOURCE_LEN = 12288
MTP_CACHE_TOKENS = 8192
TAIL_TOKENS = 64
OUTPUT_POISON = -123456789
HBM_POISON = -91
MAX_LOGICAL_HBM_SLOTS = 1 << 14
COMPREHENSIVE_BUDGETS = [0, 8192, 12288, 16256]
COMPREHENSIVE_CANDIDATES = [4096, 16384, 32768, 65536]
# The C=0 row covers the largest non-full prefill remainder.  The C=16256
# row uses 124 + four MTP3 decode tokens, exactly reaching slot 16383.
COMPREHENSIVE_PREFILL_TAILS = [127, 63, 95, 124]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


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


def physical(table: torch.Tensor, row: int, tokens: torch.Tensor) -> torch.Tensor:
    return physical_token_rows(table, row, tokens, BLOCK_SIZE)


def make_packed_bytes(blocks: int, generator: torch.Generator) -> torch.Tensor:
    nope = torch.randint(
        -3,
        4,
        (blocks, BLOCK_SIZE, 1, NOPE_DIM),
        generator=generator,
        dtype=torch.int16,
    ).float().to(torch.float8_e4m3fn)
    rope = torch.empty(
        (blocks, BLOCK_SIZE, 1, ROPE_DIM), dtype=torch.float32
    ).uniform_(-0.5, 0.5, generator=generator).to(torch.bfloat16)
    scales = torch.empty(
        (blocks, BLOCK_SIZE, 1, NOPE_DIM // BLOCK_SIZE), dtype=torch.float32
    ).uniform_(0.02, 0.08, generator=generator)
    packed = torch.cat(
        (
            nope.contiguous().view(torch.uint8),
            rope.contiguous().view(torch.uint8),
            scales.contiguous().view(torch.uint8),
        ),
        dim=-1,
    )
    if packed.size(-1) != PACKED_DIM:
        raise AssertionError(f"packed C8 row has {packed.size(-1)} bytes")
    return packed.view(torch.int8).contiguous()


def initialize_hbm(
    *,
    dram: torch.Tensor,
    dram_table: torch.Tensor,
    hbm_table: torch.Tensor,
    pool: torch.Tensor,
    req_entries: torch.Tensor,
    budgets: list[int],
    candidates: list[int],
    actual_lengths: list[int],
) -> torch.Tensor:
    hbm = torch.full(
        (int(hbm_table.max()) + 1, BLOCK_SIZE, 1, PACKED_DIM),
        HBM_POISON,
        dtype=torch.int8,
    )
    dram_rows = dram.view(-1, PACKED_DIM)
    hbm_rows = hbm.view(-1, PACKED_DIM)
    pool_cpu = pool.cpu()
    req_cpu = req_entries.cpu()
    for row, (budget, candidate, actual) in enumerate(
        zip(budgets, candidates, actual_lengths)
    ):
        if budget == 0:
            sources = torch.arange(actual, dtype=torch.int64)
            destinations = sources
        else:
            state = pool_cpu[int(req_cpu[row]), :candidate]
            sources = (state >= 0).nonzero().flatten().to(torch.int64)
            destinations = state[sources].to(torch.int64)
        hbm_rows[physical(hbm_table, row, destinations)] = dram_rows[
            physical(dram_table, row, sources)
        ]
        if budget and actual > candidate:
            tail_sources = torch.arange(candidate, actual, dtype=torch.int64)
            tail_destinations = torch.arange(
                budget, budget + actual - candidate, dtype=torch.int64
            )
            hbm_rows[physical(hbm_table, row, tail_destinations)] = dram_rows[
                physical(dram_table, row, tail_sources)
            ]
    return hbm


def decode_packed(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = packed.contiguous().view(torch.uint8)
    nope = raw[..., :NOPE_DIM].contiguous().view(torch.float8_e4m3fn).reshape(
        *raw.shape[:-1], NOPE_DIM
    )
    rope = raw[..., NOPE_DIM : NOPE_DIM + ROPE_DIM * 2].contiguous().view(
        torch.bfloat16
    ).reshape(*raw.shape[:-1], ROPE_DIM)
    scales = raw[..., NOPE_DIM + ROPE_DIM * 2 :].contiguous().view(
        torch.float32
    ).reshape(*raw.shape[:-1], NOPE_DIM // BLOCK_SIZE)
    value = (
        nope.float() * scales.repeat_interleave(BLOCK_SIZE, dim=-1)
    ).to(torch.bfloat16)
    return torch.cat((value, rope), dim=-1).float(), value.float()


def active_attention_slots(
    topk_slots: torch.Tensor,
    budgets: torch.Tensor,
    resident_lengths: torch.Tensor,
) -> list[torch.Tensor]:
    topk_cpu = topk_slots.cpu().view(topk_slots.size(0), TOPK)
    result: list[torch.Tensor] = []
    for row, (budget, resident) in enumerate(
        zip(budgets.cpu().tolist(), resident_lengths.cpu().tolist())
    ):
        if budget == 0:
            result.append(torch.arange(resident, dtype=torch.int64))
        else:
            result.append(
                torch.cat(
                    (
                        topk_cpu[row].to(torch.int64),
                        torch.arange(budget, resident, dtype=torch.int64),
                    )
                )
            )
    return result


def attention_golden(
    *,
    query: torch.Tensor,
    hbm: torch.Tensor,
    topk_slots: torch.Tensor,
    budgets: torch.Tensor,
    resident_lengths: torch.Tensor,
    block_table_cpu: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    key, value = decode_packed(hbm.cpu())
    flat_key = key.view(-1, QUERY_DIM)
    flat_value = value.view(-1, NOPE_DIM)
    query_cpu = query.cpu().float()
    outputs: list[torch.Tensor] = []
    for row, slots in enumerate(
        active_attention_slots(topk_slots, budgets, resident_lengths)
    ):
        rows = physical(block_table_cpu, row, slots)
        scores = query_cpu[row] @ flat_key[rows].T * scale
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(
            probabilities.to(torch.bfloat16).float() @ flat_value[rows]
        )
    return torch.stack(outputs)


def assert_copy(
    *,
    hbm: torch.Tensor,
    dram_cpu: torch.Tensor,
    hbm_table_cpu: torch.Tensor,
    dram_table_cpu: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> None:
    sources = source_ids.cpu().view(source_ids.size(0), -1)
    destinations = destination_slots.cpu().view(source_ids.size(0), -1)
    hbm_rows = hbm.view(torch.int8).view(-1, PACKED_DIM)
    dram_rows = dram_cpu.view(-1, PACKED_DIM)
    for row, count in enumerate(miss_counts.cpu().tolist()):
        if count == 0:
            continue
        source_rows = physical(dram_table_cpu, row, sources[row, :count])
        destination_rows = physical(
            hbm_table_cpu, row, destinations[row, :count]
        )
        if not torch.equal(
            hbm_rows[destination_rows.to(hbm.device)].cpu(),
            dram_rows[source_rows],
        ):
            raise AssertionError(f"row {row}: packed DRAM->HBM bytes differ")


def assert_candidate_boundary(
    *,
    pool: torch.Tensor,
    initial_pool: torch.Tensor,
    req_entries: torch.Tensor,
    budgets: list[int],
    candidates: list[int],
) -> None:
    pool_cpu = pool.cpu()
    initial_cpu = initial_pool.cpu()
    for row, (budget, candidate) in enumerate(zip(budgets, candidates)):
        request = int(req_entries[row].cpu())
        assert_pool_row(pool_cpu[request], candidate, budget)
        boundary = 0 if budget == 0 else candidate
        if not torch.equal(
            pool_cpu[request, boundary:], initial_cpu[request, boundary:]
        ):
            raise AssertionError(
                f"row {row}: cache state changed outside candidate prefix"
            )


def assert_dense_tail(
    *,
    hbm: torch.Tensor,
    dram_cpu: torch.Tensor,
    hbm_table_cpu: torch.Tensor,
    dram_table_cpu: torch.Tensor,
    budgets: list[int],
    candidates: list[int],
    actual_lengths: list[int],
) -> None:
    hbm_rows = hbm.view(torch.int8).view(-1, PACKED_DIM)
    dram_rows = dram_cpu.view(-1, PACKED_DIM)
    for row, (budget, candidate, actual) in enumerate(
        zip(budgets, candidates, actual_lengths)
    ):
        if actual <= candidate:
            continue
        sources = torch.arange(candidate, actual, dtype=torch.int64)
        destinations = (
            sources
            if budget == 0
            else torch.arange(
                budget, budget + actual - candidate, dtype=torch.int64
            )
        )
        source_rows = physical(dram_table_cpu, row, sources)
        destination_rows = physical(hbm_table_cpu, row, destinations)
        if not torch.equal(
            hbm_rows[destination_rows.to(hbm.device)].cpu(),
            dram_rows[source_rows],
        ):
            raise AssertionError(f"row {row}: dense tail bytes changed")


def validate_tail_profile(
    *,
    budgets: list[int],
    candidates: list[int],
    prefill_tails: list[int],
    decode_tokens: list[int],
) -> None:
    if not (
        len(budgets)
        == len(candidates)
        == len(prefill_tails)
        == len(decode_tokens)
    ):
        raise ValueError("tail profile lengths differ")
    for row, (budget, candidate, prefill_tail, decode) in enumerate(
        zip(budgets, candidates, prefill_tails, decode_tokens)
    ):
        if candidate <= 0 or candidate % BLOCK_SIZE:
            raise ValueError(f"row {row}: candidate must be block-aligned")
        if not 0 < prefill_tail < BLOCK_SIZE:
            raise ValueError(f"row {row}: prefill tail must be non-full")
        if decode <= 0:
            raise ValueError(f"row {row}: decode token count must be positive")
        if budget and budget + prefill_tail + decode > MAX_LOGICAL_HBM_SLOTS:
            raise ValueError(f"row {row}: dense tail exceeds 14-bit HBM slots")


def assert_non_mtp_no_offload(
    *,
    budgets: list[int],
    sources: torch.Tensor,
    slots: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    sources_cpu = sources.cpu().view(len(budgets), TOPK)
    slots_cpu = slots.cpu().view(len(budgets), TOPK)
    counts_cpu = counts.cpu()
    for row, budget in enumerate(budgets):
        if budget != 0:
            continue
        if (
            int(counts_cpu[row]) != 0
            or bool((sources_cpu[row] != -1).any())
            or bool((slots_cpu[row] != -1).any())
        ):
            raise AssertionError(f"row {row}: C=0 request is not a strict no-op")


def assert_mtp_no_offload(
    *,
    budgets: list[int],
    query_counts: list[int],
    topk_sources: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_miss_counts: torch.Tensor,
    copy_sources: torch.Tensor,
    copy_slots: torch.Tensor,
    copy_counts: torch.Tensor,
) -> None:
    topk_sources_cpu = topk_sources.cpu().view(-1, TOPK)
    topk_slots_cpu = topk_slots.cpu().view(-1, TOPK)
    topk_miss_cpu = topk_miss_counts.cpu()
    copy_sources_cpu = copy_sources.cpu()
    copy_slots_cpu = copy_slots.cpu()
    copy_counts_cpu = copy_counts.cpu()
    begin = 0
    for request, (budget, query_count) in enumerate(
        zip(budgets, query_counts)
    ):
        end = begin + query_count
        if budget == 0 and (
            bool((topk_sources_cpu[begin:end] != -1).any())
            or bool((topk_slots_cpu[begin:end] != -1).any())
            or bool((topk_miss_cpu[begin:end] != 0).any())
            or int(copy_counts_cpu[request]) != 0
            or bool((copy_sources_cpu[request] != OUTPUT_POISON).any())
            or bool((copy_slots_cpu[request] != OUTPUT_POISON).any())
        ):
            raise AssertionError(
                f"request {request}: C=0 MTP3 request is not a strict no-op"
            )
        begin = end


def run_case(
    *, case_name: str, device: torch.device, replays: int, seed: int
) -> None:
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    if case_name == "pure-long":
        budgets = [CACHE_TOKENS, CACHE_TOKENS]
        candidates = [SOURCE_LEN, SOURCE_LEN]
        prompt_lengths = [candidate + TAIL_TOKENS for candidate in candidates]
        source_capacity = SOURCE_LEN
        miss_range = (256, 512)
    elif case_name == "mixed":
        budgets = [0, CACHE_TOKENS]
        candidates = [TOPK, SOURCE_LEN]
        prompt_lengths = [candidate + TAIL_TOKENS for candidate in candidates]
        source_capacity = SOURCE_LEN
        miss_range = (256, 512)
    elif case_name == "comprehensive":
        budgets = COMPREHENSIVE_BUDGETS.copy()
        candidates = COMPREHENSIVE_CANDIDATES.copy()
        prefill_tails = COMPREHENSIVE_PREFILL_TAILS.copy()
        decode_tokens = [1] * len(budgets)
        validate_tail_profile(
            budgets=budgets,
            candidates=candidates,
            prefill_tails=prefill_tails,
            decode_tokens=decode_tokens,
        )
        prompt_lengths = [
            candidate + prefill_tail + decode
            for candidate, prefill_tail, decode in zip(
                candidates, prefill_tails, decode_tokens
            )
        ]
        source_capacity = max(candidates)
        miss_range = (100, 300)
    else:
        raise ValueError(f"unknown non-MTP graph case: {case_name}")
    batch = len(budgets)
    post_candidate_tokens = [
        prompt - candidate
        for prompt, candidate in zip(prompt_lengths, candidates)
    ]
    resident_lengths_cpu = [
        prompt if budget == 0 else budget + prompt - candidate
        for budget, candidate, prompt in zip(
            budgets, candidates, prompt_lengths
        )
    ]
    lidu_case = make_case(
        device=device,
        batch=batch,
        source_len=source_capacity,
        heads=32,
        budgets=budgets,
        miss_range=miss_range,
        pool_extra=3,
        seed=seed,
        candidate_lens_cpu=candidates,
    )
    dram_table_cpu, dram_blocks = private_table(
        batch,
        math.ceil(max(prompt_lengths) / BLOCK_SIZE),
        generator,
    )
    hbm_table_cpu, _ = private_table(
        batch,
        math.ceil(max(resident_lengths_cpu) / BLOCK_SIZE),
        generator,
    )
    dram_cpu = make_packed_bytes(dram_blocks, generator)
    initial_hbm_cpu = initialize_hbm(
        dram=dram_cpu,
        dram_table=dram_table_cpu,
        hbm_table=hbm_table_cpu,
        pool=lidu_case.initial_pool,
        req_entries=lidu_case.req_entries,
        budgets=budgets,
        candidates=candidates,
        actual_lengths=prompt_lengths,
    )
    dram = swapped_from_cpu(dram_cpu, device)
    initial_hbm = initial_hbm_cpu.view(torch.float8_e4m3fn).to(device)
    dram_table = dram_table_cpu.to(device)
    hbm_table = hbm_table_cpu.to(device)
    resident_lengths = torch.tensor(
        resident_lengths_cpu, dtype=torch.int32, device=device
    )
    attention_query = torch.empty(
        (batch, 8, QUERY_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-0.5, 0.5)
    scale = QUERY_DIM**-0.5

    def make_state(
        hbm_seed: torch.Tensor | None = None,
        pool_seed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        hbm = initial_hbm.clone() if hbm_seed is None else hbm_seed.clone()
        pool = (
            lidu_case.initial_pool.clone()
            if pool_seed is None
            else pool_seed.clone()
        )
        sources = torch.full(
            (batch, 1, TOPK), OUTPUT_POISON, dtype=torch.int32, device=device
        )
        destinations = torch.full_like(sources, OUTPUT_POISON)
        counts = torch.full(
            (batch,), OUTPUT_POISON, dtype=torch.int32, device=device
        )
        attention = torch.empty(
            (batch, attention_query.size(1), NOPE_DIM),
            dtype=attention_query.dtype,
            device=device,
        )
        return hbm, pool, sources, destinations, counts, attention

    def chain(state: tuple[torch.Tensor, ...]) -> None:
        hbm, pool, sources, destinations, counts, attention = state
        torch.ops.nanovllm_dsa.fused_li_manage_c8.default(
            lidu_case.query,
            lidu_case.weights,
            lidu_case.key,
            lidu_case.query_scale,
            lidu_case.key_scale,
            lidu_case.block_table,
            lidu_case.candidate_lens,
            lidu_case.cache_tokens,
            lidu_case.req_entries,
            pool,
            sources,
            destinations,
            counts,
        )
        torch.ops.nanovllm_dsa.kvcache_scatter_copy_c8.default(
            sources,
            destinations,
            counts,
            hbm_table,
            dram_table,
            hbm.view(torch.int8),
            dram,
        )
        torch.ops.nanovllm_dsa.sparse_tail_attention_c8.default(
            attention_query,
            lidu_case.actual_q,
            resident_lengths,
            lidu_case.cache_tokens,
            destinations,
            hbm_table,
            hbm,
            scale,
            attention,
        )

    eager_state = make_state()
    chain(eager_state)
    torch.npu.synchronize()
    expected_sources = eager_state[2].cpu()
    expected_destinations = eager_state[3].cpu()
    expected_counts = eager_state[4].cpu()
    long_rows = lidu_case.cache_tokens.cpu() > 0
    if bool((expected_counts[long_rows] <= 0).any()):
        raise AssertionError(
            f"long rows require nonzero misses, got {expected_counts.tolist()}"
        )
    assert_non_mtp_no_offload(
        budgets=budgets,
        sources=eager_state[2],
        slots=eager_state[3],
        counts=eager_state[4],
    )
    assert_copy(
        hbm=eager_state[0],
        dram_cpu=dram_cpu,
        hbm_table_cpu=hbm_table_cpu,
        dram_table_cpu=dram_table_cpu,
        source_ids=eager_state[2],
        destination_slots=eager_state[3],
        miss_counts=eager_state[4],
    )
    assert_candidate_boundary(
        pool=eager_state[1],
        initial_pool=lidu_case.initial_pool,
        req_entries=lidu_case.req_entries,
        budgets=budgets,
        candidates=candidates,
    )
    assert_dense_tail(
        hbm=eager_state[0],
        dram_cpu=dram_cpu,
        hbm_table_cpu=hbm_table_cpu,
        dram_table_cpu=dram_table_cpu,
        budgets=budgets,
        candidates=candidates,
        actual_lengths=prompt_lengths,
    )
    eager_golden = attention_golden(
        query=attention_query,
        hbm=eager_state[0],
        topk_slots=eager_state[3],
        budgets=lidu_case.cache_tokens,
        resident_lengths=resident_lengths,
        block_table_cpu=hbm_table_cpu,
        scale=scale,
    )
    torch.testing.assert_close(
        eager_state[5].cpu().float(), eager_golden, atol=0.08, rtol=0.03
    )

    graph_state = make_state(eager_state[0], eager_state[1])
    stable_ptrs = tuple(tensor.data_ptr() for tensor in graph_state)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        chain(graph_state)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    if bool((graph_state[4].cpu() != 0).any()):
        raise AssertionError("first replay of warmed C8 cache must be zero-miss")

    samples_us: list[float] = []
    for replay in range(replays):
        graph_state[0].copy_(initial_hbm)
        graph_state[1].copy_(lidu_case.initial_pool)
        for tensor in graph_state[2:5]:
            tensor.fill_(OUTPUT_POISON)
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)
        if not torch.equal(graph_state[2].cpu(), expected_sources):
            raise AssertionError(f"replay {replay}: source IDs changed")
        if not torch.equal(graph_state[3].cpu(), expected_destinations):
            raise AssertionError(f"replay {replay}: destination slots changed")
        if not torch.equal(graph_state[4].cpu(), expected_counts):
            raise AssertionError(f"replay {replay}: miss counts changed")
        assert_non_mtp_no_offload(
            budgets=budgets,
            sources=graph_state[2],
            slots=graph_state[3],
            counts=graph_state[4],
        )
        assert_copy(
            hbm=graph_state[0],
            dram_cpu=dram_cpu,
            hbm_table_cpu=hbm_table_cpu,
            dram_table_cpu=dram_table_cpu,
            source_ids=graph_state[2],
            destination_slots=graph_state[3],
            miss_counts=graph_state[4],
        )
        assert_candidate_boundary(
            pool=graph_state[1],
            initial_pool=lidu_case.initial_pool,
            req_entries=lidu_case.req_entries,
            budgets=budgets,
            candidates=candidates,
        )
        assert_dense_tail(
            hbm=graph_state[0],
            dram_cpu=dram_cpu,
            hbm_table_cpu=hbm_table_cpu,
            dram_table_cpu=dram_table_cpu,
            budgets=budgets,
            candidates=candidates,
            actual_lengths=prompt_lengths,
        )
        golden = attention_golden(
            query=attention_query,
            hbm=graph_state[0],
            topk_slots=graph_state[3],
            budgets=lidu_case.cache_tokens,
            resident_lengths=resident_lengths,
            block_table_cpu=hbm_table_cpu,
            scale=scale,
        )
        torch.testing.assert_close(
            graph_state[5].cpu().float(), golden, atol=0.08, rtol=0.03
        )
        if tuple(tensor.data_ptr() for tensor in graph_state) != stable_ptrs:
            raise AssertionError("C8 graph replay changed a caller-owned address")
    print(
        "A5_C8_SPLIT_OFFLOAD_GRAPH_CHECK "
        f"case={case_name} budgets={budgets} candidates={candidates} "
        f"post_candidate_dense_tokens={post_candidate_tokens} "
        f"misses={expected_counts.tolist()} replays={replays} "
        f"avg_replay_us={statistics.mean(samples_us):.3f} "
        "capture_zero_miss=1 replay_nonzero_miss=1 caller_owned_outputs=1 "
        "dram_to_hbm_exact=1 cache_budget_exact=1 candidate_boundary=1 "
        "dense_tail_exact=1 "
        "attention_golden=1 stable_addresses=1 "
        f"comprehensive={int(case_name == 'comprehensive')} ok=1",
        flush=True,
    )


def attention_golden_mtp(
    *,
    query: torch.Tensor,
    hbm: torch.Tensor,
    topk_slots: torch.Tensor,
    budgets: torch.Tensor,
    resident_lengths: torch.Tensor,
    query_counts: list[int],
    block_table_cpu: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    key, value = decode_packed(hbm.cpu())
    flat_key = key.view(-1, QUERY_DIM)
    flat_value = value.view(-1, NOPE_DIM)
    query_cpu = query.cpu().float()
    topk_cpu = topk_slots.cpu().view(query.size(0), TOPK)
    outputs: list[torch.Tensor] = []
    packed_row = 0
    for request, query_count in enumerate(query_counts):
        budget = int(budgets[request].cpu())
        resident = int(resident_lengths[request].cpu())
        dense_prefix = resident - query_count
        for path in range(query_count):
            if budget == 0:
                slots = torch.arange(
                    dense_prefix + path + 1, dtype=torch.int64
                )
            else:
                slots = torch.cat(
                    (
                        topk_cpu[packed_row].to(torch.int64),
                        torch.arange(
                            budget,
                            dense_prefix + path + 1,
                            dtype=torch.int64,
                        ),
                    )
                )
            rows = physical(block_table_cpu, request, slots)
            scores = query_cpu[packed_row] @ flat_key[rows].T * scale
            probabilities = torch.softmax(scores, dim=-1)
            outputs.append(
                probabilities.to(torch.bfloat16).float() @ flat_value[rows]
            )
            packed_row += 1
    return torch.stack(outputs)


def run_mtp_case(
    *, case_name: str, device: torch.device, replays: int, seed: int
) -> None:
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    if case_name == "pure-long":
        query_counts = [4, 4]
        budgets = [MTP_CACHE_TOKENS, MTP_CACHE_TOKENS]
        candidates = [MTP_SOURCE_LEN, MTP_SOURCE_LEN]
        prefill_tails = [0, 0]
        source_capacity = MTP_SOURCE_LEN
    elif case_name == "mixed":
        query_counts = [4, 4]
        budgets = [0, MTP_CACHE_TOKENS]
        candidates = [TOPK, MTP_SOURCE_LEN]
        prefill_tails = [0, 0]
        source_capacity = MTP_SOURCE_LEN
    elif case_name == "comprehensive":
        query_counts = [4, 4, 4, 4]
        budgets = COMPREHENSIVE_BUDGETS.copy()
        candidates = COMPREHENSIVE_CANDIDATES.copy()
        prefill_tails = COMPREHENSIVE_PREFILL_TAILS.copy()
        source_capacity = max(candidates)
    else:
        raise ValueError(f"unknown MTP graph case: {case_name}")
    batch = len(query_counts)
    if case_name == "comprehensive":
        validate_tail_profile(
            budgets=budgets,
            candidates=candidates,
            prefill_tails=prefill_tails,
            decode_tokens=query_counts,
        )
    # All MTP top-k rows of one request share a cache mapping.  The union
    # capacity is therefore four times the per-query top-2048 capacity.
    source_lengths = [
        candidate + prefill_tail + query_count
        for candidate, prefill_tail, query_count in zip(
            candidates, prefill_tails, query_counts
        )
    ]
    lidu_case = make_mtp_case(
        device=device,
        batch=batch,
        heads=32,
        source_len=source_capacity,
        budgets=budgets,
        per_query_misses=100 if case_name == "comprehensive" else 200,
        union_misses=300,
        query_noise=0.25,
        pool_extra=3,
        seed=seed,
        candidate_lens_cpu=candidates,
    )
    dram_table_cpu, dram_blocks = private_table(
        batch,
        math.ceil(max(source_lengths) / BLOCK_SIZE),
        generator,
    )
    resident_lengths_cpu = [
        source if budget == 0 else budget + source - candidate
        for source, budget, candidate in zip(
            source_lengths, budgets, candidates
        )
    ]
    hbm_table_cpu, _ = private_table(
        batch,
        math.ceil(max(resident_lengths_cpu) / BLOCK_SIZE),
        generator,
    )
    dram_cpu = make_packed_bytes(dram_blocks, generator)
    initial_hbm_cpu = initialize_hbm(
        dram=dram_cpu,
        dram_table=dram_table_cpu,
        hbm_table=hbm_table_cpu,
        pool=lidu_case.initial_pool,
        req_entries=lidu_case.req_entries,
        budgets=budgets,
        candidates=candidates,
        actual_lengths=source_lengths,
    )
    dram = swapped_from_cpu(dram_cpu, device)
    initial_hbm = initial_hbm_cpu.view(torch.float8_e4m3fn).to(device)
    dram_table = dram_table_cpu.to(device)
    hbm_table = hbm_table_cpu.to(device)
    resident_lengths = torch.tensor(
        resident_lengths_cpu, dtype=torch.int32, device=device
    )
    packed_queries = sum(query_counts)
    attention_query = torch.empty(
        (packed_queries, 8, QUERY_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-0.5, 0.5)
    scale = QUERY_DIM**-0.5

    def make_state(
        hbm_seed: torch.Tensor | None = None,
        pool_seed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        hbm = initial_hbm.clone() if hbm_seed is None else hbm_seed.clone()
        pool = (
            lidu_case.initial_pool.clone()
            if pool_seed is None
            else pool_seed.clone()
        )
        (
            topk_sources,
            topk_slots,
            topk_miss_count,
            copy_sources,
            copy_slots,
            copy_counts,
        ) = mtp_output_buffers(lidu_case)
        attention = torch.empty(
            (packed_queries, attention_query.size(1), NOPE_DIM),
            dtype=attention_query.dtype,
            device=device,
        )
        return (
            hbm,
            pool,
            topk_sources,
            topk_slots,
            topk_miss_count,
            copy_sources,
            copy_slots,
            copy_counts,
            attention,
        )

    def chain(state: tuple[torch.Tensor, ...]) -> None:
        (
            hbm,
            pool,
            topk_sources,
            topk_slots,
            topk_miss_count,
            copy_sources,
            copy_slots,
            copy_counts,
            output,
        ) = state
        torch.ops.nanovllm_dsa.fused_li_manage_mtp_c8.default(
            lidu_case.query,
            lidu_case.weights,
            lidu_case.key,
            lidu_case.query_scale,
            lidu_case.key_scale,
            lidu_case.actual_q,
            lidu_case.block_table,
            lidu_case.candidate_lens,
            lidu_case.cache_tokens,
            lidu_case.req_entries,
            pool,
            topk_sources,
            topk_slots,
            topk_miss_count,
            copy_sources,
            copy_slots,
            copy_counts,
        )
        torch.ops.nanovllm_dsa.kvcache_scatter_copy_c8.default(
            copy_sources,
            copy_slots,
            copy_counts,
            hbm_table,
            dram_table,
            hbm.view(torch.int8),
            dram,
        )
        torch.ops.nanovllm_dsa.sparse_tail_attention_c8.default(
            attention_query,
            lidu_case.actual_q,
            resident_lengths,
            lidu_case.cache_tokens,
            topk_slots,
            hbm_table,
            hbm,
            scale,
            output,
        )

    eager_state = make_state()
    chain(eager_state)
    torch.npu.synchronize()
    expected = tuple(tensor.cpu() for tensor in eager_state[2:8])
    long_rows = lidu_case.cache_tokens.cpu() > 0
    if bool((expected[5][long_rows] <= 0).any()):
        raise AssertionError("long MTP rows must exercise nonzero union misses")
    assert_mtp_no_offload(
        budgets=budgets,
        query_counts=query_counts,
        topk_sources=eager_state[2],
        topk_slots=eager_state[3],
        topk_miss_counts=eager_state[4],
        copy_sources=eager_state[5],
        copy_slots=eager_state[6],
        copy_counts=eager_state[7],
    )
    assert_copy(
        hbm=eager_state[0],
        dram_cpu=dram_cpu,
        hbm_table_cpu=hbm_table_cpu,
        dram_table_cpu=dram_table_cpu,
        source_ids=eager_state[5],
        destination_slots=eager_state[6],
        miss_counts=eager_state[7],
    )
    assert_candidate_boundary(
        pool=eager_state[1],
        initial_pool=lidu_case.initial_pool,
        req_entries=lidu_case.req_entries,
        budgets=budgets,
        candidates=candidates,
    )
    assert_dense_tail(
        hbm=eager_state[0],
        dram_cpu=dram_cpu,
        hbm_table_cpu=hbm_table_cpu,
        dram_table_cpu=dram_table_cpu,
        budgets=budgets,
        candidates=candidates,
        actual_lengths=source_lengths,
    )
    golden = attention_golden_mtp(
        query=attention_query,
        hbm=eager_state[0],
        topk_slots=eager_state[3],
        budgets=lidu_case.cache_tokens,
        resident_lengths=resident_lengths,
        query_counts=query_counts,
        block_table_cpu=hbm_table_cpu,
        scale=scale,
    )
    torch.testing.assert_close(
        eager_state[8].cpu().float(), golden, atol=0.08, rtol=0.03
    )

    graph_state = make_state(eager_state[0], eager_state[1])
    stable_ptrs = tuple(tensor.data_ptr() for tensor in graph_state)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        chain(graph_state)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    if bool((graph_state[4].cpu() != 0).any()) or bool(
        (graph_state[7].cpu() != 0).any()
    ):
        raise AssertionError("warmed MTP graph replay must be zero-miss")

    samples_us: list[float] = []
    for replay in range(replays):
        graph_state[0].copy_(initial_hbm)
        graph_state[1].copy_(lidu_case.initial_pool)
        for tensor in graph_state[2:8]:
            tensor.fill_(OUTPUT_POISON)
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)
        if any(
            not torch.equal(graph_state[index].cpu(), expected[index - 2])
            for index in (2, 3, 4, 7)
        ):
            raise AssertionError(f"MTP replay {replay}: LIM metadata changed")
        for request, count in enumerate(expected[5].tolist()):
            if not torch.equal(
                graph_state[5][request, :count].cpu(),
                expected[3][request, :count],
            ) or not torch.equal(
                graph_state[6][request, :count].cpu(),
                expected[4][request, :count],
            ):
                raise AssertionError(
                    f"MTP replay {replay}: union copy prefix changed"
                )
        assert_mtp_no_offload(
            budgets=budgets,
            query_counts=query_counts,
            topk_sources=graph_state[2],
            topk_slots=graph_state[3],
            topk_miss_counts=graph_state[4],
            copy_sources=graph_state[5],
            copy_slots=graph_state[6],
            copy_counts=graph_state[7],
        )
        assert_copy(
            hbm=graph_state[0],
            dram_cpu=dram_cpu,
            hbm_table_cpu=hbm_table_cpu,
            dram_table_cpu=dram_table_cpu,
            source_ids=graph_state[5],
            destination_slots=graph_state[6],
            miss_counts=graph_state[7],
        )
        assert_candidate_boundary(
            pool=graph_state[1],
            initial_pool=lidu_case.initial_pool,
            req_entries=lidu_case.req_entries,
            budgets=budgets,
            candidates=candidates,
        )
        assert_dense_tail(
            hbm=graph_state[0],
            dram_cpu=dram_cpu,
            hbm_table_cpu=hbm_table_cpu,
            dram_table_cpu=dram_table_cpu,
            budgets=budgets,
            candidates=candidates,
            actual_lengths=source_lengths,
        )
        golden = attention_golden_mtp(
            query=attention_query,
            hbm=graph_state[0],
            topk_slots=graph_state[3],
            budgets=lidu_case.cache_tokens,
            resident_lengths=resident_lengths,
            query_counts=query_counts,
            block_table_cpu=hbm_table_cpu,
            scale=scale,
        )
        torch.testing.assert_close(
            graph_state[8].cpu().float(), golden, atol=0.08, rtol=0.03
        )
        if tuple(tensor.data_ptr() for tensor in graph_state) != stable_ptrs:
            raise AssertionError("MTP C8 graph changed a caller-owned address")
    print(
        "A5_C8_MTP_SPLIT_OFFLOAD_GRAPH_CHECK "
        f"case={case_name} query_counts={query_counts} budgets={budgets} "
        f"candidates={candidates} prefill_tails={prefill_tails} "
        f"misses={expected[5].tolist()} "
        f"replays={replays} avg_replay_us={statistics.mean(samples_us):.3f} "
        "capture_zero_miss=1 replay_nonzero_miss=1 union_copy=1 "
        "per_query_topk=1 causal_tail=1 dram_to_hbm_exact=1 "
        "cache_budget_exact=1 candidate_boundary=1 dense_tail_exact=1 "
        "attention_golden=1 "
        "stable_addresses=1 "
        f"comprehensive={int(case_name == 'comprehensive')} ok=1",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.replays <= 0:
        raise ValueError("--replays must be positive")
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    require_a5(device, args.allow_non_a5)
    case_names = ("pure-long", "mixed", "comprehensive")
    for index, case_name in enumerate(case_names):
        run_case(
            case_name=case_name,
            device=device,
            replays=args.replays,
            seed=args.seed + index * 1000,
        )
    for index, case_name in enumerate(case_names):
        run_mtp_case(
            case_name=case_name,
            device=device,
            replays=args.replays,
            seed=args.seed + 10000 + index * 1000,
        )
    print("A5_C8_SPLIT_OFFLOAD_GRAPH_UT_OK", flush=True)


if __name__ == "__main__":
    main()
