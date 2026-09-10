#!/usr/bin/env python3
"""Debug helper: dump pool diff for the heterogeneous candidates/budgets case.

Usage: python3 _dbg_hetero.py [--full] [--variant ORIG|UNIFORM|SINGLE|BIGFIRST]
By default runs all four case variants in compact summary mode.  --full keeps
the original verbose pool/query dump for the ORIG variant.

For --variant SINGLE the run also decodes the C8_MTP_DUMP_LD_SLOTS kernel
backfill (slot s -> inactive output row s).  Only meaningful when the kernel
macro is flipped to 1; otherwise those rows hold the OUTPUT_POISON sentinel and
the decode will read as FAIL.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import nanovllm_dsa_a5  # noqa: F401
import test_fused_li_manage_mtp_c8 as T
from _lidu_utils import MAX_CACHE_TOKENS, TOPK

BLOCK = 128
AIC_CORE_NUM = 24


def compute_split(case) -> list[dict]:
    """Replicate kernel ComputeSplitInfo: per active request, covering cores.

    Returns rows of {b, rs, re, cand, cores, slot_of_core}, where cores = list of
    (cubeCoreIdx, local_block_lo, local_block_hi) covering [rs,re) global
    candidate blocks (block pool laid out row-major over active requests), and
    slot_of_core mirrors the kernel LD slot assignment: rows touched by >=2 cores
    get `touches` consecutive slots starting at slotCtr (row-major over rows).
    """
    budgets = case.cache_tokens.cpu().tolist()
    cands = case.candidate_lens.cpu().tolist()
    rows: list[dict] = []
    acc = 0
    for b, (bud, cand) in enumerate(zip(budgets, cands)):
        if bud == 0:
            continue
        w = (cand + BLOCK - 1) // BLOCK
        rows.append({"b": b, "rs": acc, "re": acc + w, "cand": cand})
        acc += w
    total = acc
    core_num = min(total, AIC_CORE_NUM)
    min_block = total // core_num
    deal = total % core_num

    def rng(c: int) -> tuple[int, int]:
        return (c * min_block + min(c, deal),
                (c + 1) * min_block + min(c + 1, deal))

    slot_ctr = 0
    for row in rows:
        rs, re = row["rs"], row["re"]
        covers: list[tuple[int, int, int]] = []
        for c in range(core_num):
            cb, ce = rng(c)
            lo, hi = max(cb, rs), min(ce, re)
            if lo < hi:
                covers.append((c, lo - rs, hi - rs))
        row["cores"] = covers
        # kernel: rows with touches>=2 occupy `touches` consecutive LD slots,
        # slot = base + (core - firstCoveringCore), base advances row-major.
        slot_of_core: dict[int, int] = {}
        if len(covers) >= 2:
            base = slot_ctr
            slot_ctr += len(covers)
            first = covers[0][0]
            for c, _, _ in covers:
                slot_of_core[c] = base + (c - first)
        row["slot_of_core"] = slot_of_core
    return rows


def slot_of_local_block(row: dict, local_block: int) -> int | None:
    """Return the LD slot that owns request-local candidate block, or None."""
    for c, blo, bhi in row["cores"]:
        if blo <= local_block < bhi:
            return row["slot_of_core"].get(c)
    return None


VARIANTS = {
    # original heterogeneous case (req1..5 in pool order)
    "ORIG": dict(
        budgets=[0, 8192, 12288, 8192, 12288, MAX_CACHE_TOKENS],
        candidate_lens_cpu=[4096, 12288, 20096, 12288, 16384, 19968],
    ),
    # all active requests identical to the (perfect) req1 config
    "UNIFORM": dict(
        budgets=[0, 8192, 8192, 8192, 8192, 8192],
        candidate_lens_cpu=[4096, 12288, 12288, 12288, 12288, 12288],
    ),
    # a single active request that is huge (spans 6 cores by itself)
    "SINGLE": dict(
        budgets=[0, 0, 0, 0, 0, MAX_CACHE_TOKENS],
        candidate_lens_cpu=[4096, 12288, 20096, 12288, 16384, 19968],
    ),
    # same (budget,cand) pairs as ORIG but big 20096-config is FIRST row
    "BIGFIRST": dict(
        budgets=[0, 12288, 8192, 8192, 12288, MAX_CACHE_TOKENS],
        candidate_lens_cpu=[4096, 20096, 12288, 12288, 16384, 19968],
    ),
}


QUERY_COUNT = 4
UNION_CAPACITY = 8192


def run_isolated(case, batch_row: int) -> bool:
    """Run one request truly alone (batch=1, bIdx=0, prefix sums = 0).

    Mirrors check_case's single-request section.  If an in-batch-failing request
    becomes correct here, the bug is batch-position/prefix dependent; if it still
    fails, the bug is inherent to that request's data or the single-row split.
    """
    q0 = batch_row * QUERY_COUNT
    q1 = q0 + QUERY_COUNT
    opt = {"dtype": torch.int32, "device": case.query.device}
    single_pool = case.initial_pool.clone()
    single_actual_q = torch.tensor([q1 - q0], dtype=torch.int32,
                                   device=case.query.device)
    single = (
        torch.empty((QUERY_COUNT, 1, TOPK), **opt),
        torch.empty((QUERY_COUNT, 1, TOPK), **opt),
        torch.empty((QUERY_COUNT,), **opt),
        torch.empty((1, UNION_CAPACITY), **opt),
        torch.empty((1, UNION_CAPACITY), **opt),
        torch.empty((1,), **opt),
    )
    torch.ops.nanovllm_dsa.fused_li_manage_mtp_c8.default(
        case.query[q0:q1].contiguous(),
        case.weights[q0:q1].contiguous(),
        case.key,
        case.query_scale[q0:q1].contiguous(),
        case.key_scale,
        single_actual_q,
        case.block_table[batch_row:batch_row + 1].contiguous(),
        case.candidate_lens[batch_row:batch_row + 1],
        case.cache_tokens[batch_row:batch_row + 1],
        case.req_entries[batch_row:batch_row + 1],
        single_pool,
        *single,
    )
    torch.npu.synchronize()
    src = single[0].reshape(QUERY_COUNT, TOPK).cpu()
    ref = case.native_topk[q0:q1].reshape(QUERY_COUNT, TOPK).cpu()
    miss = single[2].reshape(QUERY_COUNT).cpu().tolist()
    slots = single[1].reshape(QUERY_COUNT, TOPK).cpu()
    print(f"  ISOLATE req{batch_row} as bIdx=0 (batch=1):")
    for i, m in enumerate(miss):
        sl_row = slots[i].tolist()
        sl_set = set(sl_row)
        n_neg1_sl = sum(1 for v in sl_row if v == -1)
        n_dup_sl = len(sl_row) != len(sl_set)
        print(f"      missCount q{i}={m} slots uniq{len(sl_set)}/dup{int(n_dup_sl)}"
              f" n_-1{n_neg1_sl}")
    all_ok = True
    for i in range(QUERY_COUNT):
        row = src[i].tolist()
        op = set(row)
        rset = set(ref[i].tolist())
        inter = len(op & rset)
        missing = len(rset - op)
        dup = len(row) != len(op)
        n_neg1 = sum(1 for v in row if v == -1)
        pos_spur = [v for v in op if v >= 0 and v not in rset]
        ok = inter == TOPK and not dup
        all_ok &= ok
        print(f"      q{i} uniq{len(op)}/i{inter}/mis{missing}/dup{int(dup)}"
              f" n_-1{n_neg1} spur_pos{len(pos_spur)}"
              f"{' OK' if ok else ' FAIL'}")
        if not ok and pos_spur:
            ps = sorted(pos_spur)
            print(f"          spur_head={ps[:8]} spur_tail={ps[-4:]}"
                  f" valrange=[{min(ps)},{max(ps)}]")
    print(f"  ISOLATE req{batch_row} VERDICT={'PASS' if all_ok else 'FAIL'}")
    return all_ok


def run_case(name: str, budgets: list[int], cands: list[int], full: bool) -> None:
    case = T.make_case(
        device=torch.device("npu:0"),
        batch=6,
        heads=32,
        source_len=20096,
        budgets=budgets,
        candidate_lens_cpu=cands,
        per_query_misses=64,
        union_misses=100,
        query_noise=0.25,
        pool_extra=7,
        seed=1707,
    )
    split = compute_split(case)
    pool = case.initial_pool.clone()
    old_pool = pool.clone()
    outputs = T.launch(case, pool)
    torch.npu.synchronize()
    (topk_sources, topk_slots, topk_miss_count,
     miss_sources, miss_slots, miss_counts) = outputs
    old_cpu = old_pool.cpu()
    new_cpu = pool.cpu()
    ms_cpu = miss_sources.cpu()
    mc_cpu = miss_counts.cpu()
    tsc = topk_sources.reshape(case.query.size(0), TOPK).cpu()
    tmcc = topk_miss_count.cpu()
    ref_cpu = case.native_topk.cpu().reshape(case.query.size(0), TOPK)

    print(f"\n##### VARIANT {name}")
    print("req_entries:", case.req_entries_cpu.tolist())
    print("budgets:", budgets)
    print("candidate_lens:", cands)
    per_req: dict[int, dict] = {b: {} for b in range(6)}

    for batch_row, (begin, end) in enumerate(T.query_ranges(case.actual_q_cpu)):
        pool_row = int(case.req_entries_cpu[batch_row])
        budget = int(case.cache_tokens[batch_row].cpu())
        cand = int(case.candidate_lens[batch_row].cpu())
        count = int(mc_cpu[batch_row])
        old_row = old_cpu[pool_row]
        new_row = new_cpu[pool_row]
        valid_old = int((old_row[:cand] >= 0).sum())
        valid_new = int((new_row[:cand] >= 0).sum())
        installed = ((old_row < 0) & (new_row >= 0)).nonzero().flatten()
        per_req[batch_row]["cand"] = cand
        per_req[batch_row]["budget"] = budget
        per_req[batch_row]["valid"] = (valid_old, valid_new)
        per_req[batch_row]["union_count"] = count
        per_req[batch_row]["installed"] = installed.numel()

        qlines: list[str] = []
        fm: set[int] = set()
        req_row = next((r for r in split if r["b"] == batch_row), None)
        miss_slots: dict[int, int] = {}   # slot -> #missing native tokens
        miss_blocks: set[int] = set()     # local candidate block indices w/ miss
        for q in range(begin, end):
            row = tsc[q].to(torch.int64)
            ref = ref_cpu[q].to(torch.int64)
            op_set = set(row.tolist())
            ref_set = set(ref.tolist())
            dup = bool(row.numel() != len(op_set))
            qlines.append(
                f"uniq{len(op_set)}/i{len(op_set & ref_set)}"
                f"/mis{len(ref_set - op_set)}/dup{int(dup)}"
                f"/mc{int(tmcc[q])}")
            if req_row is not None:
                for core, llo, lhi in req_row["cores"]:
                    lo, hi = llo * BLOCK, lhi * BLOCK
                    ref_in = len({t for t in ref_set if lo <= t < hi})
                    op_in = len({t for t in op_set if lo <= t < hi})
                    if ref_in and op_in == 0:
                        fm.add(core)
                for t in ref_set - op_set:
                    miss_blocks.add(t // BLOCK)
                    sl = slot_of_local_block(req_row, t // BLOCK)
                    miss_slots[sl] = miss_slots.get(sl, 0) + 1
        if full:
            us = ms_cpu[batch_row, :count].long() if count else None
            print(f"--- req{batch_row} pool_row={pool_row} budget={budget} "
                  f"cand={cand} union_count={count} "
                  f"valid_old={valid_old} valid_new={valid_new} "
                  f"installed(n={installed.numel()})")
            if us is not None and us.numel():
                print(f"    miss_sources(n={count}) "
                      f"in_budget={int((us < budget).sum())} "
                      f"in_cand={int((us < cand).sum())} "
                      f"min={int(us.min())} max={int(us.max())}")
        # compact summary
        if req_row is not None:
            cores_desc = ",".join(
                f"c{c}[{llo}:{lhi}]" for c, llo, lhi in req_row["cores"])
            print(f"  req{batch_row} bud{budget} cand{cand} "
                  f"slots{n_covers(req_row)} {cores_desc}")
        else:
            print(f"  req{batch_row} bud{budget} cand{cand} (inactive)")
        for i, q in enumerate(range(begin, end)):
            print(f"      q{q} {qlines[i]}")
        if fm:
            print(f"      FULLY_MISSING cores={sorted(fm)}")
        if miss_slots:
            core_of_slot = {s: c for c, s in req_row["slot_of_core"].items()}
            descs = []
            for sl in sorted(miss_slots):
                c = core_of_slot.get(sl, -1)
                llo = lhi = -1
                for cc, blo, bhi in req_row["cores"]:
                    if cc == c:
                        llo, lhi = blo, bhi
                descs.append(f"s{sl}(core{c},blk[{llo},{lhi})"
                             f" tok[{llo*BLOCK},{lhi*BLOCK}))x{miss_slots[sl]}")
            rngs = []
            for blk in sorted(miss_blocks):
                if rngs and blk == rngs[-1][1]:
                    rngs[-1][1] = blk + 1
                else:
                    rngs.append([blk, blk + 1])
            print("      MISS_SLOTS " + ", ".join(descs))
            print("      MISS_BLOCKS " +
                  ", ".join(f"[{a},{b})tok[{a*BLOCK},{b*BLOCK})"
                            for a, b in rngs))
        # coverage per covering core: of the native-top tokens that live in a
        # core's block range, how many make it into the (4-route) output.
        if req_row is not None:
            cov = []
            for core, llo, lhi in req_row["cores"]:
                lo, hi = llo * BLOCK, lhi * BLOCK
                exp = got = 0
                for q in range(begin, end):
                    ref = set(ref_cpu[q].tolist())
                    op = set(tsc[q].tolist())
                    rng_ref = {t for t in ref if lo <= t < hi}
                    exp += len(rng_ref)
                    got += len(rng_ref & op)
                cov.append(f"c{core}[{llo}:{lhi}] {got}/{exp}")
            print("      COVERAGE " + ", ".join(cov))


    # 旧 GM 行 dump 已废弃（非活动请求 -1 预填污染，无法判定触发与否）。核内
    # [DUMPW]/[DUMPL] 标量才是当前证据。仅当显式设置 C8_GM_DUMP=1 时才解码 GM 行。
    if os.environ.get("C8_GM_DUMP") == "1":
        decode_ld_slot_dump(name, split, tsc)

    print("  --- isolated single-request re-runs (bIdx=0, batch=1) ---")
    for batch_row, (begin, end) in enumerate(T.query_ranges(case.actual_q_cpu)):
        if int(case.cache_tokens[batch_row].cpu()) == 0:
            continue
        run_isolated(case, batch_row)


def n_covers(row: dict) -> int:
    return len(row["cores"])


def decode_ld_slot_dump(name: str, split: list[dict], tsc) -> None:
    """Decode the C8_MTP_DUMP_LD_SLOTS backfill (SINGLE variant only).

    The writer core copies, before storing a slot, the route-0 index row into
    output rows 0..19 (slot s -> row s).  In SINGLE the only active request is the
    last one, so rows 0..19 belong to inactive requests and are free.  Each
    covering core c writes slot c with its whole block window (window < 2048
    tokens, so the full window is present, ascending and gapless).  This checks
    each row against the corresponding core's window.
    """
    if name != "SINGLE":
        return
    act = next((r for r in split if r["cand"]), None)
    if act is None:
        return
    win = {c: (llo, lhi) for c, llo, lhi in act["cores"]}
    poison = T.OUTPUT_POISON
    print("  --- writer LD slot dump decode (slot s = core s, route 0) "
          "[needs C8_MTP_DUMP_LD_SLOTS=1] ---")
    print("      row states: NO_DUMP=poison 全部未覆盖; EMPTY=dump 写了内容但无正 token;"
          " OK=窗口全集")
    n_ok = n_chk = n_fired = 0
    for s in range(0, min(20, act["re"])):
        if s not in win:
            print(f"    slot{s}: no covering core, skip")
            continue
        llo, lhi = win[s]
        vals = [int(v) for v in tsc[s].tolist()]
        pos = [v for v in vals if v >= 0]
        neg1 = sum(1 for v in vals if v == -1)
        n_poison = sum(1 for v in vals if v == poison)
        fired = n_poison == 0
        n_fired += int(fired)
        n_chk += 1
        exp_count = (lhi - llo) * BLOCK
        contiguous = all(b - a == 1 for a, b in zip(pos, pos[1:]))
        no_dup = len(pos) == len(set(pos))
        cnt_ok = len(pos) == exp_count
        dev = ""
        if pos:
            dev = f" start_dev={pos[0] - llo * BLOCK:+d}"
        ok = fired and contiguous and no_dup and cnt_ok and len(pos) > 0
        n_ok += int(ok)
        state = ("NO_DUMP(poison)" if not fired else
                 ("EMPTY(-1 only)" if not pos else
                  ("OK" if ok else "BAD")))
        print(f"    slot{s} core{s} blk[{llo},{lhi}) exp={exp_count} "
              f"n_pos={len(pos)} n_-1={neg1} n_poison={n_poison} "
              f"contig={int(contiguous)} nodup={int(no_dup)}{dev} "
              f"STATE={state}")
    print(f"  SLOT_DECODE VERDICT={n_ok}/{n_chk} ok, fired={n_fired}/{n_chk} "
          f"(fired=0 表示 dump 路径未执行/或此运行是旧版本)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="verbose dump for the ORIG variant only")
    ap.add_argument("--variant", default=None,
                    help="run a single variant: ORIG|UNIFORM|SINGLE|BIGFIRST")
    args = ap.parse_args()
    torch.npu.set_device("npu:0")
    torch.npu.config.allow_internal_format = False

    order = [args.variant] if args.variant else list(VARIANTS)
    for name in order:
        if name not in VARIANTS:
            print(f"unknown variant {name}", file=sys.stderr)
            sys.exit(2)
        run_case(name, VARIANTS[name]["budgets"],
                 VARIANTS[name]["candidate_lens_cpu"],
                 full=args.full and name == "ORIG")


if __name__ == "__main__":
    main()
