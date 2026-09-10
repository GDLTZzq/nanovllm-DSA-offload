#!/usr/bin/env python3
"""C8 fused_li_manage 分数 dump 诊断（临时脚本，配合 kernel 侧 LI_DEBUG_SCORE_DUMP 使用）。

复现混合用例（batch=6, source=32768, s2=[2048,8192,12288,20096,24576,32768],
budgets=[0,3072,6144,8192,12288,MAX_CACHE_TOKENS], miss_range=(0,300), seed+100），
launch 一次后读 pool 行 6..12:
  行 6+b : 请求 b 的阶段2 topk 读视图（uint16 分数全行，长度 = candidate_lens[b]）
  行 12  : 阶段1 写视图（每块前 16 个分数；请求 b 的块 k 位于 uint16 偏移 b*8192 + k*16）

判定（对每个 128-block）:
  write vs read 前16不一致       -> 跨核可见性 (B)（写侧数据未在 topk 读时可见）
  read 与 torch fp32 参考 Spearman 显著 < 1 -> 写侧分数错误 (A)
  read ≈ 参考 但集合仍不一致       -> topk/payload (C)

用法: python3 li_c8_score_dump_diag.py --seed 7 --device npu:0
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import nanovllm_dsa_a5
import torch_npu  # noqa: F401,E402

from _c8_lidu_case import make_case
from _lidu_utils import MAX_CACHE_TOKENS


def unpack_row(int32_row: np.ndarray) -> np.ndarray:
    """把 int32 池行解包为 uint16 数组（小端）。"""
    raw = int32_row.astype(np.uint32)
    out = np.empty(raw.size * 2, dtype=np.uint16)
    out[0::2] = raw & 0xFFFF
    out[1::2] = raw >> 16
    return out


def rank_spearman(a: np.ndarray, b: np.ndarray) -> float:
    ta = torch.from_numpy(a)
    tb = torch.from_numpy(b)
    ra = torch.argsort(torch.argsort(ta)).float()
    rb = torch.argsort(torch.argsort(tb)).float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(ra.norm() * rb.norm())
    if denom == 0.0:
        return 0.0
    return float((ra * rb).sum()) / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False

    cands = [2048, 8192, 12288, 20096, 24576, 32768]
    case = make_case(
        device,
        6,
        32768,
        32,
        [0, 3072, 6144, 8192, 12288, MAX_CACHE_TOKENS],
        (0, 300),
        7,
        args.seed + 100,
        cands,
    )
    pool = case.initial_pool.clone()
    source_ids, _, _, alias = torch.ops.nanovllm_dsa.fused_li_manage_c8.default(
        case.query,
        case.key,
        case.weights,
        case.query_scale,
        case.key_scale,
        case.actual_q,
        case.req_entries,
        pool,
        case.cache_tokens,
        case.candidate_lens,
        case.block_table,
    )
    torch.npu.synchronize()
    pool_cpu = pool.cpu().numpy()

    print(f"seed={args.seed} mixed case launch done", flush=True)
    print(f"pool shape: {tuple(pool.shape)}  (cacheSlotsSize={pool.shape[1]})", flush=True)
    for row in range(pool.shape[0]):
        r = pool_cpu[row]
        nz = int((r != 0).sum())
        first = [int(x) for x in r[:8]]
        print(f"  pool row {row}: nonzero_int32={nz} first8={first}", flush=True)
    for b in range(6):
        ok = bool(
            (
                torch.sort(case.native_topk[b]).values
                == torch.sort(source_ids[b, 0]).values
            ).all()
        )
        print(
            f"  r{b}: candidate={cands[b]} top-2048 集合 {'一致' if ok else '不一致 !!!'}",
            flush=True,
        )

    read = [unpack_row(pool_cpu[6 + b])[: cands[b]] for b in range(6)]
    wr_all = unpack_row(pool_cpu[12])
    write = [
        wr_all[b * 8192 : b * 8192 + (cands[b] // 128) * 16].reshape(-1, 16)
        for b in range(6)
    ]

    # torch fp32 参考分数: score[b,t] = kScale[phys(t)] * Σ_h relu(q8[b,h,:]·k8[phys(t),t%128,0,:]) * w[b,h] * qScale[b,h]
    q = case.query.float()  # (6,32,128)
    k = case.key.float()  # (blocks,128,1,128)
    w = case.weights.float()  # (6,32)
    qs = case.query_scale  # (6,32)
    ks = case.key_scale  # (blocks,128,1)
    bt = case.block_table  # (6,blocks)

    def ref_scores(b: int) -> np.ndarray:
        t_idx = torch.arange(cands[b], device=device)
        blk = bt[b].repeat_interleave(128)[: cands[b]]
        tpos = t_idx % 128
        ksel = k[blk, tpos, 0, :]  # (T,128)
        qk = torch.einsum("hd,td->ht", q[b], ksel)  # (32,T)
        relu_ = torch.relu(qk)
        s = torch.einsum("ht,h->t", relu_, w[b] * qs[b])
        s = s * ks[blk, tpos, 0]
        return s.cpu().numpy()

    ref = [ref_scores(b) for b in range(6)]
    print("", flush=True)
    for b in range(6):
        r = read[b]
        rf = ref[b]
        wr = write[b]
        nb = len(r) // 128
        bad_wr = []
        bad_spear = []
        bad_top = []
        for blk in range(nb):
            seg = r[blk * 128 : (blk + 1) * 128].astype(np.float32)
            refseg = rf[blk * 128 : (blk + 1) * 128]
            wseg = wr[blk]
            wdiff = int((seg[:16] != wseg).sum())
            if wdiff:
                bad_wr.append((blk, wdiff))
            sp = rank_spearman(seg, refseg)
            if sp < 0.95:
                bad_spear.append((blk, round(sp, 4)))
            read_top = set(np.argsort(-seg)[:16].tolist())
            ref_top = set(np.argsort(-refseg)[:16].tolist())
            ov = len(read_top & ref_top)
            if ov < 16:
                bad_top.append((blk, ov))
        print(f"=== r{b} (cand={cands[b]}, blocks={nb}) ===", flush=True)
        print(f"  写读不一致块 (write!=read 前16): {bad_wr if bad_wr else '无'}", flush=True)
        print(f"  Spearman<0.95 块 (读 vs 参考): {bad_spear if bad_spear else '无'}", flush=True)
        print(f"  top16 不全同块 (读 vs 参考):   {bad_top if bad_top else '无'}", flush=True)
        if b == 2:  # r2 已知失败行: 输出逐块明细
            for blk in range(nb):
                seg = r[blk * 128 : (blk + 1) * 128].astype(np.float32)
                refseg = rf[blk * 128 : (blk + 1) * 128]
                wdiff = int((seg[:16] != wr[blk]).sum())
                sp = rank_spearman(seg, refseg)
                wsp = rank_spearman(wr[blk].astype(np.float32), refseg[:16])
                if wdiff or sp < 0.99:
                    print(
                        f"    blk{blk:3d} (t {blk*128}-{blk*128+127}): "
                        f"wr_rd_mismatch={wdiff} spearman={sp:.4f} wr_vs_ref_sp={wsp:.4f}",
                        flush=True,
                    )
                    print(
                        f"      read [:16]={[int(x) for x in seg[:16]]}",
                        flush=True,
                    )
                    print(
                        f"      write[:16]={[int(x) for x in wr[blk]]}",
                        flush=True,
                    )
                    print(
                        f"      ref  [:16]={[round(float(x), 3) for x in refseg[:16]]}",
                        flush=True,
                    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
