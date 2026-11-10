# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fraunhofer IOSB
#
# SCAHI - Slicing with Coverage-Aware Hyper Inference.
# If you use SCAHI, please cite:
#   S. Becker, J. Bayer, D. Münch, W. Hübner. When Does Sliced Inference Pay Off? Resolution, Tiling and Dense Fusion
#   for Small Object Detection. arXiv:2611.12657, 2026.
"""Dense scale-space fusion F2 on the GPU (torch, float64) - extension study E2 ("fair latency").

Same algorithm and arithmetic as :func:`scahi.fusion.scalespace.fuse_image` for variant F2 (the CPU version stays the
reference; parity test ``tests/test_scalespace_torch.py`` / job ``e2_fusion_parity``):
prediction table (max prob >= tau_splat, F2 weight = normalised border window of the pass), trilinear splat into sparse
cells keyed by (sigma bin, ix, iy), analytic coverage C (number of levels covering the cell centre), 3x3x3 local maxima via
sorted-key lookup, top 2K peaks in descending field order, neighbourhood membership (sigma bins J-1..J+1, Chebyshev
distance <= 1.5 g_J; the CPU version uses a KD-tree), each strongest member reported once (first peak in field order),
IoU cluster, per-pass maximum of w * p_c summed over passes / C, (w p_max)-weighted box, up to 3 classes per peak.

Memory (host-a brief 2026-11-01, task 3; replaces the dense peak x prediction membership matrix that needed ~90 GiB on dense
DOTA images): splatting and peak search were already sparse; membership is computed in **spatial blocks** of the peak
centres (``block`` px core) against the predictions whose centre lies in the core plus a halo of 1.5 * max g_J of the
block's peaks (+1 px against rounding of the block index) - exact, because membership depends only on the centre
distance (Chebyshev <= 1.5 g_J) and the sigma bin; IoU clustering and scores use members only. Peaks of a block are
processed in chunks so that peaks x candidates stays below ``pair_budget``. The per-pass maximum is reduced over the
(peak, pass) slots that occur instead of a dense [peaks x passes x nc] array, and the border windows are evaluated in
chunks of points. Results are those of the dense version (same comparisons; float64 sums may differ in the last bits).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from scahi.fusion.scalespace import EPS, SIGMA0, CachedLevel, FusionParams, key_bits

DT = torch.float64


def _t(a, dev, dtype=DT):
    if isinstance(a, torch.Tensor):  # GPU-resident levels (optimised pipeline)
        return a.to(device=dev, dtype=dtype)
    return torch.as_tensor(np.asarray(a), device=dev, dtype=dtype)


def _key(j: torch.Tensor, ix: torch.Tensor, iy: torch.Tensor, nb: int = 13) -> torch.Tensor:
    """As ``scalespace._key``: ``nb`` bits per spatial index (13 = VisDrone, unchanged; 15 for DOTA-size images)."""
    m = (1 << nb) - 1
    return (j.long() << (2 * nb)) | (ix.clamp(0, m).long() << nb) | iy.clamp(0, m).long()


def _cell(j: torch.Tensor, p: FusionParams) -> torch.Tensor:
    return torch.clamp(SIGMA0 * 2.0 ** ((j.to(DT) + 0.5) / p.bins_per_oct) / p.cell_factor, min=1.0)


BORDER_BUDGET = 1 << 25  # points x tiles per chunk of the border windows (float64 temporaries)


def _border(x: torch.Tensor, y: torch.Tensor, tiles: torch.Tensor, W: int, H: int, overlap: float):
    """Normalised border window [n, T] and number of containing tiles [n] (as ``scalespace.border_norm``)."""
    step = max(1, BORDER_BUDGET // max(len(tiles), 1))
    if len(x) > step:  # rows are independent: chunking does not change any value
        parts = [_border(x[i:i + step], y[i:i + step], tiles, W, H, overlap) for i in range(0, len(x), step)]
        return torch.cat([a for a, _ in parts]), torch.cat([b for _, b in parts])
    x1, y1, x2, y2 = (tiles[:, i][None] for i in range(4))
    dx, dy = overlap * (x2 - x1), overlap * (y2 - y1)
    X, Y = x[:, None], y[:, None]
    inside = (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)

    def ramp(d, w):
        return torch.sin(0.5 * math.pi * torch.clamp(d / torch.clamp(w, min=1e-6), 0, 1)) ** 2

    one = torch.ones((), dtype=DT, device=x.device)
    wx = torch.where(x1 > 0, ramp(X - x1, dx), one) * torch.where(x2 < W, ramp(x2 - X, dx), one)
    wy = torch.where(y1 > 0, ramp(Y - y1, dy), one) * torch.where(y2 < H, ramp(y2 - Y, dy), one)
    r = torch.where(inside, wx * wy, torch.zeros((), dtype=DT, device=x.device))
    s = r.sum(1, keepdim=True)
    return torch.where(s > 0, r / torch.clamp(s, min=1e-12), torch.zeros((), dtype=DT, device=x.device)), inside.sum(1)


def _border_own(x: torch.Tensor, y: torch.Tensor, tile: torch.Tensor, tiles: torch.Tensor, W: int, H: int,
                overlap: float) -> torch.Tensor:
    """Normalised border window of each point at its own tile (column ``tile`` of ``_border``), in chunks."""
    step = max(1, BORDER_BUDGET // max(len(tiles), 1))
    out = []
    for i in range(0, len(x), step):
        bn, _ = _border(x[i:i + step], y[i:i + step], tiles, W, H, overlap)
        out.append(bn[torch.arange(len(bn), device=x.device), tile[i:i + step]])
    return torch.cat(out) if out else x.new_zeros(0)


def _rc(t: torch.Tensor) -> torch.Tensor:
    return 0.5 - 0.5 * torch.cos(math.pi * torch.clamp(t, 0, 1))


def _validity_raw(sigma_in: torch.Tensor, p: FusionParams) -> torch.Tensor:
    s = torch.log2(torch.clamp(sigma_in, min=1e-6))
    la_lo, la, la_hi, lb, lb_hi = (math.log2(x) for x in (p.a_lo, p.a, p.a_hi, p.b, p.b_hi))
    up = torch.where(s < la, 0.5 * (s - la_lo) / (la - la_lo), 0.5 + 0.5 * (s - la) / (la_hi - la))
    return torch.where(s >= lb, 1.0 - _rc((s - lb) / (lb_hi - lb)), _rc(up))


def _validity_levels(sigma: torch.Tensor, levels: list[CachedLevel], p: FusionParams, empirical: bool) -> torch.Tensor:
    """[n, K] normalised validity windows (as ``scalespace.validity_levels``; gain = median of the level's gains)."""
    cols = []
    for lv in levels:
        si = sigma * float(np.median(lv.gains))
        v = _validity_raw(si, p)
        if empirical and lv.cal.emp_quality is not None:
            e = torch.as_tensor(np.asarray(lv.cal.emp_edges, float), device=sigma.device, dtype=DT)
            q_all = torch.as_tensor(np.asarray(lv.cal.emp_quality, float), device=sigma.device, dtype=DT)
            idx = torch.clamp(torch.bucketize(si, e, right=True) - 1, 0, len(e) - 2)
            q = q_all[idx]
            v = torch.where(torch.isfinite(q) & (si >= e[0]) & (si < e[-1]), q, v)
        cols.append(v)
    raw = torch.stack(cols, 1)
    return raw / torch.clamp(raw.sum(1, keepdim=True), min=1.0)


BLOCK = 1024  # spatial block core of the membership stage (px)
PAIR_BUDGET = 1 << 24  # peaks x candidate predictions per membership chunk


def _members(pk_x: torch.Tensor, pk_y: torch.Tensor, pk_j: torch.Tensor, pk_r: torch.Tensor, cx: torch.Tensor,
             cy: torch.Tensor, pj: torch.Tensor, m: torch.Tensor, boxes: torch.Tensor, iou_cluster: float,
             block: int, pair_budget: int, stats: dict | None = None):
    """Blocked membership of the peaks (in their given order).

    Returns ``has`` [P] (peak has members), ``top`` [P] (global index of the strongest member; ties: lowest index, as the
    dense version) and the member pairs (peak position, prediction index) that pass the IoU cluster with ``top``.
    """
    dev = cx.device
    P = len(pk_x)
    has = torch.zeros(P, dtype=torch.bool, device=dev)
    top = torch.zeros(P, dtype=torch.long, device=dev)
    pis, nis = [], []
    bx, by = torch.floor(pk_x / block).long(), torch.floor(pk_y / block).long()
    bid = (bx - bx.min()) * (int(by.max() - by.min()) + 1) + (by - by.min())
    order = torch.argsort(bid, stable=True)  # within a block: peaks keep their field order
    ub, cnt = torch.unique_consecutive(bid[order], return_counts=True)
    ninf = torch.full((), -math.inf, dtype=DT, device=dev)
    for blk in torch.split(order, cnt.tolist()):
        x0, y0 = float(bx[blk[0]]) * block, float(by[blk[0]]) * block
        h = float(pk_r[blk].max()) + 1.0  # halo: largest membership radius of the block's peaks (+1 px rounding)
        cand = torch.nonzero((cx >= x0 - h) & (cx < x0 + block + h) & (cy >= y0 - h) & (cy < y0 + block + h)).flatten()
        if stats is not None:
            stats["blocks"] = stats.get("blocks", 0) + 1
            stats["max_halo"] = max(stats.get("max_halo", 0.0), h)
            stats["max_cand"] = max(stats.get("max_cand", 0), len(cand))
        if not len(cand):
            continue
        ccx, ccy, cpj, cm = cx[cand], cy[cand], pj[cand], m[cand]
        step = max(1, pair_budget // len(cand))
        for ch in torch.split(blk, step):
            r = pk_r[ch][:, None]
            mem = ((cpj[None] - pk_j[ch][:, None]).abs() <= 1) & ((ccx[None] - pk_x[ch][:, None]).abs() <= r) \
                & ((ccy[None] - pk_y[ch][:, None]).abs() <= r)
            hc = mem.any(1)
            t = cand[torch.where(mem, cm[None], ninf).argmax(1)]
            has[ch], top[ch] = hc, t
            pi, ni = torch.nonzero(mem, as_tuple=True)
            pi, ni = ch[pi], cand[ni]
            bt, bb = boxes[top[pi]], boxes[ni]
            ix = torch.clamp(torch.minimum(bt[:, 2], bb[:, 2]) - torch.maximum(bt[:, 0], bb[:, 0]), min=0)
            iy = torch.clamp(torch.minimum(bt[:, 3], bb[:, 3]) - torch.maximum(bt[:, 1], bb[:, 1]), min=0)
            inter = ix * iy
            ua = (bt[:, 2] - bt[:, 0]) * (bt[:, 3] - bt[:, 1]) + (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1]) - inter
            keep = inter / torch.clamp(ua, min=1e-9) >= iou_cluster
            pis.append(pi[keep]), nis.append(ni[keep])
    e = torch.zeros(0, dtype=torch.long, device=dev)
    return has, top, (torch.cat(pis) if pis else e), (torch.cat(nis) if nis else e)


def fuse_image_f2(levels: list[CachedLevel], W: int, H: int, p: FusionParams, nc: int = 10,
                  device: str = "cuda", variant: str = "F2", block: int = BLOCK, pair_budget: int = PAIR_BUDGET,
                  stats: dict | None = None) -> pd.DataFrame:
    """F2 (or F3e) detections (x, y, w, h, score, category_id 1-based) of one image, computed on ``device``.
    ``stats`` (optional dict) receives block counts, the largest halo and the peak GPU memory of the call."""
    assert variant in ("F2", "F3e"), variant
    dev = torch.device(device)
    if stats is not None and dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    empty = pd.DataFrame(columns=["x", "y", "w", "h", "score", "category_id"])
    # ---- prediction table ------------------------------------------------------------------------------------
    cx_l, cy_l, lw_l, lh_l, sg_l, ps_l, pr_l, w_l = [], [], [], [], [], [], [], []
    tiles_t = []
    for k, lv in enumerate(levels):
        pr = _t(lv.probs, dev)
        tiles_t.append(None if lv.whole else _t(lv.tiles, dev))
        keep = pr.max(1).values >= p.tau_splat
        if not bool(keep.any()):
            continue
        b = _t(lv.xyxy, dev)[keep]
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        w_, h_ = torch.clamp(b[:, 2] - b[:, 0], min=1e-3), torch.clamp(b[:, 3] - b[:, 1], min=1e-3)
        tile = _t(lv.tile, dev, torch.long)[keep]
        if lv.whole:
            w = torch.ones(len(cx), dtype=DT, device=dev)
        else:
            w = _border_own(cx, cy, tile, tiles_t[-1], W, H, lv.overlap)
        sg = torch.sqrt(w_ * h_)
        if variant == "F3e":
            w = w * _validity_levels(sg, levels, p, True)[:, k]
        cx_l.append(cx), cy_l.append(cy), lw_l.append(torch.log(w_)), lh_l.append(torch.log(h_))
        sg_l.append(sg), ps_l.append(k * 10000 + tile), pr_l.append(pr[keep]), w_l.append(w)
    if not cx_l:
        return empty
    cx, cy, lw, lh, sig = (torch.cat(v) for v in (cx_l, cy_l, lw_l, lh_l, sg_l))
    passes, probs, w = torch.cat(ps_l), torch.cat(pr_l), torch.cat(w_l)
    pmax = probs.max(1).values
    m = w * pmax
    # ---- trilinear splat ---------------------------------------------------------------------------------------
    sb = (torch.log2(torch.clamp(sig, min=SIGMA0)) - math.log2(SIGMA0)) * p.bins_per_oct
    j0 = torch.floor(sb - 0.5).long()
    kbits = key_bits(W, H)
    keys, vals = [], []
    for dj in (0, 1):
        j = j0 + dj
        wj = (1 - (sb - 0.5 - j0)) if dj == 0 else (sb - 0.5 - j0)
        g = _cell(j, p)
        fx, fy = cx / g - 0.5, cy / g - 0.5
        ix0, iy0 = torch.floor(fx).long(), torch.floor(fy).long()
        ok = j >= 0
        for dx in (0, 1):
            wx = (1 - (fx - ix0)) if dx == 0 else (fx - ix0)
            for dy in (0, 1):
                wy = (1 - (fy - iy0)) if dy == 0 else (fy - iy0)
                keys.append(_key(j[ok], ix0[ok] + dx, iy0[ok] + dy, kbits))
                vals.append((m * wj * wx * wy)[ok])
    uk, inv = torch.unique(torch.cat(keys), sorted=True, return_inverse=True)
    num = torch.zeros(len(uk), dtype=DT, device=dev).index_add_(0, inv, torch.cat(vals))
    cj, cix, ciy = (uk >> (2 * kbits)), (uk >> kbits) & ((1 << kbits) - 1), uk & ((1 << kbits) - 1)
    g = _cell(cj, p)
    xc, yc = (cix.to(DT) + 0.5) * g, (ciy.to(DT) + 0.5) * g
    # ---- coverage (F2: number of levels covering the cell centre) -------------------------------------------------
    cov = []
    for lv, tt in zip(levels, tiles_t):
        if lv.whole:
            cov.append(torch.ones(len(uk), dtype=DT, device=dev))
        else:
            _, n_in = _border(xc, yc, tt, W, H, lv.overlap)
            cov.append((n_in > 0).to(DT))
    cov = torch.stack(cov, 1)  # [cells, K]
    if variant == "F2":
        C = cov.sum(1)
    else:  # F3e: sum of the (empirical) validity of the covering levels at the cell's sigma
        sig_c = SIGMA0 * 2.0 ** ((cj.to(DT) + 0.5) / p.bins_per_oct)
        C = (cov * _validity_levels(sig_c, levels, p, True)).sum(1)
    fld = num / (C + EPS)
    # ---- 3x3x3 local maxima ----------------------------------------------------------------------------------------
    is_max = fld >= p.tau_peak
    ninf = torch.tensor(-math.inf, dtype=DT, device=dev)
    for dj in (-1, 0, 1):
        jn = cj + dj
        gn = _cell(jn, p)
        bx, by = torch.floor(xc / gn).long(), torch.floor(yc / gn).long()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dj == dx == dy == 0:
                    continue
                kk = _key(torch.clamp(jn, min=0), bx + dx, by + dy, kbits)
                pos = torch.clamp(torch.searchsorted(uk, kk), 0, len(uk) - 1)
                hit = (uk[pos] == kk) & (jn >= 0)
                is_max &= fld >= torch.where(hit, fld[pos], ninf)
    pk = torch.nonzero(is_max).flatten()
    if not len(pk):
        return empty
    pk = pk[torch.argsort(-fld[pk], stable=True)][: max(p.top_k * 2, p.top_k)]
    # ---- membership (blocked): sigma bins J-1..J+1, Chebyshev distance <= 1.5 g_J; IoU cluster with the top -----
    pj = torch.round(sb - 0.5).long()
    hw, hh = torch.exp(lw) / 2, torch.exp(lh) / 2
    boxes = torch.stack([cx - hw, cy - hh, cx + hw, cy + hh], 1)
    has, top, pi, ni = _members(xc[pk], yc[pk], cj[pk], 1.5 * g[pk], cx, cy, pj, m, boxes, p.iou_cluster, block,
                                pair_budget, stats)
    # each strongest member is reported once (first peak in descending field order)
    seen: set[int] = set()
    keep_pk = []
    for i, (h_, t_) in enumerate(zip(has.tolist(), top.tolist())):
        if h_ and t_ not in seen:
            seen.add(t_)
            keep_pk.append(i)
    if not keep_pk:
        return empty
    kp = torch.tensor(keep_pk, device=dev)
    newpos = torch.full((len(pk),), -1, dtype=torch.long, device=dev)
    newpos[kp] = torch.arange(len(kp), device=dev)
    sel_pairs = newpos[pi] >= 0
    pi, ni = newpos[pi[sel_pairs]], ni[sel_pairs]
    Cp = C[pk[kp]]
    P = len(kp)
    # ---- per-pass maximum of w * p_c, summed over passes (over the occurring (peak, pass) slots) ----------------------
    up, pinv = torch.unique(passes, return_inverse=True)
    wp = w[ni, None] * probs[ni]  # [M, nc]
    us, sinv = torch.unique(pi * len(up) + pinv[ni], return_inverse=True)
    mx = torch.zeros((len(us), nc), dtype=DT, device=dev)
    mx = mx.scatter_reduce(0, sinv[:, None].expand(-1, nc), wp, reduce="amax", include_self=True)
    score = torch.zeros((P, nc), dtype=DT, device=dev).index_add_(0, us // len(up), mx) / (Cp[:, None] + EPS)
    # ---- box: (w p_max)-weighted mean of (cx, cy, log w, log h) ------------------------------------------------------
    bwm = m[ni]

    def wsum(v):
        return torch.zeros(P, dtype=DT, device=dev).index_add_(0, pi, bwm * v)

    sw = wsum(torch.ones_like(bwm))
    okp = sw > 0
    bcx, bcy = wsum(cx[ni]) / sw, wsum(cy[ni]) / sw
    bwid, bhei = torch.exp(wsum(lw[ni]) / sw), torch.exp(wsum(lh[ni]) / sw)
    sc, cl = torch.sort(score, dim=1, descending=True, stable=True)
    sc, cl = sc[:, : p.classes_per_peak], cl[:, : p.classes_per_peak]
    sel = okp[:, None] & (sc > 0)
    rows = torch.nonzero(sel, as_tuple=True)[0]
    out = torch.stack([bcx[rows] - bwid[rows] / 2, bcy[rows] - bhei[rows] / 2, bwid[rows], bhei[rows], sc[sel],
                       (cl[sel] + 1).to(DT)], 1).cpu().numpy()
    d = pd.DataFrame(out, columns=["x", "y", "w", "h", "score", "category_id"]).astype({"category_id": int})
    if stats is not None and dev.type == "cuda":
        stats["peak_mem_bytes"] = max(stats.get("peak_mem_bytes", 0), torch.cuda.max_memory_allocated(dev))
    if p.final_nms and len(d):
        from scahi.fusion.scalespace import _nms

        d = _nms(d, 0.7)
    # same final sort as the CPU reference (pandas default): detections with equal scores at the top_k cut are then
    # kept in the same order (rows are built in the same peak / class order as there)
    return d.sort_values("score", ascending=False).head(p.top_k)
