# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fraunhofer IOSB
#
# SCAHI - Slicing with Coverage-Aware Hyper Inference.
# If you use SCAHI, please cite:
#   S. Becker, J. Bayer, D. Münch, W. Hübner. When Does Sliced Inference Pay Off? Resolution, Tiling and Dense Fusion
#   for Small Object Detection. arXiv:2611.12657, 2026.
"""Phase-5 fusion in the global scale space S(x, y, log2 sigma) (AGENT_BRIEF Section 8, AUTORUN_BRIEF Section 5).

Input: the raw cache of one image (``scahi.infer.cache_mz``), several levels (= one stack), one branch.

* **Predictions:** every cached anchor with max class probability >= ``tau_splat``: centre (cx, cy), size
  sigma = sqrt(w h) (global px), class probabilities p, DFL std, pass = (level, tile).
* **Windows** (normalised so that they form a partition of unity where covered):
  - border window b_t(x, y): per tile a sin^2 ramp over the nominal overlap width at interior tile edges (no taper at
    image borders), divided by the sum over the level's tiles; whole-image passes: 1;
  - validity window v_k(sigma) in sigma_in = sigma * gain_k (network-input px of the pass): raised cosine in log2 from
    0 at ``a_lo`` = 6 via 0.5 at ``a`` = 8 to 1 at ``a_hi`` = 14, flat up to ``b`` = 192, raised cosine down to 0 at
    ``b_hi`` = 256; divided by max(sum over levels, 1).
* **Weights per variant:** F1 w = 1; F2 w = b; F3 w = b v; F3e w = b v_e (empirical validity per level from calib);
  F4 = F3 * f(u), f(u) = exp(-u / u_med_k), u = mean DFL std / sigma; F5 = F4 with per-level temperature on the logits.
* **Coverage** (analytic, from geometry only): F1 C = number of passes covering (x, y); F2 C = number of levels
  covering; F3/F4/F5 C = sum_k v_k(sigma) over levels covering (x, y) (F3e: v_e). F2n (component ablation, exploratory,
  2026-10-30): weights of F2, no coverage normalisation (C = 1: field and score are plain sums over passes).
* **Volume:** sparse pyramid of sigma bins (``bins_per_oct`` per octave from 2 px) with spatial cells of
  g_j = max(1, sigma_j / ``cell_factor``) px; w * max_c p splatted trilinearly; field = numerator / (C + eps).
* **Peaks:** 3x3x3 local maxima in (x, y, sigma-bin) of the field above ``tau_peak`` (neighbouring sigma bins are
  compared at the same position); top-K <= 500.
* **Score and box of a peak:** predictions whose nearest cell lies in the peak's 3x3x3 neighbourhood and whose box
  overlaps the neighbourhood's strongest prediction (IoU >= ``iou_cluster``; peaks with the same strongest prediction
  are reported once); per pass the
  maximum of w * p_c, summed over passes, divided by C at the peak (= window-weighted mean over the covering passes:
  a pass that covers the location but predicts nothing counts as negative evidence); box = (w * p_max)-weighted mean of
  (cx, cy, log w, log h). Up to 3 classes per peak (descending score). Optional final class-wise NMS (IoU 0.7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial import cKDTree

EPS = 1e-6
SIGMA0 = 2.0  # lower end of the sigma axis (px)
VARIANTS = ("F1", "F2", "F3", "F3e", "F4", "F5", "F2n")


@dataclass(frozen=True)
class FusionParams:
    tau_splat: float = 0.01
    tau_peak: float = 0.001
    cell_factor: float = 4.0
    bins_per_oct: int = 4
    a_lo: float = 6.0
    a: float = 8.0
    a_hi: float = 14.0
    b: float = 192.0
    b_hi: float = 256.0
    top_k: int = 500
    classes_per_peak: int = 3
    iou_cluster: float = 0.5
    final_nms: bool = False


@dataclass
class LevelCal:
    """Calib-estimated per-level quantities (F3e, F4, F5)."""

    u_med: float = 1.0
    temperature: float = 1.0
    emp_edges: NDArray | None = None  # sigma_in bin edges (px)
    emp_quality: NDArray | None = None  # quality per bin in [0, 1] (NaN = no data)


@dataclass
class CachedLevel:
    key: str
    xyxy: NDArray
    probs: NDArray
    std: NDArray
    tile: NDArray
    tiles: NDArray  # [T, 4]
    gains: NDArray  # [T]
    whole: bool
    overlap: float
    cal: LevelCal = field(default_factory=LevelCal)
    # the detector's own per-pass detections (FCOS caches, preregistration addendum 3): keys xyxy, score, cls, tile;
    # None for the YOLO caches (their per-pass detections are computed from the dense outputs)
    dets: dict[str, NDArray] | None = None


# ------------------------------------------------------------------------------------------------ windows

def _rc(t: NDArray) -> NDArray:
    """Raised cosine 0 -> 1 on t in [0, 1]."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def validity_raw(sigma_in: NDArray, p: FusionParams) -> NDArray:
    s = np.log2(np.maximum(sigma_in, 1e-6))
    la_lo, la, la_hi, lb, lb_hi = (math.log2(x) for x in (p.a_lo, p.a, p.a_hi, p.b, p.b_hi))
    up = np.where(s < la, 0.5 * (s - la_lo) / (la - la_lo), 0.5 + 0.5 * (s - la) / (la_hi - la))  # 0 at a_lo, .5 at a, 1 at a_hi
    v = _rc(up)
    down = 1.0 - _rc((s - lb) / (lb_hi - lb))
    return np.where(s >= lb, down, v)


def border_raw(x: NDArray, y: NDArray, tiles: NDArray, W: int, H: int, overlap: float) -> NDArray:
    """[n, T] raw sin^2 border ramps of points (x, y) for every tile of a level."""
    x1, y1, x2, y2 = (tiles[:, i][None].astype(float) for i in range(4))
    dx, dy = overlap * (x2 - x1), overlap * (y2 - y1)
    X, Y = x[:, None], y[:, None]
    inside = (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)
    ramp = lambda d, w: np.sin(0.5 * np.pi * np.clip(d / np.maximum(w, 1e-6), 0, 1)) ** 2  # noqa: E731
    wx = np.where(x1 > 0, ramp(X - x1, dx), 1.0) * np.where(x2 < W, ramp(x2 - X, dx), 1.0)
    wy = np.where(y1 > 0, ramp(Y - y1, dy), 1.0) * np.where(y2 < H, ramp(y2 - Y, dy), 1.0)
    return np.where(inside, wx * wy, 0.0)


def border_norm(x: NDArray, y: NDArray, lv: CachedLevel, W: int, H: int) -> tuple[NDArray, NDArray]:
    """Normalised border window per (point, tile) [n, T] and the number of tiles containing each point [n]."""
    if lv.whole:
        n = len(x)
        return np.ones((n, 1)), np.ones(n)
    r = border_raw(x, y, lv.tiles, W, H, lv.overlap)
    s = r.sum(1, keepdims=True)
    inside = ((x[:, None] >= lv.tiles[None, :, 0]) & (x[:, None] <= lv.tiles[None, :, 2])
              & (y[:, None] >= lv.tiles[None, :, 1]) & (y[:, None] <= lv.tiles[None, :, 3]))
    return np.where(s > 0, r / np.maximum(s, 1e-12), 0.0), inside.sum(1).astype(float)


def level_gain(lv: CachedLevel, x: NDArray, y: NDArray) -> NDArray:
    """Gain (input px per original px) of the level at points; tiles of a level share the gain up to rounding."""
    return np.full(len(x), float(np.median(lv.gains)))


def validity_levels(sigma: NDArray, x: NDArray, y: NDArray, levels: list[CachedLevel], p: FusionParams,
                    empirical: bool) -> NDArray:
    """[n, K] normalised validity windows of all levels at (x, y, sigma)."""
    raw = np.stack([_validity_one(sigma * level_gain(lv, x, y), lv, p, empirical) for lv in levels], 1)
    return raw / np.maximum(raw.sum(1, keepdims=True), 1.0)


def _validity_one(sigma_in: NDArray, lv: CachedLevel, p: FusionParams, empirical: bool) -> NDArray:
    v = validity_raw(sigma_in, p)
    if empirical and lv.cal.emp_quality is not None:
        e = lv.cal.emp_edges
        idx = np.clip(np.digitize(sigma_in, e) - 1, 0, len(e) - 2)
        q = lv.cal.emp_quality[idx]
        v = np.where(np.isfinite(q) & (sigma_in >= e[0]) & (sigma_in < e[-1]), q, v)
    return v


# ---------------------------------------------------------------------------------------------- weights

def prediction_table(levels: list[CachedLevel], W: int, H: int, variant: str, p: FusionParams) -> pd.DataFrame:
    """All predictions above ``tau_splat`` with their weight w, pass id and probabilities."""
    rows = []
    for k, lv in enumerate(levels):
        pr = lv.probs
        if variant == "F5" and lv.cal.temperature != 1.0:
            lg = np.log(np.clip(pr, 1e-7, 1 - 1e-7)) - np.log(np.clip(1 - pr, 1e-7, 1))
            pr = 1.0 / (1.0 + np.exp(-lg / lv.cal.temperature))
        keep = pr.max(1) >= p.tau_splat
        if not keep.any():
            continue
        b = lv.xyxy[keep].astype(float)
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        w_, h_ = np.maximum(b[:, 2] - b[:, 0], 1e-3), np.maximum(b[:, 3] - b[:, 1], 1e-3)
        sig = np.sqrt(w_ * h_)
        df = pd.DataFrame({"cx": cx, "cy": cy, "lw": np.log(w_), "lh": np.log(h_), "sigma": sig,
                           "pass": k * 10000 + lv.tile[keep].astype(int), "level": k})
        df["probs"] = list(pr[keep])
        if variant == "F1":
            w = np.ones(len(df))
        else:
            bn, _ = border_norm(cx, cy, lv, W, H)
            w = bn[np.arange(len(df)), 0 if lv.whole else lv.tile[keep].astype(int)]
            if variant in ("F3", "F3e", "F4", "F5"):
                w = w * validity_levels(sig, cx, cy, levels, p, variant == "F3e")[:, k]
            if variant in ("F4", "F5"):
                u = lv.std[keep].astype(float).mean(1) / np.maximum(sig, 1.0)
                w = w * np.exp(-u / max(lv.cal.u_med, 1e-6))
        df["w"] = w
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def coverage(x: NDArray, y: NDArray, sigma: NDArray, levels: list[CachedLevel], W: int, H: int, variant: str,
             p: FusionParams) -> NDArray:
    """Analytic coverage C at points (see module docstring)."""
    if variant == "F1":
        return sum(border_norm(x, y, lv, W, H)[1] for lv in levels)
    if variant == "F2n":
        return np.ones(len(x)) - EPS  # C + EPS = 1 exactly
    covered = np.stack([border_norm(x, y, lv, W, H)[1] > 0 for lv in levels], 1).astype(float)  # [n, K]
    if variant == "F2":
        return covered.sum(1)
    v = validity_levels(sigma, x, y, levels, p, variant == "F3e")
    return (covered * v).sum(1)


# -------------------------------------------------------------------------------------- volume and peaks

def _bins(sigma: NDArray, p: FusionParams) -> NDArray:
    return (np.log2(np.maximum(sigma, SIGMA0)) - math.log2(SIGMA0)) * p.bins_per_oct


def _cell(j: NDArray, p: FusionParams) -> NDArray:
    sig_j = SIGMA0 * 2.0 ** ((j + 0.5) / p.bins_per_oct)
    return np.maximum(1.0, sig_j / p.cell_factor)


def _key(j: NDArray, ix: NDArray, iy: NDArray, nb: int = 13) -> NDArray:
    """Pack (sigma bin, cell x, cell y) into int64 with ``nb`` bits per spatial index (indices clipped to 2^nb - 1)."""
    m = (1 << nb) - 1
    return (j.astype(np.int64) << (2 * nb)) | (np.clip(ix, 0, m).astype(np.int64) << nb) | np.clip(iy, 0, m).astype(np.int64)


def _unkey(uk: NDArray, nb: int) -> tuple[NDArray, NDArray, NDArray]:
    m = (1 << nb) - 1
    return (uk >> (2 * nb)).astype(int), ((uk >> nb) & m).astype(int), (uk & m).astype(int)


def key_bits(W: int, H: int) -> int:
    """13 bits per spatial index (VisDrone, unchanged) unless the image needs more cells (>= 1 px each): 15 bits
    (DOTA, long side up to 13 383 px; Phase D 2026-10-30)."""
    return 13 if max(W, H) + 2 <= (1 << 13) - 1 else 15


def fuse_image(levels: list[CachedLevel], W: int, H: int, variant: str, p: FusionParams, nc: int = 10,
               box_source: str = "mean") -> pd.DataFrame:
    """Detections (x, y, w, h, score, category_id 1-based) of one image. ``box_source`` "top" = box of the strongest
    cluster member instead of the weighted mean (exploratory ablation, 2026-10-27; scores and peaks unchanged)."""
    P = prediction_table(levels, W, H, variant, p)
    if P.empty:
        return pd.DataFrame(columns=["x", "y", "w", "h", "score", "category_id"])
    probs = np.stack(P["probs"].to_numpy())
    pmax = probs.max(1)
    m = (P["w"].to_numpy() * pmax)
    kbits = key_bits(W, H)
    # trilinear splat in (sigma-bin, x / g_j, y / g_j) with the cell size of the lower sigma bin
    sb = _bins(P["sigma"].to_numpy(), p)
    j0 = np.floor(sb - 0.5).astype(int)
    keys, vals = [], []
    for dj in (0, 1):
        j = j0 + dj
        wj = np.where(dj == 0, 1 - (sb - 0.5 - j0), sb - 0.5 - j0)
        g = _cell(j, p)
        fx, fy = P["cx"].to_numpy() / g - 0.5, P["cy"].to_numpy() / g - 0.5
        ix0, iy0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        for dx in (0, 1):
            wx = np.where(dx == 0, 1 - (fx - ix0), fx - ix0)
            for dy in (0, 1):
                wy = np.where(dy == 0, 1 - (fy - iy0), fy - iy0)
                ok = (j >= 0)
                keys.append(_key(j[ok], ix0[ok] + dx, iy0[ok] + dy, kbits))
                vals.append((m * wj * wx * wy)[ok])
    keys, vals = np.concatenate(keys), np.concatenate(vals)
    uk, inv = np.unique(keys, return_inverse=True)
    num = np.bincount(inv, vals)
    cj, cix, ciy = _unkey(uk, kbits)
    g = _cell(cj, p)
    xc, yc = (cix + 0.5) * g, (ciy + 0.5) * g
    sig_c = SIGMA0 * 2.0 ** ((cj + 0.5) / p.bins_per_oct)
    C = coverage(xc, yc, sig_c, levels, W, H, variant, p)
    fld = num / (C + EPS)
    # 3x3x3 local maxima (sparse lookup; neighbouring sigma bins at the same position)
    order = np.argsort(uk)
    is_max = fld >= p.tau_peak
    for dj in (-1, 0, 1):
        jn = cj + dj
        gn = _cell(jn, p)
        bx, by = np.floor(xc / gn).astype(int), np.floor(yc / gn).astype(int)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dj == dx == dy == 0:
                    continue
                kk = _key(np.maximum(jn, 0), bx + dx, by + dy, kbits)
                pos = np.searchsorted(uk, kk)
                pos = np.clip(pos, 0, len(uk) - 1)
                hit = (uk[pos] == kk) & (jn >= 0)
                nb = np.where(hit, fld[pos], -np.inf)
                is_max &= fld >= nb
    del order
    pk = np.flatnonzero(is_max)
    if not len(pk):
        return pd.DataFrame(columns=["x", "y", "w", "h", "score", "category_id"])
    pk = pk[np.argsort(-fld[pk])][: max(p.top_k * 2, p.top_k)]
    # neighbourhood membership: predictions in sigma bins J-1..J+1 within 1.5 cells (Chebyshev) of the peak
    pj = np.round(sb - 0.5).astype(int)
    det = []
    xy = P[["cx", "cy"]].to_numpy()
    by_bin = {j: np.flatnonzero(pj == j) for j in np.unique(pj)}
    trees = {j: cKDTree(xy[idx]) for j, idx in by_bin.items()}
    C_pk = C[pk]
    w = P["w"].to_numpy()
    passes = P["pass"].to_numpy()
    hw, hh = np.exp(P["lw"].to_numpy()) / 2, np.exp(P["lh"].to_numpy()) / 2
    boxes = np.stack([P["cx"].to_numpy() - hw, P["cy"].to_numpy() - hh, P["cx"].to_numpy() + hw, P["cy"].to_numpy() + hh], 1)
    seen_top: set[int] = set()
    for n, c in enumerate(pk):
        J, gJ = cj[c], g[c]
        members = []
        for jj in (J - 1, J, J + 1):
            if jj in trees:
                hit = trees[jj].query_ball_point([xc[c], yc[c]], r=1.5 * gJ, p=np.inf)
                members.append(by_bin[jj][hit])
        idx = np.concatenate(members) if members else np.zeros(0, int)
        if not len(idx):
            continue
        # cluster: members overlapping the strongest member (IoU >= iou_cluster) - keeps neighbouring objects apart
        top = idx[np.argmax(w[idx] * pmax[idx])]
        if top in seen_top:
            continue
        seen_top.add(top)
        idx = idx[_iou_one(boxes[top], boxes[idx]) >= p.iou_cluster]
        wp = w[idx, None] * probs[idx]  # [m, nc]
        # per pass max, summed over passes
        ps = passes[idx]
        up, pinv = np.unique(ps, return_inverse=True)
        mx = np.zeros((len(up), nc))
        np.maximum.at(mx, pinv, wp)
        score = mx.sum(0) / (C_pk[n] + EPS)
        bw = w[idx] * pmax[idx]
        sw = bw.sum()
        if sw <= 0:
            continue
        cx = (bw * P["cx"].to_numpy()[idx]).sum() / sw
        cy = (bw * P["cy"].to_numpy()[idx]).sum() / sw
        bwid = math.exp((bw * P["lw"].to_numpy()[idx]).sum() / sw)
        bhei = math.exp((bw * P["lh"].to_numpy()[idx]).sum() / sw)
        if box_source == "top":
            cx, cy = P["cx"].to_numpy()[top], P["cy"].to_numpy()[top]
            bwid, bhei = math.exp(P["lw"].to_numpy()[top]), math.exp(P["lh"].to_numpy()[top])
        for cl in np.argsort(-score)[: p.classes_per_peak]:
            if score[cl] > 0:
                det.append((cx - bwid / 2, cy - bhei / 2, bwid, bhei, float(score[cl]), int(cl) + 1))
    out = pd.DataFrame(det, columns=["x", "y", "w", "h", "score", "category_id"])
    if p.final_nms and len(out):
        out = _nms(out, 0.7)
    return out.sort_values("score", ascending=False).head(p.top_k)


def _iou_one(b: NDArray, bs: NDArray) -> NDArray:
    ix = np.clip(np.minimum(b[2], bs[:, 2]) - np.maximum(b[0], bs[:, 0]), 0, None)
    iy = np.clip(np.minimum(b[3], bs[:, 3]) - np.maximum(b[1], bs[:, 1]), 0, None)
    inter = ix * iy
    ua = (b[2] - b[0]) * (b[3] - b[1]) + (bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def volume(levels: list[CachedLevel], W: int, H: int, variant: str, p: FusionParams) -> dict[str, NDArray]:
    """Sparse scale-space field S = numerator / (C + eps) of one image (visualisation; same splatting and coverage as
    ``fuse_image``): cell sigma-bin ``j``, cell centre ``x``, ``y`` (px), cell size ``g`` and field value ``S``."""
    P = prediction_table(levels, W, H, variant, p)
    if P.empty:
        return {k: np.zeros(0) for k in ("j", "x", "y", "g", "S", "sigma")}
    pmax = np.stack(P["probs"].to_numpy()).max(1)
    m = P["w"].to_numpy() * pmax
    kbits = key_bits(W, H)
    sb = _bins(P["sigma"].to_numpy(), p)
    j0 = np.floor(sb - 0.5).astype(int)
    keys, vals = [], []
    for dj in (0, 1):
        j = j0 + dj
        wj = np.where(dj == 0, 1 - (sb - 0.5 - j0), sb - 0.5 - j0)
        g = _cell(j, p)
        fx, fy = P["cx"].to_numpy() / g - 0.5, P["cy"].to_numpy() / g - 0.5
        ix0, iy0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        for dx in (0, 1):
            wx = np.where(dx == 0, 1 - (fx - ix0), fx - ix0)
            for dy in (0, 1):
                wy = np.where(dy == 0, 1 - (fy - iy0), fy - iy0)
                ok = j >= 0
                keys.append(_key(j[ok], ix0[ok] + dx, iy0[ok] + dy, kbits))
                vals.append((m * wj * wx * wy)[ok])
    uk, inv = np.unique(np.concatenate(keys), return_inverse=True)
    num = np.bincount(inv, np.concatenate(vals))
    cj, cix, ciy = _unkey(uk, kbits)
    g = _cell(cj, p)
    xc, yc = (cix + 0.5) * g, (ciy + 0.5) * g
    sig = SIGMA0 * 2.0 ** ((cj + 0.5) / p.bins_per_oct)
    C = coverage(xc, yc, sig, levels, W, H, variant, p)
    return {"j": cj, "x": xc, "y": yc, "g": g, "S": num / (C + EPS), "sigma": sig}


def _nms(d: pd.DataFrame, iou: float) -> pd.DataFrame:
    import torch
    from torchvision.ops import batched_nms

    b = torch.tensor(d[["x", "y", "w", "h"]].to_numpy(), dtype=torch.float64)
    b[:, 2:] += b[:, :2]
    keep = batched_nms(b, torch.tensor(d.score.to_numpy()), torch.tensor(d.category_id.to_numpy()), iou).numpy()
    return d.iloc[keep]


def load_levels(cache_dirs: list[tuple[str, Any]], stem: str, branch: str, overlap: float,
                cals: dict[str, LevelCal] | None = None, dets: bool = True) -> tuple[list[CachedLevel], int, int]:
    """Read the cached levels of one image: ``cache_dirs`` = [(level_key, directory)]; ``dets=False`` skips the
    per-pass detections (only box fusion reads them)."""
    levels, W, H = [], 0, 0
    for key, d in cache_dirs:
        z = np.load(f"{d}/{stem}.npz")
        W, H = (int(v) for v in z["image_wh"])
        pd_ = ({k: z[f"{branch}_det_{k}"] for k in ("xyxy", "score", "cls", "tile")}
               if dets and f"{branch}_det_xyxy" in z.files else None)
        levels.append(CachedLevel(key, z[f"{branch}_xyxy"], z[f"{branch}_probs"], z[f"{branch}_std"], z[f"{branch}_tile"],
                                  z["tiles"], z["gains"], bool(int(z["n_input"]) == 0), overlap,
                                  (cals or {}).get(key, LevelCal()), pd_))
    return levels, W, H
