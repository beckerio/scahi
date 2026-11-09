# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase-5 driver: F0 box fusion and dense fusion F1-F5 from the Phase-4 cache, calib estimation, calib tuning.

Tuning grids (equal budget of 24 configurations per fusion variant, calib only):
* F0-NMS / F0-GREEDYNMM (SAHI classes): match metric {IOU, IOS} x threshold {0.3 .. 0.8} x conf {0.01, 0.1};
* F0-WBF (ensemble_boxes): iou_thr {0.3 .. 0.8} x skip_box_thr {0.001, 0.01, 0.05, 0.1};
* F1-F5, F3e: tau_splat {0.005, 0.02} x tau_peak {0.001, 0.01} x cell_factor {3, 5} x bins_per_oct {3, 4, 5}.
The overlap is fixed by the cache (= overlap chosen for B-MZ-native on calib, PREREGISTRATION 2).

Per-level calib estimates (from the per-pass detections of each level, deduplicated inside the level with NMS/IOS 0.8,
matched to GT after the toolkit drop rule, IoU >= 0.5, operating point 0.25): F3e quality per half-octave sigma_in bin
= F1-score of precision and recall, divided by its maximum (bins with < 30 GT / detections: no data -> analytic
window); F4 u_med = median relative DFL std of predictions with p >= 0.1; F5 temperature minimising the BCE of matched
(1) / unmatched (0) detections over a log grid of [0.25, 4].
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scahi.config import REPO_ROOT, load_class_setting
from scahi.eval.detections import DET_COLUMNS, finalize_detections
from scahi.eval.evaluate import evaluate, prepare_images
from scahi.fusion import scalespace as ss
from scahi.infer import cache_mz as cm

CACHE_ROOT = REPO_ROOT / "cache"
F0_VARIANTS = ("F0-NMS", "F0-GREEDYNMM", "F0-WBF")
ALL_VARIANTS = (*F0_VARIANTS, *ss.VARIANTS)
THRS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def grid(variant: str) -> list[dict[str, Any]]:
    if variant in ("F0-NMS", "F0-GREEDYNMM"):
        return [{"metric": m, "thr": t, "conf": c} for m in ("IOU", "IOS") for t in THRS for c in (0.01, 0.1)]
    if variant == "F0-WBF":
        return [{"thr": t, "conf": c} for t in THRS for c in (0.001, 0.01, 0.05, 0.1)]
    return [{"tau_splat": a, "tau_peak": b, "cell_factor": c, "bins_per_oct": d}
            for a, b, c, d in itertools.product((0.005, 0.02), (0.001, 0.01), (3.0, 5.0), (3, 4, 5))]


def level_dirs(weights: Path, stack: str, split: str, overlap: float) -> list[tuple[str, Path]]:
    return [(k, cm.cache_dir(CACHE_ROOT, weights, k, overlap, split)) for k in cm.STACK_LEVELS[stack]]


def load_split(weights: Path, stack: str, split: str, overlap: float, branch: str, stems: list[str],
               cals: dict[str, ss.LevelCal] | None = None) -> dict[str, tuple[list[ss.CachedLevel], int, int]]:
    dirs = level_dirs(weights, stack, split, overlap)
    return {s: ss.load_levels(dirs, s, branch, overlap, cals) for s in stems}


# ---------------------------------------------------------------------------------------------------- F0

def pass_dets(lv: ss.CachedLevel, branch: str, conf: float, W: int, H: int, max_det: int = cm.PASS_MAX_DET,
              nms_kw: dict | None = None) -> pd.DataFrame:
    """Per-pass detections, boxes clipped to the image (as the Ultralytics predictor and SAHI do). Caches with the
    detector's own per-pass detections (FCOS) use them (score >= conf, at most ``max_det`` per pass)."""
    if lv.dets is not None:
        e = lv.dets
        d = pd.DataFrame({"x1": e["xyxy"][:, 0].astype(np.float64), "y1": e["xyxy"][:, 1].astype(np.float64),
                          "x2": e["xyxy"][:, 2].astype(np.float64), "y2": e["xyxy"][:, 3].astype(np.float64),
                          "score": e["score"].astype(np.float64), "cls": e["cls"].astype(int), "tile": e["tile"].astype(int)})
        d = d[d.score >= conf].groupby("tile", sort=False).head(max_det)  # stored in descending score per pass
    else:
        z = {f"{branch}_xyxy": lv.xyxy, f"{branch}_probs": lv.probs, f"{branch}_tile": lv.tile}
        d = cm.pass_detections(z, branch, conf, max_det=max_det, nms_kw=nms_kw)
    d[["x1", "x2"]] = d[["x1", "x2"]].clip(0, W)
    d[["y1", "y2"]] = d[["y1", "y2"]].clip(0, H)
    return d[(d.x2 > d.x1) & (d.y2 > d.y1)]


def clip_xywh(d: pd.DataFrame, W: int, H: int) -> pd.DataFrame:
    """Clip final x, y, w, h detections to the image (all fusion variants, as predictor/SAHI outputs)."""
    if d.empty:
        return d
    x1, y1 = d.x.clip(0, W), d.y.clip(0, H)
    x2, y2 = (d.x + d.w).clip(0, W), (d.y + d.h).clip(0, H)
    d = d.assign(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
    return d[(d.w > 0) & (d.h > 0)]


def f0_image(levels: list[ss.CachedLevel], W: int, H: int, branch: str, variant: str, cfg: dict[str, Any],
             raw: pd.DataFrame | None = None, max_det: int = cm.PASS_MAX_DET, nms_kw: dict | None = None) -> pd.DataFrame:
    """Box fusion of all passes of one image (``raw`` = concatenated per-pass detections at conf 0.001, optional)."""
    if raw is None:
        raw = pd.concat([pass_dets(lv, branch, 0.001, W, H, max_det, nms_kw) for lv in levels], ignore_index=True)
    r = raw[raw.score >= cfg["conf"]]
    if r.empty:
        return pd.DataFrame(columns=["x", "y", "w", "h", "score", "category_id"])
    a = r[["x1", "y1", "x2", "y2", "score", "cls"]].to_numpy(np.float64)
    if variant == "F0-NMS":
        from sahi.postprocess.combine import batched_nms

        keep = batched_nms(a, match_metric=cfg["metric"], match_threshold=cfg["thr"]) if len(a) > 1 else [0]
        k = a[np.asarray(keep, int)]
    elif variant == "F0-GREEDYNMM":
        k = _greedynmm(a, cfg["metric"], cfg["thr"], W, H)
    else:
        from ensemble_boxes import weighted_boxes_fusion

        b = np.clip(a[:, :4] / [W, H, W, H], 0, 1)
        fb, fs, fl = weighted_boxes_fusion([b.tolist()], [a[:, 4].tolist()], [a[:, 5].tolist()], iou_thr=cfg["thr"],
                                           skip_box_thr=cfg["conf"])
        k = np.column_stack([np.asarray(fb).reshape(-1, 4) * [W, H, W, H], fs, fl]) if len(fs) else np.zeros((0, 6))
    return pd.DataFrame({"x": k[:, 0], "y": k[:, 1], "w": k[:, 2] - k[:, 0], "h": k[:, 3] - k[:, 1], "score": k[:, 4],
                         "category_id": k[:, 5].astype(int) + 1})


def _greedynmm(a: np.ndarray, metric: str, thr: float, W: int, H: int) -> np.ndarray:
    """SAHI's GreedyNMMPostprocess (class-aware) on ObjectPredictions built from the array."""
    from sahi.postprocess.combine import GreedyNMMPostprocess
    from sahi.prediction import ObjectPrediction

    preds = [ObjectPrediction(bbox=[float(x1), float(y1), float(x2), float(y2)], category_id=int(c), score=float(s),
                              full_shape=[H, W]) for x1, y1, x2, y2, s, c in a]
    out = GreedyNMMPostprocess(match_threshold=thr, match_metric=metric, class_agnostic=False)(preds) if len(preds) > 1 else preds
    return np.array([[*p.bbox.to_xyxy(), p.score.value, p.category.id] for p in out], np.float64).reshape(-1, 6)


# --------------------------------------------------------------------------------------------- calibrate

def _match(coco: dict, dets: pd.DataFrame, cs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedy per image and class (IoU >= 0.5) after the toolkit drop rule: detections (score, sigma, tp, excl) and
    non-ignore GT (sigma, best matched score)."""
    from scahi.eval.matching import match_image

    D, G = [], []
    for im in prepare_images(coco, dets, cs, "visdrone_ignore"):
        d, g = match_image(im)
        d["image_id"] = g["image_id"] = im["image_id"]
        D.append(d)
        G.append(g)
    return pd.concat(D, ignore_index=True), pd.concat(G, ignore_index=True)


def calibrate(data: dict[str, tuple[list[ss.CachedLevel], int, int]], stem_to_id: dict[str, int], coco: dict,
              branch: str, level_keys: tuple[str, ...]) -> dict[str, ss.LevelCal]:
    cs = load_class_setting("visdrone10")
    cals = {}
    for k, key in enumerate(level_keys):
        rows, us, gains = [], [], {}
        for stem, (levels, W, H) in data.items():
            lv = levels[k]
            gains[stem_to_id[stem]] = float(np.median(lv.gains))
            raw = pass_dets(lv, branch, 0.01, W, H)
            if len(raw):
                from scahi.infer.multizoom import fuse

                raw = raw.assign(image_id=stem_to_id[stem], level=k)
                d = fuse(raw, 0.8, 0.01, "IOS")
                rows.append(d)
            pm = lv.probs.max(1) >= 0.1
            if pm.any():
                b = lv.xyxy[pm]
                sig = np.sqrt(np.maximum(b[:, 2] - b[:, 0], 1e-3) * np.maximum(b[:, 3] - b[:, 1], 1e-3))
                us.append(lv.std[pm].mean(1) / np.maximum(sig, 1.0))
        dets = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=DET_COLUMNS)
        D, G = _match(coco, dets, cs)
        D = D[~D.excl].copy()
        D["sigma_in"] = D.sigma * D.image_id.map(gains)
        G = G.assign(sigma_in=G.sigma * G.image_id.map(gains))
        # temperature (F5)
        s = np.clip(D.score.to_numpy(), 1e-6, 1 - 1e-6)
        y = D.tp.to_numpy(float)
        lg = np.log(s) - np.log(1 - s)
        Ts = np.exp(np.linspace(math.log(0.25), math.log(4), 41))
        bce = [-(y * np.log(np.clip(_sig(lg / T), 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - _sig(lg / T), 1e-9, 1))).mean()
               for T in Ts]
        T = float(Ts[int(np.argmin(bce))]) if len(y) else 1.0
        # empirical validity (F3e): F1-score per half-octave sigma_in bin, normalised by its maximum
        edges = 2.0 ** np.arange(1, 10.01, 0.5)
        q = np.full(len(edges) - 1, np.nan)
        op = D[D.score >= 0.25]
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            gs = G[(G.sigma_in >= lo) & (G.sigma_in < hi)]
            ds = op[(op.sigma_in >= lo) & (op.sigma_in < hi)]
            if len(gs) >= 30 and len(ds) >= 30:
                r, pr = float((gs.score >= 0.25).mean()), float(ds.tp.mean())
                q[i] = 2 * r * pr / (r + pr) if r + pr > 0 else 0.0
        if np.isfinite(q).any():
            q = q / np.nanmax(q)
        cals[key] = ss.LevelCal(u_med=float(np.median(np.concatenate(us))) if us else 1.0, temperature=T,
                                emp_edges=edges, emp_quality=q)
    return cals


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def cal_to_json(cals: dict[str, ss.LevelCal]) -> dict[str, Any]:
    return {k: {"u_med": c.u_med, "temperature": c.temperature,
                "emp_edges": None if c.emp_edges is None else c.emp_edges.tolist(),
                "emp_quality": None if c.emp_quality is None else [None if not np.isfinite(v) else float(v) for v in c.emp_quality]}
            for k, c in cals.items()}


def cal_from_json(d: dict[str, Any]) -> dict[str, ss.LevelCal]:
    return {k: ss.LevelCal(v["u_med"], v["temperature"], None if v["emp_edges"] is None else np.array(v["emp_edges"]),
                           None if v["emp_quality"] is None else np.array([np.nan if x is None else x for x in v["emp_quality"]]))
            for k, v in d.items()}


# ------------------------------------------------------------------------------------------- run / tune

def fuse_split(data: dict[str, tuple[list[ss.CachedLevel], int, int]], stem_to_id: dict[str, int], branch: str,
               variant: str, cfg: dict[str, Any], raw_cache: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    out = []
    for stem, (levels, W, H) in data.items():
        if variant in F0_VARIANTS:
            d = f0_image(levels, W, H, branch, variant, cfg, None if raw_cache is None else raw_cache[stem])
        else:
            d = ss.fuse_image(levels, W, H, variant, ss.FusionParams(**cfg))
        d = clip_xywh(d, W, H)
        d["image_id"] = stem_to_id[stem]
        out.append(d)
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=DET_COLUMNS)
    return finalize_detections(df[DET_COLUMNS].astype({"image_id": int, "category_id": int}))


def tune(data, stem_to_id, coco, branch: str, variant: str) -> list[dict[str, Any]]:
    cs = load_class_setting("visdrone10")
    raw_cache = None
    if variant in F0_VARIANTS:
        raw_cache = {s: pd.concat([pass_dets(lv, branch, 0.001, W, H) for lv in lvls], ignore_index=True)
                     for s, (lvls, W, H) in data.items()}
    res = []
    for cfg in grid(variant):
        d = fuse_split(data, stem_to_id, branch, variant, cfg, raw_cache)
        r = evaluate(coco, d, cs, "visdrone_ignore", port_only=True)["metrics"]
        res.append({**cfg, "AP": r["AP"], "AP50": r["AP50"], "source": "toolkit_port"})
    return res


def input_megapixels(stack: str, sizes: list[tuple[int, int]], overlap: float) -> float:
    """Mean network-input megapixels per image (incl. padding) of a stack."""
    tot = []
    for W, H in sizes:
        mp = 0.0
        for k in cm.STACK_LEVELS[stack]:
            for _, (h, w) in cm.level_passes(cm.LEVELS[k], W, H, overlap):
                mp += h * w / 1e6
        tot.append(mp)
    return float(np.mean(tot))
