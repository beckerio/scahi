# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end multi-zoom inference with fusion (no cache) for the Phase-5 latency job.

Per image: image decoding, the forward passes of all levels of the stack (unfused model, ``raw_predict_level``:
letterbox, forward, DFL decoding, anchors >= 0.001 to CPU), then the fusion (F0 box fusion or dense F1-F5).
Note: the unfused model computes both YOLOv10 heads; a deployed single-branch model would be slightly faster. The
box-fusion references (B-FULL, B-SAHI, B-MZ) are timed with the Ultralytics predictor (fused, one branch).
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from scahi.fusion import run as R
from scahi.fusion import scalespace as ss
from scahi.infer import cache_mz as cm


def predict_image(model: torch.nn.Module, img: np.ndarray, stack: str, overlap: float, branch: str, variant: str,
                  cfg: dict[str, Any], cals: dict[str, ss.LevelCal]) -> pd.DataFrame:
    H, W = img.shape[:2]
    levels = []
    for k in cm.STACK_LEVELS[stack]:
        z = cm.raw_predict_level(model, img, cm.LEVELS[k], overlap)
        levels.append(ss.CachedLevel(k, z[f"{branch}_xyxy"], z[f"{branch}_probs"], z[f"{branch}_std"], z[f"{branch}_tile"],
                                     z["tiles"], z["gains"], cm.LEVELS[k].n_input is None, overlap, cals.get(k, ss.LevelCal())))
    if variant in R.F0_VARIANTS:
        return R.f0_image(levels, W, H, branch, variant, cfg)
    return R.clip_xywh(ss.fuse_image(levels, W, H, variant, ss.FusionParams(**cfg)), W, H)


def measure(weights: str, images: pd.DataFrame, stack: str, overlap: float, branch: str, variant: str,
            cfg: dict[str, Any], cals: dict[str, ss.LevelCal], n_warmup: int = 3) -> dict[str, float]:
    from scahi.infer.raw_head import load_unfused
    from scahi.machine import check_exclusive

    model = load_unfused(weights)
    paths = images.path.tolist()
    for p in paths[:n_warmup]:
        predict_image(model, cv2.imread(p), stack, overlap, branch, variant, cfg, cals)
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for n, p in enumerate(paths):
        if n % 100 == 0:
            check_exclusive()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        predict_image(model, cv2.imread(p), stack, overlap, branch, variant, cfg, cals)
        torch.cuda.synchronize()
        lat.append(1000 * (time.perf_counter() - t0))
    check_exclusive()
    return {"latency_ms_mean": float(np.mean(lat)), "latency_ms_p95": float(np.percentile(lat, 95)),
            "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20, "n_images": len(lat)}


def measure_split(weights: str, images: pd.DataFrame, stack: str, overlap: float, branch: str, variant: str,
                  cfg: dict[str, Any], cals: dict[str, ss.LevelCal], n_warmup: int = 3) -> dict[str, float]:
    """Like :func:`measure`, but each image is timed in two parts: (1) decoding + all forward passes with DFL decoding
    and transfer to the CPU (``raw_predict_level``, GPU part), (2) the fusion on the CPU (F0 or dense). Used only to
    state the fusion share of the end-to-end latency (2026-10-24); the reported latencies remain those of ``measure``."""
    from scahi.infer.raw_head import load_unfused
    from scahi.machine import check_exclusive

    model = load_unfused(weights)
    paths = images.path.tolist()

    def one(p: str) -> tuple[float, float]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = cv2.imread(p)
        H, W = img.shape[:2]
        levels = []
        for k in cm.STACK_LEVELS[stack]:
            z = cm.raw_predict_level(model, img, cm.LEVELS[k], overlap)
            levels.append(ss.CachedLevel(k, z[f"{branch}_xyxy"], z[f"{branch}_probs"], z[f"{branch}_std"],
                                         z[f"{branch}_tile"], z["tiles"], z["gains"], cm.LEVELS[k].n_input is None,
                                         overlap, cals.get(k, ss.LevelCal())))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if variant in R.F0_VARIANTS:
            R.f0_image(levels, W, H, branch, variant, cfg)
        else:
            R.clip_xywh(ss.fuse_image(levels, W, H, variant, ss.FusionParams(**cfg)), W, H)
        t2 = time.perf_counter()
        return 1000 * (t1 - t0), 1000 * (t2 - t1)

    for p in paths[:n_warmup]:
        one(p)
    ts = []
    for n, p in enumerate(paths):
        if n % 100 == 0:
            check_exclusive()
        ts.append(one(p))
    check_exclusive()
    a = np.array(ts)
    return {"passes_ms_mean": float(a[:, 0].mean()), "fusion_ms_mean": float(a[:, 1].mean()),
            "total_ms_mean": float(a.sum(1).mean()), "fusion_share": float(a[:, 1].sum() / a.sum()),
            "fusion_ms_p95": float(np.percentile(a[:, 1], 95)), "n_images": len(ts)}
