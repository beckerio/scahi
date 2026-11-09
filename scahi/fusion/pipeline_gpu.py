# SPDX-License-Identifier: AGPL-3.0-or-later
"""Optimised end-to-end path (extension E2, "fair latency"): passes and dense fusion (F2, F3e) on the GPU, nothing via the host.

Per image: decoding of the image, per level the passes (same geometry and batching as ``cache_mz.raw_predict_level``),
**only the needed head** (``forward_raw_branch``: YOLOv10 o2m; single-head models their head), DFL decoding and the
mapping to image pixels on the GPU (rounded to float32 like the cache), then ``scalespace_torch.fuse_image_f2``.
Timed in two parts (passes incl. decoding / fusion). Parity with the reference path (cache + CPU F2):
``scripts/e2_pipeline_parity.py``.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from scahi.fusion import scalespace as ss
from scahi.fusion.scalespace_torch import fuse_image_f2
from scahi.infer import cache_mz as cm
from scahi.infer.cache import SCORE_MIN
from scahi.infer.dfl_decode import decode
from scahi.infer.preprocess import letterbox, to_tensor
from scahi.infer.raw_head import forward_raw_branch


@torch.no_grad()
def gpu_levels(model: torch.nn.Module, img: np.ndarray, stack: str, overlap: float, branch: str,
               batch: int = 16, cals: dict[str, ss.LevelCal] | None = None) -> tuple[list[ss.CachedLevel], int, int]:
    H, W = img.shape[:2]
    levels = []
    for lk in cm.STACK_LEVELS[stack]:
        level = cm.LEVELS[lk]
        passes = cm.level_passes(level, W, H, overlap)
        xs, ps, ts, gains = [], [], [], []
        for s in range(0, len(passes), batch):
            chunk = passes[s:s + batch]
            for sh in sorted({shp for _, shp in chunk}):
                sub = [(k, t) for k, (t, shp) in enumerate(chunk) if shp == sh]
                ims, lbs = [], []
                for _, (x1, y1, x2, y2) in sub:
                    im, lb = letterbox(img[y1:y2, x1:x2], sh)
                    ims.append(im)
                    lbs.append(lb)
                raw = forward_raw_branch(model, to_tensor(ims, "cuda"), branch)
                d = decode(raw.boxes[branch], raw.scores[branch], raw.anchors, raw.strides, raw.reg_max)
                keep = d.probs.max(-1)[0] >= SCORE_MIN
                for i, ((k, (x1, y1, _, _)), lb) in enumerate(zip(sub, lbs)):
                    kk = keep[i]
                    px, py = lb.pad_xy
                    off = torch.tensor([px, py, px, py], dtype=torch.float64, device="cuda")
                    org = torch.tensor([x1, y1, x1, y1], dtype=torch.float64, device="cuda")
                    xs.append(((d.xyxy[i][kk].double() - off) / lb.gain + org).float())
                    ps.append(d.probs[i][kk])
                    ts.append(torch.full((int(kk.sum()),), s + k, dtype=torch.long, device="cuda"))
                gains += [(s + k, lb.gain) for (k, _), lb in zip(sub, lbs)]
        z = torch.zeros(0, device="cuda")
        levels.append(ss.CachedLevel(lk, torch.cat(xs) if xs else z.view(0, 4), torch.cat(ps) if ps else z.view(0, 10), None,
                                     torch.cat(ts) if ts else z.long(), np.array([t for t, _ in passes], np.int32).reshape(-1, 4),
                                     np.array([g for _, g in sorted(gains)]), level.n_input is None, overlap,
                                     (cals or {}).get(lk, ss.LevelCal())))
    return levels, W, H


def predict_image(model: torch.nn.Module, img: np.ndarray, stack: str, overlap: float, branch: str,
                  cfg: dict[str, Any], variant: str = "F2",
                  cals: dict[str, ss.LevelCal] | None = None) -> tuple[pd.DataFrame, float, float]:
    """F2 detections of one image plus (passes ms, fusion ms); the caller times image decoding as part of the passes."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    levels, W, H = gpu_levels(model, img, stack, overlap, branch, cals=cals)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    from scahi.fusion.run import clip_xywh

    d = clip_xywh(fuse_image_f2(levels, W, H, ss.FusionParams(**cfg), variant=variant), W, H)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return d, 1000 * (t1 - t0), 1000 * (t2 - t1)


def measure(weights: str, images: pd.DataFrame, stack: str, overlap: float, branch: str, cfg: dict[str, Any],
            n_warmup: int = 3, variant: str = "F2", cals: dict[str, ss.LevelCal] | None = None) -> dict[str, float]:
    """Optimised end-to-end latency (image decoding + passes + fusion), split into passes and fusion."""
    from scahi.infer.raw_head import load_unfused
    from scahi.machine import check_exclusive

    model = load_unfused(weights)
    paths = images.path.tolist()
    for p in paths[:n_warmup]:
        predict_image(model, cv2.imread(p), stack, overlap, branch, cfg, variant, cals)
    torch.cuda.reset_peak_memory_stats()
    tp, tf = [], []
    for n, p in enumerate(paths):
        if n % 100 == 0:
            check_exclusive()
        t0 = time.perf_counter()
        img = cv2.imread(p)
        t_read = 1000 * (time.perf_counter() - t0)
        _, a, b = predict_image(model, img, stack, overlap, branch, cfg, variant, cals)
        tp.append(t_read + a)
        tf.append(b)
    check_exclusive()
    tp, tf = np.array(tp), np.array(tf)
    tot = tp + tf
    return {"latency_ms_mean": float(tot.mean()), "latency_ms_p95": float(np.percentile(tot, 95)),
            "passes_ms_mean": float(tp.mean()), "fusion_ms_mean": float(tf.mean()), "fusion_share": float(tf.sum() / tot.sum()),
            "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20, "n_images": len(tot), "path": "optimised (GPU F2, one head)"}
