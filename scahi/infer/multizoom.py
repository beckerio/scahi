# SPDX-License-Identifier: AGPL-3.0-or-later
"""B-MZ baseline: multi-zoom inference with the standard Ultralytics predictor per pass + box fusion.

A stack is a list of absolute zooms (network-input px per original px). The lowest zoom is the whole-image pass
(input long side = zoom x long image side, rounded *up* to a multiple of 32 by Ultralytics' ``check_imgsz``, so
the effective zoom can be slightly higher, e.g. 0.64 -> 1248/1920 = 0.65). Higher zooms run on tiles of
``T = n_input / zoom`` original px (``scahi.data.zoom.tile_grid``), letterboxed by the predictor to ``n_input``.
Per pass the branch is chosen as for B-FULL (o2o: ``nms=False``; o2m: Ultralytics NMS, iou 0.7). Detections of
all passes (global px) are fused with SAHI's own ``batched_nms`` (class-aware, metric IOS/IOU), i.e. exactly the
operation of ``NMSPostprocess`` - SAHI and Ultralytics code are used unchanged.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pandas as pd

from scahi.data.zoom import tile_grid
from scahi.eval.detections import DET_COLUMNS, finalize_detections
from scahi.infer.full import BRANCH_NMS_ARG, PRED_MAX_DET

RAW_COLUMNS = ["image_id", "x1", "y1", "x2", "y2", "score", "cls", "level", "tile"]


@dataclass(frozen=True)
class Level:
    zoom: float  # nominal absolute zoom (for a fixed-imgsz whole-image pass: imgsz / 1920)
    n_input: int | None  # None = whole-image pass
    imgsz: int | None = None  # whole-image pass with a fixed predictor imgsz (exactly the B-FULL pass) instead of zoom


@dataclass(frozen=True)
class Stack:
    name: str
    levels: tuple[Level, ...]


STACKS = {
    # primary multi-zoom baseline (user decision 2026-10-23): the B-FULL 1280 pass as trained + z=2 tiles
    "native": Stack("native {B-FULL 1280 pass, 2} N=1024@z2", (Level(1280 / 1920, None, 1280), Level(2.0, 1024))),
    # B-MZ-1536-native (user decision 2026-10-23, afternoon): the B-FULL 1536 pass as trained + z=2 tiles
    "native1536": Stack("native1536 {B-FULL 1536 pass, 2} N=1024@z2", (Level(1536 / 1920, None, 1536), Level(2.0, 1024))),
    "native2048": Stack("native2048 {B-FULL 2048 pass, 2} N=1024@z2", (Level(2048 / 1920, None, 2048), Level(2.0, 1024))),
    "S4": Stack("S4 {0.64,2} N=1024@z2", (Level(0.64, None), Level(2.0, 1024))),
    "S1": Stack("S1 {1/3,1,2} N=640", (Level(1 / 3, None), Level(1.0, 640), Level(2.0, 640))),
    "S5": Stack("S5 {1.0 (native),2} N=640", (Level(1.0, None), Level(2.0, 640))),
}


def whole_imgsz(zoom: float, w: int, h: int) -> int:
    """Predictor ``imgsz`` of a whole-image pass (multiple of 32, as ``check_imgsz`` would round it)."""
    return int(math.ceil(zoom * max(w, h) / 32) * 32)


def passes_for(stack: Stack, w: int, h: int, overlap: float) -> list[tuple[int, list[tuple[int, int, int, int]], int]]:
    """Per level: (level index, tiles as (x1, y1, x2, y2), predictor imgsz)."""
    out = []
    for k, lv in enumerate(stack.levels):
        if lv.n_input is None:
            out.append((k, [(0, 0, w, h)], lv.imgsz or whole_imgsz(lv.zoom, w, h)))
        else:
            tiles = tile_grid(w, h, lv.zoom, lv.n_input, overlap)
            out.append((k, [(t.x1, t.y1, t.x2, t.y2) for t in tiles], lv.n_input))
    return out


def passes_per_image(stack: Stack, w: int, h: int, overlap: float) -> int:
    return sum(len(t) for _, t, _ in passes_for(stack, w, h, overlap))


def _raw_rows(iid: int, res: Any, k: int, tiles: list[tuple[int, int, int, int]], tile0: int, w: int, h: int) -> list[tuple]:
    rows = []
    for j, (r, (x1, y1, _, _)) in enumerate(zip(res, tiles)):
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy().astype(np.float64) + [x1, y1, x1, y1]
        np.clip(xyxy, 0, [w, h, w, h], out=xyxy)
        for (a, bb, c, d), s, cl in zip(xyxy, b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)):
            rows.append((iid, a, bb, c, d, float(s), int(cl), k, tile0 + j))
    return rows


def predict_raw(model: Any, images: pd.DataFrame, stack: Stack, overlap: float, branch: str, conf: float,
                batch: int = 32, measure: bool = False) -> tuple[pd.DataFrame, dict[str, float]]:
    """Unfused per-pass detections of all images (global px); timing per image incl. image decoding."""
    import torch

    kw = {"conf": conf, "max_det": PRED_MAX_DET, "verbose": False, "nms": BRANCH_NMS_ARG[branch]}
    rows: list[tuple] = []
    lat: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    first = True
    for iid, path in zip(images.image_id, images.path):
        if measure:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = cv2.imread(path)
        h, w = img.shape[:2]
        tile0 = 0
        for k, tiles, imgsz in passes_for(stack, w, h, overlap):
            crops = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles]
            res = []
            for s in range(0, len(crops), batch):
                res += model.predict(crops[s:s + batch], imgsz=imgsz, batch=batch, **kw)
            rows += _raw_rows(iid, res, k, tiles, tile0, w, h)
            tile0 += len(tiles)
        if measure:
            torch.cuda.synchronize()
            if not first:  # the first image includes model warm-up
                lat.append(1000 * (time.perf_counter() - t0))
        first = False
    info: dict[str, float] = {"peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}
    if lat:
        info.update(latency_ms_mean=float(np.mean(lat)), latency_ms_p95=float(np.percentile(lat, 95)))
    return pd.DataFrame(rows, columns=RAW_COLUMNS), info


def fuse(raw: pd.DataFrame, match_thr: float, conf: float, metric: str = "IOS") -> pd.DataFrame:
    """Class-aware NMS over all passes with SAHI's ``batched_nms`` (= ``NMSPostprocess``)."""
    from sahi.postprocess.combine import batched_nms

    out = []
    r = raw[raw.score >= conf]
    for iid, g in r.groupby("image_id", sort=False):
        a = g[["x1", "y1", "x2", "y2", "score", "cls"]].to_numpy(np.float64)
        keep = batched_nms(a, match_metric=metric, match_threshold=match_thr) if len(a) > 1 else [0]
        k = a[np.asarray(keep, int)]
        out.append(np.column_stack([np.full(len(k), iid), k[:, 0], k[:, 1], k[:, 2] - k[:, 0], k[:, 3] - k[:, 1],
                                    k[:, 4], k[:, 5] + 1]))
    df = pd.DataFrame(np.concatenate(out) if out else np.zeros((0, 7)), columns=DET_COLUMNS)
    df = df.astype({"image_id": int, "category_id": int})
    return finalize_detections(df)


def predict_fused(model: Any, images: pd.DataFrame, stack: Stack, overlap: float, branch: str, conf: float,
                  match_thr: float, metric: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Final val run: per image passes + fusion, timed end to end (one image at a time). Returns fused
    detections, the unfused per-pass detections and timing."""
    import torch

    from scahi.machine import check_exclusive

    dets, raws, lat = [], [], []
    peak = 0.0
    for n in range(len(images)):
        if n % 100 == 0:
            check_exclusive()
        one = images.iloc[[n]]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        raw, info = predict_raw(model, one, stack, overlap, branch, conf)
        d = fuse(raw, match_thr, conf, metric)
        torch.cuda.synchronize()
        if n:
            lat.append(1000 * (time.perf_counter() - t0))
        peak = max(peak, info["peak_mem_mb"])
        dets.append(d)
        raws.append(raw)
    check_exclusive()
    df = finalize_detections(pd.concat(dets, ignore_index=True))
    return df, pd.concat(raws, ignore_index=True), {"latency_ms_mean": float(np.mean(lat)), "latency_ms_p95": float(np.percentile(lat, 95)),
                "peak_mem_mb": peak}
