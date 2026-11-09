# SPDX-License-Identifier: AGPL-3.0-or-later
"""pycocotools-based metrics (overall, per class, per size bin incl. AP50)."""

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SIZE_EDGES = (0, 8, 16, 32, 64, 128, np.inf)
SIZE_LABELS = ("<8", "8-16", "16-32", "32-64", "64-128", ">128")
# extra ranges appended after the bins (indices of the bins unchanged); "<16" = preregistered criterion (b)
EXTRA_RANGES = {"<16": (0, 16)}


def _mean_valid(x: np.ndarray) -> float:
    v = x[x > -1]
    return float(v.mean() * 100) if v.size else float("nan")


def coco_metrics(
    images: list[dict], annotations: list[dict], categories: list[dict], detections: list[dict],
    max_dets: tuple[int, ...] = (1, 10, 100, 500),
) -> dict[str, Any]:
    """COCO metrics in percent. Size bins are defined on sigma = sqrt(w*h), i.e. area = sigma^2."""
    gt = COCO()
    gt.dataset = {"images": images, "annotations": annotations, "categories": categories}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(detections) if detections else COCO()
        if not detections:
            dt.dataset = {"images": images, "annotations": [], "categories": categories}
            dt.createIndex()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = list(max_dets)
        ev.params.areaRng = ([[0, 1e10]] + [[lo**2, min(hi**2, 1e10)] for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:])]
                             + [[lo**2, hi**2] for lo, hi in EXTRA_RANGES.values()])
        ev.params.areaRngLbl = ["all", *SIZE_LABELS, *EXTRA_RANGES]
        ev.evaluate()
        ev.accumulate()
    P = ev.eval["precision"]  # [T, R, K, A, M]
    R = ev.eval["recall"]  # [T, K, A, M]
    cat_ids = ev.params.catIds
    out: dict[str, Any] = {
        "AP": _mean_valid(P[:, :, :, 0, -1]),
        "AP50": _mean_valid(P[0, :, :, 0, -1]),
        "AP75": _mean_valid(P[5, :, :, 0, -1]),
        **{f"AR{m}": _mean_valid(R[:, :, 0, i]) for i, m in enumerate(max_dets)},
        "AP_per_class": {int(c): _mean_valid(P[:, :, k, 0, -1]) for k, c in enumerate(cat_ids)},
        "AP_by_size": {lbl: _mean_valid(P[:, :, :, a + 1, -1]) for a, lbl in enumerate((*SIZE_LABELS, *EXTRA_RANGES))},
        "AP50_by_size": {lbl: _mean_valid(P[0, :, :, a + 1, -1]) for a, lbl in enumerate((*SIZE_LABELS, *EXTRA_RANGES))},
        "AR_by_size": {lbl: _mean_valid(R[:, :, a + 1, -1]) for a, lbl in enumerate((*SIZE_LABELS, *EXTRA_RANGES))},
        "max_dets": list(max_dets),
    }
    return out
