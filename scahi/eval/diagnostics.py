# SPDX-License-Identifier: AGPL-3.0-or-later
"""Error diagnostics: duplicate rate, fragment rate, FP rate vs. distance to tile borders.

Definitions (per class, detections with ``score >= conf``, greedy matching by score at IoU 0.5):

* duplicate: unmatched detection with IoU >= 0.5 to a GT of its class that is already matched.
* fragment: unmatched, non-duplicate detection with IoF (intersection / detection area) >= 0.8
  to a GT of its class while IoU < 0.5 to every GT of its class.
* Rates are relative to the number of considered detections.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from scahi.data.boxes import iof_matrix, iou_matrix, xywh_to_xyxy

BORDER_EDGES = (0, 8, 16, 32, 64, 128, np.inf)


def classify_detections(
    gt_xyxy: NDArray, gt_cls: NDArray, dt_xyxy: NDArray, dt_score: NDArray, dt_cls: NDArray,
    iou_thr: float = 0.5, frag_iof: float = 0.8,
) -> NDArray[np.int8]:
    """Per detection: 1 = TP, 0 = other FP, 2 = duplicate, 3 = fragment."""
    out = np.zeros(len(dt_cls), np.int8)
    for c in np.unique(dt_cls):
        d_idx = np.nonzero(dt_cls == c)[0]
        d_idx = d_idx[np.argsort(-dt_score[d_idx], kind="stable")]
        g = gt_xyxy[gt_cls == c]
        if len(g) == 0:
            continue
        iou = iou_matrix(dt_xyxy[d_idx], g)
        iof = iof_matrix(dt_xyxy[d_idx], g)
        matched = np.zeros(len(g), bool)
        for k, d in enumerate(d_idx):
            cand = np.where(~matched, iou[k], -1.0)
            j = int(np.argmax(cand))
            if cand[j] >= iou_thr:
                matched[j] = True
                out[d] = 1
            elif (iou[k] >= iou_thr).any():
                out[d] = 2
            elif (iof[k] >= frag_iof).any():
                out[d] = 3
    return out


def border_distance(points: NDArray, tiles_xyxy: NDArray, width: int, height: int) -> NDArray[np.float64]:
    """Distance of each point to the nearest *internal* tile edge (edges on the image border excluded)."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    best = np.full(len(pts), np.inf)
    eps = 1e-6
    for x1, y1, x2, y2 in np.asarray(tiles_xyxy, np.float64).reshape(-1, 4):
        edges = []
        if x1 > eps:
            edges.append(("v", x1, y1, y2))
        if x2 < width - eps:
            edges.append(("v", x2, y1, y2))
        if y1 > eps:
            edges.append(("h", y1, x1, x2))
        if y2 < height - eps:
            edges.append(("h", y2, x1, x2))
        for kind, pos, a, b in edges:
            if kind == "v":
                dx = np.abs(pts[:, 0] - pos)
                dy = np.clip(np.maximum(a - pts[:, 1], pts[:, 1] - b), 0, None)
            else:
                dy = np.abs(pts[:, 1] - pos)
                dx = np.clip(np.maximum(a - pts[:, 0], pts[:, 0] - b), 0, None)
            best = np.minimum(best, np.hypot(dx, dy))
    return best


def diagnostics(
    per_image: list[dict[str, Any]], conf: float, tiles: dict[int, NDArray] | None = None,
) -> dict[str, Any]:
    """``per_image`` items: image_id, width, height, gt_xywh, gt_cls, dt_xywh, dt_score, dt_cls."""
    n_det = n_dup = n_frag = n_tp = 0
    dist_all, fp_all = [], []
    for im in per_image:
        sel = im["dt_score"] >= conf
        d_xyxy = xywh_to_xyxy(im["dt_xywh"][sel])
        lab = classify_detections(xywh_to_xyxy(im["gt_xywh"]), im["gt_cls"], d_xyxy, im["dt_score"][sel], im["dt_cls"][sel])
        n_det += len(lab)
        n_tp += int((lab == 1).sum())
        n_dup += int((lab == 2).sum())
        n_frag += int((lab == 3).sum())
        if tiles is not None and im["image_id"] in tiles and len(lab):
            ctr = np.stack([(d_xyxy[:, 0] + d_xyxy[:, 2]) / 2, (d_xyxy[:, 1] + d_xyxy[:, 3]) / 2], 1)
            dist_all.append(border_distance(ctr, tiles[im["image_id"]], im["width"], im["height"]))
            fp_all.append(lab != 1)
    out: dict[str, Any] = {
        "conf": conf,
        "n_det": n_det,
        "n_tp": n_tp,
        "dup_rate": n_dup / max(n_det, 1),
        "frag_rate": n_frag / max(n_det, 1),
        "fp_by_border_dist": {},
    }
    if dist_all:
        dist, fp = np.concatenate(dist_all), np.concatenate(fp_all)
        for lo, hi in zip(BORDER_EDGES[:-1], BORDER_EDGES[1:]):
            m = (dist >= lo) & (dist < hi)
            out["fp_by_border_dist"][f"{lo}-{hi}"] = {"n": int(m.sum()), "fp_rate": float(fp[m].mean()) if m.any() else None}
    return out
