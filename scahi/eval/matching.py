# SPDX-License-Identifier: AGPL-3.0-or-later
"""Greedy per-image matching for diagnostics (collapse plot, degradation test, per-level fusion calibration).

Per image and class, detections in descending score order are matched to the free non-ignore GT with the highest IoU
(>= ``IOU``); detections matching an ignore GT (score 0) are flagged ``excl``. Own diagnostic, not the toolkit AP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scahi.data.boxes import iou_matrix, xywh_to_xyxy

IOU = 0.5


def match_image(im: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per detection (score, sigma, tp, excl), per non-ignore GT (sigma, matched score)."""
    gt_xyxy, dt_xyxy = xywh_to_xyxy(im["gt_xywh"]), xywh_to_xyxy(im["dt_xywh"])
    tp, excl = np.zeros(len(dt_xyxy), bool), np.zeros(len(dt_xyxy), bool)
    gt_score = np.full(len(gt_xyxy), -1.0)
    for c in np.union1d(im["gt_cls"], im["dt_cls"]):
        gi, di = np.flatnonzero(im["gt_cls"] == c), np.flatnonzero(im["dt_cls"] == c)
        if not len(di) or not len(gi):
            continue
        di = di[np.argsort(-im["dt_score"][di], kind="stable")]
        iou = iou_matrix(dt_xyxy[di], gt_xyxy[gi])
        ign = im["gt_ignore"][gi]
        used = np.zeros(len(gi), bool)
        for k, d in enumerate(di):
            cand = np.where(~used & ~ign & (iou[k] >= IOU), iou[k], -1)
            j = int(np.argmax(cand))
            if cand[j] >= 0:
                used[j], tp[d], gt_score[gi[j]] = True, True, im["dt_score"][d]
            elif ign.any() and (iou[k][ign] >= IOU).any():
                excl[d] = True
    s = lambda b: np.sqrt(np.clip(b[:, 2], 0, None) * np.clip(b[:, 3], 0, None))  # noqa: E731
    dets = pd.DataFrame({"score": im["dt_score"], "sigma": s(im["dt_xywh"]), "tp": tp, "excl": excl})
    gts = pd.DataFrame({"sigma": s(im["gt_xywh"]), "score": gt_score, "gt_idx": np.arange(len(gt_xyxy))})[~im["gt_ignore"]]
    return dets, gts
