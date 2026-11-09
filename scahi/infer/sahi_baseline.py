# SPDX-License-Identifier: AGPL-3.0-or-later
"""SAHI-library baselines (B-SAHI with GREEDYNMM / NMS, B-WBF, sahi-default info row).

The library is used unmodified. For tuning, one library pass per overlap collects the
*unmerged* predictions (slices + standard full-image pass) through a postprocess class that is
registered at runtime under the name ``IDENTITY``; GREEDYNMM/NMS are then applied with SAHI's own
postprocess classes, which is exactly what ``get_sliced_prediction`` does at its end
(``merge_buffer_length=None``). Final val rows run plain ``get_sliced_prediction``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scahi.eval.detections import DET_COLUMNS, finalize_detections

SLICE = 640
OVERLAPS = (0.1, 0.2, 0.3)
MATCH_THRS = (0.3, 0.5, 0.7)
CONFS = (0.01, 0.1)
MERGES = ("GREEDYNMM", "NMS", "WBF")
# Tuning grid v2 (user decision 2026-10-22): match metric tuned as well; equal budget of 72 configs per merge.
GRID2_MERGES = ("GREEDYNMM", "NMM", "NMS", "WBF")
GRID2_METRICS = ("IOU", "IOS")
GRID2_THRS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
GRID2_CONFS = (0.01, 0.1)
GRID2_WBF_SKIP = (0.001, 0.01, 0.05, 0.1)


def _register_identity() -> None:
    from sahi import predict as sp
    from sahi.postprocess.combine import PostprocessPredictions

    class IdentityPostprocess(PostprocessPredictions):
        def __call__(self, object_predictions: list) -> list:
            return object_predictions

    sp.POSTPROCESS_NAME_TO_CLASS.setdefault("IDENTITY", IdentityPostprocess)


def load_model(weights: str | Path, conf: float, branch: str = "o2o") -> Any:
    """SAHI Ultralytics model; ``branch='o2o'`` selects the YOLOv10 NMS-free one-to-one head.

    The Ultralytics predictor sets ``end2end = (args.nms is False)``, so the head mode is decided by the
    ``nms`` argument, which SAHI does not pass; it is injected via the YOLO object's ``overrides``,
    which Ultralytics merges into every predict call."""
    from sahi import AutoDetectionModel

    dm = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights),
                                            confidence_threshold=conf, device="cuda:0", image_size=SLICE)
    if branch == "o2o":
        dm.model.overrides["nms"] = False
    return dm


def _to_rows(image_id: int, preds: list) -> list[tuple]:
    rows = []
    for p in preds:
        x1, y1, x2, y2 = p.bbox.to_xyxy()
        rows.append((image_id, float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(p.score.value), int(p.category.id) + 1))
    return rows


def sliced_predict(
    model: Any, images: pd.DataFrame, overlap: float, postprocess: str, match_thr: float,
    force: bool = True, measure: bool = False, match_metric: str = "IOS",
) -> tuple[pd.DataFrame | dict[int, list], dict[str, float]]:
    """Library call per image. ``postprocess='IDENTITY'`` returns raw prediction objects per image."""
    import torch
    from sahi.predict import get_sliced_prediction

    _register_identity()
    raw: dict[int, list] = {}
    rows: list[tuple] = []
    lat: list[float] = []
    n_slices: list[int] = []
    from scahi.machine import check_exclusive

    torch.cuda.reset_peak_memory_stats()
    for n, (iid, path) in enumerate(zip(images.image_id, images.path)):
        if measure and n % 100 == 0:
            check_exclusive()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = get_sliced_prediction(path, model, slice_height=SLICE, slice_width=SLICE, overlap_height_ratio=overlap,
                                  overlap_width_ratio=overlap, perform_standard_pred=True, postprocess_type=postprocess,
                                  postprocess_match_threshold=match_thr, postprocess_match_metric=match_metric,
                                  force_postprocess_type=force, verbose=0)
        torch.cuda.synchronize()
        lat.append(1000 * (time.perf_counter() - t0))
        n_slices.append(r.durations_in_seconds.get("slice_count", 0) if isinstance(r.durations_in_seconds, dict) else 0)
        if postprocess == "IDENTITY":
            raw[iid] = r.object_prediction_list
        else:
            rows += _to_rows(iid, r.object_prediction_list)
    info = {"peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}
    if measure:
        check_exclusive()
        info.update(latency_ms_mean=float(np.mean(lat)), latency_ms_p95=float(np.percentile(lat, 95)))
    if postprocess == "IDENTITY":
        return raw, info
    return finalize_detections(pd.DataFrame(rows, columns=DET_COLUMNS)), info


def merge_offline(raw: dict[int, list], merge: str, match_thr: float, conf: float, sizes: dict[int, tuple[int, int]],
                  metric: str = "IOS") -> pd.DataFrame:
    """Apply GREEDYNMM/NMM/NMS (SAHI classes, class-aware, ``metric`` IOS = library default or IOU) or WBF
    (``match_thr`` = iou_thr, ``conf`` = skip_box_thr) to raw predictions."""
    rows: list[tuple] = []
    if merge in ("GREEDYNMM", "NMM", "NMS"):
        from sahi.predict import POSTPROCESS_NAME_TO_CLASS

        post = POSTPROCESS_NAME_TO_CLASS[merge](match_threshold=match_thr, match_metric=metric, class_agnostic=False)
        for iid, preds in raw.items():
            keep = [p for p in preds if p.score.value >= conf]
            rows += _to_rows(iid, post(keep) if len(keep) > 1 else keep)
    elif merge == "WBF":
        from ensemble_boxes import weighted_boxes_fusion

        for iid, preds in raw.items():
            W, H = sizes[iid]
            keep = [p for p in preds if p.score.value >= conf]
            if not keep:
                continue
            b = np.array([p.bbox.to_xyxy() for p in keep], np.float64) / [W, H, W, H]
            s = np.array([p.score.value for p in keep])
            lab = np.array([p.category.id for p in keep])
            fb, fs, fl = weighted_boxes_fusion([np.clip(b, 0, 1).tolist()], [s.tolist()], [lab.tolist()],
                                               iou_thr=match_thr, skip_box_thr=conf)
            for (x1, y1, x2, y2), sc, c in zip(np.asarray(fb) * [W, H, W, H], fs, fl):
                rows.append((iid, x1, y1, x2 - x1, y2 - y1, float(sc), int(c) + 1))
    else:
        raise ValueError(merge)
    return finalize_detections(pd.DataFrame(rows, columns=DET_COLUMNS))


def slices_per_image(width: int, height: int, overlap: float) -> int:
    from sahi.slicing import get_slice_bboxes

    return len(get_slice_bboxes(height, width, SLICE, SLICE, False, overlap, overlap)) + 1  # + standard pass
