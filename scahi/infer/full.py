# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full-image inference with the standard Ultralytics predictor (baselines B-FULL)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scahi.eval.detections import DET_COLUMNS, finalize_detections

PRED_CONF = 0.001
PRED_MAX_DET = 500
# Ultralytics 8.4.183 loads YOLOv10 checkpoints with end2end=False: the default predictor fuses away the
# one-to-one branch and runs one-to-many + NMS. ``nms=False`` selects the native NMS-free o2o head.
BRANCH_NMS_ARG = {"o2o": False, "o2m": None}


def image_table(coco: dict[str, Any], image_root: Path | None = None) -> pd.DataFrame:
    """``image_id, stem, path``; ``image_root`` overrides the original image directory."""
    rows = []
    for i in coco["images"]:
        stem = Path(i["file_name"]).stem
        path = Path(i["path"]) if image_root is None else image_root / i["file_name"]
        rows.append((i["id"], stem, str(path)))
    return pd.DataFrame(rows, columns=["image_id", "stem", "path"])


def predict_full(
    weights: str | Path, images: pd.DataFrame, imgsz: int, conf: float = PRED_CONF, batch: int = 16,
    measure_latency: bool = False, branch: str = "o2o",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Predict every image; returns canonical detections (sorted, <= 500/image) and timing info.

    ``branch='o2o'``: NMS-free one-to-one head (YOLOv10 native); ``'o2m'``: one-to-many head + NMS
    (Ultralytics 8.4 default for these checkpoints)."""
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kw = {"imgsz": imgsz, "conf": conf, "max_det": PRED_MAX_DET, "verbose": False, "nms": BRANCH_NMS_ARG[branch]}
    rows: list[tuple] = []
    lat: list[float] = []
    paths = images.path.tolist()
    ids = images.image_id.tolist()
    torch.cuda.reset_peak_memory_stats()
    if measure_latency:  # batch 1, synchronous timing per image (end-to-end incl. pre/post-processing)
        from scahi.machine import check_exclusive

        model.predict(paths[0], **kw)
        for n, (iid, p) in enumerate(zip(ids, paths)):
            if n % 100 == 0:
                check_exclusive()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = model.predict(p, **kw)[0]
            torch.cuda.synchronize()
            lat.append(1000 * (time.perf_counter() - t0))
            rows += _rows(iid, r)
    else:
        k = 0
        while k < len(paths):
            try:
                res = model.predict(paths[k:k + batch], batch=batch, **kw)
            except torch.cuda.OutOfMemoryError:  # YOLO12 at 2048 inside training (calib eval): smaller chunks
                if batch == 1:
                    raise
                batch = max(batch // 2, 1)
                torch.cuda.empty_cache()
                print(f"[predict_full] OOM -> batch {batch}", flush=True)
                continue
            for iid, r in zip(ids[k:k + batch], res):
                rows += _rows(iid, r)
            k += batch
    if measure_latency:
        check_exclusive()
    df = finalize_detections(pd.DataFrame(rows, columns=DET_COLUMNS))
    info = {"peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}
    if lat:
        info.update(latency_ms_mean=float(np.mean(lat)), latency_ms_p95=float(np.percentile(lat, 95)))
    return df, info


def _rows(image_id: int, r: Any) -> list[tuple]:
    b = r.boxes
    if b is None or len(b) == 0:
        return []
    xyxy = b.xyxy.cpu().numpy().astype(np.float64)
    conf = b.conf.cpu().numpy().astype(np.float64)
    cls = b.cls.cpu().numpy().astype(np.int64)
    return [(image_id, x1, y1, x2 - x1, y2 - y1, s, c + 1) for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls)]


def gflops_per_image(weights: str | Path, imgsz: int, width: int = 1920, height: int = 1080) -> float:
    """GFLOPs of one letterboxed forward pass (rect inference: long side = imgsz, short side padded to 32)."""
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    m = YOLO(str(weights)).model
    g640 = get_flops(m, 640)  # square 640x640
    s = imgsz / max(width, height)
    h = int(np.ceil(height * s / 32) * 32)
    w = int(np.ceil(width * s / 32) * 32)
    return float(g640 * (h * w) / (640 * 640))
