# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact GFLOPs per image from the pass geometry (extension study, descriptive; user request 2026-10-26).

GFLOPs of one forward pass are counted with Ultralytics' ``get_flops`` (THOP, stride-aware extrapolation) at the exact
letterboxed input shape (h, w) on the **fused** model; for YOLOv10 fusion keeps one head, and o2o and o2m heads have
identical architecture, so the count holds for both branches. Counting per shape (instead of a constant GFLOPs per
megapixel, as in the finished study) matters for YOLO12, whose area attention does not scale linearly with the pixels.
Results are cached per weights file in ``runs/ext/flops/<weights_id>.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scahi.config import REPO_ROOT

CACHE = REPO_ROOT / "runs/ext/flops"


class FlopCounter:
    def __init__(self, weights: str | Path) -> None:
        from scahi.infer.cache import weights_id

        self.weights = Path(weights)
        CACHE.mkdir(parents=True, exist_ok=True)
        self.file = CACHE / f"{weights_id(weights)}.json"
        self.table: dict[str, float] = json.loads(self.file.read_text()) if self.file.exists() else {}
        self._model = None

    def __call__(self, h: int, w: int) -> float:
        k = f"{h}x{w}"
        if k not in self.table:
            from ultralytics import YOLO
            from ultralytics.utils.torch_utils import get_flops

            if self._model is None:
                self._model = YOLO(str(self.weights)).model.fuse(verbose=False).cpu().float().eval()
            self.table[k] = float(get_flops(self._model, [h, w]))
            self.file.write_text(json.dumps(self.table, indent=1, sort_keys=True))
        return self.table[k]


def single_pass(fc: FlopCounter, sizes_wh: list[tuple[int, int]], imgsz: int) -> float:
    from scahi.infer.preprocess import full_image_shape

    return float(np.mean([fc(*full_image_shape(H, W, imgsz)) for W, H in sizes_wh]))


def stack(fc: FlopCounter, sizes_wh: list[tuple[int, int]], stack_key: str, overlap: float) -> float:
    from scahi.infer import cache_mz as cm

    return float(np.mean([sum(fc(*shp) for k in cm.STACK_LEVELS[stack_key] for _, shp in cm.level_passes(cm.LEVELS[k], W, H, overlap))
                          for W, H in sizes_wh]))


def sahi(fc: FlopCounter, sizes_wh: list[tuple[int, int]], overlap: float, slice_px: int = 640) -> float:
    """Slices of ``slice_px`` (square input) + the standard full-image pass at ``slice_px``."""
    from sahi.slicing import get_slice_bboxes

    from scahi.infer.preprocess import full_image_shape

    out = []
    for W, H in sizes_wh:
        n = len(get_slice_bboxes(H, W, slice_px, slice_px, False, overlap, overlap))
        out.append(n * fc(slice_px, slice_px) + fc(*full_image_shape(H, W, slice_px)))
    return float(np.mean(out))


def describe() -> dict[str, Any]:
    return {"method": "Ultralytics get_flops on the fused model at the exact letterboxed input shape, summed over the "
                      "passes of each image, mean over the images of the split"}
