# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tiling: SAHI-style training chips (B-SAHI-FT) and helpers shared with the zoom-stack plan.

Chip rule for B-SAHI-FT (mirrors ``sahi.slicing.slice_coco`` defaults): slice geometry from
``sahi.slicing.get_slice_bboxes``; an object is kept in a chip if at least ``min_area_ratio``
(0.1) of its area is visible, with the box clipped to the chip; ignore boxes are discarded
(Ultralytics default). Chips without objects are kept (SAHI default ``ignore_negative_samples=False``).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from scahi.data.boxes import xyxy_to_yolo, yolo_to_xyxy

CHIP_JPEG_QUALITY = 95
MIN_AREA_RATIO = 0.1


def slice_boxes(width: int, height: int, size: int, overlap: float) -> list[list[int]]:
    from sahi.slicing import get_slice_bboxes

    return get_slice_bboxes(height, width, size, size, False, overlap, overlap)


def boxes_in_chip(
    boxes_xyxy: NDArray, chip_xyxy: tuple[int, int, int, int], min_area_ratio: float = MIN_AREA_RATIO
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Keep-mask and boxes clipped to the chip, in chip coordinates."""
    b = np.asarray(boxes_xyxy, np.float64).reshape(-1, 4)
    cx1, cy1, cx2, cy2 = chip_xyxy
    clipped = np.stack([np.clip(b[:, 0], cx1, cx2), np.clip(b[:, 1], cy1, cy2),
                        np.clip(b[:, 2], cx1, cx2), np.clip(b[:, 3], cy1, cy2)], 1)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    vis = np.clip(clipped[:, 2] - clipped[:, 0], 0, None) * np.clip(clipped[:, 3] - clipped[:, 1], 0, None)
    keep = (vis > 0) & (vis >= min_area_ratio * np.maximum(area, 1e-9))
    return keep, clipped - [cx1, cy1, cx1, cy1]


def _chip_one(job: tuple[str, str, str, int, float]) -> list[str]:
    img_path, lbl_path, out_root, size, overlap = job
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    stem = Path(img_path).stem
    lines = Path(lbl_path).read_text().split("\n") if Path(lbl_path).exists() else []
    arr = np.array([[float(v) for v in ln.split()] for ln in lines if ln.strip()]).reshape(-1, 5)
    cls, xyxy = arr[:, 0].astype(int), yolo_to_xyxy(arr[:, 1:], W, H)
    out = []
    for x1, y1, x2, y2 in slice_boxes(W, H, size, overlap):
        name = f"{stem}_{x1}_{y1}"
        keep, cb = boxes_in_chip(xyxy, (x1, y1, x2, y2))
        cw, ch = x2 - x1, y2 - y1
        yl = xyxy_to_yolo(cb[keep], cw, ch) if keep.any() else np.zeros((0, 4))
        ip = Path(out_root) / "images" / f"{name}.jpg"
        cv2.imwrite(str(ip), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, CHIP_JPEG_QUALITY])
        (Path(out_root) / "labels" / f"{name}.txt").write_text(
            "".join(f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f}\n" for c, (a, b, w, h) in zip(cls[keep], yl)))
        out.append(str(ip))
    return out


def generate_sahi_chips(image_paths: list[str], label_dir_of: callable, out_root: Path, size: int = 640,
                        overlap: float = 0.2, workers: int = 16) -> list[str]:
    """Write chips to ``out_root/{images,labels}`` and return the chip image paths (sorted)."""
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)
    jobs = [(p, str(label_dir_of(p)), str(out_root), size, overlap) for p in image_paths]
    with ProcessPoolExecutor(workers) as ex:
        chips = [c for r in ex.map(_chip_one, jobs, chunksize=8) for c in r]
    return sorted(chips)
