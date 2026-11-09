# SPDX-License-Identifier: AGPL-3.0-or-later
"""Box coordinate conversions and overlap measures.

Conventions: ``xyxy`` boxes are ``[x1, y1, x2, y2]`` in continuous pixel coordinates
(the box covers ``[x1, x2) x [y1, y2)``); ``xywh`` boxes are ``[left, top, width, height]``
(VisDrone / COCO); YOLO boxes are ``[cx, cy, w, h]`` normalised by image width/height.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def xywh_to_xyxy(b: NDArray) -> FloatArray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)


def xyxy_to_xywh(b: NDArray) -> FloatArray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)


def yolo_to_xyxy(b: NDArray, img_w: float, img_h: float) -> FloatArray:
    """Normalised ``[cx, cy, w, h]`` -> absolute ``xyxy`` pixels."""
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    cx, cy = b[:, 0] * img_w, b[:, 1] * img_h
    w, h = b[:, 2] * img_w, b[:, 3] * img_h
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def xyxy_to_yolo(b: NDArray, img_w: float, img_h: float) -> FloatArray:
    """Absolute ``xyxy`` pixels -> normalised ``[cx, cy, w, h]``."""
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack(
        [
            (b[:, 0] + b[:, 2]) / 2 / img_w,
            (b[:, 1] + b[:, 3]) / 2 / img_h,
            (b[:, 2] - b[:, 0]) / img_w,
            (b[:, 3] - b[:, 1]) / img_h,
        ],
        axis=1,
    )


def box_sigma(b_xyxy: NDArray) -> FloatArray:
    """Geometric-mean box size ``sigma = sqrt(w * h)`` in pixels."""
    b = np.asarray(b_xyxy, dtype=np.float64).reshape(-1, 4)
    w = np.clip(b[:, 2] - b[:, 0], 0, None)
    h = np.clip(b[:, 3] - b[:, 1], 0, None)
    return np.sqrt(w * h)


def _intersection(a: FloatArray, b: FloatArray) -> FloatArray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    return wh[..., 0] * wh[..., 1]


def _area(b: FloatArray) -> FloatArray:
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def iou_matrix(a_xyxy: NDArray, b_xyxy: NDArray) -> FloatArray:
    """Pairwise IoU, shape ``[len(a), len(b)]``."""
    a = np.asarray(a_xyxy, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b_xyxy, dtype=np.float64).reshape(-1, 4)
    inter = _intersection(a, b)
    union = _area(a)[:, None] + _area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def iof_matrix(a_xyxy: NDArray, b_xyxy: NDArray) -> FloatArray:
    """Pairwise intersection over the area of ``a`` (foreground), shape ``[len(a), len(b)]``."""
    a = np.asarray(a_xyxy, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b_xyxy, dtype=np.float64).reshape(-1, 4)
    inter = _intersection(a, b)
    area_a = _area(a)[:, None]
    return np.where(area_a > 0, inter / np.maximum(area_a, 1e-12), 0.0)
