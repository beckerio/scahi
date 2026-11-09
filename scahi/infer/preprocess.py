# SPDX-License-Identifier: AGPL-3.0-or-later
"""Own letterbox preprocessing (the Ultralytics predictor is bypassed for raw-head access).

Letterbox convention = Ultralytics ``LetterBox`` (predictor and augmented training): bilinear resize by
``gain``, centred placement with padding value 114 (``left = round(dw - 0.1)``, ``top = round(dh - 0.1)``).
Mapping back: ``x_img = (x_input - pad) / gain``. Matching the predictor matters: INTER_AREA vs.
INTER_LINEAR at 1/3 scale changes small-object scores noticeably.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

PAD_VALUE = 114
STRIDE_MULTIPLE = 32


@dataclass(frozen=True)
class Letterbox:
    gain: float
    in_hw: tuple[int, int]  # network input (H, W)
    src_hw: tuple[int, int]  # source crop (H, W)
    pad_xy: tuple[int, int] = (0, 0)  # left, top padding in input pixels


def letterbox(img: NDArray[np.uint8], out_hw: tuple[int, int]) -> tuple[NDArray[np.uint8], Letterbox]:
    """Resize ``img`` (BGR) to fit ``out_hw`` keeping the aspect ratio; centred padding (Ultralytics LetterBox)."""
    h, w = img.shape[:2]
    gain = min(out_hw[0] / h, out_hw[1] / w)
    nh, nw = int(round(h * gain)), int(round(w * gain))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else img
    dw, dh = (out_hw[1] - nw) / 2, (out_hw[0] - nh) / 2
    left, top = int(round(dw - 0.1)), int(round(dh - 0.1))
    canvas = np.full((out_hw[0], out_hw[1], 3), PAD_VALUE, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, Letterbox(gain, out_hw, (h, w), (left, top))


def full_image_shape(h: int, w: int, long_side: int) -> tuple[int, int]:
    """Input shape for a whole-image pass: long side = ``long_side``, both sides padded to a multiple of 32."""
    g = long_side / max(h, w)
    return (int(np.ceil(h * g / STRIDE_MULTIPLE) * STRIDE_MULTIPLE), int(np.ceil(w * g / STRIDE_MULTIPLE) * STRIDE_MULTIPLE))


def to_tensor(batch: list[NDArray[np.uint8]], device: str | torch.device = "cuda") -> torch.Tensor:
    """BGR uint8 HWC images (same shape) -> float RGB NCHW in [0, 1]."""
    x = np.stack([b[..., ::-1] for b in batch]).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(x)).to(device).float() / 255.0


def input_to_source(xyxy: NDArray, lb: Letterbox, offset_xy: tuple[float, float] = (0.0, 0.0)) -> NDArray:
    """Network-input pixels -> source pixels (+ tile offset in global pixels)."""
    px, py = lb.pad_xy
    b = (np.asarray(xyxy, np.float64) - [px, py, px, py]) / lb.gain
    return b + np.array([offset_xy[0], offset_xy[1], offset_xy[0], offset_xy[1]])
