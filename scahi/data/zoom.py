# SPDX-License-Identifier: AGPL-3.0-or-later
"""Zoom-stack geometry and planning (Phase 3, ``scahi tile --plan``).

A zoom ``z`` is *absolute* (network-input pixels per original pixel): a tile of ``T = N / z`` original
pixels is resized to the network input ``N``. An object of size ``sigma`` (original px) appears with
``sigma * z`` input px and is valid for zoom ``z`` iff ``sigma * z`` lies in the band ``[a, b]``.
A zoom whose tile would cover the whole image is the whole-image pass: the image is resized by ``z``
and padded to a multiple of 32 (no tiling).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Tile:
    x1: int
    y1: int
    x2: int
    y2: int
    zoom: float
    whole_image: bool


def tile_grid(width: int, height: int, zoom: float, n_input: int = 640, overlap: float = 0.25) -> list[Tile]:
    """Tiles of ``T = n_input / zoom`` original px with fractional ``overlap``; edge tiles are shifted inward
    so every tile has full size (as in SAHI). One whole-image tile if ``T >= max(W, H)``."""
    T = n_input / zoom
    if T >= max(width, height) - 1e-6:
        return [Tile(0, 0, width, height, zoom, True)]

    def starts(length: int) -> list[int]:
        t = int(round(T))
        if t >= length:
            return [0]
        step = max(1, int(round(t * (1 - overlap))))
        s = list(range(0, length - t, step)) + [length - t]
        return sorted(set(s))

    t = int(round(T))
    tw, th = min(t, width), min(t, height)
    return [Tile(x, y, x + tw, y + th, zoom, False) for y in starts(height) for x in starts(width)]


def band_log2(zoom: float, a: float, b: float) -> tuple[float, float]:
    """Valid object sizes of a zoom in log2(original px)."""
    return math.log2(a / zoom), math.log2(b / zoom)


def plan_table(
    sigmas: NDArray, zooms: list[float], a: float, b: float, n_input: int = 640, overlap: float = 0.25,
    image_wh: tuple[int, int] = (1920, 1080),
) -> dict[str, object]:
    """Coverage/cost table for a zoom stack on a GT size distribution (train_main/calib only)."""
    zooms = sorted(zooms)
    s = np.asarray(sigmas, np.float64)
    s = s[s > 0]
    full_cost = None
    rows = []
    covered_any = np.zeros(len(s), bool)
    for k, z in enumerate(zooms):
        lo, hi = a / z, b / z
        cov = (s >= lo) & (s <= hi)
        covered_any |= cov
        tiles = tile_grid(*image_wh, z, n_input, overlap)
        px = sum((t.x2 - t.x1) * (t.y2 - t.y1) * z * z for t in tiles)  # input pixels processed
        full_cost = full_cost or px
        ov_next = None
        if k + 1 < len(zooms):
            lo1, hi1 = band_log2(z, a, b)
            lo2, hi2 = band_log2(zooms[k + 1], a, b)
            ov_next = min(hi1, hi2) - max(lo1, lo2)
        rows.append({"zoom": z, "tile_px": round(n_input / z, 1), "whole_image": tiles[0].whole_image,
                     "passes_per_image": len(tiles), "input_megapixels": px / 1e6, "relative_cost": None,
                     "sigma_band_px": [lo, hi], "share_gt_in_band": float(cov.mean()),
                     "overlap_octaves_with_next": ov_next})
    total = sum(r["input_megapixels"] for r in rows)
    for r in rows:
        r["relative_cost"] = r["input_megapixels"] / rows[0]["input_megapixels"]
    return {"zooms": zooms, "a": a, "b": b, "n_input": n_input, "overlap": overlap, "image_wh": list(image_wh),
            "rows": rows, "share_gt_covered": float(covered_any.mean()),
            "share_gt_below": float((s < a / max(zooms)).mean()), "share_gt_above": float((s > b / min(zooms)).mean()),
            "total_input_megapixels": total, "min_adjacent_overlap_octaves":
                min((r["overlap_octaves_with_next"] for r in rows[:-1]), default=None)}


def design_geometric(sigmas: NDArray, a: float, b: float, z_min: float, min_overlap_oct: float = 0.5,
                     cover_quantile: float = 0.01) -> list[float]:
    """Geometric zoom stack from ``z_min`` (whole image) up to the zoom that brings the ``cover_quantile``
    object size into the band, with adjacent bands overlapping by at least ``min_overlap_oct`` octaves."""
    s = np.asarray(sigmas, np.float64)
    z_max = max(z_min, a / np.quantile(s[s > 0], cover_quantile))
    width_oct = math.log2(b / a)
    max_ratio = 2 ** (width_oct - min_overlap_oct)
    n = max(1, math.ceil(math.log(z_max / z_min) / math.log(max_ratio))) if z_max > z_min else 0
    return [z_min * (z_max / z_min) ** (k / n) for k in range(n + 1)] if n else [z_min]
