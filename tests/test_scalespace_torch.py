# SPDX-License-Identifier: AGPL-3.0-or-later
"""Blocked torch F2 fusion (scahi.fusion.scalespace_torch) against the CPU reference (scalespace.fuse_image).

Synthetic image with a whole-image level and a tiled level; block sizes far below the image size force peaks and
members on different sides of block borders (halo path). Runs on the GPU if available, else on the CPU (torch).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scahi.fusion import scalespace as ss
from scahi.fusion.scalespace_torch import fuse_image_f2

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _levels(seed: int = 0, W: int = 1500, H: int = 1100, nc: int = 5) -> tuple[list[ss.CachedLevel], int, int]:
    rng = np.random.default_rng(seed)
    objs = np.c_[rng.uniform(0, W, 80), rng.uniform(0, H, 80), np.exp(rng.uniform(np.log(4), np.log(300), 80))]
    cls = rng.integers(0, nc, 80)

    def preds(n_per: int, tiles: np.ndarray) -> tuple[np.ndarray, ...]:
        xyxy, probs, tile = [], [], []
        for (x, y, s), c in zip(objs, cls):
            for _ in range(n_per):
                cx, cy = x + rng.normal(0, 0.15 * s), y + rng.normal(0, 0.15 * s)
                w, h = s * np.exp(rng.normal(0, 0.2)), s * np.exp(rng.normal(0, 0.2))
                inside = np.flatnonzero((tiles[:, 0] <= cx) & (cx <= tiles[:, 2]) & (tiles[:, 1] <= cy) & (cy <= tiles[:, 3]))
                if not len(inside):
                    continue
                p = rng.uniform(0, 0.05, nc)
                p[c] = rng.uniform(0.05, 0.95)
                xyxy.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]), probs.append(p), tile.append(rng.choice(inside))
        n = len(xyxy)
        return np.array(xyxy, np.float32), np.array(probs, np.float32), np.array(tile), np.zeros((n, 4), np.float32)

    whole = np.array([[0, 0, W, H]], float)
    step = 512 - 154
    xs = sorted({*range(0, W - 512, step), W - 512})
    ys = sorted({*range(0, H - 512, step), H - 512})
    tiles = np.array([[x, y, x + 512, y + 512] for x in xs for y in ys], float)
    a, b, c, d = preds(6, whole)
    lv0 = ss.CachedLevel("w", a, b, d, c, whole, np.array([1.0]), True, 0.3)
    a, b, c, d = preds(8, tiles)
    lv1 = ss.CachedLevel("t", a, b, d, c, tiles, np.full(len(tiles), 2.0), False, 0.3)
    return [lv0, lv1], W, H


def _sorted(d) -> np.ndarray:
    return d.sort_values(["score", "x", "y"], ascending=[False, True, True])[["x", "y", "w", "h", "score", "category_id"]].to_numpy(float)


@pytest.mark.parametrize("block", [64, 200, 1024, 4096])
@pytest.mark.parametrize("seed", [0, 1])
def test_blocked_matches_cpu_reference(block: int, seed: int) -> None:
    levels, W, H = _levels(seed)
    p = ss.FusionParams(tau_splat=0.005, tau_peak=0.001, cell_factor=5.0, bins_per_oct=5, top_k=5000)
    ref = ss.fuse_image(levels, W, H, "F2", p, nc=5)
    st: dict = {}
    got = fuse_image_f2(levels, W, H, p, nc=5, device=DEV, block=block, pair_budget=4096, stats=st)
    assert len(ref) > 50 and len(got) == len(ref)
    a, b = _sorted(ref), _sorted(got)
    assert np.abs(a[:, :5] - b[:, :5]).max() < 1e-6
    assert (a[:, 5] == b[:, 5]).all()
    if block < 1024:
        assert st["blocks"] > 1


def test_block_size_does_not_change_result() -> None:
    levels, W, H = _levels(2)
    p = ss.FusionParams(tau_splat=0.00125, tau_peak=0.0001, cell_factor=9.0, bins_per_oct=7, top_k=5000)
    outs = [_sorted(fuse_image_f2(levels, W, H, p, nc=5, device=DEV, block=b)) for b in (50, 300, 10000)]
    for o in outs[1:]:
        assert o.shape == outs[0].shape and np.abs(o - outs[0]).max() < 1e-9
