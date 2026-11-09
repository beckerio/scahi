# SPDX-License-Identifier: AGPL-3.0-or-later
"""SCAHI on dense predictions of an arbitrary detector: arrays in, boxes out (no Ultralytics needed).

    python examples/scahi_arrays.py

Each pass contributes, for every dense position above a low score: a box (x1, y1, x2, y2 in image pixels) and the
class probabilities. Passes are grouped in levels (here one whole-image pass and one level of two overlapping tiles);
each level knows its tile rectangles. The example uses synthetic predictions around three objects.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scahi.fusion.scalespace import CachedLevel, FusionParams, fuse_image  # noqa: E402

W, H, NC = 1200, 800, 3
OBJECTS = [(300, 400, 20, 1), (610, 380, 14, 0), (900, 200, 40, 2)]  # cx, cy, size, class
rng = np.random.default_rng(0)


def dense(tile: tuple[int, int, int, int], n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Jittered dense predictions of one pass for the objects inside ``tile``."""
    boxes, probs = [], []
    for cx, cy, s, c in OBJECTS:
        if tile[0] <= cx < tile[2] and tile[1] <= cy < tile[3]:
            x = cx + rng.normal(0, 0.15 * s, n)
            y = cy + rng.normal(0, 0.15 * s, n)
            w = s * np.exp(rng.normal(0, 0.1, n))
            boxes.append(np.stack([x - w / 2, y - w / 2, x + w / 2, y + w / 2], 1))
            p = np.full((n, NC), 0.01)
            p[:, c] = rng.uniform(0.3, 0.9, n)
            probs.append(p)
    return np.concatenate(boxes).astype(np.float32), np.concatenate(probs).astype(np.float32)


def level(tiles: list[tuple[int, int, int, int]], whole: bool) -> CachedLevel:
    parts = [dense(t) for t in tiles]
    xyxy = np.concatenate([b for b, _ in parts])
    probs = np.concatenate([p for _, p in parts])
    tile = np.concatenate([np.full(len(b), i, np.int16) for i, (b, _) in enumerate(parts)])
    return CachedLevel(key="whole" if whole else "tiles", xyxy=xyxy, probs=probs,
                       std=np.zeros((len(xyxy), 4), np.float32),  # only used by the uncertainty variants
                       tile=tile, tiles=np.asarray(tiles, np.int32), gains=np.ones(len(tiles)), whole=whole,
                       overlap=0.3)


levels = [level([(0, 0, W, H)], whole=True),                          # one whole-image pass
          level([(0, 0, 700, H), (500, 0, W, H)], whole=False)]       # two tiles overlapping by 200 px
det = fuse_image(levels, W, H, "F2", FusionParams(tau_splat=0.005, tau_peak=0.001, cell_factor=5.0, bins_per_oct=5),
                 nc=NC)
print(det[det.score >= 0.25].round(1).to_string(index=False))
