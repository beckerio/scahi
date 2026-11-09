# SPDX-License-Identifier: AGPL-3.0-or-later
import math

import numpy as np
import pytest

from scahi.data.zoom import band_log2, design_geometric, plan_table, tile_grid


def test_whole_image_tile() -> None:
    t = tile_grid(1920, 1080, 640 / 1920)
    assert len(t) == 1 and t[0].whole_image and (t[0].x2, t[0].y2) == (1920, 1080)


def test_tile_grid_covers_and_overlaps() -> None:
    tiles = tile_grid(1920, 1080, 1.0, 640, 0.25)  # T = 640
    assert all(t.x2 - t.x1 == 640 and t.y2 - t.y1 == 640 for t in tiles)
    cover = np.zeros((1080, 1920), bool)
    for t in tiles:
        cover[t.y1:t.y2, t.x1:t.x2] = True
    assert cover.all()
    xs = sorted({t.x1 for t in tiles})
    assert xs[0] == 0 and xs[-1] == 1280 and all(b - a <= 480 for a, b in zip(xs, xs[1:]))


def test_band_and_overlap() -> None:
    lo, hi = band_log2(2.0, 12, 192)
    assert 2**lo == pytest.approx(6) and 2**hi == pytest.approx(96)
    p = plan_table(np.array([4.0, 10, 30, 100, 300]), [1 / 3, 1, 2], 12, 192)
    assert p["rows"][0]["whole_image"] and p["rows"][0]["passes_per_image"] == 1
    assert p["share_gt_covered"] == pytest.approx(0.8)  # 4 px is below 12/2 = 6 px
    assert p["rows"][1]["overlap_octaves_with_next"] == pytest.approx(math.log2(96 / 12))


def test_design_geometric_overlap() -> None:
    z = design_geometric(np.full(100, 3.0), 12, 192, 1 / 3, 0.5, 0.01)
    assert z[0] == pytest.approx(1 / 3) and z[-1] == pytest.approx(4.0)
    ratios = [b / a for a, b in zip(z, z[1:])]
    assert max(ratios) <= 2 ** (4 - 0.5) + 1e-9
