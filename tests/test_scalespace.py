# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import pandas as pd
import pytest

from scahi.eval.matching import match_image
from scahi.fusion import scalespace as ss

P = ss.FusionParams()
W, H = 1000, 400


def _level(boxes, cls, p, tile=None, tiles=None, whole=True, key="w"):
    n = len(boxes)
    probs = np.zeros((n, 10))
    probs[np.arange(n), cls] = p
    tiles = np.array([[0, 0, W, H]], float) if tiles is None else np.asarray(tiles, float)
    return ss.CachedLevel(key=key, xyxy=np.asarray(boxes, float), probs=probs, std=np.ones((n, 4)),
                          tile=np.zeros(n, int) if tile is None else np.asarray(tile), tiles=tiles,
                          gains=np.ones(len(tiles)), whole=whole, overlap=0.3)


def test_validity_window_anchors():
    v = ss.validity_raw(np.array([P.a_lo, P.a, P.a_hi, 100.0, P.b, P.b_hi, 4.0, 400.0]), P)
    assert v[:6] == pytest.approx([0.0, 0.5, 1.0, 1.0, 1.0, 0.0], abs=1e-9)
    assert v[6] == 0.0 and v[7] == 0.0  # below a_lo and above b_hi


def test_border_ramp_only_at_inner_tile_edges():
    tiles = np.array([[0, 0, 600, 400], [400, 0, 1000, 400]], float)
    r = ss.border_raw(np.array([0.0, 300.0, 600.0]), np.array([200.0] * 3), tiles, W, H, 0.3)
    assert r[0, 0] == pytest.approx(1.0)  # image border: no ramp
    assert r[1, 0] == pytest.approx(1.0)
    assert r[2, 0] == pytest.approx(0.0)  # inner edge of tile 0
    assert r[0, 1] == 0.0  # outside tile 1


def test_border_windows_are_a_partition_of_unity():
    lv = _level([[0, 0, 1, 1]], [0], 0.5, tiles=[[0, 0, 600, 400], [400, 0, 1000, 400]], whole=False)
    x = np.linspace(0, 1000, 41)
    bn, cnt = ss.border_norm(x, np.full_like(x, 200.0), lv, W, H)
    assert bn.sum(1) == pytest.approx(np.ones_like(x))
    assert cnt[(x > 400) & (x < 600)].min() == 2


def test_single_box_is_returned_unchanged():
    d = ss.fuse_image([_level([[100, 100, 140, 130]], [2], 0.9)], W, H, "F2", P)
    assert len(d) == 1
    r = d.iloc[0]
    assert (r.x, r.y, r.w, r.h) == pytest.approx((100, 100, 40, 30))
    assert r.category_id == 3 and r.score == pytest.approx(0.9, rel=1e-5)


def test_duplicate_from_overlapping_tiles_is_fused_without_score_inflation():
    b = [[490, 190, 530, 220]] * 2  # same object seen by both tiles, centre in the overlap
    lv = _level(b, [0, 0], 0.8, tile=[0, 1], tiles=[[0, 0, 600, 400], [400, 0, 1000, 400]], whole=False)
    d = ss.fuse_image([lv], W, H, "F2", P)
    assert len(d) == 1
    assert d.iloc[0].score == pytest.approx(0.8, rel=1e-4)  # border weights sum to 1 -> no double counting


def test_separate_objects_stay_separate_and_box_source_top():
    boxes = [[100, 100, 130, 130], [600, 300, 630, 330], [101, 101, 133, 133]]
    probs_cls = [1, 1, 1]
    lv = _level(boxes, probs_cls, 0.6)
    lv.probs[2, 1] = 0.3  # weaker duplicate of the first object
    d = ss.fuse_image([lv], W, H, "F2", P)
    assert len(d) == 2 and set(d.category_id) == {2}
    t = ss.fuse_image([lv], W, H, "F2", P, box_source="top").sort_values("x")
    assert (t.iloc[0].x, t.iloc[0].w) == pytest.approx((100, 30))  # strongest member's box, not the mean


def test_empty_and_below_threshold():
    cols = ["x", "y", "w", "h", "score", "category_id"]
    assert list(ss.fuse_image([_level(np.zeros((0, 4)), [], 0.5)], W, H, "F2", P).columns) == cols
    assert ss.fuse_image([_level([[0, 0, 20, 20]], [0], P.tau_splat / 2)], W, H, "F2", P).empty


def test_match_image_tp_excl_and_fn():
    im = {"gt_xywh": np.array([[0, 0, 10, 10], [50, 50, 10, 10], [100, 100, 10, 10]], float),
          "gt_cls": np.array([1, 1, 1]), "gt_ignore": np.array([False, False, True]),
          "dt_xywh": np.array([[0, 0, 10, 10], [1, 1, 10, 10], [100, 100, 10, 10], [300, 300, 5, 5]], float),
          "dt_score": np.array([0.9, 0.8, 0.7, 0.6]), "dt_cls": np.array([1, 1, 1, 1])}
    dets, gts = match_image(im)
    assert dets.tp.tolist() == [True, False, False, False]  # second hit on the same GT is a duplicate
    assert dets.excl.tolist() == [False, False, True, False]  # matches the ignore GT
    assert len(gts) == 2 and gts.score.tolist() == [0.9, -1.0]  # ignore GT not listed; second GT missed
    assert isinstance(dets, pd.DataFrame)
