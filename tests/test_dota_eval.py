# SPDX-License-Identifier: AGPL-3.0-or-later
"""DOTA evaluation port (scahi/dota/evaluate.py): devkit semantics on hand-made cases; bootstrap weights."""

import numpy as np
import pandas as pd
import pytest

from scahi.dota import evaluate as EV


def _gt(*boxes, cls=0, difficult=()):
    b = np.array(boxes, float).reshape(-1, 4)
    return b, np.full(len(b), cls), np.array([i in difficult for i in range(len(b))])


def _dets(rows):
    return pd.DataFrame(rows, columns=EV.DET_COLS)


def ap(gt, rows, cap=None, w=None):
    return EV.class_ap(EV.build(gt, _dets(rows), cap, sizes=False), 0, w=None if w is None else np.array(w, float))


def test_perfect_and_duplicate():
    gt = {"a": _gt([0, 0, 9, 9])}
    assert ap(gt, [("a", 0, 0.9, 0, 0, 9, 9)]) == pytest.approx(1.0)
    # duplicate after the TP is a FP; recall 1 reached at precision 1 -> VOC07 AP still 1
    assert ap(gt, [("a", 0, 0.9, 0, 0, 9, 9), ("a", 0, 0.8, 0, 0, 9, 9)]) == pytest.approx(1.0)
    # duplicate BEFORE the TP (higher score, lower IoU but > 0.5 -> it takes the GT): second is FP
    r = ap(gt, [("a", 0, 0.5, 0, 0, 9, 9), ("a", 0, 0.9, 1, 1, 9, 9)])
    assert r == pytest.approx(1.0)


def test_strict_threshold_and_plus_one():
    # +1 convention: GT 0..9 (area 100), det 0..4 x 0..9 -> IoU 50/100 = 0.5 exactly -> not > 0.5 -> FP
    gt = {"a": _gt([0, 0, 9, 9])}
    assert ap(gt, [("a", 0, 0.9, 0, 0, 4, 9)]) == pytest.approx(0.0)
    assert ap(gt, [("a", 0, 0.9, 0, 0, 5, 9)]) == pytest.approx(1.0)  # IoU 0.6


def test_difficult_is_neither_tp_nor_fp():
    gt = {"a": _gt([0, 0, 9, 9], [50, 50, 59, 59], difficult=(1,))}
    # det on the difficult GT ranks first: ignored, so precision stays 1 at recall 1
    assert ap(gt, [("a", 0, 0.95, 50, 50, 59, 59), ("a", 0, 0.9, 0, 0, 9, 9)]) == pytest.approx(1.0)
    # only difficult GT of the class -> no AP (left out of the mean)
    assert ap({"a": _gt([0, 0, 9, 9], difficult=(0,))}, [("a", 0, 0.9, 0, 0, 9, 9)]) is None


def test_argmax_gt_already_matched_is_fp():
    # two GT; the second det overlaps GT0 most (already matched) -> FP even though GT1 is free (devkit rule)
    gt = {"a": _gt([0, 0, 9, 9], [6, 0, 15, 9])}
    rows = [("a", 0, 0.9, 0, 0, 9, 9), ("a", 0, 0.8, 1, 0, 10, 9)]
    c = EV.build(gt, _dets(rows), None, sizes=False)
    st, _ = c.dets[(0, 0, -1)]
    assert st.tolist() == [1, 0]


def test_cap_keeps_top_per_image():
    gt = {"a": _gt([0, 0, 9, 9])}
    rows = [("a", 0, 0.1, 0, 0, 9, 9), ("a", 0, 0.9, 100, 100, 109, 109)]
    assert ap(gt, rows) == pytest.approx(0.5)  # TP found second: precision 0.5 at recall 1 -> 11pt AP 0.5
    assert ap(gt, rows, cap=1) == pytest.approx(0.0)  # the TP is cut by the cap


def test_weights_one_is_default_and_integer_weights_scale_npos():
    gt = {"a": _gt([0, 0, 9, 9]), "b": _gt([0, 0, 9, 9])}
    rows = [("a", 0, 0.9, 0, 0, 9, 9), ("b", 0, 0.8, 50, 50, 59, 59)]
    assert ap(gt, rows, w=[1, 1]) == ap(gt, rows)
    assert ap(gt, rows, w=[1, 1]) == pytest.approx(6 / 11)  # recall 0.5 at precision 1 -> points 0..0.5
    assert ap(gt, rows, w=[1, 0]) == pytest.approx(1.0)  # image b dropped
    assert ap(gt, rows, w=[0, 1]) == pytest.approx(0.0)


def test_size_bins_ignore_outside():
    gt = {"a": _gt([0, 0, 4, 4], [100, 100, 139, 139])}  # sigma 4 (<8) and 39 (32-64)
    rows = [("a", 0, 0.9, 100, 100, 139, 139), ("a", 0, 0.8, 0, 0, 4, 4), ("a", 0, 0.7, 300, 300, 303, 303)]
    c = EV.build(gt, _dets(rows), None, sizes=True)
    k = EV.SIZE_LABELS.index("32-64")
    assert EV.class_ap(c, 0, 0, k) == pytest.approx(1.0)  # the small TP and the small FP are ignored in this bin
    k8 = EV.SIZE_LABELS.index("<8")
    st, _ = c.dets[(0, 0, k8)]
    assert sorted(st.tolist()) == [-1, 0, 1]  # big TP ignored (GT outside bin), small TP, small FP counted
