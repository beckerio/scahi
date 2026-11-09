# SPDX-License-Identifier: AGPL-3.0-or-later
"""B-MZ: pass geometry and equivalence of the array fusion with SAHI's NMSPostprocess."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scahi.infer import multizoom as mz


def test_whole_imgsz_rounds_up_to_32() -> None:
    assert mz.whole_imgsz(0.64, 1920, 1080) == 1248
    assert mz.whole_imgsz(1 / 3, 1920, 1080) == 640
    assert mz.whole_imgsz(1280 / 1920, 1920, 1080) == 1280


def test_passes_s4_geometry() -> None:
    p = mz.passes_for(mz.STACKS["S4"], 1920, 1080, 0.25)
    assert p[0] == (0, [(0, 0, 1920, 1080)], 1248)
    k, tiles, imgsz = p[1]
    assert imgsz == 1024 and all((x2 - x1, y2 - y1) == (512, 512) for x1, y1, x2, y2 in tiles)
    xs = {t[0] for t in tiles} | {t[2] for t in tiles}
    assert min(xs) == 0 and max(xs) == 1920


@pytest.mark.parametrize("metric", ["IOS", "IOU"])
def test_fuse_equals_sahi_nms_postprocess(metric: str) -> None:
    from sahi.postprocess.combine import NMSPostprocess
    from sahi.prediction import ObjectPrediction

    rng = np.random.default_rng(0)
    n = 300
    xy = rng.uniform(0, 400, (n, 2))
    wh = rng.uniform(5, 60, (n, 2))
    raw = pd.DataFrame({"image_id": 1, "x1": xy[:, 0], "y1": xy[:, 1], "x2": xy[:, 0] + wh[:, 0], "y2": xy[:, 1] + wh[:, 1],
                        "score": rng.uniform(0.01, 1, n), "cls": rng.integers(0, 3, n), "level": 0, "tile": 0})
    got = mz.fuse(raw, 0.5, 0.05, metric)
    keep = raw[raw.score >= 0.05]
    preds = [ObjectPrediction(bbox=[r.x1, r.y1, r.x2, r.y2], category_id=int(r.cls), score=float(r.score),
                              full_shape=[1000, 1000]) for r in keep.itertuples()]
    ref = NMSPostprocess(match_threshold=0.5, match_metric=metric, class_agnostic=False)(preds)
    ref_s = np.sort([p.score.value for p in ref])
    assert len(got) == len(ref) and np.allclose(np.sort(got.score.values), ref_s)


def test_native_stack_uses_bfull_pass() -> None:
    for w, h in ((1920, 1080), (1360, 765), (960, 540)):
        p = mz.passes_for(mz.STACKS["native"], w, h, 0.2)
        assert p[0] == (0, [(0, 0, w, h)], 1280)
        assert p[1][2] == 1024 and all((x2 - x1, y2 - y1) == (512, 512) for x1, y1, x2, y2 in p[1][1])
