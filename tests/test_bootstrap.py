# SPDX-License-Identifier: AGPL-3.0-or-later
"""Weighted re-evaluation (bootstrap) reproduces the toolkit port and pycocotools exactly for unit weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scahi.config import load_class_setting
from scahi.eval import bootstrap as B
from scahi.eval.evaluate import evaluate


def _synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    images, anns, rows = [], [], []
    for i in range(1, 13):
        images.append({"id": i, "file_name": f"{i:04d}.jpg", "width": 400, "height": 300})
        for _ in range(rng.integers(1, 15)):
            x, y, w, h = rng.uniform(0, 300), rng.uniform(0, 200), rng.uniform(3, 60), rng.uniform(3, 60)
            c = int(rng.integers(1, 4))
            anns.append({"id": len(anns) + 1, "image_id": i, "bbox": [x, y, w, h], "area": w * h, "category_id": c,
                         "score": 1, "orig_category": c, "truncation": 0, "occlusion": 0})
            if rng.random() < 0.8:
                rows.append((i, x + rng.normal(0, 3), y + rng.normal(0, 3), w, h, rng.uniform(0.1, 1), c))
        for _ in range(5):
            rows.append((i, rng.uniform(0, 300), rng.uniform(0, 200), 20, 20, rng.uniform(0, 0.5), int(rng.integers(1, 4))))
    cats = [{"id": c, "name": n} for c, n in enumerate(load_class_setting("visdrone10").classes, 1)]
    coco = {"images": images, "annotations": anns, "ignore": [], "categories": cats}
    return coco, pd.DataFrame(rows, columns=["image_id", "x", "y", "w", "h", "score", "category_id"])


def test_unit_weights_reproduce_port_and_coco() -> None:
    cs = load_class_setting("visdrone10")
    coco, d = _synthetic()
    r = evaluate(coco, d, cs, "visdrone_ignore")
    ims = B.prepared(coco, d, cs)
    pc = B.port_cache(ims, tuple(sorted(c["id"] for c in coco["categories"])))
    assert abs(B.port_ap(pc) - r["metrics"]["AP"]) < 1e-9
    cc = B.coco_cache(ims, coco)
    assert abs(B.coco_ap(cc) - r["metrics"]["coco"]["AP"]) < 1e-9
    assert abs(B.coco_ap(cc, "<16") - r["metrics"]["AP_by_size"]["<16"]) < 1e-9


def test_paired_bootstrap_of_identical_methods_is_zero() -> None:
    cs = load_class_setting("visdrone10")
    coco, d = _synthetic(1)
    pc = B.port_cache(B.prepared(coco, d, cs), tuple(sorted(c["id"] for c in coco["categories"])))
    res = B.paired(lambda w: B.port_ap(pc, w), lambda w: B.port_ap(pc, w), len(coco["images"]), n=50)
    assert res["delta"] == 0 and res["ci95"] == [0.0, 0.0] and not res["excludes_zero"]
