# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inputs of the supplement figures for YOLO26s / YOLO26m (exploratory, post hoc; descriptive, no verdict).

    python scripts/ext26_figure_data.py [--workers 6]     # -> runs/ext/yolo26_figdata.json

Computed from the stored detections of the single held-out evaluation (runs/ext/<model>/{val,testdev}/<row>/), with the
functions used for the five extension detectors:
  decomposition (fig_ext_waterfall): per model and N, main branch o2m, A = single pass (o2o, as for the two-branch
    YOLOv10m), B = SCAHI on the single pass alone (row D), C = pass + tiles with the pre-specified SCAHI setting, XG =
    refined grid row C; steps B - A, C - B, XG - C with paired image-bootstrap 95 % CI (1000 resamples, seed 0; the
    statistics of scripts/testdev_analysis.py) on val and test-dev;
  recall (fig_ext_recall): recall@0.5 of crowded / isolated ground truth on val, all detections (the per-image rule of
    scripts/ext_posthoc.py: crowded = a same-class neighbour with IoU > 0.3), for A, B, C and XG.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from testdev_analysis import _d, _row_stats  # noqa: E402

OUT = REPO / "runs/ext/yolo26_figdata.json"
MODELS = ("yolo26s", "yolo26m")
SIZES = (1280, 2048)
SUB = {"val": "val", "test-dev": "testdev"}


def rows(m: str, sz: int) -> dict[str, str]:
    return {"A": f"E_{m}_full{sz}_o2o", "B": f"D_{m}_single{sz}_F2_o2m", "C": f"E_{m}_mz{sz}_F2_o2m",
            "XG": f"XG_{m}_{sz}_o2m_C"}


def _stats(args):
    split, m, rid = args
    _, _, o = _row_stats((split, rid, str(REPO / f"runs/ext/{m}/{SUB[split]}/{rid}"), False))
    return (split, rid), {"AP0": o["AP0"], "AP_vec": o["AP_vec"]}


def _recall(args):
    import ext_posthoc as P

    from scahi.config import load_class_setting
    from scahi.eval import bootstrap as Bt
    from scahi.eval.evaluate import load_coco_gt

    m, rid = args
    coco = load_coco_gt(REPO / "data/canonical/visdrone10/coco/val.json")
    ims = Bt.prepared(coco, pd.read_parquet(REPO / f"runs/ext/{m}/val/{rid}/dets_val.parquet"),
                      load_class_setting("visdrone10"))
    per = [P.per_image(im, False) for im in ims]
    return rid, {g: 100.0 * sum(i[f"m_{g}"] for i in per) / sum(i[f"n_{g}"] for i in per) for g in ("crowd", "iso")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from scahi import ext as E

    jobs = sorted({(sp, m, r) for m in MODELS for sz in SIZES for r in rows(m, sz).values() for sp in SUB})
    with ProcessPoolExecutor(a.workers, mp_context=E._spawn()) as ex:
        st = dict(ex.map(_stats, jobs))
        rec = dict(ex.map(_recall, sorted({(m, r) for m in MODELS for sz in SIZES for r in rows(m, sz).values()})))
    cs = []
    for m in MODELS:
        for sz in SIZES:
            r = rows(m, sz)
            for fam, name, x, y in (("decomposition", "B − A", "B", "A"), ("decomposition", "C − B", "C", "B"),
                                    ("X-grid", "C − F2 C", "XG", "C")):
                c = {"family": fam, "model": m, "s": sz, "branch": "o2m", "contrast": name, "role": "main",
                     "a": r[x], "b": r[y], "td_a": r[x], "td_b": r[y]}
                for sp in SUB:
                    sa, sb = st[(sp, r[x])], st[(sp, r[y])]
                    d = _d(sa["AP0"], sb["AP0"], sa["AP_vec"], sb["AP_vec"])
                    d.pop("excludes_zero")
                    c[sp] = d
                cs.append(c)
    recall = {str(sz): {m: {g: {k: rec[rid][g] for k, rid in rows(m, sz).items()} for g in ("crowd", "iso")}
                        for m in MODELS} for sz in SIZES}
    OUT.write_text(json.dumps({"label": "exploratory, post hoc (YOLO26 addenda; figure inputs)",
                               "procedure": "paired image bootstrap, 1000 resamples, seed 0 (scripts/testdev_analysis.py);"
                                            " recall@0.5 crowded / isolated GT as scripts/ext_posthoc.py, val, all "
                                            "detections",
                               "rows": {f"{m}/{sz}": rows(m, sz) for m in MODELS for sz in SIZES},
                               "contrasts": cs, "recall50_all": recall}, indent=1, default=float))
    print(f"wrote {OUT.relative_to(REPO)}: {len(cs)} contrasts")


if __name__ == "__main__":
    main()
