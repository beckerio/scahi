# SPDX-License-Identifier: AGPL-3.0-or-later
"""Track B helpers: the calib tuning steps of the extension study that the generic pipeline CLI does not cover, and the
paired comparison behind the verdicts. All tuning on calib, caches with Ultralytics' input preparation.

    python tutorials/yolo26m_visdrone/paper_way.py tune-nms --weights W --stack native2048 --branch o2o
    python tutorials/yolo26m_visdrone/paper_way.py refined-grid --weights W --stack native2048 --branch o2m --row C
    python tutorials/yolo26m_visdrone/paper_way.py compare --a A.parquet --b B.parquet --split val [--ci-lower]

tune-nms: box fusion of all passes, NMS with intersection over the smaller box (IOS), 12 points (threshold 0.3-0.8 x
conf 0.01 / 0.1), selection max AP on calib (the extension's grid; the generic ``tune --variant F0-NMS`` adds IOU and is
the main study's grid). The pre-specified SCAHI setting is tuned with ``aligned.py tune --variant F2`` (24 points).
refined-grid: the frozen refined-grid procedure (configs/ext/xgrid_frozen.json) for row B (whole-image pass alone) or
row C (whole-image pass + tiles): round 1 of the grid, fusion time of every variant relative to the pre-specified
setting measured in the same process, selection = max calib AP among variants within 2x that time. If the selection lies
on the edge of the grid and no variant hit the time limit, the frozen procedure extends the grid outwards (reported here;
not needed for YOLO26m: every case hit the limit in round 1). The refined grid is exploratory (post hoc on VisDrone).
compare: paired bootstrap over the images (1000 resamples, seed 0) of the toolkit AP; delta, 95 % CI and the verdict rule
of the pre-registration (delta > eps; with --ci-lower also the lower CI bound > 0, as for E-H3).
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import pandas as pd

import common as C

NMS_GRID = [{"metric": "IOS", "thr": t, "conf": c} for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8) for c in (0.01, 0.1)]
PARAMS = ("tau_splat", "tau_peak", "cell_factor", "bins_per_oct")
REF = {"iou_cluster": 0.5, "tau_splat": 0.005, "tau_peak": 0.001, "cell_factor": 5.0, "bins_per_oct": 5}
OUT = C.REPO / "out/yolo26m_tuning"


def _calib(weights: str, stack: str, branch: str, levels: int | None = None):
    from scahi import reproduce
    from scahi.fusion import run as R
    from scahi.infer.full import image_table

    coco = reproduce.split_coco("calib")
    imgs = image_table(coco)
    with C.ultralytics_caches():
        data = R.load_split(Path(weights), stack, "calib", 0.3, branch, list(imgs.stem))
    if levels:  # row B: the whole-image pass alone (first level of the stack)
        data = {s: (lv[:levels], W, H) for s, (lv, W, H) in data.items()}
    return coco, dict(zip(imgs.stem, imgs.image_id)), data


def _ap(coco, d) -> float:
    from scahi.config import load_class_setting
    from scahi.eval.evaluate import evaluate

    return evaluate(coco, d, load_class_setting(C.SETTING), "visdrone_ignore", port_only=True)["metrics"]["AP"]


def cmd_tune_nms(a) -> None:
    from scahi.fusion import run as R

    coco, s2i, data = _calib(a.weights, a.stack, a.branch)
    raw = {s: pd.concat([R.pass_dets(lv, a.branch, 0.001, W, H) for lv in lvls], ignore_index=True)
           for s, (lvls, W, H) in data.items()}
    res = [{**cfg, "AP": _ap(coco, R.fuse_split(data, s2i, a.branch, "F0-NMS", cfg, raw))} for cfg in NMS_GRID]
    best = max(res, key=lambda r: r["AP"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"nms_{a.stack}_{a.branch}.json").write_text(json.dumps({"grid": res, "best": best}, indent=1))
    print("best on calib:", json.dumps(best))


def cmd_refined_grid(a) -> None:
    from scahi.fusion import run as R

    fz = json.loads((C.REPO / "configs/ext/xgrid_frozen.json").read_text())
    grid, cap = fz["grid"][a.row], fz["time_cap_vs_F2"]
    coco, s2i, data = _calib(a.weights, a.stack, a.branch, levels=1 if a.row == "B" else None)

    def run(v):
        t0 = time.perf_counter()
        d = R.fuse_split(data, s2i, a.branch, "F2", v)
        return _ap(coco, d), (time.perf_counter() - t0) / len(data)

    ap_ref, t_ref = run(REF)
    rows = [{**REF, "AP": ap_ref, "time_ratio": 1.0}]
    for x in itertools.product(*(grid[p] for p in PARAMS)):
        v = {"iou_cluster": 0.5, **dict(zip(PARAMS, x))}
        ap_, t = run(v)
        rows.append({**v, "AP": ap_, "time_ratio": t / t_ref})
        print(f"  {x}: AP {ap_:.3f}, time {t / t_ref:.2f}x", flush=True)
    ok = [r for r in rows if r["time_ratio"] <= cap] or rows
    sel = max(ok, key=lambda r: r["AP"])
    capped = any(r["time_ratio"] > cap for r in rows)
    outer = {p: (min if p.startswith("tau") else max)(float(v) for v in grid[p]) for p in PARAMS}  # outward direction
    edges = [p for p in PARAMS if len(grid[p]) > 1 and float(sel[p]) == outer[p]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"refined_{a.row}_{a.stack}_{a.branch}.json").write_text(json.dumps(
        {"rows": rows, "selected": sel, "time_cap_hit": capped, "outer_edges": edges}, indent=1))
    print("selected on calib:", json.dumps({p: sel[p] for p in PARAMS}), f"AP {sel['AP']:.3f}",
          f"(pre-specified setting {ap_ref:.3f}); time limit hit: {capped}; outer edges: {edges or 'none'}")
    if edges and not capped:
        print("selection on the outer edge without hitting the time limit: the frozen procedure adds an outward round "
              "(configs/ext/xgrid_frozen.json, 'rounds'); extend the grid accordingly")


def _vec(path: str, split: str):
    from scahi.config import load_class_setting
    from scahi.eval import bootstrap as B

    coco, _ = C.split_data(split)
    cats = tuple(sorted(c["id"] for c in coco["categories"]))
    ims = B.prepared(coco, pd.read_parquet(path), load_class_setting(C.SETTING))
    pc = B.port_cache(ims, cats)
    W = B.resample_weights(len(coco["images"]))
    import numpy as np

    return B.port_ap(pc), np.array([B.port_ap(pc, w) for w in W])


def cmd_compare(a) -> None:
    import numpy as np

    a0, va = _vec(a.a, a.split)
    b0, vb = _vec(a.b, a.split)
    d = va - vb
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    met = (a0 - b0) > a.eps and (lo > 0 if a.ci_lower else True)
    print(json.dumps({"AP_a": a0, "AP_b": b0, "delta": a0 - b0, "ci95": [lo, hi], "eps": a.eps,
                      "rule": "delta > eps" + (" and lower CI bound > 0" if a.ci_lower else ""), "met": bool(met)},
                     indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tune-nms", "refined-grid"):
        p = sub.add_parser(name)
        p.add_argument("--weights", required=True)
        p.add_argument("--stack", required=True, choices=("native", "native2048"))
        p.add_argument("--branch", default="o2m", choices=("o2m", "o2o"))
        if name == "refined-grid":
            p.add_argument("--row", required=True, choices=("B", "C"))
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--split", default="val", choices=("val", "test-dev"))
    p.add_argument("--eps", type=float, default=0.30)
    p.add_argument("--ci-lower", action="store_true")
    a = ap.parse_args()
    {"tune-nms": cmd_tune_nms, "refined-grid": cmd_refined_grid, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
