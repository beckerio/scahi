# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO26s / YOLO26m values of the paper, regenerated from the stored result files (no GPU, data or weights).

    python scripts/ext26_paper_values.py

Inputs (shipped result files): the freezes configs/ext/frozen_rows_ext26.json (YOLO26m) and frozen_rows_ext26s.json
(YOLO26s); the single held-out evaluation runs/ext/<model>/{val,testdev}/<row>/metrics.json; the verdicts
runs/ext/<model>/verdicts.json (paired bootstrap over the images, 1000 resamples, seed 0, eps 0.30; computed by
scripts/ext26_verdicts.py from the detections, which are not shipped); the single-pass GFLOPs of runs/ext/
scaling_points.json (val) and of the equal-compute comparison (val / test-dev, verdicts.json of YOLO26s).

Outputs in reports/ext26/:
  tab1_yolo26.csv         main paper Tab. 1, YOLO26 rows (val AP; Delta Q1 on val and test-dev)
  fig4_yolo26.csv         main paper Fig. 4 values (2048 px decomposition; AP on val and test-dev)
  equal_compute.csv       YOLO26s 2048 px vs. YOLO26m 1280 px single pass (AP, Delta with CI, GFLOPs per image)
  gflops_single.csv       single-pass GFLOPs per image on val (Fig. 3)
  tab_s_y26.csv           supplement Tab. s-y26 (verdicts per detector)
  single_pass_gate.csv    YOLO26s single-pass rows: calib tuning value (batch 16), batch-1 replay, difference
  YOLO26_PAPER_VALUES.md  all of the above as tables
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports/ext26"
MODELS = {"yolo26s": "configs/ext/frozen_rows_ext26s.json", "yolo26m": "configs/ext/frozen_rows_ext26.json"}
LABEL = {"yolo26s": "YOLO26s", "yolo26m": "YOLO26m"}
SPLITS = {"val": "val", "test-dev": "testdev"}
TAB1 = [("single pass 1280 (o2o)", "E_{m}_full1280_o2o"), ("single pass 2048 (o2o)", "E_{m}_full2048_o2o"),
        ("SAHI (o2o)", "E_{m}_sahi_o2o"), ("1280 px + tiles, box fusion (o2o; Q1 row)", "E_{m}_mz1280_nms-ios_o2o"),
        ("2048 px + tiles, SCAHI pre-specified (o2m)", "E_{m}_mz2048_F2_o2m"),
        ("2048 px + tiles, SCAHI refined grid, row C (o2m; exploratory)", "XG_{m}_2048_o2m_C")]
FIG4 = [("single pass 2048 (o2o)", "E_{m}_full2048_o2o"),
        ("SCAHI on the single pass alone, row D (o2m, pre-specified)", "D_{m}_single2048_F2_o2m"),
        ("2048 px + tiles, SCAHI pre-specified (o2m)", "E_{m}_mz2048_F2_o2m"),
        ("2048 px + tiles, refined grid, row C (o2m)", "XG_{m}_2048_o2m_C")]
STATUS = {"erfüllt": "met", "nicht erfüllt": "not met", "nicht prüfbar": "not testable"}  # verdict labels of the files
QUESTIONS = [("Q1", "E-H1"), ("Q2, 1280 px", "E-H2 s=1280"), ("Q2, 2048 px", "E-H2 s=2048"), ("Q3", "E-H3")]


def _j(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


def ap(m: str, row: str, split: str) -> float:
    return _j(f"runs/ext/{m}/{SPLITS[split]}/{row}/metrics.json")["metrics"]["AP"]


def _csv(name: str, header: list[str], rows: list[list]) -> None:
    with (OUT / name).open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def _f(x: float, nd: int = 2, sign: bool = False) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-0.00", "0.00") if not sign else s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ver = {m: _j(f"runs/ext/{m}/verdicts.json") for m in MODELS}
    L = ["# YOLO26s / YOLO26m: values of the paper (regenerated from the stored results)", "",
         "AP@[.5:.95], VisDrone toolkit protocol (ignored regions, at most 500 detections per image). Single held-out "
         "evaluation of the frozen rows (val once, exact val replay, test-dev once).", ""]

    # Tab. 1
    rows = []
    for m in MODELS:
        h = ver[m]["splits"]
        q1v, q1t = h["val"]["hypotheses"]["E-H1"], h["test-dev"]["hypotheses"]["E-H1"]
        rows.append([LABEL[m], *[_f(ap(m, r.format(m=m), "val")) for _, r in TAB1],
                     _f(q1v["delta"], sign=True), _f(q1t["delta"], sign=True)])
    hdr = ["detector", *[c for c, _ in TAB1], "Delta Q1 val", "Delta Q1 test-dev"]
    _csv("tab1_yolo26.csv", hdr, rows)
    L += ["## Main paper Tab. 1 (VisDrone val; Delta Q1 on val and test-dev)", "", "| " + " | ".join(hdr) + " |",
          "|" + "---|" * len(hdr)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]

    # Fig. 4
    rows = [[LABEL[m], c, _f(ap(m, r.format(m=m), "test-dev")), _f(ap(m, r.format(m=m), "val")), r.format(m=m)]
            for m in MODELS for c, r in FIG4]
    hdr = ["detector", "configuration", "AP test-dev", "AP val", "frozen row"]
    _csv("fig4_yolo26.csv", hdr, rows)
    L += ["## Main paper Fig. 4 (2048 px decomposition)", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    L += ["| " + " | ".join(r) + " |" for r in rows] + [""]

    # GFLOPs (Fig. 3) and equal-compute comparison
    sp = {(p["model"], p["imgsz"]): p for p in _j("runs/ext/scaling_points.json")["points"]}
    rows = [[LABEL[m], str(s), _f(sp[(m, s)]["gflops"]), _f(sp[(m, s)]["val_AP"])] for m in MODELS for s in (1280, 2048)]
    hdr = ["detector", "input size [px]", "GFLOPs per image (val)", "AP val (o2o)"]
    _csv("gflops_single.csv", hdr, rows)
    L += ["## Main paper Fig. 3: single-pass GFLOPs per image (val, exact per input shape)", "",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]
    ec = ver["yolo26s"]["equal_compute"]
    rows = []
    for split, x in ec["splits"].items():
        g = x["gflops_per_img"]
        rows.append([split, _f(x["a"]), _f(x["b"]), _f(x["delta"], sign=True), _f(x["ci95"][0], sign=True),
                     _f(x["ci95"][1], sign=True), _f(g[ec["row_a"]]), _f(g[ec["row_b"]]), _f(x["gflops_ratio_a_over_b"])])
    hdr = ["split", "YOLO26s 2048 AP", "YOLO26m 1280 AP", "Delta", "CI low", "CI high", "GFLOPs YOLO26s 2048",
           "GFLOPs YOLO26m 1280", "GFLOPs ratio"]
    _csv("equal_compute.csv", hdr, rows)
    L += ["## Equal-compute comparison (descriptive): single pass YOLO26s 2048 px vs. YOLO26m 1280 px, both o2o", "",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]

    # Tab. s-y26
    rows = []
    for q, k in QUESTIONS:
        r = [q]
        for m in MODELS:
            v, t = ver[m]["splits"]["val"]["hypotheses"][k], ver[m]["splits"]["test-dev"]["hypotheses"][k]
            r += [_f(v["delta"], sign=True), _f(v["ci95"][0]), _f(v["ci95"][1]), _f(t["delta"], sign=True), STATUS.get(v["status"], v["status"])]
        rows.append(r)
    hdr = ["question"] + [f"{LABEL[m]} {c}" for m in MODELS for c in ("val Delta", "CI low", "CI high",
                                                                        "test-dev Delta", "verdict (val)")]
    _csv("tab_s_y26.csv", hdr, rows)
    L += ["## Supplement Tab. s-y26 (Delta AP with 95 % CI on val, Delta on test-dev; verdict on val, eps 0.30; "
          "Q3 also needs CI low > 0)", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    L += ["| " + " | ".join(r) + " |" for r in rows] + [""]

    # YOLO26s single-pass gate (documented deviation)
    fz = _j(MODELS["yolo26s"])
    rows = [[r["id"], f"{r['calib_AP']:.4f}", f"{r['calib_AP_batch1']:.4f}", f"{r['calib_AP_batch1'] - r['calib_AP']:+.4f}"]
            for r in fz["rows"] if r["kind"] == "single"]
    hdr = ["row", "calib AP tuning (batch 16)", "calib AP replay (batch 1)", "difference"]
    _csv("single_pass_gate.csv", hdr, rows)
    L += ["## YOLO26s single-pass gate (not met as specified; choice unchanged, batch-1 values = reference)", "",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(r) + " |" for r in rows]
    L += ["", fz["single_pass_gate"]["decision"]]
    (OUT / "YOLO26_PAPER_VALUES.md").write_text("\n".join(L) + "\n")
    print("written to", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
