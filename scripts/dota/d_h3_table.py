# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compact D-H3 table (tiles + SCAHI, refined grid − single pass, both at S_max) under four metrics, all four models, read
from runs/dota/val/results.json and results_new.json (addendum 2: YOLO11s, YOLO12s; paired bootstrap over DOTA-val images, 1000 resamples, seed 0, percentile 95 % CI; no new
evaluation).

    python scripts/dota/d_h3_table.py   -> reports/dota/tab_dota_dh3_metrics.{csv,md}
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports/dota/tab_dota_dh3_metrics"
METRICS = (("mAP50", "mAP50, primary (difficult ignored, no cap)"), ("mAP50_all_objects", "mAP50, all objects"),
           ("mAP50_cap4000", "mAP50, cap 4000"), ("AP", "AP@[.5:.95]"))
NAME = {"yolov10s": "YOLOv10s", "yolov10m": "YOLOv10m", "yolo11s": "YOLOv11s", "yolo12s": "YOLOv12s", "fcos": "FCOS"}
FCOS = REPO / "runs/dota/val/results_fcos.json"  # addendum 3, added once it exists


def main() -> None:
    res = json.loads((REPO / "runs/dota/val/results.json").read_text())
    new = json.loads((REPO / "runs/dota/val/results_new.json").read_text())
    smax = new["s_max"]
    recs = []
    models = [(m, per) for r in (res, new) for m, per in r["hypotheses"]["D-H3"]["models"].items()]
    if FCOS.exists():
        fc = json.loads(FCOS.read_text())
        smax = {**smax, fc["model"]: fc["s_max"]}
        models.append((fc["model"], fc["hypotheses"]["D-H3"]))
    for m, per in models:
        for k, label in METRICS:
            x = per[k]
            recs.append({"detector": NAME[m], "size_s_max": smax[m], "metric": label, "single_pass": x["b"],
                         "scahi_refined": x["a"], "delta": x["delta"], "ci_lo": x["ci95"][0], "ci_hi": x["ci95"][1],
                         "verdict_metric": k == "mAP50"})
    df = pd.DataFrame(recs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT.with_suffix(".csv"), index=False, float_format="%.4f")
    L = ["# DOTA-val, Q3 (D-H3): tiles + SCAHI, refined grid − single pass, both at S_max", "",
         "Contrast RG_{S_max} − A_{S_max} per detector; S_max = largest training size that passes the memory smoke "
         "(2048; **YOLOv12s: 1024**, its 2048 training does not fit, so its D-H3 value is measured on the 1024 pass, where "
         "the single pass is weaker – not directly comparable with the 2048 values). YOLOv11s / YOLOv12s: addendum 2, "
         "values from runs/dota/val/results_new.json; single-head models use their only head throughout. "
         f"Paired bootstrap over {res['n_images']} DOTA-val images ({res['bootstrap']['n']} resamples, seed "
         f"{res['bootstrap']['seed']}, {res['bootstrap']['ci']}); YOLOv10s / YOLOv10m from runs/dota/val/results.json. "
         + ("FCOS (addendum 3, one head, S_max = 2048): runs/dota/val/results_fcos.json. " if FCOS.exists() else "")
         + "Only the primary metric carries the preregistered verdict; the others are secondary, without verdict.", "",
         "| detector | size (S_max) | metric | single pass | tiles + SCAHI, refined grid | Δ | 95 % CI |",
         "|---|---:|---|---:|---:|---:|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.detector} | {r.size_s_max}{' (S_max = 1024)' if r.size_s_max == 1024 else ''} | {r.metric} | {r.single_pass:.2f} | {r.scahi_refined:.2f} | {r.delta:+.2f} | "
                 f"[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] |")
    OUT.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
