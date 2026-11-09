# SPDX-License-Identifier: AGPL-3.0-or-later
"""DOTA-val, all frozen rows of the four detectors in paper terms (YOLOv10s / YOLOv10m: runs/dota/val/results.json;
YOLO11s / YOLO12s, addendum 2: results_new.json; FCOS, addendum 3: results_fcos.json once it exists). No new evaluation.

    python scripts/dota/d_rows_table.py   -> reports/dota/tab_dota_rows_all_models.{csv,md}
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports/dota/tab_dota_rows_all_models"
NAME = {"yolov10s": "YOLOv10s", "yolov10m": "YOLOv10m", "yolo11s": "YOLOv11s", "yolo12s": "YOLOv12s", "fcos": "FCOS"}
FCOS = REPO / "runs/dota/val/results_fcos.json"
FROZEN_FCOS = REPO / "configs/dota/frozen_rows_fcos.json"  # inference size chosen for X_fcos_2048
TERM = {"A": "single pass", "X": "single pass of the 2048 weights, inference size selected on calib (2048 / 2560 / 3072; 2048 chosen)",
        "NMS-o2o": "tiles + box fusion (o2o head)", "NMS-o2m": "tiles + box fusion", "F2": "tiles + SCAHI",
        "RG": "tiles + SCAHI, refined grid", "B-F2": "SCAHI, single pass (no tiles)",
        "B-RG": "SCAHI, single pass (no tiles), refined grid", "SAHI": "SAHI"}
ORDER = list(TERM)


def main() -> None:
    recs = []
    val = REPO / "runs/dota/val"
    srcs = ((val / "results.json", ""), (val / "results_new.json", "addendum 2")) + (((FCOS, "addendum 3"),) if FCOS.exists() else ())
    x_fcos = (next(r["imgsz"] for r in json.loads(FROZEN_FCOS.read_text())["rows"] if r["row"] == "X")
              if FCOS.exists() else None)
    for f, add in srcs:
        res = json.loads(f.read_text())
        for rid, r in res["rows"].items():
            kind, m, *s = rid.split("_")
            p = r["primary"]
            method = TERM[kind] if (kind, m) != ("X", "fcos") else TERM[kind].replace("2048 chosen", f"{x_fcos} chosen")
            recs.append({"detector": NAME[m], "size": int(s[0]) if s else 1024, "method": method, "row_id": rid,
                         "mAP50": r["mAP50"], "mAP50_cap4000": r["mAP50_cap4000"],
                         "mAP50_all_objects": r["mAP50_all_objects"], "AP": p["AP"], "source": add or "original freeze"})
    df = pd.DataFrame(recs)
    df["_o"] = df.row_id.str.split("_").str[0].map(ORDER.index)
    df["_m"] = df.detector.map(list(NAME.values()).index)
    df = df.sort_values(["_m", "size", "_o"]).drop(columns=["_o", "_m"])
    df.to_csv(OUT.with_suffix(".csv"), index=False, float_format="%.4f")
    L = [f"# DOTA-val, all frozen rows of the {'five' if FCOS.exists() else 'four'} detectors", "",
         "Devkit port; primary mAP50 = difficult GT ignored, no cap; secondaries: cap 4000, all objects (difficult as "
         "positives), AP@[.5:.95]. Sizes are training = inference sizes (SAHI: 1024 weights). YOLOv12s: S_max = 1024 "
         "(no 2048 rows). YOLOv10: single pass and SAHI on the o2o head, SCAHI and box fusion on the o2m head (unless marked o2o); "
         "YOLOv11s / YOLOv12s have one (dense) head, used by every row."
         + (" FCOS (addendum 3): one head, score sqrt(p · centerness), no SAHI row; values from runs/dota/val/results_fcos.json."
            if FCOS.exists() else ""), "",
         "| detector | size | method | mAP50 | cap 4000 | all objects | AP@[.5:.95] | source |",
         "|---|---:|---|---:|---:|---:|---:|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.detector} | {r['size']} | {r.method} | {r.mAP50:.2f} | {r.mAP50_cap4000:.2f} | "
                 f"{r.mAP50_all_objects:.2f} | {r.AP:.2f} | {r.source} |")
    OUT.with_suffix(".md").write_text("\n".join(L) + "\n")
    print(f"{OUT}.md: {len(df)} rows")


if __name__ == "__main__":
    main()
