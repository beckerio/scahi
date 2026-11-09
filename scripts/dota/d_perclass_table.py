# SPDX-License-Identifier: AGPL-3.0-or-later
"""DOTA-val per-class mAP50 of the frozen rows at S_max (supplement table; values read from runs/dota/val/results.json and
results_new.json (addendum 2),
devkit port, primary protocol = difficult GT ignored, no cap).

    python scripts/dota/d_perclass_table.py

Rows: single pass 2048 (A), tiles + box fusion 2048 (NMS-o2m, same head as SCAHI; the Q2 comparator), tiles + SCAHI
pre-specified (F2) and refined grid (RG), all four models at S_max (2048; YOLO12s 1024). Output: reports/dota/tab_dota_perclass_map50.{csv,md}.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports/dota"
METHODS = (("A", "single pass"), ("NMS-o2m", "tiles + box fusion"), ("F2", "tiles + SCAHI"),
           ("RG", "tiles + SCAHI, refined grid"))
MODELS = (("yolov10s", "YOLOv10s"), ("yolov10m", "YOLOv10m"), ("yolo11s", "YOLOv11s"), ("yolo12s", "YOLOv12s"))
FCOS = REPO / "runs/dota/val/results_fcos.json"  # addendum 3, added once it exists


def main() -> None:
    res = json.loads((REPO / "runs/dota/val/results.json").read_text())
    new = json.loads((REPO / "runs/dota/val/results_new.json").read_text())
    rows = {**res["rows"], **new["rows"]}
    smax, models = new["s_max"], MODELS
    if FCOS.exists():
        fc = json.loads(FCOS.read_text())
        rows, smax, models = {**rows, **fc["rows"]}, {**smax, fc["model"]: fc["s_max"]}, MODELS + ((fc["model"], "FCOS"),)
    recs = []
    for m, mname in models:
        for key, term in METHODS:
            rid = f"{key}_{m}_{smax[m]}"
            p = rows[rid]["primary"]
            recs.append({"detector": mname, "size_s_max": smax[m], "method": term, "row_id": rid, **p["AP50_per_class"],
                         "mAP50": p["mAP50"]})
    df = pd.DataFrame(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "tab_dota_perclass_map50.csv", index=False, float_format="%.4f")
    cls = [c for c in df.columns if c not in ("detector", "size_s_max", "method", "row_id")]
    L = ["# DOTA-val per-class AP50, frozen rows at S_max", "",
         f"Devkit port, difficult GT ignored, no cap; {res['n_images']} images; values from runs/dota/val/results.json. "
         "Box fusion = class-wise NMS across passes on the o2m head (same head as SCAHI; YOLOv11s / YOLOv12s: their only head). "
         "S_max = 2048, except YOLOv12s: 1024 (its 2048 training does not fit; addendum 2). Best per class and detector in bold (ties at the shown precision all bold)."
         + (" FCOS (addendum 3, one head): runs/dota/val/results_fcos.json." if FCOS.exists() else ""), ""]
    for _, mname in models:
        t = df[df.detector == mname].set_index("method")[cls].T
        s = smax[dict(map(reversed, models))[mname]]
        L += [f"## {mname}, {s} px" + (" (S_max = 1024)" if s == 1024 else ""), "", "| class | " + " | ".join(t.columns) + " |", "|---|" + "---:|" * len(t.columns)]
        for c, r in t.iterrows():
            best = round(r.max(), 1)
            L.append(f"| {c} | " + " | ".join(f"**{v:.1f}**" if round(v, 1) == best else f"{v:.1f}" for v in r) + " |")
        L.append("")
    (OUT / "tab_dota_perclass_map50.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
