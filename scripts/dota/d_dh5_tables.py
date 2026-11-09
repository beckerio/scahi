# SPDX-License-Identifier: AGPL-3.0-or-later
"""DOTA-val tables of addendum 4 (D-H5: SCAHI on the passes of the frozen SAHI rows), values read from
runs/dota/val/results_dh5.json and configs/dota/frozen_rows_dh5.json (stored results only; nothing is evaluated).

    python scripts/dota/d_dh5_tables.py

Outputs (reports/dota/): tab_dota_dh5.md (all three tables) and
- tab_dota_dh5_rows.csv: main-table rows SAHI (frozen, stored evaluation), SAHI passes + SCAHI (F2) and
  SAHI passes + SCAHI, refined grid (RG), YOLOv10s / YOLOv10m; primary mAP50 and the secondaries.
- tab_dota_dh5_contrasts.csv: D-H5 (SAHI-RG − SAHI, Holm over the two models) and the secondary contrast
  SAHI-F2 − SAHI (no verdict), every stored metric with paired bootstrap CI.
- tab_dota_dh5_sizebins.csv: AP50 and AP@[.5:.95] per GT size bin (primary protocol; all-objects in the CSV).
- tab_dota_dh5_parameters.csv: SCAHI parameters selected on calib (frozen rows of addendum 4).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports/dota"
RES = "runs/dota/val/results_dh5.json"
MODELS = (("yolov10s", "YOLOv10s"), ("yolov10m", "YOLOv10m"))
METHODS = (("SAHI", "SAHI"), ("SAHI-F2", "SAHI passes + SCAHI"), ("SAHI-RG", "SAHI passes + SCAHI, refined grid"))
METRICS = (("mAP50", "mAP50"), ("mAP50_cap4000", "cap 4000"), ("mAP50_all_objects", "all objects"),
           ("AP", "AP@[.5:.95]"), ("mAP50_vehicles", "mAP50 vehicles"), ("R50_crowd", "R50 crowd"),
           ("R50_iso", "R50 isolated"))
BINS = ("<8", "8-16", "16-32", "32-64", "64-128", ">128")


def f2(v: float) -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"


def rows_table(res: dict) -> list[str]:
    recs = []
    for m, mname in MODELS:
        for key, term in METHODS:
            r = res["rows"][f"{key}_{m}"]
            recs.append({"detector": mname, "size": 1024, "method": term, "row_id": r["id"], "mAP50": r["primary"]["mAP50"],
                         "mAP50_cap4000": r["mAP50_cap4000"], "mAP50_all_objects": r["all_objects"]["mAP50"],
                         "AP": r["primary"]["AP"], "mAP50_vehicles": r["primary"]["mAP50_vehicles"],
                         "R50_crowd": r["R50_crowd"], "R50_iso": r["R50_iso"], "n_dets": r["n_dets"],
                         "images_over_cap": r["images_over_cap"],
                         "source": "original freeze (stored evaluation)" if key == "SAHI" else "addendum 4"})
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "tab_dota_dh5_rows.csv", index=False, float_format="%.4f")
    L = ["# DOTA-val, addendum 4 (D-H5): SCAHI on the passes of the frozen SAHI rows", "",
         f"Devkit port, {res['n_images']} images; primary mAP50 = difficult GT ignored, no cap; secondaries: cap 4000, "
         "all objects (difficult as positives), AP@[.5:.95]. 1024 weights of the frozen SAHI rows; SAHI's passes "
         "(640 px slices, frozen overlap 0.1 for YOLOv10s / 0.2 for YOLOv10m, plus SAHI's standard pass at long side "
         "640); SAHI on the o2o head, SCAHI on the o2m head of the same forward. SAHI rows: stored evaluation of the "
         f"original freeze (not re-evaluated). Pipeline limit {res['pipeline_limit']}. Values from {RES}.", "",
         "| detector | method | mAP50 | cap 4000 | all objects | AP@[.5:.95] | mAP50 vehicles | detections | images > cap |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, x in df.iterrows():
        L.append(f"| {x.detector} | {x.method} | {f2(x.mAP50)} | {f2(x.mAP50_cap4000)} | {f2(x.mAP50_all_objects)} | "
                 f"{f2(x.AP)} | {f2(x.mAP50_vehicles)} | {x.n_dets:,} | {x.images_over_cap} |")
    return L


def contrasts_table(res: dict) -> list[str]:
    b = res["bootstrap"]
    h = res["hypotheses"]["D-H5"]
    recs = []
    for name, src, kind in (("D-H5: " + h["contrast"], h["models"], "hypothesis"),
                            ("SAHI-F2 − SAHI", res["secondary"]["SAHI-F2 − SAHI"], "secondary (no verdict)")):
        for m, mname in MODELS:
            for key, term in METRICS:
                c = src[m][key]
                hm = src[m].get("holm", {}) if key == "mAP50" and kind == "hypothesis" else {}
                recs.append({"contrast": name, "kind": kind, "detector": mname, "metric": term, "a": c["a"], "b": c["b"],
                             "delta": c["delta"], "ci95_lo": c["ci95"][0], "ci95_hi": c["ci95"][1],
                             "p_one_sided": c["p_one_sided"], "p_holm": hm.get("p_holm"),
                             "met": src[m]["met"] if hm else None})
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "tab_dota_dh5_contrasts.csv", index=False, float_format="%.6g")
    L = ["", "## Contrasts (Δ [95 % CI], one-sided p)", "",
         f"Paired bootstrap over images ({b['n']} resamples, seed {b['seed']}, {b['ci']}); SAHI uses its stored "
         f"bootstrap weights. D-H5 criterion: Δ mAP50 > {res['eps']:.2f}, Holm over the two models, holds if met for "
         f"both → **{'holds' if h['holds'] else 'does not hold'}**. SAHI-F2 − SAHI: secondary, no verdict.", "",
         "| contrast | detector | metric | A | B | Δ [95 % CI] | p (one-sided) | Holm p | met |",
         "|---|---|---|---:|---:|---|---:|---:|---|"]
    for _, x in df.iterrows():
        holm = "" if x.p_holm is None or pd.isna(x.p_holm) else f"{x.p_holm:.3f}"
        met = "" if x.met is None or pd.isna(x.met) else ("yes" if x.met else "no")
        L.append(f"| {x.contrast} | {x.detector} | {x.metric} | {f2(x.a)} | {f2(x.b)} | {x.delta:+.2f} "
                 f"[{x.ci95_lo:+.2f}, {x.ci95_hi:+.2f}] | {x.p_one_sided:.3f} | {holm} | {met} |")
    return L


def sizebins_table(res: dict) -> list[str]:
    recs = []
    for m, mname in MODELS:
        for key, term in METHODS:
            r = res["rows"][f"{key}_{m}"]
            for proto, src in (("primary", r["primary"]), ("all objects", r["all_objects"])):
                for met, k in (("AP50", "AP50_by_size"), ("AP@[.5:.95]", "AP_by_size")):
                    recs.append({"detector": mname, "method": term, "row_id": r["id"], "protocol": proto, "metric": met,
                                 **{bn: src[k][bn] for bn in BINS}})
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "tab_dota_dh5_sizebins.csv", index=False, float_format="%.4f", na_rep="")
    L = ["", "## Size bins (GT box size in px, primary protocol)", "",
         "Bin < 8 px: not defined in the primary protocol (stored as NaN, shown as –; same for the frozen SAHI "
         "rows); all-objects values for every bin in tab_dota_dh5_sizebins.csv."]
    for met in ("AP50", "AP@[.5:.95]"):
        L += ["", f"### {met}", "", "| detector | method | " + " | ".join(BINS) + " |", "|---|---|" + "---:|" * len(BINS)]
        for _, x in df[(df.protocol == "primary") & (df.metric == met)].iterrows():
            L.append(f"| {x.detector} | {x.method} | " + " | ".join(f2(x[bn]) for bn in BINS) + " |")
    return L


def parameters_table() -> None:
    fr = json.loads((REPO / "configs/dota/frozen_rows_dh5.json").read_text())
    recs = [{"dataset": "DOTA", "detector": dict(MODELS)[r["model"]], "s": r["s"],
             "setting": "pre-specified" if r["row"] == "SAHI-F2" else "refined grid",
             "passes": f"SAHI passes (640 px slices, overlap {r['overlap']}, + standard pass)", "head": r["branch"],
             **{k: r["cfg"][k] for k in ("cell_factor", "bins_per_oct", "tau_splat", "tau_peak", "iou_cluster")},
             "calib_mAP50": r["calib_mAP50"], "code_row": r["id"]} for r in fr["rows"]]
    pd.DataFrame(recs).to_csv(OUT / "tab_dota_dh5_parameters.csv", index=False)


def main() -> None:
    res = json.loads((REPO / RES).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    L = rows_table(res) + contrasts_table(res) + sizebins_table(res)
    L += ["", "Latency of these rows: not measured yet (exclusive run on host-a pending)."]
    (OUT / "tab_dota_dh5.md").write_text("\n".join(L) + "\n")
    parameters_table()
    print("\n".join(L))


if __name__ == "__main__":
    main()
