# SPDX-License-Identifier: AGPL-3.0-or-later
"""DOTA-val tables of addendum 5 (D-H6: SAHI and SCAHI on SAHI's passes for the single-head detectors YOLO11s /
YOLO12s), values read from runs/dota/val/results_a5.json, configs/dota/frozen_rows_a5.json, the calib tuning and the
gate files (stored results only; nothing is evaluated). Same layout as scripts/dota/d_dh5_tables.py.

    python scripts/dota/d_a5_tables.py

Outputs (reports/dota/): tab_dota_a5.md (all tables) and tab_dota_a5_{rows,contrasts,sizebins,parameters,
sensitivity}.csv.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports/dota"
RES = "runs/dota/val/results_a5.json"
MODELS = (("yolo11s", "YOLOv11s"), ("yolo12s", "YOLOv12s"))
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
                         "source": "addendum 5"})
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "tab_dota_a5_rows.csv", index=False, float_format="%.4f")
    L = ["# DOTA-val, addendum 5 (D-H6): SAHI vs. SCAHI on SAHI's passes, single-head detectors", "",
         f"Devkit port, {res['n_images']} images; primary mAP50 = difficult GT ignored, no cap; secondaries: cap 4000, "
         "all objects (difficult as positives), AP@[.5:.95]. 1024 weights (YOLOv11s, YOLOv12s; one dense head). SAHI "
         "tuned on calib with the procedure and grid of the frozen YOLOv10 SAHI rows (selected for both: overlap 0.2, "
         "NMS / IOU 0.4, conf 0.01; pass postprocess Ultralytics NMS, IoU 0.7); SCAHI rows on exactly the passes of the "
         "selected SAHI configuration (640 px slices + SAHI's standard pass at long side 640), same head. All six rows "
         f"frozen and approved {res['approved_at']}, evaluated once. Pipeline limit of the SCAHI rows "
         f"{res['pipeline_limit']}. Values from {RES}.", "",
         "| detector | method | mAP50 | cap 4000 | all objects | AP@[.5:.95] | mAP50 vehicles | detections | images > cap |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, x in df.iterrows():
        L.append(f"| {x.detector} | {x.method} | {f2(x.mAP50)} | {f2(x.mAP50_cap4000)} | {f2(x.mAP50_all_objects)} | "
                 f"{f2(x.AP)} | {f2(x.mAP50_vehicles)} | {x.n_dets:,} | {x.images_over_cap} |")
    rows = res["rows"]
    L += ["", f"Images at the pipeline limit {res['pipeline_limit']:,}: "
          + ", ".join(f"{r} {rows[r]['images_at_limit']}" for r in rows) + ". SAHI rows have no per-image limit; "
          "images with >= 20,000 detections (informational; limit 200,000 not reached, no truncation): "
          + ", ".join(f"{r} {rows[r]['images_ge_20000_dets']} (max {rows[r]['max_dets_per_image']:,} per image)"
                      for r in rows if "images_ge_20000_dets" in rows[r]) + "."]
    return L


def contrasts_table(res: dict) -> list[str]:
    b = res["bootstrap"]
    h = res["hypotheses"]["D-H6"]
    recs = []
    for name, src, kind in (("D-H6: " + h["contrast"], h["models"], "hypothesis"),
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
    df.to_csv(OUT / "tab_dota_a5_contrasts.csv", index=False, float_format="%.6g")
    L = ["", "## Contrasts (Δ [95 % CI], one-sided p)", "",
         f"Paired bootstrap over images ({b['n']} resamples, seed {b['seed']}, {b['ci']}). D-H6 criterion: Δ mAP50 > "
         f"{res['eps']:.2f}, Holm over YOLOv11s and YOLOv12s, holds if met for both → "
         f"**{'holds' if h['holds'] else 'does not hold'}**. SAHI-F2 − SAHI: secondary, no verdict. The YOLOv10 D-H5 "
         "verdict (addendum 4) is unchanged.", "",
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
    df.to_csv(OUT / "tab_dota_a5_sizebins.csv", index=False, float_format="%.4f", na_rep="")
    L = ["", "## Size bins (GT box size in px, primary protocol)", "",
         "Bin < 8 px: not defined in the primary protocol (stored as NaN, shown as –); all-objects values for every "
         "bin in tab_dota_a5_sizebins.csv."]
    for met in ("AP50", "AP@[.5:.95]"):
        L += ["", f"### {met}", "", "| detector | method | " + " | ".join(BINS) + " |", "|---|---|" + "---:|" * len(BINS)]
        for _, x in df[(df.protocol == "primary") & (df.metric == met)].iterrows():
            L.append(f"| {x.detector} | {x.method} | " + " | ".join(f2(x[bn]) for bn in BINS) + " |")
    return L


def parameters_table() -> None:
    fr = json.loads((REPO / "configs/dota/frozen_rows_a5.json").read_text())
    recs = []
    for r in fr["rows"]:
        base = {"dataset": "DOTA", "detector": dict(MODELS)[r["model"]], "s": r["s"], "code_row": r["id"],
                "calib_mAP50": r["calib_mAP50"]}
        if r["kind"] == "a5_sahi":
            recs.append({**base, "setting": "SAHI (calib-tuned, grid v2)", **r["cfg"]})
        else:
            recs.append({**base, "setting": "pre-specified" if r["row"] == "SAHI-F2" else "refined grid",
                         "passes": f"SAHI passes (640 px slices, overlap {r['overlap']}, + standard pass)",
                         "head": "single dense head",
                         **{k: r["cfg"][k] for k in ("cell_factor", "bins_per_oct", "tau_splat", "tau_peak",
                                                     "iou_cluster")}})
    pd.DataFrame(recs).to_csv(OUT / "tab_dota_a5_parameters.csv", index=False)


def sensitivity_table() -> list[str]:
    """2x time rule: the selected RG setting and every variant with higher calib mAP50 whose time ratio is in [1.8, 2.2]
    (scahi.dota.time_rule, as in the freeze proposal); calib only."""
    import sys

    sys.path.insert(0, str(REPO))
    from scahi.dota import time_rule as TR

    recs = []
    for m, mname in MODELS:
        t = json.loads((REPO / f"runs/dota/fusion/{m}_1024/a5/tuning.json").read_text())
        sv = TR.sensitivity(t["RG"]["rows"])
        for kind, xs in (("selected", [sv["selected"]]), ("higher mAP50, ratio in [1.8, 2.2]", sv["higher_mAP50_in_band"])):
            for x in xs:
                recs.append({"detector": mname, "kind": kind, **{k: x[k] for k in ("tau_splat", "tau_peak", "cell_factor",
                             "bins_per_oct")}, "calib_mAP50": x["mAP50"], "time_ratio": x["time_ratio_vs_ref"],
                             "admissible": x["admissible"]})
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "tab_dota_a5_sensitivity.csv", index=False, float_format="%.6g")
    L = ["", "## Sensitivity of the 2x time rule (calib, report only; selection rule unchanged)", "",
         "Refined grid: the selected setting and every variant with higher calib mAP50 whose fusion time ratio vs. the "
         "reference lies in [1.8, 2.2] (timing noise at the cap about ±10 %).", "",
         "| detector | setting | τ_splat | τ_peak | cell | bins | calib mAP50 | time ratio | admissible |",
         "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for _, x in df.iterrows():
        L.append(f"| {x.detector} | {x.kind} | {x.tau_splat:g} | {x.tau_peak:g} | {x.cell_factor:g} | "
                 f"{x.bins_per_oct:g} | {x.calib_mAP50:.3f} | {x.time_ratio:.3f} | {'yes' if x.admissible else 'no'} |")
    return L


def raw_comparison() -> list[str]:
    L = ["", "## SAHI's own raw predictions vs. the SCAHI cache path (descriptive, no metric)", "",
         "Per image: pass detections as the cache path forms them vs. SAHI's raw predictions of the selected "
         "configuration (gate rule of addendum 5; the DOTA-val part was computed after the freeze approval).", "",
         "| detector | split | images | dets cache | dets SAHI | images same count | box px p99 (median) | box px max |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for m, mname in MODELS:
        g = json.loads((REPO / f"runs/dota/fusion/{m}_1024/sahi_scahi_gate.json").read_text())
        for sp in ("calib", "val"):
            x = g[f"raw_vs_frozen_{sp}"]
            L.append(f"| {mname} | {'DOTA-val' if sp == 'val' else sp} | {x['images']} | {x['dets_cache']:,} | "
                     f"{x['dets_frozen']:,} | {x['images_same_count']} | {x['box_px_p99_median']:.4f} | "
                     f"{x['box_px_max']:.2f} |")
    return L


def main() -> None:
    res = json.loads((REPO / RES).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    L = rows_table(res) + contrasts_table(res) + sizebins_table(res) + sensitivity_table() + raw_comparison()
    L += ["", "Latency of these rows: not measured (no exclusive window before the move to host-b; addendum 5, item 8)."]
    (OUT / "tab_dota_a5.md").write_text("\n".join(L) + "\n")
    parameters_table()
    print("\n".join(L))


if __name__ == "__main__":
    main()
