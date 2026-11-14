# SPDX-License-Identifier: AGPL-3.0-or-later
"""Readable report of the box back-mapping check of all frozen rows (supplement Sec. 1: "the recomputed values are listed
with the code").

    python scripts/backmap_report.py report    # -> reports/BACKMAP_CHECK.md and reports/backmap_check.json

Inputs (shipped result files; computed in the research repo by scripts/mapcheck_all.py, scripts/mapcheck_heldout.py,
scripts/mapcheck_brg.py and, for the five rows of the original check, scahi/mapcheck.py):
runs/mapcheck/all/{inventory,check,heldout,serial,investigate_B-RG_yolov10m_2048}.json, runs/mapcheck/{visdrone,dota}.json.
Report mode only reads these files (no GPU, data, caches or weights).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IN = REPO / "runs/mapcheck/all"
OUT_MD = REPO / "reports/BACKMAP_CHECK.md"
OUT_JSON = REPO / "reports/backmap_check.json"
RULE = 0.05

MODEL = {"yolov10s": "YOLOv10s", "yolov10m": "YOLOv10m", "yolo11s": "YOLOv11s", "yolo11m": "YOLOv11m",
         "yolo12s": "YOLOv12s", "yolo12m": "YOLOv12m"}
TWO_HEADS = ("yolov10s", "yolov10m")
TRAIN = {"T0": "tile training, ignore regions dropped", "T1": "tile training, ignore regions painted grey",
         "T2": "tile training, loss-masked", "T3": "tile training, loss-masked + truncation ignore",
         "T4": "tile training, scale-normalised"}
LEVEL = {"w0.64": "whole-image pass at zoom 0.64", "w1.0": "whole-image pass at native resolution",
         "w1024": "1024 px whole-image pass", "w1280": "1280 px whole-image pass", "w1536": "1536 px whole-image pass",
         "w2048": "2048 px whole-image pass", "t2.0n1024": "zoom-2 tiles (1024 px input)",
         "t1.0n1024": "native-resolution tiles (1024 px input)"}
FUSION = {"F0-NMS": "box fusion (NMS)", "F0-GREEDYNMM": "box fusion (greedy NMM)", "F0-WBF": "box fusion (WBF)",
          "F2": "SCAHI, pre-specified setting", "F3e": "SCAHI with validity windows"}
DOTA_ROW = {"NMS-o2o": ("box fusion (NMS)", "o2o"), "NMS-o2m": ("box fusion (NMS)", "o2m"),
            "F2": ("SCAHI, setting from VisDrone", "o2m"), "RG": ("SCAHI, refined grid", "o2m"),
            "B-F2": ("SCAHI, setting from VisDrone", "o2m"), "B-RG": ("SCAHI, refined grid", "o2m")}
DETAIL = {"size_rule": "size rule", "letterbox_gain": "letterbox factor", "back_mapping": "per-axis back-mapping"}


def _head(model: str, branch: str) -> str:
    if model not in TWO_HEADS:
        return ""
    return ", one-to-one head" if branch == "o2o" else ", one-to-many head"


def describe(x: dict) -> dict:
    """Plain description of a row: dataset, detector / weights, passes, fusion, head."""
    i, sp = x["id"], x["spec"]
    if x["ds"] == "dota":
        row, m, s = re.match(r"(.+)_(yolo\w+?)_(\d+)$", i).groups()
        fus, br = DOTA_ROW[row]
        passes = f"{s} px whole-image pass" + ("" if row.startswith("B-") else " + native-resolution tiles")
        return {"dataset": "DOTA v1.5", "detector": MODEL[m], "weights": f"trained at {s} px",
                "configuration": f"{passes}, {fus}{_head(m, br)}", "metric": "mAP50 (devkit, VOC07)"}
    w = x["weights"]
    if "/runs/ext/" in "/" + w:
        m, s = re.search(r"train/(\w+?)_(\d+)_s0", w).groups()
        det, wt = MODEL[m], f"trained at {s} px"
    else:
        det = "YOLOv10s"
        mt = re.search(r"_(BFULL(\d+)|P3(T\d)_train_s(\d))", w)
        if mt.group(2):
            wt = f"single-pass training at {mt.group(2)} px"
        else:
            wt = TRAIN[mt.group(3)] + ("" if mt.group(4) == "0" else f", seed {mt.group(4)}")
        m = "yolov10s"
    passes = " + ".join(LEVEL[lv] for lv in sp["levels"])
    fus = "SCAHI, refined grid" if i.startswith("XG_") else FUSION[sp["variant"]]
    return {"dataset": "VisDrone", "detector": det, "weights": wt,
            "configuration": f"{passes}, {fus}{_head(m, sp['branch'])}", "metric": "AP@[.5:.95] (toolkit protocol)"}


def role(x: dict) -> str:
    if x["ds"] == "dota":  # main paper: Tab. 3 (box fusion one-to-many head, SCAHI refined grid), ablation (F2 at 2048)
        g = x["group"]
        if g.startswith("B-"):
            return "decomposition (no verdict)"
        return "main paper" if g in ("NMS-o2m", "RG") or (g == "F2" and x["id"].endswith("_2048")) else \
            "pre-specified row"
    if x["main_paper"] or x["id"] == "X1":  # X1: Tab. 2 "2048 + tiles, box fusion" (covered by the original check)
        return "main paper"
    g = x["group"]
    if g in ("prereg", "preregistered", "reference"):
        return "pre-specified row"
    if g == "seed":
        return "seed repeat"
    return "exploratory (no verdict)"


def _f(v, nd=4, sign=False):
    if v is not None and abs(v) < 0.5 * 10 ** -nd:
        v = 0.0  # no negative zero in the tables
    return "–" if v is None else (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}")


def report() -> None:
    inv = json.loads((IN / "inventory.json").read_text())
    chk = json.loads((IN / "check.json").read_text())["rows"]
    ho = json.loads((IN / "heldout.json").read_text())["rows"]
    ser = json.loads((IN / "serial.json").read_text())
    brg = json.loads((IN / "investigate_B-RG_yolov10m_2048.json").read_text())
    orig = {r["id"]: r for ds in ("visdrone", "dota")
            for r in json.loads((REPO / f"runs/mapcheck/{ds}.json").read_text())["rows"]}
    rows = []
    for x in inv["rows"]:
        d = describe(x)
        if x["covered_by_original_check"]:
            o = orig[x["id"]]
            mk = "AP" if x["ds"] == "visdrone" else "mAP50"
            fz, al = o["frozen"][mk], o["aligned"][mk]
            src = "original check (2026-11-03)"
        else:
            c = chk[x["id"]]
            fz, al, src = c["frozen"]["value"], c["aligned"]["value"], "this check"
        r = {"id": x["id"], **d, "role": role(x), "calib_frozen": fz, "calib_recomputed": al, "delta": al - fz,
             "at_or_above_0.05": abs(al - fz) >= RULE, "computed_in": src,
             "per_detail": chk.get(x["id"], {}).get("delta_per_detail") or None}
        if x["id"] in ho:
            r["held_out"] = {sp: {k: v[k] for k in ("original_stored", "corrected", "delta")}
                             for sp, v in ho[x["id"]]["splits"].items()}
        rows.append(r)
    key = lambda r: (r["dataset"], r["detector"], r["configuration"], r["weights"])  # noqa: E731
    rows.sort(key=key)
    big = [r for r in rows if r["at_or_above_0.05"]]
    mainver = [r for r in rows if r["role"] in ("main paper", "pre-specified row")]
    summ = {"n_rows": len(rows), "n_original_check": sum(r["computed_in"] != "this check" for r in rows),
            "n_at_or_above_0.05": len(big),
            "max_abs_delta_main_paper_or_pre_specified": max(abs(r["delta"]) for r in mainver),
            "max_abs_delta_held_out_rows_at_or_above": max(abs(v["delta"]) for r in big for v in r["held_out"].values()),
            "serial_cross_check_identical": all(v[t]["identical"] for v in ser.values() for t in ("frozen", "aligned"))}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1) + "\n")

    L = ["# Box back-mapping check of all frozen configurations", "",
         "Descriptive check after the held-out evaluations. Frozen values, selections and verdicts are kept; this report "
         "lists the recomputed values.", "",
         "## What was checked", "",
         "The cached whole-image passes map boxes from the network input back to the image with one scale factor. "
         "Ultralytics' own predictor differs in three details: (1) **size rule** - it rounds the resized image side "
         "before padding to a multiple of 32; (2) **letterbox factor** - it resizes with min(L/h, L/w) for the target "
         "size L; (3) **per-axis back-mapping** - it divides each axis by its own effective factor (rounded side / "
         "side). The detections differ by fractions of a pixel, and only for images whose resized side is rounded "
         "(on VisDrone at 1280 px: the 1400 x 788 images). Tiles of 512 px scaled to 1024 px are not affected on "
         "VisDrone (recomputed tiles are bit-identical); on DOTA some edge tiles differ.", "",
         "## Rule (fixed before any number of the check existed)", "",
         "Each configuration is recomputed on the calibration split with all three details aligned to Ultralytics at "
         "once, without re-tuning. If the change is below 0.05 (VisDrone: AP@[.5:.95]; DOTA: mAP50), nothing changes and "
         "the size of the deviation is documented. If it is 0.05 or larger, the configuration is recomputed on the "
         "held-out splits (VisDrone val and test-dev, DOTA-val) with the aligned conventions, again without re-tuning, "
         "and original and corrected values are reported side by side; pre-specified verdicts stay as originally "
         "computed. Only for configurations reaching 0.05 is the contribution of each detail reported separately.", "",
         "## Scope", "",
         f"- {len(rows)} frozen configurations whose detections rest on a cached whole-image pass (box fusion or SCAHI "
         f"on the whole-image pass with or without tiles); {summ['n_original_check']} of them were already covered by "
         "the original check.",
         "- Not affected and therefore not recomputed: single passes, multi-zoom inference with the predictor and SAHI "
         "(the Ultralytics predictor maps boxes itself); SCAHI on SAHI's passes on DOTA (its caches were built with "
         "Ultralytics' preparation); FCOS (torchvision resizes and maps back per axis itself).",
         "- Every configuration is fused twice through the same code path as its frozen evaluation; only the source of "
         "the cached passes differs. The value from the frozen caches reproduces the stored calibration value exactly "
         "for every configuration that has one. DOTA values on calib use the calibration selection limit of 20 000 "
         "detections per image (as stored), DOTA-val values the evaluation limit of 200 000.",
         "- Parallel computation was cross-checked: one configuration per study recomputed in a single process gave "
         f"bit-identical values ({'yes' if summ['serial_cross_check_identical'] else 'NO'}).", "",
         "## Summary", "",
         f"- Configurations behind numbers of the main paper or a pre-specified verdict: largest change "
         f"{summ['max_abs_delta_main_paper_or_pre_specified']:.4f} (below 0.05).",
         f"- {len(big)} exploratory or decomposition configurations reach 0.05 on calib; on the held-out splits they "
         f"change by at most {summ['max_abs_delta_held_out_rows_at_or_above']:.3f}. None of them enters a verdict.",
         "- Frozen values are kept in all tables; the corrected values are listed below.", "",
         "## Configurations reaching 0.05 on calib: held-out values", "",
         "| dataset | detector | weights | configuration | calib change | split | frozen | recomputed | change |",
         "|---|---|---|---|---:|---|---:|---:|---:|"]
    for r in big:
        for sp, v in r["held_out"].items():
            L.append(f"| {r['dataset']} | {r['detector']} | {r['weights']} | {r['configuration']} | "
                     f"{_f(r['delta'], sign=True)} | {sp} | {v['original_stored']:.2f} | {v['corrected']:.2f} | "
                     f"{_f(v['delta'], 3, True)} |")
    L += ["", "VisDrone val contains no 1400 x 788 image, so the 1280 px configurations are unchanged on val.", "",
          "Contribution of each detail on calib (one detail aligned at a time, the others as frozen):", "",
          "| dataset | detector | configuration | all three | size rule | letterbox factor | per-axis back-mapping |",
          "|---|---|---|---:|---:|---:|---:|"]
    for r in big:
        p = r["per_detail"]
        L.append(f"| {r['dataset']} | {r['detector']} | {r['configuration']} | {_f(r['delta'], sign=True)} | "
                 + " | ".join(_f(p[k], sign=True) for k in DETAIL) + " |")
    b = brg
    top = b["per_image_swap"]["top"][0]
    L += ["", "**DOTA, YOLOv10m, 2048 px pass alone + SCAHI refined grid (calib change "
          f"{b['delta']:+.3f}).** Almost the whole change sits in one class and one image: helicopter AP50 "
          f"{b['per_class_AP50']['helicopter']['frozen']:.2f} -> {b['per_class_AP50']['helicopter']['aligned']:.2f} "
          "(the calibration split has 4 helicopter objects); replacing the detections of one image at a time, image "
          f"{top['image']} alone accounts for {top['delta']:+.3f}. There the highest-scoring helicopter detection has an "
          "IoU of 0.500021 with a ground-truth box (true positive) on the frozen pass and 0.499968 (false positive) after "
          "a shift of 0.37 px, which moves three of the 11 interpolation points of the VOC07 precision curve. The "
          f"detection limit is not involved (change {b['limit']['delta_limit1e7']:+.3f} with a limit of 10 000 000 and "
          f"{b['limit']['delta_without_at_limit_images']:+.3f} without the images at the limit). On DOTA-val the "
          "configuration changes by "
          f"{next(r for r in big if r['id'] == 'B-RG_yolov10m_2048')['held_out']['val']['delta']:+.3f}.", ""]
    for ds in ("VisDrone", "DOTA v1.5"):
        L += [f"## All configurations, {ds} (calib)", "",
              "| detector | weights | configuration | used in | frozen | recomputed | change |",
              "|---|---|---|---|---:|---:|---:|"]
        for r in rows:
            if r["dataset"] != ds:
                continue
            mark = " (original check)" if r["computed_in"] != "this check" else ""
            L.append(f"| {r['detector']} | {r['weights']} | {r['configuration']} | {r['role']}{mark} | "
                     f"{r['calib_frozen']:.4f} | {r['calib_recomputed']:.4f} | {_f(r['delta'], sign=True)}"
                     f"{' **>= 0.05**' if r['at_or_above_0.05'] else ''} |")
        L.append("")
    L += ["Metrics: VisDrone AP@[.5:.95] with the toolkit protocol (ignored regions, at most 500 detections per image); "
          "DOTA mAP50 with the devkit (VOC07, no cap). Data: reports/backmap_check.json.", ""]
    OUT_MD.write_text("\n".join(L))
    print(f"{OUT_MD.relative_to(REPO)}: {len(rows)} configurations, {len(big)} >= 0.05")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
