# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase D figures (DOTA-val, one run) from runs/dota/val/results.json (YOLOv10s / YOLOv10m) and results_new.json
(addendum 2: YOLO11s / YOLO12s) and the training curves; no new experiments. Four detectors since 2026-11-01 (host-a
brief, task 5); YOLO12s is trained only at 1024 px (largest trainable size, memory smoke) and marked so.

    python scripts/dota/d_figures.py

Style: scripts/figstyle.py (as the paper figures), paper terms only (no internal row labels): single pass, tiles + box
fusion, SCAHI, SCAHI refined grid, SCAHI whole image; questions Q1-Q3 (Q1 = resolution vs. tiling = D-H1, Q2 = fusion
with identical passes = D-H2a/b, Q3 = SCAHI vs. single pass = D-H3). Models in fixed categorical slots (YOLOv10s blue,
YOLOv10m orange, YOLO11s aqua, YOLO12s yellow = slots 1-4 of the reference palette) plus one marker shape per model and a
legend. The refined grid is labelled "pre-specified on DOTA" (on VisDrone it was
post hoc; user 2026-10-30); Q4 is not pre-registered on DOTA. Figures -> reports/figures/dota/ (PDF + PNG + CSV of the plotted values).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import figstyle as FS  # noqa: E402
from figstyle import AQUA, BLUE, FULL_W, INK2, MUTED, ORANGE  # noqa: E402

OUT = REPO / "reports/figures/dota"
VAL = REPO / "runs/dota/val/results.json"
VAL_NEW = REPO / "runs/dota/val/results_new.json"
YELLOW = "#eda100"  # categorical slot 4 of the reference palette (dataviz), after blue, orange, aqua
MODELS = ("yolov10s", "yolov10m", "yolo11s", "yolo12s")
NAME = {"yolov10s": "YOLOv10s", "yolov10m": "YOLOv10m", "yolo11s": "YOLOv11s", "yolo12s": "YOLOv12s (1024 px only)"}
COL = {"yolov10s": BLUE, "yolov10m": ORANGE, "yolo11s": AQUA, "yolo12s": YELLOW}
MARK = {"yolov10s": "o", "yolov10m": "s", "yolo11s": "^", "yolo12s": "v"}
ROWS = ("A", "NMS-o2o", "NMS-o2m", "F2", "RG", "B-F2", "B-RG")
PAPER = {"A": "single pass", "NMS-o2o": "tiles + box fusion (o2o)", "NMS-o2m": "tiles + box fusion (o2m)",
         "F2": "tiles + SCAHI", "RG": "tiles + SCAHI (refined grid,\npre-specified on DOTA)", "B-F2": "SCAHI, whole image only",
         "B-RG": "SCAHI, whole image only (refined\ngrid, pre-specified on DOTA)", "SAHI": "SAHI (1024 weights)"}
QUESTION = {"D-H1": ("Q1", "single pass 2048 vs.\ntiles + box fusion 1024"),
            "D-H2a": ("Q2", "SCAHI (refined grid, pre-specified\non DOTA) vs. box fusion, same passes, 1024"),
            "D-H2b": ("Q2", "SCAHI (refined grid, pre-specified\non DOTA) vs. box fusion, same passes, 2048"),
            "D-H3": ("Q3", "SCAHI (refined grid, pre-specified\non DOTA; tiles) vs. single pass, 2048")}


def save(fig, name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=FS.DPI_PNG)
    pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    plt.close(fig)
    print("wrote", name)


def hypotheses(R: dict) -> None:
    """Forest plot: Δ mAP50 with 95 % CI per hypothesis and model; lines at 0 and ε. Not testable (YOLO12s without
    2048 weights: Q1, Q2 at 2048) marked as such; YOLO12s Q3 is measured on its 1024 pass (S_max)."""
    hs = list(R["hypotheses"])
    fig, ax = plt.subplots(figsize=(FULL_W * 0.66, 0.78 * len(hs) + 0.7))
    rows = []
    off = {m: (j - 1.5) * 0.17 for j, m in enumerate(MODELS)}
    labelled: set[str] = set()
    for i, h in enumerate(hs):
        for m in MODELS:
            e = R["hypotheses"][h]["models"][m]
            y = i + off[m]
            if not e.get("testable", True):
                ax.text(0, y, f"  {NAME[m].split(' (')[0]}: not testable (no 2048 weights)", color=INK2, fontsize=5.6,
                        va="center")
                rows.append({"question": QUESTION[h][0], "contrast": QUESTION[h][1].replace("\n", " "), "model": m,
                             "delta": None, "ci_lo": None, "ci_hi": None, "note": "not testable"})
                continue
            c = e["mAP50"]
            ax.errorbar(c["delta"], y, xerr=[[c["delta"] - c["ci95"][0]], [c["ci95"][1] - c["delta"]]], fmt=MARK[m],
                        color=COL[m], ms=3.6, lw=1.1, capsize=0, label=None if m in labelled else NAME[m], zorder=3)
            labelled.add(m)
            note = "1024 pass (S_max)" if m == "yolo12s" and h == "D-H3" else ""
            if note:
                ax.text(c["ci95"][1], y, f"  {note}", color=INK2, fontsize=5.6, va="center")
            rows.append({"question": QUESTION[h][0], "contrast": QUESTION[h][1].replace("\n", " "), "model": m,
                         "delta": c["delta"], "ci_lo": c["ci95"][0], "ci_hi": c["ci95"][1], "note": note})
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(R["eps"], color=INK2, lw=0.8, ls="--")
    ax.text(R["eps"], -0.8, " ε", color=INK2, fontsize=7, va="center")
    ax.set_yticks(range(len(hs)), [f"{QUESTION[h][0]}: {QUESTION[h][1]}" for h in hs], fontsize=6.3)
    ax.set_ylim(len(hs) - 0.5, -1)
    ax.set_xlabel("Δ mAP50 on DOTA-val (points, 95 % CI)")
    h_, l_ = ax.get_legend_handles_labels()
    o = [l_.index(NAME[m]) for m in MODELS if NAME[m] in l_]
    ax.legend([h_[k] for k in o], [l_[k] for k in o], frameon=False, fontsize=6.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=4, handletextpad=0.3)
    save(fig, "fig_dota_hypotheses", rows)


def rows_plot(R: dict) -> None:
    """mAP50 of every frozen row; panels s = 1024 / 2048, models as coloured points (shape per model). Rows that do not
    exist for a model (single-head models: no o2o box fusion, no SAHI; YOLO12s: no 2048 weights) are left out."""
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, 2.35), sharey=True)
    rows = []
    off = {m: (j - 1.5) * 0.12 for j, m in enumerate(MODELS)}
    for ax, s in zip(axs, (1024, 2048)):
        for m in MODELS:
            pts = [(k, R["rows"][f"{r}_{m}_{s}"]["mAP50"]) for k, r in enumerate(ROWS) if f"{r}_{m}_{s}" in R["rows"]]
            if not pts:
                continue
            ax.plot([k + off[m] for k, _ in pts], [y for _, y in pts], MARK[m], color=COL[m], ms=4, label=NAME[m])
            rows += [{"s": s, "model": m, "method": PAPER[ROWS[k]].replace("\n", " "), "mAP50": y} for k, y in pts]
            if s == 1024 and f"SAHI_{m}" in R["rows"]:
                y = R["rows"][f"SAHI_{m}"]["mAP50"]
                ax.plot(len(ROWS) + off[m], y, MARK[m], color=COL[m], ms=4, mfc="white", mew=1.2)
                rows.append({"s": s, "model": m, "method": PAPER["SAHI"], "mAP50": y})
        lab = [PAPER[r] for r in ROWS] + ([PAPER["SAHI"]] if s == 1024 else [])
        ax.set_xticks(range(len(lab)), lab, rotation=40, ha="right", fontsize=6.5)
        ax.set_title(f"weights trained at {s} px", fontsize=8)
    axs[0].set_ylabel("mAP50 on DOTA-val")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.8, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=4)
    save(fig, "fig_dota_rows", rows)


def calib_curves() -> None:
    """calib mAP50 over epochs (model selection), panels by training size."""
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, 2.0), sharey=True)
    rows = []
    for ax, s in zip(axs, (1024, 2048)):
        for m in (x for x in MODELS if (REPO / f"runs/dota/train/{x}_{s}_s0").exists()):
            c = json.loads((REPO / f"runs/dota/train/{m}_{s}_s0/ultralytics/train/weights/calib_curve.json").read_text())
            ax.plot([x["epoch"] for x in c], [x["mAP50"] for x in c], "-", marker=MARK[m], ms=2.5, lw=1.2, color=COL[m],
                    label=NAME[m])
            rows += [{"s": s, "model": m, "epoch": x["epoch"], "mAP50": x["mAP50"]} for x in c]
        ax.set_title(f"training at {s} px", fontsize=8)
        ax.set_xlabel("epoch")
    axs[0].set_ylabel("mAP50 on calib")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.8, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=4)
    save(fig, "fig_dota_calib_curves", rows)


def main() -> None:
    FS.apply()
    R = json.loads(VAL.read_text())
    N = json.loads(VAL_NEW.read_text())  # addendum 2: rows and contrasts of YOLO11s / YOLO12s (same evaluation)
    R["rows"] = {**R["rows"], **N["rows"]}
    for h, v in N["hypotheses"].items():
        R["hypotheses"][h]["models"].update(v["models"])
    hypotheses(R)
    rows_plot(R)
    calib_curves()


if __name__ == "__main__":
    main()
