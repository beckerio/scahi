# SPDX-License-Identifier: AGPL-3.0-or-later
"""reports/ABLATION.md, one table + one figure per dataset (PDF + PNG + CSV) from runs/ablation/{visdrone,dota}.json.

    python scripts/ablation_report.py

Exploratory, post hoc (scripts/ablation_components.py); nothing here evaluates detections.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import figstyle as FS  # noqa: E402
from figstyle import AQUA, BLUE, FULL_W, INK2, MUTED, ORANGE  # noqa: E402

from ablation_components import ARM_TEXT, CF_SWEEP  # noqa: E402

FIG = REPO / "reports/figures/ablation"
OUT = REPO / "reports/ABLATION.md"
NAME = {"yolov10s": "YOLOv10s", "yolov10m": "YOLOv10m", "yolo11s": "YOLOv11s", "yolo12s": "YOLOv12s", "yolo11m": "YOLOv11m"}
DS = {"visdrone": {"title": "VisDrone (val)", "fig_title": "VisDrone val, 2048 px weights; refined grid post hoc", "primary": "AP", "prim_text": "AP (Toolkit-Port, = AP@[.5:.95])",
                   "second": "AP50", "split": "VisDrone-val"},
      "dota": {"title": "DOTA v1.5 HBB (DOTA-val, post hoc)",
               "fig_title": "DOTA v1.5 HBB, DOTA-val, 2048 px weights; refined grid pre-specified on DOTA", "primary": "mAP50", "prim_text": "mAP50 (Devkit, ohne Cap)",
               "second": "AP", "split": "DOTA-val"}}
ARM_ORDER = ("a", "b", "c", "d", "e")
PAPER_ARM = {"a": "(a) no coverage\nnormalisation", "b": "(b) no border\ntaper", "d": "(d) + validity\nwindows",
             "e": "(e) refined\npeak grid"}
CMP = ("a", "b", "d", "e")
COLORS = (BLUE, ORANGE, AQUA, INK2, MUTED)  # weight sets in fixed order (ink tones beyond the 3 validated slots,
#                                             always with a legend and direct labels in the table)
MARK = ("o", "s", "D", "^", "v")


def f(x, nd: int = 2) -> str:
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def ci(c: dict) -> str:
    return f"{c['delta']:+.2f} [{c['ci95'][0]:+.2f}; {c['ci95'][1]:+.2f}]".replace(".", ",")


def table(ds: str, R: dict) -> tuple[list[str], list[dict]]:
    p, s = DS[ds]["primary"], DS[ds]["second"]
    L = [f"| Gewichte | Arm | Variante | {p} | Δ {p} gegen (c) [95 %-KI] | {s} | Recall@0,5 crowded / isolated | "
         "Δ Recall crowded gegen (c) | Dup / Frag (%) |", "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for m, arms in R["arms"].items():
        for a in (*ARM_ORDER, *(f"cf{c:g}" for c in CF_SWEEP)):
            x = arms[a]
            vc = x.get("vs_c")
            L.append(f"| {NAME[m]} | {a} | {x['variant']}{' cf ' + f(x['cfg']['cell_factor'], 0) if a.startswith('cf') else ''} | "
                     f"{f(x[p])} | {ci(vc[p]) if vc else '(Referenz)'} | {f(x[s])} | "
                     f"{f(x['R50_crowd'], 1)} / {f(x['R50_iso'], 1)} | {ci(vc['R50_crowd']) if vc else '—'} | "
                     f"{f(100 * x['dup_rate'], 1)} / {f(100 * x['frag_rate'], 1)} |")
            rows.append({"dataset": ds, "weights": m, "arm": a, "variant": x["variant"], **{k: x["cfg"].get(k) for k in (
                "tau_splat", "tau_peak", "cell_factor", "bins_per_oct")}, p: x[p], s: x[s], "R50_crowd": x["R50_crowd"],
                "R50_iso": x["R50_iso"], "dup_rate": x["dup_rate"], "frag_rate": x["frag_rate"],
                **({f"d_{p}": vc[p]["delta"], f"d_{p}_lo": vc[p]["ci95"][0], f"d_{p}_hi": vc[p]["ci95"][1]} if vc else {})})
    return L, rows


def figure(ds: str, R: dict) -> None:
    p = DS[ds]["primary"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.4), gridspec_kw={"width_ratios": [1.15, 1]})
    models = list(R["arms"])
    n = len(models)
    for i, m in enumerate(models):
        arms = R["arms"][m]
        for j, a in enumerate(CMP):
            c = arms[a]["vs_c"][p]
            x = j + (i - (n - 1) / 2) * 0.12
            a1.errorbar(x, c["delta"], yerr=[[c["delta"] - c["ci95"][0]], [c["ci95"][1] - c["delta"]]], fmt=MARK[i],
                        color=COLORS[i], ms=3.5, lw=1.0, label=NAME[m] if j == 0 else None)
        xs = list(CF_SWEEP)
        a2.plot(xs, [arms[f"cf{c:g}"][p] - arms["c"][p] for c in xs], "-", marker=MARK[i], ms=3.5, lw=1.1,
                color=COLORS[i], label=NAME[m])
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_xticks(range(len(CMP)), [PAPER_ARM[a] for a in CMP], fontsize=6.3)
    a1.set_ylabel(f"Δ {p} vs. (c) (95 % CI)")
    a1.set_title("components vs. (c) taper + coverage normalisation", fontsize=7.5)
    lo = min(R["arms"][m][a]["vs_c"][p]["ci95"][0] for m in models for a in CMP if a != "d")
    dmin = min(R["arms"][m]["d"]["vs_c"][p]["ci95"][0] for m in models)
    if dmin < 3 * min(lo, -1.0):  # (d) far off the scale of the other arms (DOTA): clip and state the values
        a1.set_ylim(min(lo, -1.0) * 1.6, None)
        vals = " / ".join(f"{R['arms'][m]['d']['vs_c'][p]['delta']:+.1f}" for m in models)
        a1.annotate(f"(d) off scale:\n{vals}", (CMP.index("d"), min(lo, -1.0) * 1.55), ha="center", va="bottom",
                    fontsize=6, color=INK2)
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_xticks(list(CF_SWEEP))
    a2.set_xlabel("cell factor of (c) (other parameters fixed)")
    a2.set_ylabel(f"{p} − {p} of (c)")
    a2.set_title("cell-factor sweep", fontsize=7.5)
    a2.legend(frameon=False, fontsize=6.5, loc="best")
    fig.suptitle(f"{DS[ds]['fig_title']} (exploratory, post hoc)", fontsize=8.5, y=1.04)
    FIG.mkdir(parents=True, exist_ok=True)
    name = f"fig_ablation_{ds}"
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=FS.DPI_PNG)
    plt.close(fig)


def main() -> None:
    FS.apply()
    L = ["# SCAHI-Bausteine: Ablation (explorativ, post hoc)", "",
         "Nutzerauftrag 2026-10-30, nach den Hauptergebnissen; **nie Grundlage eines Urteils**. Nur CPU, aus den "
         "vorhandenen Caches (`scripts/ablation_components.py`). In allen Armen dieselben Pässe: Einzelpass bei der "
         "Trainingsgröße 2048 + Kacheln, Zweig o2m.", "",
         "| Arm | Variante | Bedeutung |", "|---|---|---|"]
    from ablation_components import ARMS

    for a in ARM_ORDER:
        L.append(f"| ({a}) | {ARMS[a]} | {ARM_TEXT[a]} |")
    L += ["",
          "- (a) behält die Rand-Taper-Gewichte und lässt nur die Division durch die Abdeckung C weg (Summe der gewichteten "
          "Scores über die Pässe), sodass (a) − (c) allein die Normalisierung misst.",
          "- Parameter von (a), (b), (d): auf calib aus dem vorab festgelegten F2-Raster (24 Konfigurationen) gewählt, "
          "Kriterium die Primärmetrik des Datensatzes, genau wie die bestehenden F2-Zeilen. (c) und (e): eingefrorene "
          "calib-Auswahl. Keine neue Tuning-Dimension.",
          "- cell_factor-Sweep: (c) mit cell_factor 5 / 7 / 9 / 11 / 13, übrige Parameter wie (c), ohne Auswahl.",
          "- Gepaarter Bootstrap über Bilder (1000 Resamples, Seed 0, dieselben Resamples wie die Hauptauswertung), "
          "Perzentil-KI, Differenz gegen (c). Dup/Frag: Detektionen mit Score ≥ 0,25, Definitionen von "
          "`scahi.eval.diagnostics` (DOTA: alle GT einschließlich difficult).", ""]
    L += ["## Einordnung: Abdeckungsnormalisierung", "",
          "- In den eingefrorenen Stacks bilden die Rand-Taper über überlappende Kacheln eine Zerlegung der Eins, und beide "
          "Ebenen (Einzelpass auf das ganze Bild, Kacheln) decken das ganze Bild ab. Damit ist C konstant (C = 2), und die "
          "explizite Normalisierung skaliert die Scores nur um einen festen Faktor; sie ändert weder die Rangfolge noch die "
          "Boxen. Messbar bleibt allein, dass ohne Division mehr Peaks über `tau_peak` liegen (alle mit niedrigem Score), "
          "daher (a) − (c) ≈ 0.",
          "- „Ein Pass ohne Vorhersage zählt als Evidenz dagegen“ folgt aus der Summe über die Pässe (ein Pass, der die "
          "Stelle abdeckt, aber nichts vorhersagt, trägt 0 bei), nicht aus der Division.",
          "- (b) zeigt den Effekt der abdeckungsbewussten Gewichtung selbst: ohne Taper variiert C (Anzahl der abdeckenden "
          "Pässe), und die Gewichte bilden keine Zerlegung der Eins mehr.",
          "- Eine explizite Normalisierung wäre bei Stacks mit nur teilweiser Abdeckung wirksam (Ebenen, die nur einen "
          "Bildausschnitt abdecken); solche Stacks enthalten die eingefrorenen Zeilen nicht. Dafür gibt es kein neues "
          "Experiment.", ""]
    cf_ = REPO / "runs/ablation/coverage.json"
    if cf_.exists():
        cov = json.loads(cf_.read_text())
        L += ["C an allen fusionierten Positionen (Zellen des Volumens; `scripts/ablation_coverage.py`, "
              "`runs/ablation/coverage.json`; val):", "",
              "| Datensatz / Gewichte / Arm | Variante | Positionen im Bild | C min / max / Mittel | Anteil C = 2 | "
              "häufigste C-Werte (Anteil) | Positionen außerhalb des Bildes |", "|---|---|---|---|---|---|---|"]
        for k, v in cov.items():
            ds, m, arm = k.split("/")
            top = ", ".join(f"{f(c, 2)} ({100 * n / v['n_positions_inside']:.1f} %)".replace(".", ",")
                            for c, n in v["most_frequent_C"][:4])
            L.append(f"| {ds} / {NAME[m]} / ({arm}) | {v['variant']} | {v['n_positions_inside']} | "
                     f"{f(v['C_min'])} / {f(v['C_max'])} / {f(v['C_mean'], 3)} | {100 * v['share_C_eq_2']:.2f} % | {top} | "
                     f"{v['n_positions_outside_image']} |")
        L.append("")
    rows = []
    for ds in ("visdrone", "dota"):
        f_ = REPO / f"runs/ablation/{ds}.json"
        if not f_.exists():
            L += [f"## {DS[ds]['title']}", "", "Noch nicht berechnet.", ""]
            continue
        R = json.loads(f_.read_text())
        t, r = table(ds, R)
        rows += r
        figure(ds, R)
        L += [f"## {DS[ds]['title']}", "", f"Primärmetrik: {DS[ds]['prim_text']}. Abbildung "
              f"`reports/figures/ablation/fig_ablation_{ds}.png`.", ""]
        if ds == "dota":
            L += ["**Post-hoc-Analyse auf dem bereits einmal ausgewerteten zurückgehaltenen Satz DOTA-val.** Arm (e) "
                  "ist hier SCAHI mit dem verfeinerten Raster, **auf DOTA vorab festgelegt** (auf VisDrone war das Raster post "
                  "hoc). Die "
                  "Validity-Fenster von (d) sind die vorab festgelegten VisDrone-Parameter (Eingabe-σ 6–256 px); große "
                  "DOTA-Objekte liegen teils außerhalb, die Scores von (d) überschreiten dort 1 (Ränge und AP davon nicht "
                  "verfälscht). Die Fenster werden nicht nachträglich an DOTA angepasst. Da die Validity-Fenster nicht Teil "
                  "von SCAHI sind (schon auf VisDrone ohne Nutzen) und auf DOTA zusätzlich falsch skaliert, steht (d) hier "
                  "**nur der Vollständigkeit halber, nicht als Argument**.", ""]
        if R.get("recompute_check_cf5_equals_c"):
            L += [f"Kontrolle: der Sweep-Punkt cell_factor = 5 berechnet (c) neu; |ΔAP| gegen die bestehende Zeile: "
                  + ", ".join(f"{NAME[m]} {v:.1e}" for m, v in R["recompute_check_cf5_equals_c"].items()) + ".", ""]
        L += t + [""]
    pd.DataFrame(rows).to_csv(FIG / "ablation.csv", index=False) if rows else None
    OUT.write_text("\n".join(L))
    print(OUT)


if __name__ == "__main__":
    main()
