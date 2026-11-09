# SPDX-License-Identifier: AGPL-3.0-or-later
"""reports/D_RESULTS_NEW.md from the one DOTA-val run of the addendum-2 rows (runs/dota/val/results_new.json, latency of
the new D-H3 rows from runs/latency_breakdown/latency_new.json) and the training logs of YOLO11s / YOLO12s.

    python scripts/dota/d_results_new.py

Every number comes from the result files; nothing here evaluates detections. Reporting rules: docs/FREEZE_PROPOSAL_DOTA_NEW.md
(approved 2026-10-31, docs/DECISIONS.md) = those of the original freeze. reports/D_RESULTS.md is not touched.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from d_results import ci, f  # noqa: E402

from scahi.dota import evaluate as EV  # noqa: E402
from scahi.dota import final as F  # noqa: E402
from scahi.dota import train as T  # noqa: E402
from scahi.machine import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "reports/D_RESULTS_NEW.md"
NAME = {"yolov10s": "YOLOv10s", "yolov10m": "YOLOv10m", "yolo11s": "YOLOv11s", "yolo12s": "YOLOv12s"}
ROW_ORDER = ("A", "NMS-o2m", "F2", "RG", "B-F2", "B-RG")
LAT = REPO_ROOT / "runs/latency_breakdown/latency_new.json"
SEC = ("mAP50_cap4000", "mAP50_all_objects", "mAP50_vehicles", "AP", "R50_crowd", "R50_iso")


def ids() -> list[str]:
    out = [F._row_id(r, m, s) for m in T.NEW_MODELS for s in (1024, 2048) if s <= T.S_MAX[m] for r in ROW_ORDER]
    return out + [F._row_id("X", m, 2048) for m in T.NEW_MODELS if T.S_MAX[m] >= 2048]


def main() -> None:
    R = json.loads((F.VAL_DIR / "results_new.json").read_text())
    O = json.loads((F.VAL_DIR / "results.json").read_text())
    rows = R["rows"]
    L = ["# Phase D, Addendum 2 – YOLOv11s und YOLOv12s auf DOTA-val (v1.5, HBB)", "",
         f"DOTA-val ({R['n_images']} Bilder), **einmal ausgewertet** nach dem freigegebenen Freeze der neuen Zeilen "
         "(`docs/FREEZE_PROPOSAL_DOTA_NEW.md`, `configs/dota/frozen_rows_new.json`, Freigabe 2026-10-31, "
         f"`docs/DECISIONS.md`). Alle {len(rows)} Zeilen, kein Nachtuning; Pipeline-Grenze {R['pipeline_limit']}. "
         f"Berichtsregeln wie im ursprünglichen Freeze: primär Devkit-mAP50 ohne Cap, ε = {f(R['eps'])}, gepaarter "
         "Bootstrap über Bilder (1000 Resamples, Seed 0, Perzentil-KI), einseitiges p gegen ε; **Holm über die zwei neuen "
         "Modelle**. Ein-Kopf-Modelle: Einzelpass, Box-Fusion und SCAHI nutzen den einzigen (dichten) Kopf. "
         f"S_max (Speicher-Smoke): {', '.join(f'{NAME[m]} {s}' for m, s in R['s_max'].items())}. Die Urteile der "
         "ursprünglichen zwei Modelle (`reports/D_RESULTS.md`) bleiben unverändert. Erzeugt von "
         "`scripts/dota/d_results_new.py` aus `runs/dota/val/results_new.json`.", ""]
    L += ["## Hypothesen (primär: mAP50 ohne Cap)", "",
          "Urteil je Modell (Kriterium und Holm-p ≤ 0,05). Nicht prüfbar (YOLOv12s ohne 2048er Zeilen, S_max = 1024): "
          "zählt als nicht erfüllt (Holm-p = 1). Die Zählung über vier Modelle ist deskriptiv.", "",
          "| Hypothese | YOLOv11s | YOLOv12s | erfüllt (neu, von 2) | erfüllt (alle, von 4, deskriptiv) | "
          "ursprüngliches Urteil (YOLOv10s + YOLOv10m), unverändert |", "|---|---|---|---|---|---|"]
    for h, x in R["hypotheses"].items():
        v = ["nicht prüfbar" if not x["models"][m]["testable"] else ("erfüllt" if x["models"][m]["met"] else "nicht erfüllt")
             + (f" ({ci(x['models'][m]['mAP50'])}, auf dem 1024er Pass = S_max)" if h == "D-H3" and R["s_max"][m] == 1024
                else "") for m in T.NEW_MODELS]
        old = ", ".join(f"{NAME[m]} {'ja' if y['met'] else 'nein'}" for m, y in O["hypotheses"][h]["models"].items())
        L.append(f"| {h} | {v[0]} | {v[1]} | {x['met_new']} | {x['met_of_four_descriptive']} | "
                 f"{'gilt' if x['original_two_model_holds'] else 'gilt nicht'} ({old}) |")
    L.append("")
    for h, x in R["hypotheses"].items():
        L += [f"### {h}", "", "| Modell | Kontrast | a | b | Δ [95 %-KI] | p | Holm-p | Kriterium | erfüllt |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m, v in x["models"].items():
            if not v["testable"]:
                L.append(f"| {NAME[m]} | {v['contrast']} | — | — | nicht prüfbar | — | {f(v['holm']['p_holm'], 3)} | "
                         "nein | nein |")
                continue
            c = v["mAP50"]
            note = " (1024er Pass, S_max)" if h == "D-H3" and R["s_max"][m] == 1024 else ""
            L.append(f"| {NAME[m]}{note} | {v['contrast']} | {f(c['a'])} | {f(c['b'])} | {ci(c)} | "
                     f"{f(c['p_one_sided'], 3)} | {f(v['holm']['p_holm'], 3)} | {'ja' if v['criterion'] else 'nein'} | "
                     f"{'ja' if v['met'] else 'nein'} |")
        if h == "D-H3":
            L += ["", "YOLOv12s: D-H3 auf dem 1024er Pass (S_max = 1024; das 2048er Training passt nicht in den Speicher). "
                  "Der Einzelpass bei 1024 ist deutlich schwächer als bei 2048 (YOLOv11s: A_1024 "
                  f"{f(rows['A_yolo11s_1024']['mAP50'])}, A_2048 {f(rows['A_yolo11s_2048']['mAP50'])}; für YOLOv12s gibt es "
                  "keinen 2048er Einzelpass), daher fällt der Gewinn höher aus als bei den "
                  "2048er Modellen (" + ", ".join(
                      f"{NAME[m]} {ci(y['models'][m]['mAP50']).split(' ')[0]}"
                      for y in (O["hypotheses"]["D-H3"], x) for m in y["models"] if R["s_max"][m] == 2048)
                  + "); nicht direkt vergleichbar."]
        L += ["", "Sekundär (ohne Urteil): Δ [95 %-KI]", "",
              "| Modell | mAP50 Cap 4000 | „alle Objekte“ | Fahrzeuge mAP50 | AP@[.5:.95] | Recall@0,5 crowded | "
              "Recall@0,5 isolated |", "|---|---|---|---|---|---|---|"]
        for m, v in x["models"].items():
            if v["testable"]:
                L.append(f"| {NAME[m]} | " + " | ".join(ci(v[k]) for k in SEC) + " |")
        L.append("")
    L += ["## Sekundäre Kontraste (vorab festgelegt, ohne Urteil)", "",
          "Ohne die o2o- und SAHI-Kontraste (Ein-Kopf-Modelle, kein SAHI; Addendum 2). Δ in mAP50 (ohne Cap) [95 %-KI].",
          "", "| Kontrast | " + " | ".join(NAME[m] for m in T.NEW_MODELS) + " |", "|---|" + "---|" * len(T.NEW_MODELS)]
    for name, v in R["secondary"].items():
        L.append(f"| {name} | " + " | ".join(ci(v[m]["mAP50"]) if m in v else "—" for m in T.NEW_MODELS) + " |")
    L += ["", "Inferenz-Kreuzung (nur YOLOv11s): auf calib wurde die Trainingsgröße 2048 gewählt; X_2048 = A_2048, "
          "der Kontrast ist daher gleich D-H3.", ""]
    L += ["## Alle Zeilen", "",
          "| Zeile | mAP50 | Cap 4000 (Bilder > Cap) | alle Objekte | AP@[.5:.95] | Fahrzeuge mAP50 | klassenagn. AP50 | "
          "Recall crowded / isolated | Bilder an der Grenze | max. Det./Bild |", "|---|---|---|---|---|---|---|---|---|---|"]
    for i in ids():
        r, p = rows[i], rows[i]["primary"]
        L.append(f"| `{i}` | **{f(r['mAP50'])}** | {f(r['mAP50_cap4000'])} ({r['images_over_cap']}) | "
                 f"{f(r['mAP50_all_objects'])} | {f(p['AP'])} | {f(p['mAP50_vehicles'])} | {f(p['AP50_agnostic'])} | "
                 f"{f(r['R50_crowd'], 1)} / {f(r['R50_iso'], 1)} | {r['images_at_limit']} | {r['max_dets_per_image']} |")
    L += ["", "## Größenbins (σ = √(w·h) px)", "",
          "Primär (difficult ignoriert; der Bin < 8 hat dann keine GT: „—“) und „alle Objekte“. AP@[.5:.95] / AP50.", "",
          "| Zeile | Auswertung | " + " | ".join(EV.SIZE_LABELS) + " |", "|---|---|" + "---|" * len(EV.SIZE_LABELS)]
    for i in ids():
        for lbl, key in (("primär", "primary"), ("alle Objekte", "all_objects")):
            mt = rows[i][key]
            L.append(f"| `{i}` | {lbl} | " + " | ".join(
                f"{f(mt['AP_by_size'][b], 1)} / {f(mt['AP50_by_size'][b], 1)}" for b in EV.SIZE_LABELS) + " |")
    L += ["", "## Latenz der D-H3-Zeilen (host-b)", ""]
    L += (["Siehe `reports/tab_latency_breakdown.md` (Nachtrag mit Ankerzeilen, gleiches Verfahren, 575 W, exklusiv)."]
          if LAT.exists() else ["Noch nicht gemessen (`scripts/latency_breakdown.py measure_new`)."]) + [""]
    L += ["## Trainingsbedingungen", "",
          "| Modell | s | Batch | gewählte Epoche (calib-mAP50) | Mittel ± SD Epochen 41–60 |", "|---|---|---|---|---|"]
    for m in T.NEW_MODELS:
        for s in (1024, 2048):
            if s > T.S_MAX[m]:
                continue
            d = T.train_dir(m, s)
            c = json.loads((d / "ultralytics/train/weights/calib_curve.json").read_text())
            late = [x["mAP50"] for x in c if x["epoch"] >= 41]
            be = max(c, key=lambda x: x["mAP50"])
            b = json.loads((d / "training.json").read_text())["batch"]
            L.append(f"| {NAME[m]} | {s} | {b} | {be['epoch']} ({f(be['mAP50'])}) | "
                     f"{f(st.mean(late))} ± {f(st.stdev(late))} |")
    L += ["", "Rezept unverändert (60 Epochen, COCO-Init, Auswahl auf calib). YOLOv12s bei 2048: Speicher-Smoke mit "
          "Batch 3 und 2 fehlgeschlagen (`runs/dota/smoke.json`) → S_max = 1024 nach der präregistrierten Regel.", ""]
    OUT.write_text("\n".join(L))
    print(OUT)


if __name__ == "__main__":
    main()
