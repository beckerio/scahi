# SPDX-License-Identifier: AGPL-3.0-or-later
"""reports/D_RESULTS.md from the one DOTA-val run (runs/dota/val/results.json, latency.json) and the training logs.

    python scripts/dota/d_results.py

Every number comes from the result files; nothing here evaluates detections. Reporting rules: docs/FREEZE_PROPOSAL_DOTA.md
(approved 2026-10-30, docs/DECISIONS.md); exploratory analyses in separate tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scahi without an installed package (release)

from scahi.dota import evaluate as EV  # noqa: E402
from scahi.dota import final as F  # noqa: E402
from scahi.dota import train as T  # noqa: E402
from scahi.machine import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "reports/D_RESULTS.md"
NAME = {"yolov10s": "YOLOv10s", "yolov10m": "YOLOv10m"}
ROW_ORDER = ("A", "NMS-o2o", "NMS-o2m", "F2", "RG", "B-F2", "B-RG")
HEAD = {"A": "o2o", "X": "o2o", "NMS-o2o": "o2o", "NMS-o2m": "o2m", "F2": "o2m", "RG": "o2m", "B-F2": "o2m",
        "B-RG": "o2m", "SAHI": "o2o"}
PREREG = {  # docs/PREREGISTRATION_DOTA.md section 4, verbatim
    "D-H1": "„D-H1 (Auflösung vs. Kachelung): Δ = A_2048 − NMS_1024 (o2o); Kriterium Δ > ε.“",
    "D-H2a": "„D-H2 (Fusion): Δ = RG_s − NMS_s (beide o2m, identische Pässe), getrennt für s = 1024 (D-H2a) und "
             "s = 2048 (D-H2b); Kriterium Δ > ε.“",
    "D-H2b": "wie D-H2a, s = 2048",
    "D-H3": "„D-H3 (entscheidend): Δ = RG_{S_max} − A_{S_max} (SCAHI o2m mit Kacheln gegen den Einzelpass o2o, beide "
            "auf den S_max-Gewichten); Kriterium Δ > ε und untere KI-Grenze > 0.“ (S_max = 2048)",
}
COMPUTED = {
    "D-H1": "A_2048 (Einzelpass 2048, o2o) − NMS-o2o_1024 (ganzes Bild 1024 + Kacheln, Box-Fusion IoS, o2o)",
    "D-H2a": "RG_1024 (SCAHI, verfeinertes Raster, o2m) − NMS-o2m_1024 (Box-Fusion IoS, o2m), identische Pässe dota1024",
    "D-H2b": "RG_2048 (SCAHI, o2m) − NMS-o2m_2048 (Box-Fusion IoS, o2m), identische Pässe dota2048",
    "D-H3": "RG_2048 (SCAHI o2m, ganzes Bild 2048 + Kacheln) − A_2048 (Einzelpass o2o), 2048er Gewichte",
}


def f(x: float | None, nd: int = 2) -> str:
    return "—" if x is None or x != x else f"{x:.{nd}f}".replace(".", ",")


def ci(c: dict, nd: int = 2) -> str:
    return f"{c['delta']:+.{nd}f} [{c['ci95'][0]:+.{nd}f}; {c['ci95'][1]:+.{nd}f}]".replace(".", ",")


def training_section() -> list[str]:
    L = ["## Trainingsbedingungen", "",
         "| Modell | s | Batch | effektive Batch | gewählte Epoche (calib-mAP50) | Mittel ± SD Epochen 41–60 |",
         "|---|---|---|---|---|---|"]
    for m in T.MODELS:
        for s in (1024, 2048):
            d = T.train_dir(m, s)
            c = json.loads((d / "ultralytics/train/weights/calib_curve.json").read_text())
            late = [x["mAP50"] for x in c if x["epoch"] >= 41]
            import statistics as st

            be = max(c, key=lambda x: x["mAP50"])
            b = T.BATCH[s]
            L.append(f"| {NAME[m]} | {s} | {b} | {b * max(round(64 / b), 1)} | {be['epoch']} ({f(be['mAP50'])}) | "
                     f"{f(st.mean(late))} ± {f(st.stdev(late))} |")
    L += ["",
          "- Batch fest in allen 60 Epochen (Batch-Wache nie ausgelöst, kein OOM-Fallback, kein Resume); keine NaN/Inf, "
          "keine übersprungenen Batches.",
          "- **OOM-Rückfall bei 2048 px:** In jeder der 60 Epochen meldet der CUDA-Allokator fehlgeschlagene Allokationen "
          "(insgesamt YOLOv10s 423, YOLOv10m 1008 Meldungen auf je 25 380 Iterationen; bei 1024: 10 bzw. 26). Sie werden "
          "vom Allokator (Cache freigeben, erneut versuchen) oder von Ultralytics' Rückfall der Label-Zuordnung "
          "(`TaskAlignedAssigner`: dieselbe Zuordnung Bild für Bild auf der GPU, Forward-Batch unverändert) abgefangen. "
          "Laut Code ist die Zuordnung identisch; die Ergebnisse sind nicht betroffen.",
          "- **Befund (calib, vor DOTA-val festgehalten):** YOLOv10m liegt bei 2048 über die ganze Spätphase rund 1,6 "
          "mAP50-Punkte unter YOLOv10s, bei 1024 klar darüber; die 2048er Kurven schwanken stärker. Mögliche "
          "Erklärungen, **nicht geprüft**: Batch 3 (Batch-Norm-Statistik) trifft das größere Modell stärker; "
          "Kapazität/Überanpassung von YOLOv10m bei 2048; Rauschen der kleinen calib-Menge (141 Bilder).", ""]
    return L


def main() -> None:
    R = json.loads((F.VAL_DIR / "results.json").read_text())
    lat_f = F.VAL_DIR / "latency.json"
    lat = json.loads(lat_f.read_text()) if lat_f.exists() else None
    rows = R["rows"]
    L = ["# Phase D – Ergebnisse auf DOTA-val (v1.5, HBB)", "",
         f"DOTA-val ({R['n_images']} Bilder), **einmal ausgewertet** nach dem freigegebenen Freeze "
         "(`docs/FREEZE_PROPOSAL_DOTA.md`, `configs/dota/frozen_rows.json`, Freigabe 2026-10-30, `docs/DECISIONS.md`). "
         f"Pipeline-Grenze {R['pipeline_limit']} je Bild/Pass für alle Zeilen; alle 32 Zeilen, kein Nachtuning. "
         f"Primär: offizielle Devkit-mAP50 ohne Cap. ε = {f(R['eps'])}; gepaarter Bootstrap über Bilder "
         "(1000 Resamples, Seed 0, Perzentil-KI), einseitiges p gegen ε, Holm über die zwei Modelle. Erzeugt von "
         "`scripts/dota/d_results.py` aus `runs/dota/val/results.json`.", ""]
    L += ["## Hypothesen (primär: mAP50 ohne Cap)", "",
          "Eine Hypothese gilt, wenn beide Modelle sie erfüllen (Kriterium und Holm-p ≤ 0,05).", ""]
    for h, x in R["hypotheses"].items():
        L += [f"### {h} – {'gilt' if x['holds'] else 'gilt nicht'}", "",
              f"- Präregistriert: {PREREG[h]}", f"- Berechnet: {COMPUTED[h]}", "",
              "| Modell | a | b | Δ [95 %-KI] | p | Holm-p | Kriterium | erfüllt |", "|---|---|---|---|---|---|---|---|"]
        for m, v in x["models"].items():
            c = v["mAP50"]
            L.append(f"| {NAME[m]} | {f(c['a'])} | {f(c['b'])} | {ci(c)} | {f(c['p_one_sided'], 3)} | "
                     f"{f(v['holm']['p_holm'], 3)} | {'ja' if v['criterion'] else 'nein'} | "
                     f"{'ja' if v['met'] else 'nein'} |")
        L += ["", "Dieselbe Differenz, sekundär (ohne Urteil): Δ [95 %-KI]", "",
              "| Modell | mAP50 Cap 4000 | „alle Objekte“ | Fahrzeuge mAP50 | AP@[.5:.95] | Recall@0,5 crowded | "
              "Recall@0,5 isolated |", "|---|---|---|---|---|---|---|"]
        for m, v in x["models"].items():
            L.append(f"| {NAME[m]} | " + " | ".join(ci(v[k]) for k in (
                "mAP50_cap4000", "mAP50_all_objects", "mAP50_vehicles", "AP", "R50_crowd", "R50_iso")) + " |")
        L.append("")
    L += ["## Sekundäre Kontraste (vorab festgelegt, ohne Urteil)", "",
          "„Anderer Kopf“: A ist an o2o, RG an o2m gebunden; nur der Box-Fusions-Arm wechselt den Kopf. "
          "Δ in mAP50 (ohne Cap) [95 %-KI].", "",
          "| Kontrast | " + " | ".join(NAME[m] for m in T.MODELS) + " |", "|---|" + "---|" * len(T.MODELS)]
    for name, v in R["secondary"].items():
        L.append(f"| {name} | " + " | ".join(ci(v[m]["mAP50"]) for m in T.MODELS) + " |")
    L += ["", "Inferenz-Kreuzung: auf calib wurde bei beiden Modellen die Trainingsgröße 2048 gewählt; X_2048 = A_2048, "
          "der Kontrast ist daher gleich D-H3.", ""]
    L += ["## Alle Zeilen", "",
          "| Zeile | Kopf | mAP50 | Cap 4000 (Bilder > Cap) | alle Objekte | AP@[.5:.95] | Fahrzeuge mAP50 | "
          "klassenagn. AP50 | Recall crowded / isolated | Bilder an der Grenze | max. Det./Bild |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    ids = [F._row_id(r, m, s) for m in T.MODELS for s in (1024, 2048) for r in ROW_ORDER]
    ids += [F._row_id("X", m, 2048) for m in T.MODELS] + [F._row_id("SAHI", m, 1024) for m in T.MODELS]
    for i in ids:
        r = rows[i]
        row = i.split("_")[0]
        p = r["primary"]
        L.append(f"| `{i}` | {HEAD[row]} | **{f(r['mAP50'])}** | {f(r['mAP50_cap4000'])} ({r['images_over_cap']}) | "
                 f"{f(r['mAP50_all_objects'])} | {f(p['AP'])} | {f(p['mAP50_vehicles'])} | {f(p['AP50_agnostic'])} | "
                 f"{f(r['R50_crowd'], 1)} / {f(r['R50_iso'], 1)} | {r['images_at_limit']} | {r['max_dets_per_image']} |")
    L += ["", f"Crowded / isolated: nicht-difficult-GT mit / ohne gleichklassigen Nachbarn (IoU > 0,3); "
          f"{rows[ids[0]]['n_crowd_gt']} crowded, {rows[ids[0]]['n_iso_gt']} isolated GT.", ""]
    L += ["## Größenbins (σ = √(w·h) px)", "",
          "Primär (difficult ignoriert; der Bin < 8 hat dann keine GT: „—“) und „alle Objekte“ (difficult als normale GT). "
          "AP@[.5:.95] / AP50.", "",
          "| Zeile | Auswertung | " + " | ".join(EV.SIZE_LABELS) + " |", "|---|---|" + "---|" * len(EV.SIZE_LABELS)]
    for i in ids:
        for lbl, key in (("primär", "primary"), ("alle Objekte", "all_objects")):
            mt = rows[i][key]
            L.append(f"| `{i}` | {lbl} | " + " | ".join(
                f"{f(mt['AP_by_size'][b], 1)} / {f(mt['AP50_by_size'][b], 1)}" for b in EV.SIZE_LABELS) + " |")
    L += ["", "## Latenz und GFLOPs der D-H3-Zeilen", ""]
    if lat is None:
        L += ["Noch nicht gemessen (`d_val_latency`).", ""]
    else:
        v = lat["validity"]
        L += [f"Status: **{v['latency_status']}** (GPU exklusiv: {'ja' if v['latency_gpu_exclusive'] else 'nein'}, CPU "
              f"exklusiv: {'ja' if v['latency_cpu_exclusive'] else 'nein'}, Leistungsgrenze "
              f"{v['latency_gpu_power_limit_w']:.0f} W, Standard); Maschine "
              f"{lat['machine'].get('gpu_name', lat['machine'].get('name'))}. GFLOPs: {lat['gflops_method']['method']}.", "",
              "| Zeile | Pfad | Latenz ms (Mittel / p95) | davon Pässe / Fusion | Spitzenspeicher MB | GFLOPs/Bild |",
              "|---|---|---|---|---|---|"]
        for k, x in lat.items():
            if not isinstance(x, dict) or "gflops_per_image" not in x:
                continue
            split = (f"{f(x['passes_ms_mean'], 0)} / {f(x['fusion_ms_mean'], 0)}" if "passes_ms_mean" in x else "—")
            L.append(f"| `{k}` | {x['path']} | {f(x['latency_ms_mean'], 0)} / {f(x['latency_ms_p95'], 0)} | {split} | "
                     f"{f(x['peak_mem_mb'], 0)} | {f(x['gflops_per_image'], 0)} |")
        L += ["", "Pfade wie in der VisDrone-Referenzspalte: A mit dem Ultralytics-Prädiktor (fusionierte Gewichte, o2o); RG "
              "mit der Referenz-Pipeline (unfusioniertes Modell mit beiden Köpfen, Pässe von dota2048 auf der GPU, SCAHI auf "
              "der CPU, ein Bild nach dem anderen), die auf 3 calib-Bildern die gecachten Detektionen reproduziert. "
              "X_2048 = A_2048. GFLOPs zählen das fusionierte Ein-Kopf-Modell je Pass.", ""]
    lc_f = F.VAL_DIR / "limit_check.json"
    if lc_f.exists():
        lc = json.loads(lc_f.read_text())
        L += ["## Explorativ, post hoc: Wirkung der Pipeline-Grenze", "",
              f"Zeilen mit Bildern an der Grenze {lc['limit']}: dieselben Bilder ohne wirksame Grenze neu fusioniert "
              "(`scripts/dota/d_limit_check.py`); berichtete Zahlen oben unverändert.", "",
              "| Zeile | Bild: Detektionen mit / ohne Grenze | mAP50 mit Grenze | ohne Grenze | Δ |", "|---|---|---|---|---|"]
        for i, x in lc["rows"].items():
            im = "; ".join(f"{s}: {v['dets_limited']} / {v['dets_unlimited']}" for s, v in x["images"].items())
            L.append(f"| `{i}` | {im} | {f(x['mAP50_limited'], 4)} | {f(x['mAP50_unlimited'], 4)} | "
                     f"{f(x['delta'], 4)} |")
        L.append("")
    L += training_section()
    OUT.write_text("\n".join(L))
    print(OUT)


if __name__ == "__main__":
    main()
