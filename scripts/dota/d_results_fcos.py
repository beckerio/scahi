# SPDX-License-Identifier: AGPL-3.0-or-later
"""reports/D_RESULTS_FCOS.md from the one DOTA-val run of the FCOS rows (addendum 3; runs/dota/val/results_fcos.json) and
the FCOS training records. The earlier verdicts (results.json, results_new.json) are read, never changed.

    python scripts/dota/d_results_fcos.py

Every number comes from the result files; nothing here evaluates detections. Reporting rules: docs/PREREGISTRATION_DOTA.md
addendum 3 item 9 (one model -> no Holm; verdict with the unadjusted one-sided p and the criterion; descriptive count of
the five DOTA detectors) and the approval note in configs/dota/frozen_rows_fcos.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from d_results import ci, f  # noqa: E402
from d_rows_table import TERM  # noqa: E402  (row descriptions; B rows: "SCAHI, single pass (no tiles)", user 2026-11-08)

from scahi.dota import evaluate as EV  # noqa: E402
from scahi.dota import final as F  # noqa: E402
from scahi.dota import train as T  # noqa: E402
from scahi.machine import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "reports/D_RESULTS_FCOS.md"
NAME = {"yolov10s": "YOLOv10s", "yolov10m": "YOLOv10m", "yolo11s": "YOLOv11s", "yolo12s": "YOLOv12s", "fcos": "FCOS"}
SEC = ("mAP50_cap4000", "mAP50_all_objects", "mAP50_vehicles", "AP", "R50_crowd", "R50_iso")


def main() -> None:
    R = json.loads((F.VAL_DIR / "results_fcos.json").read_text())
    O = json.loads((F.VAL_DIR / "results.json").read_text())
    N = json.loads((F.VAL_DIR / "results_new.json").read_text())
    fz = json.loads(F.FROZEN_FCOS.read_text())
    rows, ids = R["rows"], [r["id"] for r in fz["rows"]]
    x_row = next(r for r in fz["rows"] if r["row"] == "X")
    L = ["# Phase D, Addendum 3 – FCOS auf DOTA-val (v1.5, HBB)", "",
         f"DOTA-val ({R['n_images']} Bilder), **einmal ausgewertet** nach dem freigegebenen FCOS-Freeze "
         f"(`configs/dota/frozen_rows_fcos.json`, Freigabe {R['approved_at']}, `docs/DECISIONS.md`). Alle {len(ids)} "
         f"Zeilen, kein Nachtuning; Pipeline-Grenze {R['pipeline_limit']} je Bild (Einzelpass, SCAHI-top_k) und je Pass; "
         "**Abweichung vom Wortlaut** „je Bild und je Pass“ (Addendum 3, Punkt 6): bei der Box-Fusion greift die Grenze "
         "nur je Pass, die zusammengeführte Ausgabe ist nicht begrenzt – der eingefrorene, auf calib ausgewählte und exakt "
         "nachgerechnete Pfad, wie bei allen YOLO-Box-Fusion-Zeilen (Entscheidung 2026-11-07). Primär Devkit-mAP50 ohne Cap, "
         f"ε = {f(R['eps'])}, gepaarter Bootstrap über Bilder ({R['bootstrap']['n']} Resamples, Seed "
         f"{R['bootstrap']['seed']}, Perzentil-KI), einseitiges p gegen ε. **Ein Modell → kein Holm:** Urteil mit dem "
         "unadjustierten p ≤ 0,05 und dem Kriterium (D-H3 zusätzlich untere KI-Grenze > 0). Deskriptiv: Anzahl der fünf "
         "DOTA-Detektoren, die eine Hypothese erfüllen; die Urteile der vier YOLO-Modelle (`reports/D_RESULTS.md`, "
         f"`reports/D_RESULTS_NEW.md`) bleiben unverändert. S_max = {R['s_max']}. FCOS hat einen Kopf; Einzelpass, "
         "Box-Fusion und SCAHI nutzen ihn mit dem Detektor-Score sqrt(p · Centerness); keine SAHI-Zeile. Erzeugt von "
         "`scripts/dota/d_results_fcos.py` aus `runs/dota/val/results_fcos.json`.", ""]
    L += ["## Hypothesen (primär: mAP50 ohne Cap)", "",
          "| Hypothese | Kontrast | a | b | Δ [95 %-KI] | p (einseitig) | Kriterium | erfüllt | erfüllt (von 5, deskriptiv) | "
          "YOLO-Urteile (unverändert) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for h, x in R["hypotheses"].items():
        c = x["mAP50"]
        old = ", ".join(f"{NAME[m]} {'ja' if y['met'] else 'nein'}" for res in (O, N)
                        for m, y in res["hypotheses"][h]["models"].items())
        L.append(f"| {h} | {x['contrast']} | {f(c['a'])} | {f(c['b'])} | {ci(c)} | {f(c['p_one_sided'], 3)} | "
                 f"{'ja' if x['criterion'] else 'nein'} | {'ja' if x['met'] else 'nein'} | {x['met_of_five_descriptive']} | "
                 f"{old} |")
    L += ["", "Sekundär (ohne Urteil): Δ [95 %-KI]", "",
          "| Hypothese | mAP50 Cap 4000 | „alle Objekte“ | Fahrzeuge mAP50 | AP@[.5:.95] | Recall@0,5 crowded | "
          "Recall@0,5 isolated |", "|---|---|---|---|---|---|---|"]
    for h, x in R["hypotheses"].items():
        L.append(f"| {h} | " + " | ".join(ci(x[k]) for k in SEC) + " |")
    L += ["", "## Sekundäre Kontraste (vorab festgelegt, ohne Urteil)", "",
          "Ohne die o2o- und SAHI-Kontraste (ein Kopf, keine SAHI-Zeile; Addendum 3). Die Inferenz-Kreuzung wählte auf calib "
          f"{x_row['imgsz']} px für die 2048er Gewichte (calib-mAP50 je Größe: "
          + ", ".join(f"{s}: {f(v)}" for s, v in x_row["crossing_calib_mAP50"].items())
          + "); der Vergleich RG_2048 − X_2048 ist laut Freigabe als sekundärer Vergleich zu berichten.", "",
          "| Kontrast | Δ mAP50 [95 %-KI] | Δ „alle Objekte“ | Δ AP@[.5:.95] |", "|---|---|---|---|"]
    for name, v in R["secondary"].items():
        L.append(f"| {name} | {ci(v['mAP50'])} | {ci(v['mAP50_all_objects'])} | {ci(v['AP'])} |")
    L += ["", "## Alle Zeilen", "",
          "| Zeile | Beschreibung | mAP50 | Cap 4000 (Bilder > Cap) | alle Objekte | AP@[.5:.95] | Fahrzeuge mAP50 | klassenagn. AP50 | "
          "Recall crowded / isolated | Bilder an der Grenze (Box-Fusion: Ausgabe ≥ Grenze, nicht gekürzt) | max. Det./Bild |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    term = {r["id"]: TERM[r["row"]] for r in fz["rows"]}
    term[x_row["id"]] = f"single pass of the 2048 weights at {x_row['imgsz']} px (inference size selected on calib)"
    for i in ids:
        r, p = rows[i], rows[i]["primary"]
        L.append(f"| `{i}` | {term[i]} | **{f(r['mAP50'])}** | {f(r['mAP50_cap4000'])} ({r['images_over_cap']}) | "
                 f"{f(r['mAP50_all_objects'])} | {f(p['AP'])} | {f(p['mAP50_vehicles'])} | {f(p['AP50_agnostic'])} | "
                 f"{f(r['R50_crowd'], 1)} / {f(r['R50_iso'], 1)} | {r['images_at_limit']} | {r['max_dets_per_image']} |")
    L += ["", "## Größenbins (σ = √(w·h) px)", "",
          "Primär (difficult ignoriert; der Bin < 8 hat dann keine GT: „—“) und „alle Objekte“. AP@[.5:.95] / AP50.", "",
          "| Zeile | Auswertung | " + " | ".join(EV.SIZE_LABELS) + " |", "|---|---|" + "---|" * len(EV.SIZE_LABELS)]
    for i in ids:
        for lbl, key in (("primär", "primary"), ("alle Objekte", "all_objects")):
            mt = rows[i][key]
            L.append(f"| `{i}` | {lbl} | " + " | ".join(
                f"{f(mt['AP_by_size'][b], 1)} / {f(mt['AP50_by_size'][b], 1)}" for b in EV.SIZE_LABELS) + " |")
    L += ["", "## Latenz", "", "Nicht Teil der Freigabe (Addendum 3, Punkt 10); keine FCOS-Zeile in den Latenztabellen.", "",
          "## Trainingsbedingungen", "", "| s | Batch | Akkumulation | effektive Batch | Lernrate | Auswahl auf calib |",
          "|---|---|---|---|---|---|"]
    for s in (1024, 2048):
        t = json.loads((T.train_dir(T.FCOS, s) / "training.json").read_text())
        L.append(f"| {s} | {t['batch']} | {t.get('accumulation', 1)} | {t.get('effective_batch', t['batch'])} | "
                 f"{str(t['lr']).replace('.', ',')} | {t['selection']} |")
    L += ["", "Rezept: `docs/PREREGISTRATION_DOTA.md` Addendum 3 (torchvision `fcos_resnet50_fpn`, COCO_V1, 60 Epochen, nur "
          "horizontales Spiegeln, Auswahl auf calib).", ""]
    OUT.write_text("\n".join(L))
    print(OUT)


if __name__ == "__main__":
    main()
