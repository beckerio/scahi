# SPDX-License-Identifier: AGPL-3.0-or-later
"""FREEZE_PROPOSAL_DOTA (stop c): rows from the calib artefacts, replay status -> docs/FREEZE_PROPOSAL_DOTA.md and
configs/dota/frozen_rows.json (``approved: false`` until the user approves; nothing here reads DOTA-val).

    python scripts/dota/d_freeze_proposal.py
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess

from scahi.dota import final as F
from scahi.dota import pipeline as P
from scahi.machine import REPO_ROOT

DOC = REPO_ROOT / "docs/FREEZE_PROPOSAL_DOTA.md"
ROW_TEXT = {
    "A": "Einzelpass o2o bei s; conf auf calib",
    "X": "Inferenz-Kreuzung der 2048er Gewichte (sekundär): beste Größe aus 2048 / 2560 / 3072 auf calib",
    "NMS-o2o": "Box-Fusion (IoS) der Pässe dota<s>, o2o; Schwelle × conf auf calib (12)",
    "NMS-o2m": "Box-Fusion (IoS) der Pässe dota<s>, o2m; Schwelle × conf auf calib (12)",
    "F2": "SCAHI vorab festgelegt (F2-Raster, 24) auf dota<s>, o2m",
    "RG": "SCAHI verfeinertes Raster (108 + Referenz, Zeit ≤ 2×) auf dota<s>, o2m",
    "B-F2": "Zerlegung: SCAHI nur auf dem ganzen Bild, Parameter von F2",
    "B-RG": "Zerlegung: SCAHI nur auf dem ganzen Bild, eigenes verfeinertes Raster",
    "SAHI": "SAHI auf den 1024er Gewichten, Raster v2 (Overlap × Merge × Metrik × Schwelle × conf)",
}


def _fmt_cfg(r: dict) -> str:
    if r["kind"] == "single":
        return f"imgsz {r['imgsz']}, conf {r['conf']}"
    return ", ".join(f"{k} {v}" for k, v in r["cfg"].items())


def main() -> None:
    rows = F.proposal_rows()
    rep = {}
    for k in ("cpu", "gpu"):
        f = F.FREEZE_DIR / f"replay_{k}.json"
        if f.exists():
            rep.update({x["id"]: x for x in json.loads(f.read_text())["rows"]})
    r200 = {}
    for k in ("cpu", "gpu"):
        f = F.FREEZE_DIR / f"replay_{k}_limit200000.json"
        if f.exists():
            r200.update({x["id"]: x for x in json.loads(f.read_text())["rows"]})
    rep200 = (f"{sum(1 for r in rows if r['id'] in r200 and r200[r['id']]['abs_diff'] == 0)}/{len(rows)} Zeilen mit "
              f"identischer calib-mAP50, {sum(r200[x]['images_at_limit'] for x in r200)} Bilder an der Grenze "
              "(`runs/dota/freeze/replay_*_limit200000.json`)") if r200 else "noch offen"
    d200 = [x for x in r200.values() if x["abs_diff"] > 0]
    diff200 = ("; abweichend: " + ", ".join(f"`{x['id']}` {x['replay_mAP50'] - x['calib_mAP50']:+.1e}" for x in d200)
               + " mAP50-Punkte (die Grenze behält bei SCAHI die stärksten Peaks, dann die höchsten Scores; beim Einzelpass die höchsten Scores)") if d200 else ""
    n_exact = sum(1 for r in rows if r["id"] in rep and rep[r["id"]]["abs_diff"] == 0)
    n_missing = sum(1 for r in rows if r["id"] not in rep)
    commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                            text=True).stdout.strip()
    now = dt.datetime.now().isoformat(timespec="minutes")
    F.FROZEN.parent.mkdir(parents=True, exist_ok=True)
    F.FROZEN.write_text(json.dumps({"approved": False, "generated": now, "code_commit": commit,
                                    "pipeline_limit": P.PIPE, "rows": rows}, indent=1))
    lim_rows = [r for r in rows if r.get("calib_images_at_limit")]
    L = [
        "# FREEZE_PROPOSAL_DOTA (Vorschlag, Stopp c)",
        "",
        f"Erzeugt {now} von `scripts/dota/d_freeze_proposal.py` aus den calib-Artefakten (Code-Commit `{commit}`); "
        f"Zeilen in `configs/dota/frozen_rows.json` (`approved: false`). **Keine DOTA-val-Zahl liegt vor;** DOTA-val "
        "wird erst nach der Freigabe einmal ausgewertet (`scahi.dota.final.run_val` verweigert ohne Freigabe).",
        "",
        "## Was eingefroren wird",
        "",
        "Je Modell (YOLOv10s, YOLOv10m) und s ∈ {1024, 2048} (S_max = 2048, Addendum 1): A, NMS-o2o, NMS-o2m, F2, RG, B-F2, "
        "B-RG; je Modell die Inferenz-Kreuzung X (2048er Gewichte) und SAHI (1024er Gewichte). Gewichte = calib-beste "
        "Epoche (o2o, Primärmetrik). Alle Parameter auf calib gewählt; nichts wird neu gewählt.",
        "",
        "Hypothesen (Abschnitt 4, unverändert): D-H1 A_2048 − NMS-o2o_1024; D-H2a/b RG_s − NMS-o2m_s (s = 1024 / 2048); "
        "D-H3 RG_2048 − A_2048 (Kriterium Δ > 0,30 und untere KI-Grenze > 0). Holm über die zwei Modelle; eine "
        "Hypothese gilt, wenn beide Modelle sie erfüllen.",
        "",
        "## Berichtsregeln (mit der Freigabe fest, vor DOTA-val)",
        "",
        "1. Jede eingefrorene Zeile wird berichtet, unabhängig vom Ergebnis; kein Nachtuning, keine Neuauswahl, keine "
        "zusätzlichen Zeilen nach Sicht der DOTA-val-Zahlen.",
        "2. Primär: offizielle Devkit-mAP50 ohne Cap (difficult ignoriert); sekundär Cap 4000 mit Zahl der betroffenen "
        "Bilder; je Zeile die Zahl der Bilder an der Pipeline-Grenze.",
        "3. Vorab festgelegte Sekundäranalyse „alle Objekte“ (difficult als normale GT): D-H3 mit gepaartem KI, "
        "Größenbins < 8 px und 8–16 px (AP und AP50) aller D-H3-Zeilen.",
        "4. Inferenz-Kreuzung: Δ = RG_2048 − X_2048 mit gepaartem Bootstrap-KI, sekundär, ohne Urteil.",
        "5. Sekundär ohne Urteil (Abschnitt 4): F2 statt RG; D-H1/D-H2 mit dem anderen Kopf; Fahrzeug-Teilmenge; "
        "AP@[.5:.95]; Zerlegung A → B → C; crowded vs. isolated Recall; GFLOPs und Latenz der D-H3-Zeilen (Standard-"
        "Leistungsgrenze 575 W, ohne parallele Last); SAHI gegen A_1024 und NMS-o2o_1024.",
        "6. Größenbins (AP und AP50) aller Zeilen.",
        "",
        "## Zur Entscheidung: Pipeline-Grenze (SCAHI und Einzelpass)",
        "",
        f"Die interne Grenze {P.PIPE} Detektionen je Bild (Addendum 1: darf nicht greifen) wird auf calib "
        f"erreicht: {len(lim_rows)} der vorgeschlagenen Zeilen auf 1–{max((r['calib_images_at_limit'] for r in lim_rows), default=0)} "
        "calib-Bildern (Spalte „Grenze“): SCAHI und der Einzelpass bei 2048; die Box-Fusion nie. Ohne wirksame Grenze "
        "liefert SCAHI auf calib höchstens 60 567 Detektionen je Bild, der Einzelpass höchstens 25 287. "
        "DOTA-val enthält ein Bild mit 8609 GT (calib höchstens < 4000).",
        "",
        "- **Option 1 (Empfehlung):** Grenze für alle Zeilen auf DOTA-val **200 000** (> 3 × das calib-Maximum ohne "
        "Grenze); das entspricht der Absicht von Addendum 1. Die Auswahl (bei Grenze 20 000 getroffen) bleibt eingefroren "
        f"und wird nicht wiederholt. Replay aller Zeilen mit Grenze 200 000: {rep200}{diff200}. Jede Zeile meldet weiter die Bilder an der Grenze. SAHI "
        "hat nach dem Merge keine Grenze (pro 640er-Slice höchstens 8400 Anker < Grenze); dort wird nichts gekappt.",
        f"- **Option 2:** Grenze {P.PIPE} wie präregistriert; Bilder an der Grenze werden berichtet.",
        "",
        f"## Replay auf calib (Auswertungspfad von DOTA-val, `scahi.dota.final.row_dets`; exakt verlangt)",
        "",
        f"{n_exact}/{len(rows)} Zeilen exakt; {n_missing} noch nicht wiederholt "
        "(`runs/dota/freeze/replay_cpu.json`, `replay_gpu.json`).",
        "",
        "## Zeilen",
        "",
        "| id | Zeile | Gewichte SHA-256 (12) | Wahl auf calib | calib-mAP50 | Grenze | Replay |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        x = rep.get(r["id"])
        rp = "—" if x is None else ("exakt" if x["abs_diff"] == 0 else f"Δ {x['abs_diff']:.2e}")
        L.append(f"| `{r['id']}` | {ROW_TEXT[r['row']]} | `{r['weights_sha256'][:12]}` | {_fmt_cfg(r)} | "
                 f"{r['calib_mAP50']:.3f} | {r.get('calib_images_at_limit', 0)} | {rp} |")
    L += ["", "Quellen je Zeile: `configs/dota/frozen_rows.json` (Feld `source`).", ""]
    DOC.write_text("\n".join(L))
    print(f"{DOC}: {len(rows)} rows, replay exact {n_exact}, missing {n_missing}")


if __name__ == "__main__":
    main()
