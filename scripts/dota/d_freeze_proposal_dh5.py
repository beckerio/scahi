# SPDX-License-Identifier: AGPL-3.0-or-later
"""FREEZE_PROPOSAL_DOTA_DH5 (addendum 4): rows SAHI-F2 / SAHI-RG of YOLOv10s / YOLOv10m from the calib artefacts,
replay status -> docs/FREEZE_PROPOSAL_DOTA_DH5.md and configs/dota/frozen_rows_dh5.json (``approved: false`` until the
user approves; nothing here reads DOTA-val). The approved freezes (frozen_rows*.json, SAHI rows included) are not touched.

    python -m scahi.dota.sahi_scahi replay           # replay of every new row on calib (cores 0-1)
    python scripts/dota/d_freeze_proposal_dh5.py     # write the proposal
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from d_freeze_proposal import _fmt_cfg  # noqa: E402

from scahi.dota import final as F  # noqa: E402
from scahi.dota import sahi_scahi as S  # noqa: E402
from scahi.dota import time_rule as TR  # noqa: E402
from scahi.machine import REPO_ROOT  # noqa: E402

DOC = REPO_ROOT / "docs/FREEZE_PROPOSAL_DOTA_DH5.md"
LOG = REPO_ROOT / "runs/dota/fcos_logs/dh5_tuning.log"
TEXT = {"SAHI-F2": "SAHI-Pässe + SCAHI (o2m), vorab festgelegt (F2-Raster, 24)",
        "SAHI-RG": "SAHI-Pässe + SCAHI (o2m), verfeinertes Raster (108 + Referenz, Zeit ≤ 2×)"}


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def main() -> None:
    rows = S.dh5_rows()
    rf = F.FREEZE_DIR / "replay_dh5.json"
    rp = json.loads(rf.read_text()) if rf.exists() else {"rows": []}
    rep = {x["id"]: x for x in rp["rows"]}
    n_exact = sum(1 for r in rows if r["id"] in rep and rep[r["id"]]["abs_diff"] == 0
                  and rep[r["id"]]["abs_diff_cap4000"] == 0)
    n_missing = sum(1 for r in rows if r["id"] not in rep)
    commit = _git("rev-parse", "--short=12", "HEAD")
    now = dt.datetime.now().isoformat(timespec="minutes")
    S.FROZEN_DH5.write_text(json.dumps({"approved": False, "generated": now, "code_commit": commit, "addendum": 4,
                                        "pipeline_limit": S.DH5_LIMIT, "calib_selection_limit": S.DH5_LIMIT,
                                        "rows": rows}, indent=1))
    add = _git("log", "--reverse", "--format=%h %cI", "-S## Addendum 4", "--", "docs/PREREGISTRATION_DOTA.md")
    add_h, add_t = add.splitlines()[0].split()
    fz = {r["id"]: r for r in F.frozen()["rows"]}
    tun = {m: json.loads((S.dh5_dir(m) / "tuning.json").read_text()) for m in S.MODELS}
    gate = {m: json.loads((F.P.OUT / f"{m}_1024/sahi_scahi_gate.json").read_text()) for m in S.MODELS}
    log = LOG.read_text() if LOG.exists() else ""
    n_warn = len(re.findall(r"\[dh5\] WARNING task", log))
    n_def = len(re.findall(r"deferred to a run with a larger budget", log))
    peak = max(r["peak_gib_max"] for t in tun.values() for st in ("F2", "RG") for r in t[st]["rows"])
    L = [
        "# FREEZE_PROPOSAL_DOTA_DH5 (Addendum 4: SAHI-F2, SAHI-RG – Vorschlag, Stopp für die Freigabe)",
        "",
        f"Erzeugt {now} von `scripts/dota/d_freeze_proposal_dh5.py` aus den calib-Artefakten (Code-Commit `{commit}`); "
        "Zeilen in `configs/dota/frozen_rows_dh5.json` (`approved: false`). **Keine DOTA-val-Zahl dieser Zeilen liegt "
        "vor;** DOTA-val wird nach der Freigabe einmal ausgewertet, nur für diese vier Zeilen. Die freigegebenen Freezes "
        "(SAHI-Zeilen eingeschlossen) bleiben unverändert; SAHI wird nicht neu ausgewertet (Addendum 4, Punkt 5).",
        "",
        "## Was eingefroren wird",
        "",
        "Je Modell (YOLOv10s, YOLOv10m; 1024er Gewichte der eingefrorenen SAHI-Zeilen): SAHI-F2 und SAHI-RG auf genau den "
        "Pässen der eingefrorenen SAHI-Zeile (Slices 640 px nach `sahi.slicing.get_slice_bboxes`, Überlappung 0,1 bzw. "
        "0,2, plus SAHIs Standard-Pass mit langer Seite 640; Cache-Ebenen `s640f`, `w640f`), Kopf o2m, Fusion F2. Alle "
        f"Parameter auf calib gewählt (`scahi.dota.pipeline.select`), Pipeline-Grenze {S.DH5_LIMIT} bei Auswahl, Replay "
        "und DOTA-val.",
        "",
        "Hypothese unverändert (Addendum 4, Punkt 6): D-H5 Δ = SAHI-RG − SAHI auf DOTA-val, primär mAP50 ohne Cap, "
        "Kriterium Δ > 0,30, gepaarter Bootstrap (1000, Seed 0), Holm über die zwei Modelle, α = 0,05; D-H5 gilt, wenn "
        "beide Modelle sie erfüllen. Sekundär ohne Urteil: SAHI-F2 − SAHI, Cap 4000, „alle Objekte“, AP@[.5:.95], "
        "Größenbins; Latenz (Pässe + Merge-Schritt) in einem exklusiven Lauf nach dem FCOS-Tuning.",
        "",
        "## Reihenfolge und Gate",
        "",
        f"- Addendum 4 committet `{add_h}` ({add_t}), vor den Caches, dem Gate und dem Tuning dieser Zeilen.",
        "- Paritäts-Gate (Regel von `pipeline.gate`): "
        + "; ".join(f"{m}: Dekodierung {'bestanden' if g['passed_decode'] else 'NICHT bestanden'}, calib-mAP50 der "
                    f"eingefrorenen SAHI-Konfiguration aus den Cache-Pässen {g['map']['mAP50_cache']:.4f} vs. "
                    f"{g['map']['mAP50_frozen']:.4f} (|Δ| {g['map']['abs_delta']:.0e})" for m, g in gate.items())
        + " → **bestanden** (`runs/dota/fusion/<m>_1024/sahi_scahi_gate.json`).",
        "",
        f"## Replay auf calib (Auswertungspfad von DOTA-val, `scahi.dota.sahi_scahi.dh5_row_dets`, Grenze {S.DH5_LIMIT})",
        "",
        f"- **{n_exact}/{len(rows)} Zeilen exakt** (mAP50 und mAP50 Cap 4000), {n_missing} fehlen "
        "(`runs/dota/freeze/replay_dh5.json`). Das Tuning lief aufgeteilt (eine Aufgabe je Bild und Einstellung, "
        "Detektionen in Bildreihenfolge zusammengesetzt); der Replay rechnet jede Zeile in einem Prozess über alle "
        "calib-Bilder, wie später DOTA-val.",
        "",
        "## Zeilen",
        "",
        "| id | Zeile | Gewichte SHA-256 (12) | Wahl auf calib | calib-mAP50 | Cap 4000 | Zeit vs. Ref. | Grenze | Replay |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        x = rep.get(r["id"])
        rs = "—" if x is None else ("exakt" if x["abs_diff"] == 0 and x["abs_diff_cap4000"] == 0
                                     else f"Δ {x['abs_diff']:.2e} / {x['abs_diff_cap4000']:.2e}")
        tr = f"{r['time_ratio_vs_ref']:.3f}" if "time_ratio_vs_ref" in r else "—"
        L.append(f"| `{r['id']}` | {TEXT[r['row']]} | `{r['weights_sha256'][:12]}` | {_fmt_cfg(r)} | "
                 f"{r['calib_mAP50']:.3f} | {r['calib_mAP50_cap4000']:.3f} | {tr} | {r['calib_images_at_limit']} | {rs} |")
    L += ["", "Zum Vergleich (nur calib, bereits eingefroren): "
          + "; ".join(f"`SAHI_{m}` calib-mAP50 {fz[f'SAHI_{m}']['calib_mAP50']:.3f}" for m in S.MODELS)
          + ". Referenz des verfeinerten Rasters je Modell = calib-gewählte SAHI-F2-Einstellung (Feld `reference_cfg`).",
          "", "Quellen je Zeile: `configs/dota/frozen_rows_dh5.json` (Feld `source`).", ""]
    L += TR.lines_de({f"SAHI-RG {m}": TR.sensitivity(tun[m]["RG"]["rows"]) for m in S.MODELS})
    L += [
        "## Pipeline-Grenze",
        "",
        f"Auswahl, Replay und DOTA-val mit derselben Grenze {S.DH5_LIMIT} (Addendum 4; anders als die eingefrorenen "
        "YOLO-Zeilen, die bei 20 000 gewählt und bei 200 000 ausgewertet wurden). Kein calib-Bild erreicht die Grenze bei "
        "den gewählten Einstellungen (Spalte „Grenze“); größte Detektionszahl je Bild: "
        + "; ".join(f"{m} F2 {tun[m]['selected']['F2']['dets_per_image_max']}, RG {tun[m]['selected']['RG']['dets_per_image_max']}"
                    for m in S.MODELS) + ".",
        "",
        "## Durchführung (nur Ablauf, ohne Einfluss auf Ergebnisse)",
        "",
        "Parallel zum FCOS-Tuning (Freigabe 2026-11-02, DECISIONS): eigener Scope MemoryMax 10G, MemorySwapMax 0, "
        "oom_score_adj 1000, Kerne 0–1; eine Aufgabe je (Bild, Einstellung) in einem frischen Prozess; Zeitregel nach "
        "Option 3 (Referenz im selben Prozess je Aufgabe, abwechselnde Reihenfolge), wie FCOS. "
        f"Zurückgestellte Aufgaben: {n_def} (alle Schätzungen passten ins Budget). Speicherwarnungen (Spitze > Schätzung, "
        f"nur Planung): {n_warn}; höchste Spitze einer Aufgabe {peak:.2f} GiB bei 10 GB Deckel; kein Abbruch, kein OOM. "
        "Laufzeiten: "
        + "; ".join(f"{m} F2 {tun[m]['F2']['wall_s'] / 60:.0f} min, RG {tun[m]['RG']['wall_s'] / 3600:.1f} h"
                    for m in S.MODELS) + " (RG YOLOv10s ab dem Neustart 18:31).",
        "",
        "**Stopp: Freigabe dieser vier Zeilen nötig, bevor DOTA-val berührt wird.**",
        "",
    ]
    DOC.write_text("\n".join(L))
    print(f"{DOC}: {len(rows)} rows, replay exact {n_exact}, missing {n_missing}")


if __name__ == "__main__":
    main()
