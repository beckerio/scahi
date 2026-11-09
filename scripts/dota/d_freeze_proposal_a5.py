# SPDX-License-Identifier: AGPL-3.0-or-later
"""FREEZE_PROPOSAL_DOTA_A5 (addendum 5, D-H6): rows SAHI / SAHI-F2 / SAHI-RG of YOLO11s / YOLO12s from the calib
artefacts, replay status -> docs/FREEZE_PROPOSAL_DOTA_A5.md and configs/dota/frozen_rows_a5.json (``approved: false``
until the user approves; nothing here reads DOTA-val). The approved freezes are not touched.

    python -m scahi.dota.sahi_scahi a5_replay       # replay of every row on calib (cores 0-1)
    python scripts/dota/d_freeze_proposal_a5.py     # write the proposal
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

DOC = REPO_ROOT / "docs/FREEZE_PROPOSAL_DOTA_A5.md"
LOG = REPO_ROOT / "runs/slot/a5_tune.log"
TEXT = {"SAHI": "SAHI (calib-getunt, Raster v2 wie YOLOv10)",
        "SAHI-F2": "SAHI-Pässe + SCAHI, vorab festgelegt (F2-Raster, 24)",
        "SAHI-RG": "SAHI-Pässe + SCAHI, verfeinertes Raster (108 + Referenz, Zeit ≤ 2×)"}


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def _cfg(r: dict) -> str:
    if r["kind"] != "a5_sahi":
        return _fmt_cfg(r)
    c = r["cfg"]
    return f"Überlappung {c['overlap']}, {c['merge']}/{c['metric']} {c['match_thr']}, conf {c['conf']}"


def main() -> None:
    rows = S.a5_rows()
    rf = F.FREEZE_DIR / "replay_a5.json"
    rp = json.loads(rf.read_text()) if rf.exists() else {"rows": []}
    rep = {x["id"]: x for x in rp["rows"]}
    n_exact = sum(1 for r in rows if r["id"] in rep and rep[r["id"]]["abs_diff"] == 0
                  and rep[r["id"]]["abs_diff_cap4000"] == 0)
    n_missing = sum(1 for r in rows if r["id"] not in rep)
    commit = _git("rev-parse", "--short=12", "HEAD")
    now = dt.datetime.now().isoformat(timespec="minutes")
    S.FROZEN_A5.write_text(json.dumps({"approved": False, "generated": now, "code_commit": commit, "addendum": 5,
                                       "pipeline_limit": S.DH5_LIMIT, "calib_selection_limit": S.DH5_LIMIT,
                                       "sahi_limit": "none (as the SAHI tuning)", "rows": rows}, indent=1))
    add = _git("log", "--reverse", "--format=%h %cI", "-S## Addendum 5", "--", "docs/PREREGISTRATION_DOTA.md")
    add_h, add_t = add.splitlines()[0].split()
    tun = {m: json.loads((S.dh5_dir(m) / "tuning.json").read_text()) for m in S.A5_MODELS}
    sahi = {m: json.loads((F.P.OUT / f"{m}_1024/sahi_tuning.json").read_text()) for m in S.A5_MODELS}
    gate = {m: json.loads((F.P.OUT / f"{m}_1024/sahi_scahi_gate.json").read_text()) for m in S.A5_MODELS}
    log = LOG.read_text() if LOG.exists() else ""
    n_warn = len(re.findall(r"\[dh5\] WARNING task", log))
    n_def = len(re.findall(r"deferred to a run with a larger budget", log))
    peak = max(r["peak_gib_max"] for t in tun.values() for st in ("F2", "RG") for r in t[st]["rows"])
    L = [
        "# FREEZE_PROPOSAL_DOTA_A5 (Addendum 5, D-H6: SAHI, SAHI-F2, SAHI-RG für YOLO11s / YOLO12s – Vorschlag, "
        "Stopp für die Freigabe)",
        "",
        f"Erzeugt {now} von `scripts/dota/d_freeze_proposal_a5.py` aus den calib-Artefakten (Code-Commit `{commit}`); "
        "Zeilen in `configs/dota/frozen_rows_a5.json` (`approved: false`). **Keine DOTA-val-Zahl dieser Zeilen liegt "
        "vor;** auch SAHIs eigene DOTA-val-Rohvorhersagen werden erst nach der Freigabe berechnet (Addendum 5, Punkt 10). "
        "DOTA-val wird danach einmal ausgewertet, für diese sechs Zeilen. Die bisherigen Freezes bleiben unverändert.",
        "",
        "## Was eingefroren wird",
        "",
        "Je Modell (YOLO11s, YOLO12s; 1024er Gewichte, ein dichter Kopf): **SAHI** mit der auf calib gewählten "
        "Konfiguration (Verfahren und Raster der eingefrorenen YOLOv10-SAHI-Zeilen: drei Überlappungen, Raster v2, "
        f"{len(sahi['yolo11s']['rows'])} Konfigurationen je Modell, Auswahl max. calib-mAP50; Pass-Nachverarbeitung "
        f"Ultralytics-NMS, conf 0,001, IoU {S.A5_IOU}); **SAHI-F2** und **SAHI-RG** auf genau den Pässen dieser "
        "SAHI-Konfiguration (Slices 640 px nach `sahi.slicing.get_slice_bboxes` plus SAHIs Standard-Pass mit langer "
        "Seite 640; Cache-Ebenen `s640f`, `w640f`), derselbe Kopf, Fusion F2. Alle Parameter auf calib gewählt "
        f"(`scahi.dota.pipeline.select`), Pipeline-Grenze {S.DH5_LIMIT} für die SCAHI-Zeilen bei Auswahl, Replay und "
        "DOTA-val; SAHI ohne Grenze wie im SAHI-Tuning.",
        "",
        "Hypothese unverändert (Addendum 5, Punkt 7): D-H6 Δ = SAHI-RG − SAHI auf DOTA-val, primär mAP50 ohne Cap, "
        "Kriterium Δ > 0,30, gepaarter Bootstrap (1000, Seed 0), Holm über die zwei Modelle, α = 0,05; D-H6 gilt, wenn "
        "beide Modelle sie erfüllen. Sekundär ohne Urteil: SAHI-F2 − SAHI, Cap 4000, „alle Objekte“, AP@[.5:.95], "
        "Größenbins. Das D-H5-Urteil (YOLOv10) bleibt unverändert. Latenz: kein exklusives Fenster vor Freitag "
        "(FCOS ohne weitere Pausen), daher für diese Zeilen keine Latenz (Punkt 8).",
        "",
        "## Reihenfolge und Gate",
        "",
        f"- Addendum 5 committet `{add_h}` ({add_t}), vor jeder SAHI-Rohvorhersage, jedem Cache, Gate und Tuning dieser "
        "Zeilen.",
        "- SAHI-Tuning: " + "; ".join(
            f"{m} {_cfg(next(r for r in rows if r['id'] == f'SAHI_{m}'))}, calib-mAP50 {sahi[m]['selected']['mAP50']:.3f}"
            for m in S.A5_MODELS) + ".",
        "- Paritäts-Gate (Regel von `pipeline.gate`, Addendum 5 Punkt 5): "
        + "; ".join(f"{m}: Dekodierung {'bestanden' if g['passed_decode'] else 'NICHT bestanden'}, calib-mAP50 der "
                    f"gewählten SAHI-Konfiguration aus den Cache-Pässen {g['map']['mAP50_cache']:.4f} vs. "
                    f"{g['map']['mAP50_frozen']:.4f} (|Δ| {g['map']['abs_delta']:.0e})" for m, g in gate.items())
        + " → **bestanden** (`runs/dota/fusion/<m>_1024/sahi_scahi_gate.json`). Der deskriptive DOTA-val-Vergleich "
        "mit SAHIs eigenen Rohvorhersagen folgt mit diesen im DOTA-val-Lauf (DECISIONS 2026-11-03).",
        "",
        "## Replay auf calib (Auswertungspfad von DOTA-val, `scahi.dota.sahi_scahi.a5_row_dets`)",
        "",
        f"- **{n_exact}/{len(rows)} Zeilen exakt** (mAP50 und mAP50 Cap 4000), {n_missing} fehlen "
        "(`runs/dota/freeze/replay_a5.json`). Das SCAHI-Tuning lief aufgeteilt (eine Aufgabe je Bild und Einstellung); "
        "der Replay rechnet jede Zeile in einem Prozess über alle calib-Bilder, wie später DOTA-val. SAHI-Zeilen: Merge "
        "der gespeicherten calib-Rohvorhersagen mit der gewählten Konfiguration.",
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
        L.append(f"| `{r['id']}` | {TEXT[r['row']]} | `{r['weights_sha256'][:12]}` | {_cfg(r)} | "
                 f"{r['calib_mAP50']:.3f} | {r['calib_mAP50_cap4000']:.3f} | {tr} | {r['calib_images_at_limit']} | {rs} |")
    L += ["", "Referenz des verfeinerten Rasters je Modell = calib-gewählte SAHI-F2-Einstellung (Feld `reference_cfg`). "
          "Quellen je Zeile: `configs/dota/frozen_rows_a5.json` (Feld `source`).", ""]
    L += TR.lines_de({f"SAHI-RG {m}": TR.sensitivity(tun[m]["RG"]["rows"]) for m in S.A5_MODELS})
    L += [
        "## Pipeline-Grenze",
        "",
        f"SCAHI-Zeilen: Auswahl, Replay und DOTA-val mit derselben Grenze {S.DH5_LIMIT} (wie D-H5). Kein calib-Bild "
        "erreicht sie bei den gewählten Einstellungen (Spalte „Grenze“); größte Detektionszahl je Bild: "
        + "; ".join(f"{m} F2 {tun[m]['selected']['F2']['dets_per_image_max']}, RG {tun[m]['selected']['RG']['dets_per_image_max']}"
                    for m in S.A5_MODELS) + ". SAHI-Zeilen: keine Grenze je Bild, je Pass "
        f"`max_det` {F.P.PIPE} (wie die YOLOv10-SAHI-Zeilen); größte Detektionszahl je Bild "
        + "; ".join(f"{m} {sahi[m]['selected']['dets_per_image_max']}" for m in S.A5_MODELS) + ".",
        "",
        "## Durchführung (nur Ablauf, ohne Einfluss auf Ergebnisse)",
        "",
        "Parallel zum FCOS-Tuning: SAHI-Tuning und SCAHI-Tuning im Slot (MemoryMax 10G, MemorySwapMax 0, "
        "oom_score_adj 1000, Kerne 0–1), Caches und Gate auf der GPU; eine Aufgabe je (Bild, Einstellung) in einem "
        "frischen Prozess; Zeitregel nach Option 3 (Referenz im selben Prozess je Aufgabe, abwechselnde Reihenfolge). "
        f"Zurückgestellte Aufgaben: {n_def}. Speicherwarnungen (Spitze > Schätzung, nur Planung): {n_warn}; höchste "
        f"Spitze einer Aufgabe {peak:.2f} GiB bei 10 GB Deckel; kein Abbruch, kein OOM. Laufzeiten: "
        + "; ".join(f"{m} F2 {tun[m]['F2']['wall_s'] / 60:.0f} min, RG {tun[m]['RG']['wall_s'] / 3600:.1f} h"
                    for m in S.A5_MODELS) + ". Ein Gate-Lauf brach vor dem Urteil ab (fehlende DOTA-val-Rohvorhersagen "
        "im deskriptiven Teil) und wurde ohne Änderung der Gate-Regel wiederholt (DECISIONS 2026-11-03).",
        "",
        "## Nach der Freigabe",
        "",
        "SAHI-Rohvorhersagen DOTA-val der gewählten Überlappung (GPU, ≈ 2 × 4 min), Detektionen aller sechs Zeilen über "
        "`a5_row_dets`, Auswertung wie D-H5, D-H6 (Holm über die zwei Modelle) und Sekundärkontraste. Passt der Lauf "
        "nicht mehr vor den Stopp auf host-a am Freitag, läuft er auf host-b (HANDOFF).",
        "",
        "**Stopp: Freigabe dieser sechs Zeilen nötig, bevor DOTA-val berührt wird.**",
        "",
    ]
    DOC.write_text("\n".join(L))
    print(f"{DOC}: {len(rows)} rows, replay exact {n_exact}, missing {n_missing}")


if __name__ == "__main__":
    main()
