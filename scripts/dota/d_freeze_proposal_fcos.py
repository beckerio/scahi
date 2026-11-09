# SPDX-License-Identifier: AGPL-3.0-or-later
"""FREEZE_PROPOSAL_DOTA_FCOS (addendum 3): rows of FCOS from the calib artefacts, replay status ->
docs/FREEZE_PROPOSAL_DOTA_FCOS.md and configs/dota/frozen_rows_fcos.json (``approved: false`` until the user approves;
nothing here reads DOTA-val). The approved freezes of the YOLO models (frozen_rows.json, frozen_rows_new.json) are not
touched.

    python scripts/dota/fcos_limit_check.py calib    # replay of every row on calib + effect of the limit (GPU, CPU)
    python scripts/dota/d_freeze_proposal_fcos.py     # write the proposal
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from d_freeze_proposal import ROW_TEXT, _fmt_cfg  # noqa: E402

from scahi.dota import final as F  # noqa: E402
from scahi.dota import train as T  # noqa: E402
from scahi.machine import REPO_ROOT  # noqa: E402

DOC = REPO_ROOT / "docs/FREEZE_PROPOSAL_DOTA_FCOS.md"
FROZEN_FCOS = REPO_ROOT / "configs/dota/frozen_rows_fcos.json"
LIMIT = 200_000  # addendum 3 item 6: pipeline limit for the calib tuning and DOTA-val alike; kept (user, 2026-11-01)
NO_LIMIT = 10_000_000  # scripts/dota/fcos_limit_check.py
RULE_EPS = 0.05  # per-pass limit rule (user, 2026-11-02): |Δ mAP50| < 0.05 for every row -> limit kept
MODELS = (T.FCOS,)
LIMIT_CALIB = F.FREEZE_DIR / "limit_calib_fcos.json"  # scripts/dota/fcos_limit_check.py calib (replay + limit effect)
TEXT = {**ROW_TEXT, "A": "Einzelpass (torchvision-Prädiktor, sqrt(p·ctr)) bei s; conf auf calib",
        "X": "Inferenz-Kreuzung der 2048er Gewichte (sekundär): beste Größe aus 2048 / 2560 / 3072",
        "NMS-o2m": "Box-Fusion (IoS) der Pass-Detektionen des Detektors, dota<s>; Schwelle × conf auf calib (12)",
        "F2": "SCAHI vorab festgelegt (F2-Raster, 24) auf dota<s>",
        "RG": "SCAHI verfeinertes Raster (108 + Referenz, Zeit ≤ 2×) auf dota<s>"}


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def _birth(p: Path) -> dt.datetime:
    """Creation time of a file (statx birth time; mtime if the file system does not record it)."""
    w = int(subprocess.run(["stat", "-c", "%W", str(p)], capture_output=True, text=True).stdout.strip() or 0)
    return dt.datetime.fromtimestamp(w if w > 0 else p.stat().st_mtime).astimezone()


def prereg_order() -> dict:
    """Addendum 3 commit (first commit containing its heading) vs. the first FCOS training artefacts (addendum 3 run:
    S_max smoke, training logs, gates) - the addendum must predate all of them, and its numbered items must be unchanged
    since that commit."""
    doc = "docs/PREREGISTRATION_DOTA.md"
    h, t = _git("log", "--reverse", "--format=%H %cI", "-S## Addendum 3", "--", doc).splitlines()[0].split()
    t_add = dt.datetime.fromisoformat(t)
    arts = [T.DOTA_RUNS / "fcos_smoke.json", *sorted(T.DOTA_RUNS.glob("train/fcos_*_s0/train_log.jsonl")),
            *sorted(F.P.OUT.glob("fcos_*/gate.json")), *sorted(F.P.OUT.glob("fcos_*/single_calib.json"))]
    first = {str(p.relative_to(REPO_ROOT)): _birth(p) for p in arts if p.exists()}
    code = _git("log", "--reverse", "--format=%h %cI", "--", "scahi/dota/fcos_train.py").splitlines()[0].split()

    def items(rev: str) -> str:
        s = _git("show", f"{rev}:{doc}")
        s = s[s.index("## Addendum 3"):]
        return s[s.index("\n1. "):].split("\n*Korrektur")[0].split("\n## ")[0].strip()

    return {"commit": h, "commit_time": t_add, "first_artefacts": first, "training_code": code,
            "predates_all": all(t_add < v for v in first.values()) and t_add < dt.datetime.fromisoformat(code[1]),
            "code_descends": subprocess.run(["git", "merge-base", "--is-ancestor", h, code[0]], cwd=REPO_ROOT).returncode == 0,
            "items_unchanged": items(h) == items("HEAD")}


def _limit_calib() -> dict:
    """The calib limit check: whole file, or the per-size parts (1024 first, 2048 later; user 2026-11-02) merged."""
    if LIMIT_CALIB.exists():
        return json.loads(LIMIT_CALIB.read_text())
    parts = [json.loads(f.read_text()) for f in sorted(LIMIT_CALIB.parent.glob(f"{LIMIT_CALIB.stem}_*.json"))]
    if not parts:
        return {"rows": {}}
    same = [p.get("nolimit_dense_identical", {}) for p in parts]
    return {**parts[0], "rows": {k: v for p in parts for k, v in p["rows"].items()},
            "parts": [p.get("sizes") for p in parts],
            "nolimit_dense_identical": {"n": sum(d.get("n", 0) for d in same), "all": all(d.get("all", True) for d in same),
                                        "differs": [x for d in same for x in d.get("differs", [])]}}


def precheck(rows: list, lc: dict) -> list[str]:
    """Automatic pre-check reported in the proposal (user 2026-11-05: it never sets ``approved``; the approval before the
    held-out evaluation is the authors' decision): exact calib replay and limit rule for every row, both sizes present.
    Returns the failed conditions (empty = all met)."""
    bad = []
    for r in rows:
        e = lc["rows"].get(r["id"])
        if e is None:
            bad.append(f"{r['id']}: no limit check / replay")
            continue
        if e.get("replay_abs_diff") != 0:
            bad.append(f"{r['id']}: replay |Δ| {e.get('replay_abs_diff')}")
        if not abs(e["delta"]) < RULE_EPS:
            bad.append(f"{r['id']}: limit rule |Δ mAP50| {abs(e['delta']):.3f} >= {RULE_EPS}")
    if {r["s"] for r in rows} != {1024, 2048}:
        bad.append(f"sizes {sorted({r['s'] for r in rows})}")
    return bad


def main() -> None:
    rows = F.proposal_rows(("single", "cache"), MODELS)
    lc = _limit_calib()
    bad = precheck(rows, lc)
    rep = {k: {**v, "abs_diff": v["replay_abs_diff"]} for k, v in lc["rows"].items() if "replay_abs_diff" in v}
    n_exact = sum(1 for r in rows if r["id"] in rep and rep[r["id"]]["abs_diff"] == 0)
    n_missing = sum(1 for r in rows if r["id"] not in rep)
    commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                            text=True).stdout.strip()
    now = dt.datetime.now().isoformat(timespec="minutes")
    FROZEN_FCOS.write_text(json.dumps({"approved": False, "generated": now, "code_commit": commit, "addendum": 3,
                                       "pipeline_limit": LIMIT, "calib_selection_limit": LIMIT,
                                       "precheck": {"all_met": not bad, "failed": bad}, "rows": rows},
                                      indent=1))
    smoke = json.loads((T.DOTA_RUNS / "fcos_smoke.json").read_text())
    gates = {s: json.loads((F.P.OUT / f"fcos_{s}/gate.json").read_text()) for s in (1024, 2048) if s <= T.s_max(T.FCOS)}
    tr = {s: json.loads((T.weights(T.FCOS, s).parent.parent / "training.json").read_text()) for s in gates}
    L = [
        "# FREEZE_PROPOSAL_DOTA_FCOS (Addendum 3: FCOS – Vorschlag, Stopp für die Freigabe)",
        "",
        f"Erzeugt {now} von `scripts/dota/d_freeze_proposal_fcos.py` aus den calib-Artefakten (Code-Commit `{commit}`); "
        "Zeilen in `configs/dota/frozen_rows_fcos.json` (`approved: false`). **Keine DOTA-val-Zahl von FCOS liegt vor;** "
        "DOTA-val wird nach der Freigabe einmal ausgewertet, nur für diese Zeilen. Die freigegebenen Freezes der vier "
        "YOLO-Modelle bleiben unverändert.",
        "",
        "## Automatische Vorprüfung (setzt keine Freigabe)",
        "",
        "Exakter calib-Replay jeder Zeile, Limit-Regel |Δ mAP50| < " + f"{RULE_EPS} in jeder Zeile, beide Größen (1024, 2048): "
        + ("**alle erfüllt.**" if not bad else "**nicht erfüllt:** " + "; ".join(bad)),
        "Die Freigabe vor der Auswertung auf DOTA-val trifft der Nutzer (Autoren); erst danach startet "
        "`scripts/dota/fcos_val_auto.sh`.",
        "",
        "## Was eingefroren wird",
        "",
        f"FCOS (torchvision `fcos_resnet50_fpn`, COCO_V1), s ∈ {{1024, 2048}} (S_max = {smoke['s_max']} laut Speicher-Smoke "
        "auf den 48 dichtesten train_main-Bildern, `runs/dota/fcos_smoke.json`): A, NMS (Pass-Detektionen des Detektors), "
        "F2, RG, B-F2, B-RG; Inferenz-Kreuzung X_2048. Kein SAHI (Addendum 3, Punkt 7). Score überall sqrt(p_c · ctr). "
        "Gewichte = calib-beste Epoche. Alle Parameter auf calib gewählt.",
        "",
        "Hypothesen unverändert (Addendum 3, Punkt 9): D-H1 A_2048 − NMS_1024; D-H2a/b RG_s − NMS_s; D-H3 RG_2048 − A_2048. "
        "Ein Modell → kein Holm; Urteil mit dem unadjustierten einseitigen p ≤ 0,05 und dem Kriterium (ε = 0,30; D-H3 "
        "zusätzlich untere KI-Grenze > 0). Zusätzlich deskriptiv die Anzahl der Detektoren (von fünf), die eine Hypothese "
        f"erfüllen. Berichtsregeln wie in den YOLO-Freezes. Pipeline-Grenze {LIMIT} für calib-Auswahl und DOTA-val.",
        "",
    ]
    by = {r["id"]: r for r in rows}
    x = by.get(f"X_{T.FCOS}_2048")
    xs = f"X_2048 = Einzelpass der 2048er Gewichte bei {x['imgsz']} px (auf calib gewählt)" if x else "X_2048 fehlt"
    L += [
        "## Vergleiche (Bericht nach DOTA-val)",
        "",
        "| Vergleich | Δ | Rolle |",
        "|---|---|---|",
        "| D-H1 | A_2048 − NMS_1024 | primär |",
        "| D-H2a | RG_1024 − NMS_1024 | primär |",
        "| D-H2b | RG_2048 − NMS_2048 | primär |",
        "| D-H3 | RG_2048 − A_2048 (Einzelpass bei 2048) | primär |",
        f"| Inferenz-Kreuzung, neben D-H3 berichtet | RG_2048 − X_2048 ({xs}) | sekundär, vorab festgelegt |",
        "| F2 statt RG | F2_s − NMS_s, F2_2048 − A_2048 | sekundär |",
        "| Zerlegung | B-F2_s − A_s, F2_s − B-F2_s, B-RG_s − A_s, RG_s − B-RG_s | sekundär |",
        "",
    ]
    po = prereg_order()
    t0 = min(po["first_artefacts"].values())
    L += [
        "## Präregistrierung vor dem ersten FCOS-Ergebnis",
        "",
        f"- Addendum 3 (`docs/PREREGISTRATION_DOTA.md`): Commit `{po['commit'][:12]}`, "
        f"{po['commit_time']:%Y-%m-%d %H:%M:%S %z} (Commit-Zeit aus git).",
        f"- Erster FCOS-Trainingscode (`scahi/dota/fcos_train.py`): Commit `{po['training_code'][0]}`, "
        f"{dt.datetime.fromisoformat(po['training_code'][1]):%H:%M:%S}; Nachfahre von Addendum 3: "
        f"{'ja' if po['code_descends'] else 'NEIN'}.",
        f"- Erstes Artefakt des Addendum-3-Laufs: {t0:%Y-%m-%d %H:%M:%S} ({(t0 - po['commit_time']).total_seconds() / 60:.0f} "
        "min nach dem Commit). Erstellungszeiten: "
        + ", ".join(f"`{k}` {v:%H:%M:%S}" for k, v in sorted(po["first_artefacts"].items(), key=lambda kv: kv[1])) + ".",
        f"- **Addendum 3 liegt vor jedem FCOS-Trainingsergebnis: {'ja' if po['predates_all'] else 'NEIN'}.** Punkte 1–11 "
        f"seit dem Commit unverändert: {'ja' if po['items_unchanged'] else 'NEIN'} (danach nur die datierte Korrektur der "
        "Zeitangabe in der Überschrift).",
        "- Vor dem Addendum lagen nur die Ergebnisse des Machbarkeits-Spikes vor (`runs/fcos/*.json`, 20:19–20:23, "
        "`reports/FCOS_SPIKE.md`, Commit `b85ad3f`): Decodier-Parität, Speicher, ein 2-Epochen-Smoke auf 64 Bildern "
        "(calib-mAP50 3,1, ohne Aussagekraft). Diese Gewichte werden nicht verwendet; das Addendum beruft sich auf den "
        "Spike nur für Rezept und Speichergrenzen.",
        "",
        "## Training und Paritäts-Gate",
        "",
    ]
    for s in gates:
        g = gates[s]
        L.append(f"- {s} px: {tr[s]['selection']}; Batch {tr[s]['batch']} × Akkumulation {tr[s]['accumulation']}. "
                 f"Gate {'bestanden' if g['passed'] else 'NICHT bestanden'}: Nachverarbeitung bitgleich "
                 f"{sum(r['identical'] for r in g['rows'])}/{len(g['rows'])}, calib-mAP50 eigen / torchvision "
                 f"{g['map_identity']['own']:.3f} / {g['map_identity']['torchvision']:.3f} "
                 f"(|Δ| {g['map_identity']['abs_delta']:.1e}) (`runs/dota/fusion/fcos_{s}/gate.json`).")
    L += [
        "",
        "## Replay auf calib (Auswertungspfad von DOTA-val, `scahi.dota.final.row_dets`, Grenze "
        f"{LIMIT})",
        "",
        f"- **{n_exact}/{len(rows)} Zeilen exakt**, {n_missing} fehlen (`{LIMIT_CALIB.relative_to(REPO_ROOT)}`).",
        "",
        "## Pipeline-Grenze 200 000 (Entscheidung des Nutzers, 2026-11-01, vor DOTA-val)",
        "",
        f"Die Grenze {LIMIT} bleibt für FCOS (je Bild und je Pass), wie für alle YOLO-Zeilen. **Begründung:** dieselbe "
        "Pipeline für alle Detektoren; abgeschnitten werden nur die Detektionen mit den niedrigsten Scores. Addendum 3, "
        "Punkt 6 („die Grenze darf nicht greifen“) wird damit für FCOS nicht erfüllt; Auswahl und Zeilen bleiben, wie sie "
        "sind (kein Nachtuning). Nach der einmaligen DOTA-val-Auswertung: je Zeile die Zahl der Bilder an der Grenze und "
        "explorativ mAP50 ohne Grenze für jede betroffene Zeile (wie bei P1384).",
        "",
        f"calib, je Zeile (`{LIMIT_CALIB.relative_to(REPO_ROOT)}`, `scripts/dota/fcos_limit_check.py calib`): Bilder an "
        "der Grenze je Bild, Bilder mit einem Pass an der Grenze (nur Box-Fusion, die die Pass-Detektionen nutzt) und "
        f"Δ mAP50 = ohne Grenze ({NO_LIMIT:,}) − mit Grenze; die Kandidatengrenze je FPN-Ebene (200 000, Addendum 3, "
        "Punkt 6) bleibt.",
        "",
    ]
    L += ["**Regel für die Grenze je Pass (Nutzer, 2026-11-02, festgelegt vor jeder Zahl der Limit-Prüfung):** Ist "
          f"|Δ mAP50| (ohne − mit Grenze) für **jede** FCOS-Zeile auf calib < {RULE_EPS}, bleibt {LIMIT}; sonst gilt "
          f"{NO_LIMIT:,} für **alle** FCOS-Zeilen (Box-Fusion und SCAHI sehen dieselben Detektionen). **Bedeutung, falls "
          "die Regel greift (Nutzer, 2026-11-02, vor jeder Zahl der Limit-Prüfung):** Die gewählten Einstellungen bleiben, "
          f"wie auf calib mit {LIMIT:,} getunt; nur der calib-Replay und die DOTA-val-Auswertung **aller** FCOS-Zeilen "
          f"nutzen {NO_LIMIT:,}. Kein Nachtuning – wie bei den YOLO-Zeilen (gewählt bei 20 000, ausgewertet bei 200 000).",
          ""]
    if lc["rows"]:
        mx = max(abs(v["delta"]) for v in lc["rows"].values())
        L += [f"**Ergebnis:** max |Δ mAP50| = {mx:.4f} → "
              + (f"Grenze {LIMIT} bleibt." if mx < RULE_EPS else
                 f"**Regel greift: calib-Replay und DOTA-val-Auswertung aller FCOS-Zeilen mit {NO_LIMIT:,}; Auswahl "
                 "unverändert (calib-Replay mit dieser Grenze = Spalte „mAP50 ohne Grenze“).**"), ""]
        d = lc.get("nolimit_dense_identical", {})
        L += [f"Kontrolle: dichte Ausgaben der Pässe ohne Grenze identisch mit den Caches: "
              f"{'ja' if d.get('all', True) else 'NEIN ' + str(d.get('differs'))} ({d.get('n', 0)} Pässe-Ebenen).", "",
              "| id | Bilder an der Grenze (Bild) | Bilder mit Pass an der Grenze | mAP50 mit Grenze | mAP50 ohne Grenze "
              "| Δ mAP50 |", "|---|---:|---:|---:|---:|---:|"]
        for k, v in lc["rows"].items():
            if v["images_affected"]:
                L.append(f"| `{k}` | {v['images_at_image_limit']} | {v['images_with_pass_at_limit']} | "
                         f"{v['mAP50_limited']:.4f} | {v['mAP50_unlimited']:.4f} | {v['delta']:+.2e} |")
        rest = [k for k, v in lc["rows"].items() if not v["images_affected"]]
        L += ["", "Nicht betroffen (Δ = 0): " + (", ".join(f"`{k}`" for k in rest) or "keine") + ".", ""]
    else:
        L += ["**Limit-Prüfung auf calib fehlt noch.**", ""]
    from scahi.dota import time_rule as TR

    sens = {}
    for s in (1024, 2048):
        f = F.P.OUT / f"fcos_{s}/tuning.json"
        tu = json.loads(f.read_text()) if f.exists() else {}
        for row, lab in (("C", "RG"), ("B", "B-RG")):
            st = tu.get(f"RG{row}_o2m")
            sens[f"{lab}_{s}"] = TR.sensitivity(st["rows"]) if st else None
    (F.FREEZE_DIR / "time_rule_sensitivity_fcos.json").write_text(json.dumps(sens, indent=1))
    L += TR.lines_de(sens)
    L += [
        "## Zeilen",
        "",
        f"| id | Zeile | Gewichte SHA-256 (12) | Wahl auf calib | calib-mAP50 | Grenze | Replay ({LIMIT}) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        x = rep.get(r["id"])
        rp = "—" if x is None else ("exakt" if x["abs_diff"] == 0 else f"Δ {x['abs_diff']:.2e}")
        L.append(f"| `{r['id']}` | {TEXT[r['row']]} | `{r['weights_sha256'][:12]}` | {_fmt_cfg(r)} | "
                 f"{r['calib_mAP50']:.3f} | {r.get('calib_images_at_limit', 0)} | {rp} |")
    for r in rows:
        if "crossing_calib_mAP50" in r:
            L += ["", "Inferenz-Kreuzung (calib-mAP50): "
                  + ", ".join(f"{k}: {v:.3f}" for k, v in sorted(r["crossing_calib_mAP50"].items(), key=lambda t: int(t[0])))]
    L += ["", "Quellen je Zeile: `configs/dota/frozen_rows_fcos.json` (Feld `source`).", ""]
    DOC.write_text("\n".join(L))
    print(f"{DOC}: {len(rows)} rows, replay exact {n_exact}, missing {n_missing}; pre-check "
          + ("all met" if not bad else "NOT met: " + "; ".join(bad)) + " (approved: false - the user decides)")


if __name__ == "__main__":
    {"replay-gpu": lambda: replay("gpu"), "replay-cpu": lambda: replay("cpu")}.get(
        sys.argv[1] if len(sys.argv) > 1 else "", main)()
