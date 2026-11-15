# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO26m (addendum 7c01b5d) / YOLO26s (addendum 7bc0ace) freeze proposal and its calib replay gate. Nothing on val or
test-dev.

    python scripts/ext26_freeze_proposal.py draft                 # -> configs/ext/frozen_rows_ext26.draft.json
    python scripts/ext26_freeze_proposal.py replay cache [--workers 7]
    python scripts/ext26_freeze_proposal.py replay predictor [--tag a|b]
    python scripts/ext26_freeze_proposal.py doc                   # -> docs/FREEZE_PROPOSAL_EXT26.md
    ... --prep ultralytics   # cache rows on the Ultralytics-aligned caches (cache_ualigned/; back-mapping rule applied
                             # before the freeze, DECISIONS 2026-11-14): tuning files of runs/ext/yolo26m/ualigned/,
                             # cache replay into runs/ext/yolo26m/replay_calib_ualigned/; single-pass / SAHI rows
                             # (Ultralytics predictor) unchanged, their two replays carried over from replay_calib/

Rows as for YOLOv10m in configs/ext/frozen_rows_ext.json (same ids with yolo26m, same kinds and fields): E single pass
1280 / 2048 x o2o / o2m, E-MZ-NMS and E-MZ-DENSE (F2) 1280 / 2048 x o2m / o2o, E-SAHI (1280, o2o), X-grid B / C
1280 / 2048 x o2m / o2o (exploratory), decomposition rows D (whole-image pass + F2 with row C's parameters,
exploratory). Every choice is read from the calib tuning files written by scripts/ext26_calib.py; nothing is
re-selected here. Each row carries ``calib_AP`` = the calib value of its frozen configuration as recorded at selection.

YOLO26s (``--model yolo26s --prep ultralytics``; only this preparation exists for it): same rows and fields, files
configs/ext/frozen_rows_ext26s.draft.json and docs/FREEZE_PROPOSAL_EXT26S.md, tuning files runs/ext/yolo26s/ualigned/;
its predictor rows are replayed twice (nothing to carry over).

Replay gate (calib): every frozen row is recomputed on calib through the val / test-dev code path
(scahi.ext_testdev.run_cache_row / run_predictor_row, unchanged; output root runs/ext/yolo26m/replay_calib/) in fresh
processes, and compared with ``calib_AP``. Cache rows use the same fusion code as the tuning -> exact equality required.
Predictor rows: the val / test-dev path times one image at a time (batch 1; SAHI merges online), the tuning used batch
16 (single pass) and offline merging of raw slice predictions (SAHI) - the difference is reported; exactness is
required between two independent replays (``--tag a`` / ``--tag b``).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scahi import ext as E  # noqa: E402
from scahi import ext_testdev as T  # noqa: E402

M = "yolo26m"
D = E.mdir(M)
DRAFT = REPO / "configs/ext/frozen_rows_ext26.draft.json"
DOC = REPO / "docs/FREEZE_PROPOSAL_EXT26.md"
REPLAY = D / "replay_calib"
GATE = REPLAY / "gate_replay_calib.json"
ADD = E.ADDENDUM_NEW_MODELS
P_WEIGHTS = (f"training recipe fixed by design (PREREGISTRATION_EXT 2, addendum {ADD}); epoch = max toolkit-port AP "
             "on calib (o2o) at the 36 scheduled epochs")
P_STACK = "fixed by design: pass at the training size + z = 2 tiles (N = 1024), cache overlap 0.3 (= R)"
P_BRANCH = f"fixed by design (two-branch head as YOLOv10m: both branches, o2o = hypothesis branch; addendum {ADD})"
LABEL = "YOLO26m"
OTHER_FILES = {"yolo26s": ("configs/ext/frozen_rows_ext26s.draft.json", "docs/FREEZE_PROPOSAL_EXT26S.md")}


def set_model(m: str) -> None:
    """Module globals for one model (defaults above = YOLO26m, unchanged)."""
    global M, D, DRAFT, DOC, REPLAY, GATE, ADD, P_WEIGHTS, P_BRANCH, TD, LABEL
    if m == M:
        return
    M, D, LABEL = m, E.mdir(m), "YOLO" + m[4:]
    DRAFT, DOC = (REPO / x for x in OTHER_FILES[m])
    REPLAY, TD = D / "replay_calib", D
    GATE = REPLAY / "gate_replay_calib.json"
    ADD = E.ADDENDUM_BY_MODEL[m]
    P_WEIGHTS = (f"training recipe fixed by design (PREREGISTRATION_EXT 2, addendum {ADD}); epoch = max toolkit-port "
                 "AP on calib (o2o) at the 36 scheduled epochs")
    P_BRANCH = f"fixed by design (two-branch head as YOLOv10m: both branches, o2o = hypothesis branch; addendum {ADD})"


def _epoch_ckpt(sz: int) -> Path:
    """Checkpoint of the calib-selected epoch (training.json 'selection': '... best epoch N (...)')."""
    sel = json.loads((E.train_dir(M, sz) / "training.json").read_text())["selection"]
    return E.weights(M, sz).parent / f"calib_epoch{int(sel.split('best epoch ')[1].split()[0]):03d}.pt"


LEVELS = {1280: ["w1280", "t2.0n1024"], 2048: ["w2048", "t2.0n1024"]}
TD = D  # directory of the cache-row tuning files (tuning.json, xgrid_calib.json, gate_GE3.json)
PREP = None
UROOT = REPO / "cache_ualigned"
PREP_NOTE = ("cache rows on caches with Ultralytics' input preparation (size rule, letterbox gain, per-axis back-mapping "
             "as scale_boxes; scahi.mapcheck.ultralytics_prep, cache root cache_ualigned/); back-mapping rule of "
             "DECISIONS 2026-11-02 10:15 / 10:29 applied before the freeze (DECISIONS 2026-11-14)")


def set_prep(prep: str | None) -> None:
    global TD, PREP, REPLAY, GATE
    if prep == "ultralytics":
        from scahi.fusion import run as R

        PREP, TD, REPLAY = prep, D / "ualigned", D / "replay_calib_ualigned"
        GATE = REPLAY / "gate_replay_calib.json"
        R.CACHE_ROOT = UROOT


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def draft() -> None:
    from scahi.infer import cache_mz as cm

    sha, rows = {}, []
    tuning = json.loads((TD / "tuning.json").read_text())
    xg = json.loads((TD / "xgrid_calib.json").read_text())["results"]
    tr = {sz: json.loads((E.train_dir(M, sz) / "training.json").read_text()) for sz in E.sizes(M)}
    for sz in E.sizes(M):
        w = E.weights(M, sz)
        sha[sz] = T.sha256(w)
        assert list(cm.STACK_LEVELS[E.STACK[sz]]) == LEVELS[sz], cm.STACK_LEVELS[E.STACK[sz]]
    base = lambda sz: {"model": M, "weights": _rel(E.weights(M, sz)), "weights_sha256": sha[sz],  # noqa: E731
                       "train_variant": f"E-{M}-{sz}",
                       "training": {k: tr[sz][k] for k in ("batch", "effective_batch", "oom_fallback", "selection")}}
    for sz in E.sizes(M):
        for br in ("o2m", "o2o"):
            cf = D / f"conf_tuning_{sz}_{br}.json"
            c = json.loads(cf.read_text())
            g = next(x for x in c["grid"] if x["conf"] == c["conf"])
            rows.append({"id": f"E_{M}_full{sz}_{br}", "group": "preregistered", "kind": "single", **base(sz),
                         "calib_AP": g["AP"], "calib_AP_source": f"{_rel(cf)} (predictor, batch 16)",
                         "inference": {"zoom_stack": [], "overlap": None, "fusion": "none" if br == "o2o" else "nms",
                                       "branch": br, "row": f"E-{sz}",
                                       "role": "hypothesis row" if br == "o2o" else "descriptive (single pass o2m+NMS)",
                                       "imgsz": sz, "conf": c["conf"], "conf_tuned_on": "calib", "tuning_file": _rel(cf),
                                       "addendum": ADD},
                         "provenance": {"weights": P_WEIGHTS, "imgsz": "fixed by design (training size)",
                                        "branch": P_BRANCH,
                                        "conf": f"calib ({_rel(cf)}, grid {list(E.CONF_GRID)}, max toolkit-port AP)"}})
    for sz in E.sizes(M):
        for var, tag, rowname, grid in (("F2", "F2", "E-MZ-DENSE", "F2 grid, 24 points"),
                                        ("F0-NMS", "nms-ios", "E-MZ-NMS", "NMS-IOS grid, 12 points")):
            for br in ("o2m", "o2o"):
                t = tuning[f"{sz}/{br}/{var}"]
                params = {k: v for k, v in t["best"].items() if k not in ("AP", "AP50", "source")}
                f = {"imgsz": sz, "levels": LEVELS[sz], "overlap": E.OVERLAP, "branch": br, "variant": var,
                     "params": params}
                rows.append({"id": f"E_{M}_mz{sz}_{tag}_{br}", "group": "preregistered", "kind": "cache", **base(sz),
                             "calib_AP": t["best"]["AP"], "calib_AP_source": f"{_rel(TD / 'tuning.json')} {sz}/{br}/{var}",
                             "inference": {"overlap": E.OVERLAP, "fusion": var, "branch": br,
                                           "row": f"{rowname}-{sz}", "stack": E.STACK[sz], "levels": LEVELS[sz],
                                           "params": params, "tuned_on": "calib", "addendum": ADD},
                             "fusion": f,
                             "provenance": {"weights": P_WEIGHTS, "levels / overlap": P_STACK, "branch": P_BRANCH,
                                            "params": f"calib ({grid}, max toolkit-port AP; "
                                                      f"{_rel(TD / 'tuning.json')}; budget {t['budget']})"}})
    sf = D / "sahi_tuning" / "sahi_tuning2_o2o.json"
    b = max(json.loads(sf.read_text())["best"].values(), key=lambda r: r["AP"])
    rows.append({"id": f"E_{M}_sahi_o2o", "group": "preregistered", "kind": "sahi", **base(1280),
                 "calib_AP": b["AP"], "calib_AP_source": f"{_rel(sf)} (raw slice predictions merged offline)",
                 "inference": {"zoom_stack": [], "overlap": b["overlap"],
                               "fusion": f"{b['merge'].lower()}-{b['metric'].lower()}", "branch": "o2o",
                               "row": "E-SAHI", "slice": 640, "full_image_pass": True, "match_thr": b["match_thr"],
                               "conf": b["conf"], "tuned_on": "calib", "tuning_grid": "v2", "tuning_file": _rel(sf),
                               "addendum": ADD},
                 "sahi": b,
                 "provenance": {"weights": P_WEIGHTS + " (1280 weights)", "branch": "fixed by design (o2o, as YOLOv10m)",
                                "slice / overlap / merge / match_thr / conf": f"calib (SAHI tuning grid v2, {_rel(sf)})"}})
    for sz in E.sizes(M):
        for br in ("o2m", "o2o"):
            for row in ("B", "C"):
                o = xg[f"{M}/{sz}/{br}/{row}"]
                lv = LEVELS[sz][:1] if row == "B" else LEVELS[sz]
                f = {"imgsz": sz, "levels": lv, "overlap": E.OVERLAP, "branch": br, "variant": "F2",
                     "params": o["selected"]}
                p = o["selected"]
                rows.append({"id": f"XG_{M}_{sz}_{br}_{row}", "group": "X-grid (exploratory, post hoc)", "kind": "cache",
                             **base(sz), "calib_AP": o["calib_selected"]["AP"],
                             "calib_AP_source": f"{_rel(TD / 'xgrid_calib.json')} {M}/{sz}/{br}/{row} calib_selected",
                             "inference": {"zoom_stack": [], "overlap": E.OVERLAP, "fusion": "F2", "branch": br,
                                           "row": f"X-grid-{row}-{sz}", "levels": lv, "params": p,
                                           "tuned_on": "calib (widened F2 grid, max AP, time <= 2x F2)"},
                             "fusion": f, "xgrid": {k: o[k] for k in ("time_cap_hit", "edges_unconstrained",
                                                                     "edges_under_cap", "rounds")},
                             "provenance": {"weights": P_WEIGHTS,
                                            "levels": "row B: whole-image pass only; row C: as E-MZ-DENSE",
                                            "params": "EXPLORATORY (refined grid; procedure frozen in "
                                                      "configs/ext/xgrid_frozen.json, addendum " + ADD + " item 2): "
                                                      "calib, max AP among variants with fusion time <= 2x F2"}})
    for sz in E.sizes(M):
        for br in ("o2m", "o2o"):
            t = tuning[f"{sz}/{br}/F2"]
            params = {k: v for k, v in t["best"].items() if k not in ("AP", "AP50", "source")}
            o = xg[f"{M}/{sz}/{br}/B"]
            ref = {k: o["calib_reference"][k] for k in ("tau_splat", "tau_peak", "cell_factor", "bins_per_oct")}
            same = all(float(ref[k]) == float(params[k]) for k in ref) and o["calib_reference"]["iou_cluster"] == 0.5
            f = {"imgsz": sz, "levels": LEVELS[sz][:1], "overlap": E.OVERLAP, "branch": br, "variant": "F2",
                 "params": params}
            rows.append({"id": f"D_{M}_single{sz}_F2_{br}", "group": "decomposition B (exploratory, post hoc)",
                         "kind": "cache", **base(sz),
                         "calib_AP": o["calib_reference"]["AP"] if same else None,
                         "calib_AP_source": (f"{_rel(TD / 'xgrid_calib.json')} {M}/{sz}/{br}/B calib_reference (same F2 "
                                             "parameters, whole-image pass)") if same else "none (no calib value)",
                         "inference": {"zoom_stack": [], "overlap": E.OVERLAP, "fusion": "F2", "branch": br,
                                       "row": f"D-single-{sz}", "levels": LEVELS[sz][:1], "params": params},
                         "fusion": f,
                         "provenance": {"weights": P_WEIGHTS,
                                        "levels": "whole-image pass of the C stack only (single pass + F2)",
                                        "params": "EXPLORATORY row; F2 parameters = the calib-tuned parameters of the "
                                                  f"corresponding row C ({_rel(TD / 'tuning.json')} {sz}/{br}/F2); "
                                                  "no new selection"}})
    for r in rows:  # same field as the committed extension freeze: the exp id the val / test-dev run writes to
        r["val_exp_id"] = r["id"]
        if PREP and r["kind"] == "cache":
            r["cache_prep"] = {"prep": PREP, "cache_root": _rel(UROOT), "note": PREP_NOTE}
    fz = {"status": "DRAFT - not approved; nothing evaluated on val or test-dev",
          "created": datetime.now().isoformat(timespec="minutes"), "model": M, "addendum": ADD,
          "split_for_test": "VisDrone val (548 images) once, then VisDrone test-dev (1610 images) once, after approval",
          "reporting_rules": [
              f"Pre-registered questions E-H1, E-H2 (per s), E-H3 as in the addendum: verdict per detector for {LABEL}, "
              f"paired bootstrap over val images (1000, seed 0), eps = 0.30; Holm appendix for {LABEL} alone "
              "(one-model family = unadjusted). The five-model verdicts, '>= 4 of 5' and the pooled effect are unchanged; "
              f"{LABEL} counts in no family criterion.",
              "Test-dev: the same definitions as held-out confirmation, reported next to val.",
              "Every frozen row is reported regardless of outcome; no retuning, re-selection or added rows after val.",
              "Exploratory rows (X-grid, decomposition D) in separate tables, labelled post hoc, never in a verdict.",
              "Primary protocol visdrone_ignore (toolkit port, cap 500 per image); plain as a secondary column."]
          + ([f"G-E2 (s model): E-1280 (o2o) within +-5 AP of YOLOv10s E-1280 (B-FULL 1280, val 33.66), checked after "
              "the single val evaluation; otherwise checked and reported, changes no verdict (addendum " + ADD + " item 6).",
              "Descriptive equal-compute comparison (addendum " + ADD + " item 8): single pass YOLO26s at 2048 px vs. "
              "single pass YOLO26m at 1280 px, both o2o (rows E-2048 / E-1280): GFLOPs per image (exact per input shape, "
              "scahi.flops, over the val / test-dev images) and AP on val and test-dev, delta with paired bootstrap CI "
              "(1000, seed 0); no verdict, no threshold."] if M == "yolo26s" else []),
          "weights": {str(sz): {"path": _rel(E.weights(M, sz)), "sha256": sha[sz],
                                "selected_epoch_checkpoint": _rel(_epoch_ckpt(sz))}
                      for sz in E.sizes(M)},
          "gates": {"G-E1": _rel(D / "gate_GE1.json"), "G-E3": _rel(TD / "gate_GE3.json")},
          **({"cache_prep": PREP_NOTE, "mapping_check": _rel(D / "mapping" / "check.json"),
              "superseded_draft": "configs/ext/frozen_rows_ext26.draft.json at bb1614a (single-gain caches)"}
             if PREP and M == "yolo26m" else {"cache_prep": PREP_NOTE.split("; back-mapping rule")[0] + " from the start "
                                              f"(addendum {ADD} item 4)"} if PREP else {}),
          "rows": rows}
    DRAFT.write_text(json.dumps(fz, indent=1))
    print(f"{len(rows)} rows -> {_rel(DRAFT)}")


def _fz() -> dict:
    fz = json.loads(DRAFT.read_text())
    fz["freeze_commit"] = "DRAFT (not approved)"
    return fz


def _cache_worker(args):
    T.REPLAY = REPLAY  # out_root() of the val / test-dev path reads this global; spawned workers need it set again
    if PREP:
        from scahi.fusion import run as R

        R.CACHE_ROOT = UROOT
    return T.run_cache_row(args)


def _cache_init(prep: str | None, model: str = "yolo26m") -> None:
    set_model(model)
    set_prep(prep)


def replay(part: str, workers: int, tag: str) -> None:
    fz = _fz()
    rows = [r for r in fz["rows"] if (r["kind"] == "cache") == (part == "cache")]
    T.REPLAY = REPLAY / tag if part == "predictor" else REPLAY
    T.REPLAY.mkdir(parents=True, exist_ok=True)
    if part == "cache":
        with ProcessPoolExecutor(workers, mp_context=E._spawn(), initializer=_cache_init, initargs=(PREP, M)) as ex:
            for rid in ex.map(_cache_worker, [(r, "calib", fz) for r in rows]):
                print(f"[replay] {rid}", flush=True)
    elif PREP and M == "yolo26m":  # predictor rows unchanged by the cache preparation: carry over the two replays of replay_calib/
        old = json.loads((D / "replay_calib" / "gate_replay_calib.json").read_text())["rows"]
        g = json.loads(GATE.read_text()) if GATE.exists() else {"rows": {}}
        for r in rows:
            o = old[r["id"]]
            assert o["calib_AP"] == r["calib_AP"], r["id"]
            g["rows"][r["id"]] = {**o, "carried_over_from": _rel(D / "replay_calib" / "gate_replay_calib.json")}
        GATE.write_text(json.dumps(g, indent=1))
        rows = []
    else:
        for r in rows:
            T.run_predictor_row(r, "calib", fz)
            print(f"[replay] {r['id']} ({tag})", flush=True)
    g = json.loads(GATE.read_text()) if GATE.exists() else {"rows": {}}
    for r in rows:
        ap_ = json.loads((T.REPLAY / r["val_exp_id"] / "metrics.json").read_text())["metrics"]["AP"]
        x = g["rows"].setdefault(r["id"], {"id": r["id"], "kind": r["kind"], "group": r["group"],
                                           "calib_AP": r["calib_AP"], "calib_AP_source": r["calib_AP_source"]})
        if part == "cache":
            x["replay_AP"] = ap_
            x["abs_diff"] = None if r["calib_AP"] is None else abs(ap_ - r["calib_AP"])
            x["exact"] = x["abs_diff"] == 0.0
        else:
            x[f"replay_AP_{tag}"] = ap_
            x["diff_vs_tuning"] = ap_ - r["calib_AP"]
            if "replay_AP_a" in x and "replay_AP_b" in x:
                x["exact_a_vs_b"] = x["replay_AP_a"] == x["replay_AP_b"]
    rs = list(g["rows"].values())
    cache = [x for x in rs if x["kind"] == "cache"]
    pred = [x for x in rs if x["kind"] != "cache"]
    g.update({"gate": f"{LABEL} calib replay (val / test-dev code path on calib)", "time":
              datetime.now().isoformat(timespec="minutes"), "freeze": "DRAFT (not approved)",
              "n_frozen_rows": len(fz["rows"]),
              "cache_rows": f"{sum(x.get('exact') is True for x in cache)}/{len(cache)} exact vs. the selection value"
                            f" ({sum(x.get('abs_diff') is None and 'replay_AP' in x for x in cache)} without one)",
              "predictor_rows": f"{sum(x.get('exact_a_vs_b') is True for x in pred)}/{len(pred)} identical in two "
                                "independent replays",
              "predictor_gate": "bit-identical in two independent replays AND |replay - tuning value| <= 0.05 AP "
                                "(authors 2026-11-14)",
              "predictor_rows_passed": f"{sum(x.get('exact_a_vs_b') is True and abs(x['diff_vs_tuning']) <= 0.05 for x in pred)}"
                                       f"/{len(pred)} (max |diff vs. tuning| "
                                       f"{max((abs(x['diff_vs_tuning']) for x in pred), default=float('nan')):.4f})"})
    GATE.write_text(json.dumps(g, indent=1))
    print(g["cache_rows"], "|", g["predictor_rows"], "|", g["predictor_rows_passed"])


def doc() -> None:
    fz = _fz()
    g = json.loads(GATE.read_text()) if GATE.exists() else {"rows": {}}
    f = lambda x: "–" if x is None else f"{x:.4f}"  # noqa: E731
    L = [f"# Freeze proposal {LABEL} (addendum {ADD}) – DRAFT, not approved", "",
         f"Created {fz['created']}. Nothing has been evaluated on val or test-dev. File: `{_rel(DRAFT)}`.", "",
         "## Weights", "", "| s | best_calib.pt | SHA-256 | selection |", "|---|---|---|---|"]
    for sz, w in fz["weights"].items():
        tr = next(r for r in fz["rows"] if r["weights"] == w["path"])["training"]
        L.append(f"| {sz} | `{w['path']}` | `{w['sha256']}` | {tr['selection']}; batch {tr['batch']}, "
                 f"effective {tr['effective_batch']} |")
    L += ["", "## Rows and the origin of every choice", "",
          "| id | group | choice | calib AP (selection) | replay | origin |", "|---|---|---|---|---|---|"]
    for r in fz["rows"]:
        inf = r["inference"]
        ch = (f"conf {inf['conf']}" if r["kind"] == "single" else
              f"{inf['fusion']} overlap {inf['overlap']} thr {inf['match_thr']} conf {inf['conf']}" if r["kind"] == "sahi"
              else ", ".join(f"{k} {v}" for k, v in r["fusion"]["params"].items()))
        x = g["rows"].get(r["id"], {})
        rp = (f"{f(x.get('replay_AP'))} (Δ {f(x.get('abs_diff'))})" if r["kind"] == "cache" else
              f"a {f(x.get('replay_AP_a'))} / b {f(x.get('replay_AP_b'))}")
        L.append(f"| `{r['id']}` | {r['group']} | {ch} | {f(r['calib_AP'])} | {rp} | "
                 f"{'; '.join(f'{k}: {v}' for k, v in r['provenance'].items())} |")
    if PREP and M == "yolo26m":
        L[3:3] = [f"Preparation: {PREP_NOTE}. Superseded: {fz['superseded_draft']}; mapping check "
                  f"`{fz['mapping_check']}`.", ""]
    elif PREP:
        L[3:3] = [f"Preparation: {fz['cache_prep']}.", ""]
    L += ["", "## Replay gate (calib)", "", f"- cache rows: {g.get('cache_rows', 'not run')}",
          f"- predictor rows: {g.get('predictor_rows', 'not run')}; gate: {g.get('predictor_gate', '–')}: "
          f"{g.get('predictor_rows_passed', 'not run')}", "",
          "## Reporting rules (proposed)", ""] + [f"- {x}" for x in fz["reporting_rules"]]
    DOC.write_text("\n".join(L) + "\n")
    print(_rel(DOC))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("draft", "replay", "doc"))
    ap.add_argument("part", nargs="?", choices=("cache", "predictor"))
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--tag", default="a", choices=("a", "b"))
    ap.add_argument("--prep", choices=("ultralytics",), default=None)
    ap.add_argument("--model", choices=("yolo26m", "yolo26s"), default="yolo26m")
    a = ap.parse_args()
    if a.model != "yolo26m" and a.prep != "ultralytics":
        sys.exit(f"{a.model}: only --prep ultralytics (addendum item 4)")
    set_model(a.model)
    set_prep(a.prep)
    if a.mode == "draft":
        draft()
    elif a.mode == "replay":
        replay(a.part, a.workers, a.tag)
    else:
        doc()


if __name__ == "__main__":
    main()
