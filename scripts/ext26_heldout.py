# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO26m: the single held-out evaluation of the approved freeze (authors 2026-11-14, freeze 5db78fa): val once, exact val
replay, test-dev once.

    python scripts/ext26_heldout.py freeze                         # draft -> configs/ext/frozen_rows_ext26.json (commit it)
    python scripts/ext26_heldout.py caches --split val|test-dev    # GPU: Ultralytics-aligned caches (cache_ualigned/)
    python scripts/ext26_heldout.py run --split val|test-dev --part cache|predictor [--replay] [--workers 7]
    python scripts/ext26_heldout.py gate                           # val vs. val replay, every row, exact equality

Rows are computed only from the committed freeze through the extension's val / test-dev code path
(scahi.ext_testdev.run_cache_row / run_predictor_row; for the five earlier models this path reproduced every stored val
row exactly). Cache rows read the Ultralytics-aligned caches named in the row (``cache_prep``; back-mapping rule applied
before the freeze, DECISIONS 2026-11-14). Outputs (separate from the five-model rows): runs/ext/yolo26m/val/,
runs/ext/yolo26m/val_replay/, runs/ext/yolo26m/testdev/. Every run refuses to overwrite an existing row.

YOLO26s (``--model yolo26s``; authors 2026-11-15, option (a)): same stages, files configs/ext/frozen_rows_ext26s.json and
runs/ext/yolo26s/; the freeze records the single-pass gate decision and the batch-1 calib replay values.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
FREEZE = REPO / "configs/ext/frozen_rows_ext26.json"
APPROVAL = "authors 2026-11-14: freeze of 5db78fa approved, incl. the exact val replay before test-dev"
OUT = {"val": D / "val", "val_replay": D / "val_replay", "test-dev": D / "testdev"}
GATE26S = "runs/ext/yolo26s/replay_calib_ualigned/gate_replay_calib.json"
SP_DECISION = ("authors 2026-11-15, option (a): the single-pass gate as specified is NOT met (Delta 0.079 to 0.094 > 0.05, "
               "tuning at batch 16 vs. replay at batch 1); the conf grid re-evaluated on the batch-1 detections gives the "
               "same choice (0.001) and the same ordering in all four cases (runs/ext/yolo26s/conf_batch1_check.json); "
               "the choice stands; the batch-1 values are the reference for the val replay, the tuning values are "
               "reported alongside; no rule was changed")


def set_model(m: str) -> None:
    global M, D, DRAFT, FREEZE, APPROVAL, OUT
    if m == M:
        return
    M, D = m, E.mdir(m)
    DRAFT, FREEZE = REPO / "configs/ext/frozen_rows_ext26s.draft.json", REPO / "configs/ext/frozen_rows_ext26s.json"
    APPROVAL = ("authors 2026-11-15: freeze approved on the basis of option (a) for the single-pass gate; val once, "
                "exact val replay, test-dev once")
    OUT = {"val": D / "val", "val_replay": D / "val_replay", "test-dev": D / "testdev"}


LEVELS = {1280: ["w1280", "t2.0n1024"], 2048: ["w2048", "t2.0n1024"]}


def make_freeze() -> None:
    if FREEZE.exists():
        sys.exit(f"{FREEZE.name} exists")
    fz = json.loads(DRAFT.read_text())
    fz["status"] = "APPROVED - frozen; val once, exact val replay, test-dev once"
    fz["approved_by"] = APPROVAL
    fz["draft_commit"] = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(DRAFT)], cwd=REPO,
                                        capture_output=True, text=True).stdout.strip()
    if M == "yolo26s":
        g = json.loads((REPO / GATE26S).read_text())["rows"]
        fz["single_pass_gate"] = {"decision": SP_DECISION, "replay_file": GATE26S}
        for r in fz["rows"]:
            if r["kind"] == "single":
                x = g[r["id"]]
                assert x["exact_a_vs_b"] and x["replay_AP_a"] == x["replay_AP_b"], r["id"]
                r["calib_AP_batch1"] = x["replay_AP_a"]
                r["calib_AP_note"] = ("calib_AP = tuning value (predictor batch 16); calib_AP_batch1 = replay through the "
                                      "val / test-dev path (batch 1, bit-identical twice) = reference for the val replay")
    FREEZE.write_text(json.dumps(fz, indent=1))
    print(f"{FREEZE.relative_to(REPO)} written from {fz['draft_commit'][:7]}; commit it before any run")


def freeze() -> dict:
    r = subprocess.run(["git", "status", "--porcelain", "--", str(FREEZE)], cwd=REPO, capture_output=True, text=True)
    if r.stdout.strip() or not FREEZE.exists():
        sys.exit(f"{FREEZE.name} missing or uncommitted")
    fz = json.loads(FREEZE.read_text())
    fz["freeze_commit"] = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(FREEZE)], cwd=REPO,
                                         capture_output=True, text=True).stdout.strip()
    return fz


def caches(split: str) -> None:
    from scahi import mapcheck as MC
    from scahi.infer import cache_mz as cm
    from scahi.infer.raw_head import load_unfused

    _, imgs = T.split_data(split)
    rep = {}
    for sz in E.sizes(M):
        w = E.weights(M, sz)
        m = load_unfused(w)
        for lk in LEVELS[sz]:
            with MC.ultralytics_prep():
                d = cm.cache_level(m, w, imgs, lk, E.OVERLAP, split, MC.ROOT)
            (d / "cache_state.json").write_text(json.dumps({**MC.STATE, "weights": str(w), "level": lk}, indent=1))
            rep[f"{sz}/{lk}"] = int(sum(not (d / f"{s}.npz").exists() for s in imgs.stem))
            print(f"[26ho] cache {split} {sz} {lk}: missing {rep[f'{sz}/{lk}']}", flush=True)
        del m
    (D / f"cache_complete_{split}.json").write_text(json.dumps({"missing": rep, "passed": not any(rep.values())},
                                                              indent=1))


def _root(row: dict) -> None:
    from scahi.fusion import run as R

    cp = row.get("cache_prep")
    if cp:
        R.CACHE_ROOT = REPO / cp["cache_root"]


def _cache_worker(args):
    row, split, fz, out = args
    T.TD, T.REPLAY = Path(out), Path(out)  # out_root() of the val / test-dev path (both splits -> this folder)
    _root(row)
    return T.run_cache_row((row, split, fz))


def run(split: str, part: str, replay: bool, workers: int) -> None:
    fz = freeze()
    key = "val_replay" if replay else split
    if replay and split != "val":
        sys.exit("--replay is the val replay")
    if split == "test-dev":
        g = D / "gate_val_replay.json"
        if not (g.exists() and json.loads(g.read_text()).get("passed")):
            sys.exit("val replay gate not passed: no test-dev")
    if not json.loads((D / f"cache_complete_{split}.json").read_text())["passed"]:
        sys.exit(f"{split} caches incomplete")
    out = OUT[key]
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in fz["rows"] if (r["kind"] == "cache") == (part == "cache")]
    done = [r["id"] for r in rows if (out / r["val_exp_id"] / "metrics.json").exists()]
    if done:
        sys.exit(f"rows already evaluated in {out.relative_to(REPO)} (single evaluation): {done}")
    if part == "cache":
        with ProcessPoolExecutor(workers, mp_context=E._spawn()) as ex:
            for rid in ex.map(_cache_worker, [(r, split, fz, str(out)) for r in rows]):
                print(f"[26ho] {key} {rid}", flush=True)
    else:
        T.TD, T.REPLAY = out, out
        for r in rows:
            T.run_predictor_row(r, split, fz)
            print(f"[26ho] {key} {r['id']}", flush=True)


def gate() -> None:
    fz = freeze()
    res = []
    for r in fz["rows"]:
        a = json.loads((OUT["val"] / r["val_exp_id"] / "metrics.json").read_text())["metrics"]["AP"]
        b = json.loads((OUT["val_replay"] / r["val_exp_id"] / "metrics.json").read_text())["metrics"]["AP"]
        res.append({"id": r["id"], "kind": r["kind"], "val_AP": a, "replay_AP": b, "abs_diff": abs(a - b)})
    ok = all(x["abs_diff"] == 0.0 for x in res)
    (D / "gate_val_replay.json").write_text(json.dumps({
        "gate": f"YOLO{M[4:]} val replay (fresh processes, same code path), exact equality for every frozen row",
        "freeze": fz["freeze_commit"], "time": datetime.now().isoformat(timespec="minutes"), "rows": res,
        "n_exact": sum(x["abs_diff"] == 0.0 for x in res), "n_rows": len(res), "passed": ok}, indent=1))
    print(f"val replay: {sum(x['abs_diff'] == 0.0 for x in res)}/{len(res)} exact -> {'passed' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("freeze", "caches", "run", "gate"))
    ap.add_argument("--split", choices=("val", "test-dev"), default="val")
    ap.add_argument("--part", choices=("cache", "predictor"), default="cache")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--model", choices=("yolo26m", "yolo26s"), default="yolo26m")
    a = ap.parse_args()
    set_model(a.model)
    if a.stage == "freeze":
        make_freeze()
    elif a.stage == "caches":
        caches(a.split)
    elif a.stage == "run":
        run(a.split, a.part, a.replay, a.workers)
    else:
        gate()


if __name__ == "__main__":
    main()
