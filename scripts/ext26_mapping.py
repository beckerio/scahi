# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO26m: back-mapping check (rule DECISIONS 2026-11-02 10:15 / 10:29) applied before the freeze (authors, 2026-11-14).

    python scripts/ext26_mapping.py caches               # calib caches with Ultralytics' input preparation
    python scripts/ext26_mapping.py check [--workers 4]  # calib: delta per cache row of the single-gain draft

caches: (1) all three details aligned to Ultralytics at once (``scahi.mapcheck.ultralytics_prep``: size rule, letterbox
gain, per-axis back-mapping) for every level of both stacks -> ``cache_ualigned/`` (``cache_state.json`` per level);
the tile levels are compared with the frozen single-gain caches (bit identity expected: VisDrone tiles had no
deviation, DECISIONS 2026-11-02 10:15); (2) for the per-detail report of the 10:29 rule, the whole-image levels with one
detail aligned at a time -> ``cache_ualigned_detail/<detail>/`` (tile levels linked to the frozen caches if (1) found
them bit-identical).

check: every cache row of the single-gain draft (bb1614a; copy runs/ext/yolo26m/mapping/draft_singlegain.json) fused
from the frozen and the aligned caches with its frozen calib configuration, through the same code path as the
replay (``ext_testdev._dirs`` + ``fuse_split``; visdrone_ignore, headline toolkit port, size bins pycocotools); delta = aligned - single-gain; the
single-gain value must reproduce the stored calib_AP. Rows with |delta| >= 0.05: one detail at a time as well.
Single-pass and SAHI rows use the Ultralytics predictor (its own mapping) and are not affected.
Output runs/ext/yolo26m/mapping/check.json. Calib only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scahi import ext as E  # noqa: E402
from scahi import mapcheck as MC  # noqa: E402

M = "yolo26m"
OUT = E.mdir(M) / "mapping"
SG_DRAFT = OUT / "draft_singlegain.json"
UROOT = MC.ROOT
DROOT = REPO / "cache_ualigned_detail"
DETAILS = MC.DETAILS
RULE = MC.RULE
LEVELS = {1280: ("w1280", "t2.0n1024"), 2048: ("w2048", "t2.0n1024")}


one_detail = MC.one_detail  # moved to scahi.mapcheck (shared with scripts/mapcheck_all.py)


def _same_npz(a: Path, b: Path) -> bool:
    za, zb = np.load(a), np.load(b)
    return sorted(za.files) == sorted(zb.files) and all(np.array_equal(za[k], zb[k]) for k in za.files)


def caches() -> None:
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm
    from scahi.infer.raw_head import load_unfused

    OUT.mkdir(parents=True, exist_ok=True)
    _, imgs = E._split_images("calib")
    rep = {"aligned_root": str(UROOT.relative_to(REPO)), "detail_root": str(DROOT.relative_to(REPO)), "levels": {}}
    for sz in E.sizes(M):
        w = E.weights(M, sz)
        m = load_unfused(w)
        with MC.ultralytics_prep():
            for lk in LEVELS[sz]:
                t0 = time.time()
                d = cm.cache_level(m, w, imgs, lk, E.OVERLAP, "calib", UROOT)
                (d / "cache_state.json").write_text(json.dumps({**MC.STATE, "weights": str(w), "level": lk}, indent=1))
                print(f"[26map] aligned {sz} {lk}: {time.time() - t0:.0f} s", flush=True)
        for lk in LEVELS[sz]:
            fz = cm.cache_dir(CACHE_ROOT, w, lk, E.OVERLAP, "calib")
            al = cm.cache_dir(UROOT, w, lk, E.OVERLAP, "calib")
            same = [_same_npz(fz / f"{s}.npz", al / f"{s}.npz") for s in imgs.stem]
            rep["levels"][f"{sz}/{lk}"] = {"images": len(same), "bit_identical_to_frozen": int(sum(same))}
            print(f"[26map] {sz} {lk}: {sum(same)}/{len(same)} images bit-identical to the frozen cache", flush=True)
        for det in DETAILS:
            root = DROOT / det
            with one_detail(det):
                for lk in LEVELS[sz]:
                    d = cm.cache_dir(root, w, lk, E.OVERLAP, "calib")
                    if lk.startswith("t"):
                        if rep["levels"][f"{sz}/{lk}"]["bit_identical_to_frozen"] != len(imgs):
                            sys.exit(f"{sz} {lk}: tiles differ with all details aligned; build the tile level per detail")
                        if not d.exists():
                            d.parent.mkdir(parents=True, exist_ok=True)
                            d.symlink_to(cm.cache_dir(CACHE_ROOT, w, lk, E.OVERLAP, "calib"), target_is_directory=True)
                        continue
                    t0 = time.time()
                    cm.cache_level(m, w, imgs, lk, E.OVERLAP, "calib", root)
                    (d / "cache_state.json").write_text(json.dumps({"aligned_detail": det, "others": "frozen",
                                                                    "weights": str(w), "level": lk}, indent=1))
                    print(f"[26map] detail {det} {sz} {lk}: {time.time() - t0:.0f} s", flush=True)
        del m
    (OUT / "caches.json").write_text(json.dumps(rep, indent=1))


def _fuse_eval(args) -> tuple[str, str, dict]:
    import torch

    from scahi import ext_testdev as TD
    from scahi.config import load_class_setting
    from scahi.eval.evaluate import evaluate
    from scahi.fusion import run as R
    from scahi.fusion import scalespace as ss

    torch.set_num_threads(1)
    row, tag, root = args
    f = row["fusion"]
    coco, imgs = E._split_images("calib")
    R.CACHE_ROOT = Path(root)
    dirs = TD._dirs(REPO / row["weights"], f["levels"], f["overlap"], "calib")
    data = {s: ss.load_levels(dirs, s, f["branch"], f["overlap"]) for s in imgs.stem}
    t0 = time.time()
    d = R.fuse_split(data, dict(zip(imgs.stem, imgs.image_id)), f["branch"], f["variant"], f["params"])
    m = evaluate(coco, d, load_class_setting(E.SETTING), "visdrone_ignore")["metrics"]  # headline = toolkit port; size bins
    return row["id"], tag, {"AP": m["AP"], "AP50": m["AP50"], "AP_lt16": m["AP_by_size"]["<16"],
                            "n_dets": len(d), "fuse_s": time.time() - t0}


def check(workers: int) -> None:
    from scahi.fusion.run import CACHE_ROOT

    fz = json.loads(SG_DRAFT.read_text())
    rows = [r for r in fz["rows"] if r["kind"] == "cache"]
    of = OUT / "check.json"
    res = json.loads(of.read_text())["rows"] if of.exists() else {}
    jobs = [(r, t, str(rt)) for r in rows for t, rt in (("single_gain", CACHE_ROOT), ("aligned", UROOT))
            if t not in res.get(r["id"], {})]
    with ProcessPoolExecutor(workers, mp_context=E._spawn()) as ex:
        for rid, tag, v in ex.map(_fuse_eval, jobs):
            res.setdefault(rid, {})[tag] = v
            print(f"[26map] {rid} {tag}: AP {v['AP']:.5f}", flush=True)
            _write(of, fz, rows, res)
        big = [r for r in rows if abs(res[r["id"]]["aligned"]["AP"] - res[r["id"]]["single_gain"]["AP"]) >= RULE]
        jobs = [(r, det, str(DROOT / det)) for r in big for det in DETAILS if det not in res[r["id"]]]
        for rid, tag, v in ex.map(_fuse_eval, jobs):
            res[rid][tag] = v
            print(f"[26map] {rid} {tag}: AP {v['AP']:.5f}", flush=True)
            _write(of, fz, rows, res)
    _write(of, fz, rows, res)


def _write(of: Path, fz: dict, rows: list, res: dict) -> None:
    out = {}
    for r in rows:
        v = res.get(r["id"], {})
        if "single_gain" in v and "aligned" in v:
            sg = v["single_gain"]
            v["stored_calib_AP"] = r["calib_AP"]
            v["single_gain_reproduces_stored"] = (sg["AP"] == r["calib_AP"]) if r["calib_AP"] is not None else None
            v["delta"] = v["aligned"]["AP"] - sg["AP"]
            v["delta_secondary"] = {k: v["aligned"][k] - sg[k] for k in ("AP50", "AP_lt16")}
            v["at_or_above_rule"] = abs(v["delta"]) >= RULE
            v["delta_per_detail"] = {det: v[det]["AP"] - sg["AP"] for det in DETAILS if det in v}
        out[r["id"]] = v
    done = [v for v in out.values() if "delta" in v]
    of.write_text(json.dumps({
        "rule": "DECISIONS 2026-11-02 10:15 / 10:29: |delta AP| on calib, all three details aligned at once; >= 0.05 "
                "-> bug fix (applied to YOLO26m before the freeze, authors 2026-11-14)",
        "draft": "configs/ext/frozen_rows_ext26.draft.json at bb1614a (single-gain caches)", "threshold": RULE,
        "aligned_cache_state": MC.STATE, "time": time.strftime("%F %T"), "n_rows": len(rows), "n_done": len(done),
        "n_at_or_above_rule": sum(v["at_or_above_rule"] for v in done),
        "max_abs_delta": max((abs(v["delta"]) for v in done), default=None), "rows": out}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("caches", "check"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    caches() if a.stage == "caches" else check(a.workers)


if __name__ == "__main__":
    main()
