# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO26m (addendum 7c01b5d) and YOLO26s (addendum 7bc0ace): every calib step up to the freeze, nothing on val / test-dev.

    python scripts/ext26_calib.py gpu    # G-E1, single-pass conf, SAHI grid v2, calib caches
    python scripts/ext26_calib.py cpu    # G-E3, F0-NMS (NMS-IOS) and F2 24-grid, refined grid (X-grid B / C)
    python scripts/ext26_calib.py cpu --prep ultralytics   # the same on the Ultralytics-aligned caches (cache_ualigned/,
                                                           # scripts/ext26_mapping.py) -> runs/ext/yolo26m/ualigned/

Same functions and grids as the extension's jobs (scahi.ext.e_gpu / e_fuse, scripts/ext_grid_ext.py), with their
val stages left out: the authors approve the freeze before the single val and test-dev evaluation (2026-11-13).
Outputs in runs/ext/yolo26m/ (gate_GE1.json, conf_tuning_<s>_<br>.json, sahi_tuning/, cache_complete_calib.json,
gate_GE3.json, tuning.json, xgrid_calib.json). Every stage skips work whose output exists (restartable).

YOLO26s (``--model yolo26s``, authors 2026-11-14): same functions, grids and selection; the calib caches are built with
Ultralytics' input preparation from the start (``scahi.mapcheck.ultralytics_prep``, cache_ualigned/, the convention of
the frozen YOLO26m rows), so the cpu stage runs with ``--prep ultralytics`` only -> runs/ext/yolo26s/ualigned/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scahi import ext as E  # noqa: E402

MODEL = "yolo26m"
D = E.mdir(MODEL)
UROOT = REPO / "cache_ualigned"  # scahi.mapcheck.ROOT: caches with all three details aligned to Ultralytics


def _init_root(root: str | None) -> None:
    """Pool initializer: spawned workers read the cache root from scahi.fusion.run at call time."""
    if root:
        from scahi.fusion import run as R

        R.CACHE_ROOT = Path(root)


def gpu() -> None:
    from scahi.config import load_class_setting
    from scahi.eval.evaluate import evaluate
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm
    from scahi.infer.full import predict_full
    from scahi.infer.raw_head import load_unfused
    from scahi.jobs import sahi_tune2

    for sz in E.sizes(MODEL):
        if not (E.train_dir(MODEL, sz) / "TRAIN_DONE").exists():
            sys.exit(f"training {MODEL} {sz} not finished")
    gf = D / "gate_GE1.json"
    if not gf.exists():
        gf.write_text(json.dumps(E.gate_ge1(MODEL), indent=1))
    if not json.loads(gf.read_text())["passed"]:
        sys.exit("G-E1 failed: no dense rows (addendum 7c01b5d item 5)")
    print("[26] G-E1 passed", flush=True)

    cs = load_class_setting(E.SETTING)
    coco_c, imgs_c = E._split_images("calib")
    for sz in E.sizes(MODEL):  # single-pass conf on calib (as scahi.ext.single_pass_rows, without its val row)
        for br in E.branches(MODEL):
            cf = D / f"conf_tuning_{sz}_{br}.json"
            if not cf.exists():
                dc, _ = predict_full(E.weights(MODEL, sz), imgs_c, sz, branch=br)
                grid = [{"conf": c, "AP": evaluate(coco_c, dc[dc.score >= c], cs, "visdrone_ignore",
                                                   port_only=True)["metrics"]["AP"]} for c in E.CONF_GRID]
                cf.write_text(json.dumps({"grid": grid, "conf": max(grid, key=lambda g: g["AP"])["conf"]}, indent=2))
            print(f"[26] conf {sz} {br}: {json.loads(cf.read_text())['conf']}", flush=True)

    tdir = D / "sahi_tuning"  # SAHI grid v2 on calib, 1280 weights, selection branch (as scahi.ext.sahi_row)
    tdir.mkdir(exist_ok=True)
    tuned = sahi_tune2(E.weights(MODEL, 1280), E.SETTING, tdir, E.selection_branch(MODEL))
    b = max(tuned["best"].values(), key=lambda r: r["AP"])
    print(f"[26] SAHI best on calib: {b}", flush=True)

    miss = {}  # caches of calib only; val / test-dev caches after the authors' approval of the freeze
    aligned = MODEL != "yolo26m"  # YOLO26s: Ultralytics' input preparation from the start
    from scahi import mapcheck as MC

    for sz, stack in E.STACK.items():
        w = E.weights(MODEL, sz)
        m = load_unfused(w)
        _, imgs = E._split_images("calib")
        for lk in cm.STACK_LEVELS[stack]:
            if aligned:
                with MC.ultralytics_prep():
                    d = cm.cache_level(m, w, imgs, lk, E.OVERLAP, "calib", UROOT)
                (d / "cache_state.json").write_text(json.dumps({**MC.STATE, "weights": str(w), "level": lk}, indent=1))
            else:
                d = cm.cache_level(m, w, imgs, lk, E.OVERLAP, "calib", CACHE_ROOT)
            miss[f"{sz}/calib/{lk}"] = int(sum(not (d / f"{s}.npz").exists() for s in imgs.stem))
            print(f"[26] cache {sz} {lk} calib done", flush=True)
        del m
    (D / "cache_complete_calib.json").write_text(json.dumps(
        {"missing": miss, "passed": not any(miss.values()),
         "cache_prep": "ultralytics" if aligned else "single_gain", "root": str((UROOT if aligned else CACHE_ROOT))},
        indent=1))
    if any(miss.values()):
        sys.exit(f"calib cache incomplete {miss}")
    print("[26] calib caches complete", flush=True)


def cpu(workers: int, max_rounds: int, prep: str | None = None) -> None:
    import ext_grid_ext as X
    from concurrent.futures import ProcessPoolExecutor

    global D
    root = None
    if MODEL != "yolo26m" and prep != "ultralytics":
        sys.exit(f"{MODEL}: calib caches exist with Ultralytics' preparation only; use --prep ultralytics")
    if prep == "ultralytics":  # back-mapping rule applied before the freeze (DECISIONS 2026-11-14)
        if MODEL == "yolo26m":
            mp = E.mdir(MODEL) / "mapping" / "caches.json"
            if not mp.exists():
                sys.exit("aligned calib caches missing (scripts/ext26_mapping.py caches)")
        else:
            cc = json.loads((D / "cache_complete_calib.json").read_text())
            if not (cc["passed"] and cc.get("cache_prep") == "ultralytics"):
                sys.exit("aligned calib caches incomplete")
        root, D = str(UROOT), E.mdir(MODEL) / "ualigned"
        D.mkdir(exist_ok=True)
        _init_root(root)
    elif not json.loads((D / "cache_complete_calib.json").read_text())["passed"]:
        sys.exit("calib caches incomplete")
    gf = D / "gate_GE3.json"
    g = json.loads(gf.read_text()) if gf.exists() else {sz: E.gate_ge3(MODEL, sz) for sz in E.sizes(MODEL)}
    gf.write_text(json.dumps(g, indent=1, default=str))
    if not all(v["passed"] for v in g.values()):
        sys.exit("G-E3 failed")
    print("[26] G-E3 passed", flush=True)

    tf = D / "tuning.json"  # F0-NMS (NMS-IOS grid) and F2 (24-grid) on calib, as scahi.ext.e_fuse
    tuning = json.loads(tf.read_text()) if tf.exists() else {}
    todo = [(MODEL, sz, br, v) for sz in E.sizes(MODEL) for br in E.mz_branches(MODEL) for v in ("F0-NMS", "F2")
            if f"{sz}/{br}/{v}" not in tuning]
    pmap = E._pmap(E._tune, todo) if root is None else ProcessPoolExecutor(
        workers, mp_context=E._spawn(), initializer=_init_root, initargs=(root,)).map(E._tune, todo)
    for k, res in pmap:
        tuning[k] = {"grid": res, "best": max(res, key=lambda r: r["AP"]), "budget": len(res)}
        tf.write_text(json.dumps(tuning, indent=1))
        print(f"[26] {k}: best calib AP {tuning[k]['best']['AP']:.2f} {tuning[k]['best']}", flush=True)

    # refined grid (X-grid rows B / C): frozen procedure of scripts/ext_grid_ext.py, calib rounds and selection only
    fz = json.loads((REPO / "configs/ext/xgrid_frozen.json").read_text())
    if json.loads(json.dumps(X.GRID)) != fz["grid"] or X.TIME_CAP != fz["time_cap_vs_F2"]:
        sys.exit("GRID / TIME_CAP differ from configs/ext/xgrid_frozen.json")
    xf = D / "xgrid_calib.json"
    old = json.loads(xf.read_text())["results"] if xf.exists() else {}
    cases = [(MODEL, sz, br, row) for sz in E.sizes(MODEL) for br in E.mz_branches(MODEL) for row in ("B", "C")]
    todo, calib, log = {}, {}, {}
    for c in cases:
        o = old.get("/".join(map(str, c)))
        have = {X._key(r) for r in o["calib_all"]} if o else set()
        vs = [] if o else [X.REF]
        vs += [v for v in ({"iou_cluster": 0.5, **dict(zip(X.PARAMS, x))}
                           for x in itertools.product(*(X.GRID[c[3]][p] for p in X.PARAMS))) if X._key(v) not in have]
        calib[c], log[c] = (list(o["calib_all"]), list(o["rounds"])) if o else ([], [])
        if vs:
            todo[c] = vs
    with ProcessPoolExecutor(workers, mp_context=E._spawn(), initializer=_init_root, initargs=(root,)) as ex:
        r0 = max((len(v) for v in log.values()), default=0)
        for rnd in range(r0 + 1, r0 + max_rounds + 1):
            chunks = [(*c, vs[i:i + 6]) for c, vs in todo.items() for i in range(0, len(vs), 6)]
            if not chunks:
                break
            for k, out in ex.map(X._calib, chunks):
                calib[k].extend(out)
            todo = {}
            for c in cases:
                sel, _ = X._select(calib[c])
                edges = X._edges(calib[c], sel)
                capped = any(r["time_ratio_vs_ref"] > X.TIME_CAP for r in calib[c])
                log[c].append({"round": rnd, "n_variants": len(calib[c]), "selected": {p: sel[p] for p in X.PARAMS},
                               "AP": sel["AP"], "edges": edges, "time_cap_hit": capped})
                print(f"[26] round {rnd} {c}: AP {sel['AP']:.2f} sel {[sel[p] for p in X.PARAMS]} edges {edges} "
                      f"cap {capped}", flush=True)
                if edges and not capped and rnd < r0 + max_rounds:
                    todo[c] = X._next_round(calib[c], sel, edges)
    res = {}
    for c in cases:
        s_ok, s_any = X._select(calib[c])
        res["/".join(map(str, c))] = {
            "selected": {"iou_cluster": 0.5, **{p: s_ok[p] for p in X.PARAMS}}, "calib_selected": s_ok,
            "calib_best_unconstrained": s_any,
            "calib_reference": next(r for r in calib[c] if X._key(r) == X._key(X.REF)),
            "rounds": log[c], "time_cap_hit": log[c][-1]["time_cap_hit"], "calib_all": calib[c],
            "edges_unconstrained": X._edges(calib[c], s_any), "edges_under_cap": X._edges(calib[c], s_ok)}
        xf.write_text(json.dumps({"procedure": "configs/ext/xgrid_frozen.json (scripts/ext_grid_ext.py, calib part)",
                                  "results": res}, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("gpu", "cpu"))
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--prep", choices=("ultralytics",), default=None)
    ap.add_argument("--model", choices=("yolo26m", "yolo26s"), default="yolo26m")
    a = ap.parse_args()
    global MODEL, D
    MODEL, D = a.model, E.mdir(a.model)
    D.mkdir(parents=True, exist_ok=True)
    gpu() if a.stage == "gpu" else cpu(a.workers, a.max_rounds, a.prep)


if __name__ == "__main__":
    main()
