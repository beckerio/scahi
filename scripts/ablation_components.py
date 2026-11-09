# SPDX-License-Identifier: AGPL-3.0-or-later
"""SCAHI component ablation (exploratory, post hoc; user request 2026-10-30; CPU only, from the existing caches).

    python scripts/ablation_components.py visdrone|dota [--workers 7]

Never toward any verdict. Same passes in every arm: the single pass at the training size + tiles (VisDrone: stack
native2048 = w2048 + z = 2 tiles; DOTA: dota2048 = w2048 + native 1024 tiles), branch o2m, 2048 px weights.

Arms (scahi.fusion.scalespace variants):
  (a) F2n - border taper, no coverage normalisation (field and score are plain sums of w * p over passes, C = 1);
  (b) F1  - coverage normalisation (C = number of covering passes), no border taper (w = 1);
  (c) F2  - border taper + coverage normalisation (pre-specified setting) = the existing F2 rows;
  (d) F3  - (c) + validity windows (pre-specified VisDrone window parameters);
  (e) F2 with the refined peak grid = the existing RG / X-grid rows.
(a), (b), (d): parameters selected on calib from the pre-specified F2 grid (24 configurations), max calib primary metric,
exactly as the existing F2 rows; (c) and (e) keep their frozen calib selection. Cell-factor sweep of (c): cell_factor in
{5, 7, 9, 11, 13}, other parameters at the calib values of (c), no selection.

Reported on VisDrone val (toolkit-port AP = AP@[.5:.95], AP50, AP75) and DOTA-val (devkit mAP50, AP@[.5:.95]) - DOTA-val
was already evaluated once for the frozen rows: post-hoc analysis on the held-out set. Plus recall@0.5 of crowded /
isolated GT, duplicate / fragment rates (score >= 0.25; scahi.eval.diagnostics definitions) and paired bootstrap CIs
(1000 resamples, seed 0, the resamples of the main evaluation) against (c). Output: runs/ablation/<dataset>.json + CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABEL = "exploratory, post hoc (SCAHI component ablation, user request 2026-10-30); never a hypothesis or method row"
OUT = REPO / "runs/ablation"
ARMS = {"a": "F2n", "b": "F1", "c": "F2", "d": "F3", "e": "F2"}
ARM_TEXT = {"a": "Rand-Taper, ohne Abdeckungsnormalisierung", "b": "Abdeckungsnormalisierung, ohne Rand-Taper",
            "c": "Rand-Taper + Abdeckungsnormalisierung (F2, vorab festgelegt)", "d": "(c) + Validity-Fenster",
            "e": "(c) mit verfeinertem Peak-Raster (RG)"}
TUNED = ("a", "b", "d")
CF_SWEEP = (5.0, 7.0, 9.0, 11.0, 13.0)
KEYS = ("tau_splat", "tau_peak", "cell_factor", "bins_per_oct")
SZ = 2048


def _spawn():
    from scahi.ext import _spawn as sp

    return sp()


def _cpu_slot() -> None:
    import os
    import subprocess

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        os.sched_setaffinity(0, set(range(16, 24)))
        os.nice(19)
        subprocess.run(["ionice", "-c3", "-p", str(os.getpid())], check=False)
    except OSError:
        pass


def _ci(a0: float, b0: float, va, vb) -> dict:
    d = np.asarray(va) - np.asarray(vb)
    return {"a": a0, "b": b0, "delta": a0 - b0, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


# ---------------------------------------------------------------------------------------------------- VisDrone

V_MODELS = ("yolov10s", "yolov10m", "yolo11s", "yolo12s", "yolo11m")  # every weight set at 2048 (YOLO12m: no 2048)


def v_frozen(m: str) -> dict:
    """(c) and (e) of one weight set: frozen calib selection and the existing val run."""
    from scahi import ext as E

    xg = json.loads((REPO / "configs/ext/xgrid_frozen.json").read_text())["weight_sets"][f"{m}/{SZ}/o2m/C"]
    if m == "yolov10s":
        fr = json.loads((REPO / "configs/final/frozen_rows.json").read_text())
        x2 = next(r for r in fr["rows"] if r["id"] == "X2")
        c_cfg, c_run, c_ap = dict(x2["inference"]["params"]), REPO / x2["val_run"], x2["inference"]["calib_AP"]
    else:
        b = json.loads((E.mdir(m) / "tuning.json").read_text())[f"{SZ}/o2m/F2"]["best"]
        c_cfg, c_run, c_ap = {k: b[k] for k in KEYS}, E.ROWS / E.mz_exp(m, SZ, "o2m", "F2"), b["AP"]
    return {"c": {"cfg": {**c_cfg, "iou_cluster": 0.5}, "run": str(c_run), "calib_AP": c_ap},
            "e": {"cfg": {**xg["params"], "iou_cluster": 0.5}, "run": str(REPO / xg["val_row"]), "calib_AP": xg["calib_AP"]}}


def _v_calib(args: tuple[str, str, list[dict]]) -> tuple[tuple[str, str], list[dict]]:
    from ext_cluster_diag import _load

    from scahi import ext as E
    from scahi.config import load_class_setting
    from scahi.eval.evaluate import evaluate
    from scahi.fusion import run as R

    _cpu_slot()
    m, variant, cfgs = args
    coco, s2i, data, _, _ = _load(m, SZ, "o2m", "C", "calib")
    cs = load_class_setting(E.SETTING)
    out = []
    for cfg in cfgs:
        r = evaluate(coco, R.fuse_split(data, s2i, "o2m", variant, cfg), cs, "visdrone_ignore", port_only=True)["metrics"]
        out.append({**cfg, "AP": r["AP"], "AP50": r["AP50"]})
    return (m, variant), out


def _v_val(args: tuple[str, str, str, dict]) -> str:
    from ext_cluster_diag import _load

    from scahi import ext as E
    from scahi.fusion import run as R

    _cpu_slot()
    m, name, variant, cfg = args
    exp = f"ABL_{m}_mz{SZ}_{name}_{variant}_ts{cfg['tau_splat']}_tp{cfg['tau_peak']}_cf{cfg['cell_factor']}_b{cfg['bins_per_oct']}_o2m"
    run = OUT / "visdrone" / exp
    if not (run / "metrics.json").exists():
        coco, s2i, data, lks, tiles = _load(m, SZ, "o2m", "C", "val")
        d = R.fuse_split(data, s2i, "o2m", variant, cfg)
        E.write_eval(exp, d, E.SETTING, coco, "val", {
            "model": m, "weights": E.rel(E.weights(m, SZ)), "train_variant": f"E-{m}-{SZ}",
            "inference": {"row": f"ABL-{name}", "levels": lks, "overlap": E.OVERLAP, "fusion": variant, "branch": "o2m",
                          "params": cfg, "tuned_on": "calib (F2 grid, max AP)" if name in TUNED else "fixed"},
            "training": E.training_of(m, SZ), "notes": LABEL}, root=OUT / "visdrone", phase=8, tiles=tiles or None)
    return str(run)


def _v_measures(run: str) -> tuple[str, dict]:
    from ext_cluster_diag import _recall_vecs, _val_measures
    from ext_decomp import stats

    _cpu_slot()
    _, meas = _val_measures((run, SZ))
    _, st = stats(run)
    _, rv = _recall_vecs(run)
    return run, {"meas": meas, "st": st, "rv": rv}


def visdrone(workers: int) -> dict:
    from scahi.fusion import run as R

    (OUT / "visdrone").mkdir(parents=True, exist_ok=True)
    f = OUT / "visdrone.json"
    res = json.loads(f.read_text()) if f.exists() else {"label": LABEL, "calib": {}}
    fz = {m: v_frozen(m) for m in V_MODELS}
    grid = R.grid("F2")
    todo = [(m, ARMS[a], grid[i::3]) for m in V_MODELS for a in TUNED for i in range(3)
            if f"{m}/{ARMS[a]}" not in res["calib"]]
    with ProcessPoolExecutor(workers, mp_context=_spawn()) as ex:
        part: dict[tuple, list] = {}
        for k, out in ex.map(_v_calib, todo):
            part.setdefault(k, []).extend(out)
        for (m, v), rows in part.items():
            res["calib"][f"{m}/{v}"] = rows
        f.write_text(json.dumps(res, indent=1, default=float))
        arms: dict[str, dict] = {}
        jobs = []
        for m in V_MODELS:
            arms[m] = {}
            for a in TUNED:
                best = max(res["calib"][f"{m}/{ARMS[a]}"], key=lambda r: r["AP"])
                cfg = {**{k: best[k] for k in KEYS}, "iou_cluster": 0.5}
                arms[m][a] = {"cfg": cfg, "calib_AP": best["AP"]}
                jobs.append((m, a, ARMS[a], cfg))
            arms[m].update(fz[m])
            for cf in CF_SWEEP:
                cfg = {**fz[m]["c"]["cfg"], "cell_factor": cf}
                arms[m][f"cf{cf:g}"] = {"cfg": cfg}
                jobs.append((m, f"cf{cf:g}", "F2", cfg))
        runs = list(ex.map(_v_val, jobs))
        for (m, name, _, _), run in zip(jobs, runs):
            arms[m][name]["run"] = run
        all_runs = sorted({x["run"] for m in V_MODELS for x in arms[m].values()})
        ms = dict(ex.map(_v_measures, all_runs))
    res["arms"] = {}
    for m in V_MODELS:
        c = ms[arms[m]["c"]["run"]]
        out = {}
        for name, x in arms[m].items():
            r = ms[x["run"]]
            o = {**x, "variant": ARMS.get(name, "F2"), **{k: r["st"][k] for k in ("AP", "AP50", "AP75")},
                 "R50_crowd": r["rv"]["crowd"][0], "R50_iso": r["rv"]["iso"][0],
                 "dup_rate": r["meas"]["dup_rate"], "frag_rate": r["meas"]["frag_rate"],
                 "n_det@0.25": r["meas"]["n_det@0.25"]}
            if name != "c":
                o["vs_c"] = {k: _ci(r["st"][k], c["st"][k], r["st"][k + "_vec"], c["st"][k + "_vec"])
                             for k in ("AP", "AP50", "AP75")}
                for g in ("crowd", "iso"):
                    o["vs_c"][f"R50_{g}"] = _ci(r["rv"][g][0], c["rv"][g][0], r["rv"][g][1], c["rv"][g][1])
            out[name] = o
        res["arms"][m] = out
    res["recompute_check_cf5_equals_c"] = {
        m: abs(res["arms"][m]["cf5"]["AP"] - res["arms"][m]["c"]["AP"]) for m in V_MODELS
        if res["arms"][m]["c"]["cfg"]["cell_factor"] == 5.0}
    f.write_text(json.dumps(res, indent=1, default=float))
    return res


# -------------------------------------------------------------------------------------------------------- DOTA

D_MODELS = ("yolov10s", "yolov10m")


def _d_dupfrag(dets: pd.DataFrame, conf: float = 0.25) -> dict:
    """Duplicate / fragment rates (scahi.eval.diagnostics definitions; all GT incl. difficult count as GT)."""
    from scahi.dota import evaluate as EV
    from scahi.dota import pipeline as P
    from scahi.eval.diagnostics import classify_detections

    gt = EV.gt_arrays("val", P.split_stems("val"), all_objects=True)
    d = dets[dets.score >= conf]
    n = {0: 0, 1: 0, 2: 0, 3: 0}
    for s, g in d.groupby("image", sort=False):
        b, k, _ = gt[s]
        st = classify_detections(b.astype(float), k, g[["x1", "y1", "x2", "y2"]].to_numpy(float),
                                 g["score"].to_numpy(float), g["cls"].to_numpy(int))
        for v, c in zip(*np.unique(st, return_counts=True)):
            n[int(v)] += int(c)
    tot = max(sum(n.values()), 1)
    return {"dup_rate": n[2] / tot, "frag_rate": n[3] / tot, "n_det@0.25": sum(n.values())}


def _d_val(args: tuple[str, str, str, dict]) -> tuple[str, dict]:
    import torch

    from scahi.dota import final as F
    from scahi.dota import pipeline as P

    _cpu_slot()
    torch.set_num_threads(1)
    m, name, variant, cfg = args
    rid = f"ABL-{name}_{m}_{SZ}"
    d = OUT / "dota"
    if (d / f"{rid}.json").exists():
        return rid, json.loads((d / f"{rid}.json").read_text())
    if name in ("c", "e"):  # frozen rows of the one DOTA-val run: their detections and bootstrap vectors are reused
        src = f"{'F2' if name == 'c' else 'RG'}_{m}_{SZ}"
        dets = pd.read_parquet(F.DETS_DIR / f"{src}.parquet")
        o = json.loads((F.VAL_DIR / "eval" / f"{src}.json").read_text())
        vec = dict(np.load(F.VAL_DIR / "eval" / f"{src}_boot.npz"))
        o["source"] = src
    else:
        data = P.load(m, SZ, "val", "o2m", "C")
        dets = P.fuse(data, "o2m", variant, cfg, F.frozen()["pipeline_limit"])
        o, vec = F.evaluate_row(rid, dets=dets)
        o["images_at_limit"] = P.at_limit(dets, F.frozen()["pipeline_limit"])
    o.update(_d_dupfrag(dets), variant=variant, cfg=cfg)
    np.savez(d / f"{rid}_boot.npz", **vec)
    (d / f"{rid}.json").write_text(json.dumps(o, indent=1, default=float))
    return rid, o


def dota(workers: int) -> dict:
    from scahi.dota import final as F
    from scahi.dota import pipeline as P

    _cpu_slot()
    (OUT / "dota").mkdir(parents=True, exist_ok=True)
    f = OUT / "dota.json"
    res = json.loads(f.read_text()) if f.exists() else {"label": LABEL + "; DOTA-val already evaluated once for the "
                                                                        "frozen rows (post-hoc analysis on the held-out set)",
                                                        "calib": {}}
    fz = {r["id"]: r for r in F.frozen()["rows"]}
    for m in D_MODELS:
        for a in TUNED:
            k = f"{m}/{ARMS[a]}"
            if k not in res["calib"]:  # calib selection exactly as the F2 rows (pipeline.fuse_job / select)
                res["calib"][k] = P.tune(m, SZ, "o2m", "C", ARMS[a], P.f2_grid(), workers=workers)
                f.write_text(json.dumps(res, indent=1, default=float))
    jobs, arms = [], {}
    for m in D_MODELS:
        arms[m] = {}
        for a in TUNED:
            best = P.select(res["calib"][f"{m}/{ARMS[a]}"])
            cfg = {k: best[k] for k in F.F2_KEYS}
            arms[m][a] = {"cfg": cfg, "calib_mAP50": best["mAP50"]}
            jobs.append((m, a, ARMS[a], cfg))
        c_cfg = fz[f"F2_{m}_{SZ}"]["cfg"]
        arms[m]["c"] = {"cfg": c_cfg, "calib_mAP50": fz[f"F2_{m}_{SZ}"]["calib_mAP50"]}
        arms[m]["e"] = {"cfg": fz[f"RG_{m}_{SZ}"]["cfg"], "calib_mAP50": fz[f"RG_{m}_{SZ}"]["calib_mAP50"]}
        jobs += [(m, "c", "F2", c_cfg), (m, "e", "F2", arms[m]["e"]["cfg"])]
        for cf in CF_SWEEP:
            cfg = {**c_cfg, "cell_factor": cf}
            arms[m][f"cf{cf:g}"] = {"cfg": cfg}
            jobs.append((m, f"cf{cf:g}", "F2", cfg))
    jobs.sort(key=lambda j: j[1] in ("c", "e"))  # new fusions first
    with ProcessPoolExecutor(workers, mp_context=_spawn()) as ex:
        out = dict(ex.map(_d_val, jobs))
    res["arms"] = {}
    keys = ("mAP50", "AP", "R50_crowd", "R50_iso")
    for m in D_MODELS:
        cv = dict(np.load(OUT / "dota" / f"ABL-c_{m}_{SZ}_boot.npz"))
        c = out[f"ABL-c_{m}_{SZ}"]
        res["arms"][m] = {}
        for name, x in arms[m].items():
            rid = f"ABL-{name}_{m}_{SZ}"
            o = out[rid]
            r = {**x, "variant": ARMS.get(name, "F2"), **{k: o[k] for k in keys}, "mAP50_all_objects": o["mAP50_all_objects"],
                 "dup_rate": o["dup_rate"], "frag_rate": o["frag_rate"], "n_det@0.25": o["n_det@0.25"],
                 "n_dets": o["n_dets"], "images_at_limit": o.get("images_at_limit")}
            if name != "c":
                v = dict(np.load(OUT / "dota" / f"{rid}_boot.npz"))
                r["vs_c"] = {k: _ci(o[k], c[k], v[k], cv[k]) for k in keys}
            res["arms"][m][name] = r
    f.write_text(json.dumps(res, indent=1, default=float))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=("visdrone", "dota"))
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    _cpu_slot()
    (visdrone if a.dataset == "visdrone" else dota)(a.workers)


if __name__ == "__main__":
    main()
