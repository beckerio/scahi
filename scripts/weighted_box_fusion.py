# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exploratory border-weighted box fusion (post hoc; spec docs/ANALYSIS_WEIGHTED_BOX_FUSION.md, user 2026-11-07; framing
in docs/DECISIONS.md before any number). CPU only, from the existing caches; never toward any verdict.

    python scripts/weighted_box_fusion.py visdrone|dota --variants a[,b] [--workers N] [--cores 2-23]
    python scripts/weighted_box_fusion.py wbf_probe     # time of (b) on the densest DOTA calib images (decides the order)
    python scripts/weighted_box_fusion.py report        # -> reports/ANALYSIS_WEIGHTED_BOX_FUSION.{md,csv}

Question: does SCAHI's gain over box fusion come from the coverage-aware weighting or from merging the dense outputs
before boxes are decided? Same passes as the frozen box-fusion rows at 2048 px (whole image at the training size +
z = 2 tiles), same head, the detector's own per-pass detections (``scahi.fusion.run.pass_dets``, conf 0.001). Variants:
  (a) weighted NMS: score of every per-pass detection x SCAHI's F2 weight at the box centre (``scalespace.border_norm`` of
      its own tile: border taper b_t; whole-image pass 1; no validity window), then the box-fusion merge of
      ``run.f0_image`` (SAHI batched NMS);
  (b) weighted WBF: the same weighted scores, merged with Weighted Boxes Fusion (``run.f0_image``, ensemble_boxes).
Only the box-fusion parameters are selected on calib, with the grid and rule of the frozen row: VisDrone YOLOv10s
``run.grid("F0-NMS")`` (as X1), YOLOv11s ``ext.NMS_GRID``, DOTA ``pipeline.NMS_GRID`` + ``pipeline.select``; (b) the
main-study WBF grid ``run.grid("F0-WBF")``. VisDrone YOLOv10s runs on both heads; on o2m the plain box fusion is new
(post hoc, tuned like X1). Comparisons only between rows on the same head (amendment 2026-11-07).
Nothing in scahi/ is changed (the running FCOS row imports it). Output: runs/analysis_wbf/<dataset>.json (+ rows).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABEL = ("exploratory, post hoc (border-weighted box fusion, user 2026-11-07); never toward any verdict; reported only "
         "on the authors' decision")
OUT = REPO / "runs/analysis_wbf"
SZ = 2048
VARIANT = {"a": "F0-NMS", "b": "F0-WBF"}
CORES = os.environ.get("SCAHI_WBF_CORES", "2-23")


def _cores(spec: str) -> set[int]:
    a, b = spec.split("-")
    return set(range(int(a), int(b) + 1))


def _slot() -> None:
    import subprocess

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        os.sched_setaffinity(0, _cores(CORES))
        os.nice(19 - os.nice(0))
        subprocess.run(["ionice", "-c3", "-p", str(os.getpid())], check=False)
    except OSError:
        pass


def _spawn():
    from scahi.common import spawn_context

    return spawn_context()


def _ci(a0: float, b0: float, va, vb) -> dict:
    d = np.asarray(va) - np.asarray(vb)
    return {"a": a0, "b": b0, "delta": a0 - b0, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def weighted_raw(levels, W: int, H: int, branch: str, max_det: int | None = None, nms_kw: dict | None = None,
                 weighted: bool = True) -> pd.DataFrame:
    """Concatenated per-pass detections (as the frozen box-fusion rows: ``pass_dets`` at conf 0.001), scores multiplied
    by SCAHI's F2 weight at the box centre (``border_norm`` of the detection's own tile; whole-image pass: 1)."""
    from scahi.fusion import run as R
    from scahi.fusion import scalespace as ss
    from scahi.infer import cache_mz as cm

    parts = []
    for lv in levels:
        d = R.pass_dets(lv, branch, 0.001, W, H, max_det or cm.PASS_MAX_DET, nms_kw)
        if weighted and not lv.whole and len(d):
            cx, cy = ((d.x1 + d.x2) / 2).to_numpy(float), ((d.y1 + d.y2) / 2).to_numpy(float)
            bn, _ = ss.border_norm(cx, cy, lv, W, H)
            d = d.assign(score=d.score.to_numpy(float) * bn[np.arange(len(d)), d.tile.to_numpy(int)])
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------------------------------- VisDrone

# (model, head) -> frozen rows on the same head; "plain" None = new plain box fusion (tuned like X1, post hoc)
V_SETS = {("yolov10s", "o2o"): {"plain": "X1", "F2": None, "RG": "yolov10s/2048/o2o/C", "grid": "main"},
          ("yolov10s", "o2m"): {"plain": None, "F2": "X2", "RG": "yolov10s/2048/o2m/C", "grid": "main"},
          ("yolo11s", "o2m"): {"plain": "E", "F2": "E", "RG": "yolo11s/2048/o2m/C", "grid": "ext"}}


def v_grid(model: str, head: str, arm: str) -> list[dict]:
    from scahi import ext as E
    from scahi.fusion import run as R

    if arm == "b":
        return R.grid("F0-WBF")
    return R.grid("F0-NMS") if V_SETS[(model, head)]["grid"] == "main" else E.NMS_GRID


def v_frozen(model: str, head: str) -> dict[str, str]:
    """Val runs (directories with dets_val.parquet) of the frozen rows on this head."""
    from scahi import ext as E

    s = V_SETS[(model, head)]
    fr = {r["id"]: r for r in json.loads((REPO / "configs/final/frozen_rows.json").read_text())["rows"]}
    xg = json.loads((REPO / "configs/ext/xgrid_frozen.json").read_text())["weight_sets"]
    out = {"RG": str(REPO / xg[s["RG"]]["val_row"])}
    if s["plain"] == "X1":
        out["plain"] = str(REPO / fr["X1"]["val_run"])
    elif s["plain"] == "E":
        out["plain"] = str(E.ROWS / E.mz_exp(model, SZ, head, "F0-NMS"))
    if s["F2"] == "X2":
        out["F2"] = str(REPO / fr["X2"]["val_run"])
    elif s["F2"] == "E":
        out["F2"] = str(E.ROWS / E.mz_exp(model, SZ, head, "F2"))
    return out


def _v_calib(args: tuple[str, str, str, bool, list[dict]]) -> tuple[tuple, list[dict]]:
    from ext_cluster_diag import _load

    from scahi import ext as E
    from scahi.config import load_class_setting
    from scahi.eval.evaluate import evaluate
    from scahi.fusion import run as R

    _slot()
    model, head, arm, weighted, cfgs = args
    coco, s2i, data, _, _ = _load(model, SZ, head, "C", "calib")
    raw = {s: weighted_raw(lv, W, H, head, weighted=weighted) for s, (lv, W, H) in data.items()}
    cs = load_class_setting(E.SETTING)
    out = []
    for cfg in cfgs:
        t0 = time.perf_counter()
        d = R.fuse_split(data, s2i, head, VARIANT[arm], cfg, raw)
        t = (time.perf_counter() - t0) / len(data)
        r = evaluate(coco, d, cs, "visdrone_ignore", port_only=True)["metrics"]
        out.append({**cfg, "AP": r["AP"], "AP50": r["AP50"], "fuse_s_per_image": t})
    return (model, head, arm, weighted), out


def _v_val(args: tuple[str, str, str, bool, dict]) -> tuple[tuple, str, float]:
    from ext_cluster_diag import _load

    from scahi import ext as E
    from scahi.fusion import run as R

    _slot()
    model, head, arm, weighted, cfg = args
    name = f"{'w' if weighted else 'plain'}{'NMS' if arm == 'a' else 'WBF'}"
    exp = f"WBF_{model}_mz{SZ}_{name}_{head}"
    run = OUT / "visdrone" / exp
    tf = run / "fuse_time.json"
    if not (run / "metrics.json").exists():
        coco, s2i, data, lks, tiles = _load(model, SZ, head, "C", "val")
        t0 = time.perf_counter()
        raw = {s: weighted_raw(lv, W, H, head, weighted=weighted) for s, (lv, W, H) in data.items()}
        d = R.fuse_split(data, s2i, head, VARIANT[arm], cfg, raw)
        t = (time.perf_counter() - t0) / len(data)
        E.write_eval(exp, d, E.SETTING, coco, "val", {
            "model": model, "weights": E.rel(E.weights(model, SZ)), "train_variant": f"{model}-{SZ}",
            "inference": {"row": f"WBF-{name}", "levels": lks, "overlap": E.OVERLAP, "fusion": VARIANT[arm],
                          "weighted_scores": weighted, "branch": head, "params": cfg, "tuned_on": "calib"},
            "training": E.training_of(model, SZ), "notes": LABEL}, root=OUT / "visdrone", phase=8, tiles=tiles or None)
        tf.write_text(json.dumps({"fuse_s_per_image": t, "note": "incl. weighting; informational, not exclusive"}))
    return (model, head, arm, weighted), str(run), json.loads(tf.read_text())["fuse_s_per_image"]


def _v_measures(run: str) -> tuple[str, dict]:
    from ext_cluster_diag import _recall_vecs
    from ext_decomp import stats

    _slot()
    _, st = stats(run)
    _, rv = _recall_vecs(run)
    return run, {"st": st, "rv": rv}


def visdrone(arms: tuple[str, ...], workers: int) -> dict:
    (OUT / "visdrone").mkdir(parents=True, exist_ok=True)
    f = OUT / "visdrone.json"
    res = json.loads(f.read_text()) if f.exists() else {"label": LABEL, "calib": {}, "val": {}}
    jobs = []  # (model, head, arm, weighted)
    for (m, h), s in V_SETS.items():
        jobs += [(m, h, a, True) for a in arms]
        if s["plain"] is None and "a" in arms:
            jobs.append((m, h, "a", False))  # new plain box fusion on o2m, tuned like X1
    key = lambda j: "/".join(map(str, j))  # noqa: E731
    with ProcessPoolExecutor(workers, mp_context=_spawn()) as ex:
        todo = [(*j, v_grid(j[0], j[1], j[2])[i::4]) for j in jobs if key(j) not in res["calib"] for i in range(4)]
        part: dict[str, list] = {}
        for k, out in ex.map(_v_calib, todo):
            part.setdefault(key(k), []).extend(out)
        res["calib"].update(part)
        f.write_text(json.dumps(res, indent=1, default=float))
        sel = {key(j): max(res["calib"][key(j)], key=lambda r: (r["AP"], -r["fuse_s_per_image"])) for j in jobs}
        cfgs = {k: {c: v for c, v in r.items() if c in ("metric", "thr", "conf")} for k, r in sel.items()}
        for k, rk, t in ex.map(_v_val, [(*j, cfgs[key(j)]) for j in jobs]):
            res["val"][key(k)] = {"cfg": cfgs[key(k)], "calib_AP": sel[key(k)]["AP"], "run": rk, "fuse_s_per_image": t}
        frozen = {f"{m}/{h}": v_frozen(m, h) for m, h in V_SETS}
        runs = sorted({x["run"] for x in res["val"].values()} | {r for v in frozen.values() for r in v.values()})
        ms = dict(ex.map(_v_measures, runs))
    res["frozen"] = frozen
    res["rows"] = {}
    for (m, h), s in V_SETS.items():
        rows = {n: r for n, r in frozen[f"{m}/{h}"].items()}
        if s["plain"] is None:
            rows["plain"] = res["val"][f"{m}/{h}/a/False"]["run"]
        for a in arms:
            rows[f"w{'NMS' if a == 'a' else 'WBF'}"] = res["val"][f"{m}/{h}/{a}/True"]["run"]
        out = {}
        for n, run in rows.items():
            r = ms[run]
            o = {"run": run, **{k: r["st"][k] for k in ("AP", "AP50")}, "R50_crowd": r["rv"]["crowd"][0],
                 "R50_iso": r["rv"]["iso"][0]}
            for ref in ("plain", "F2", "RG"):
                if n.startswith("w") or (n == "plain" and ref != "plain"):
                    if ref in rows and ref != n:
                        c = ms[rows[ref]]
                        o[f"vs_{ref}"] = {k: _ci(r["st"][k], c["st"][k], r["st"][k + "_vec"], c["st"][k + "_vec"])
                                          for k in ("AP", "AP50")}
                        for g in ("crowd", "iso"):
                            o[f"vs_{ref}"][f"R50_{g}"] = _ci(r["rv"][g][0], c["rv"][g][0], r["rv"][g][1], c["rv"][g][1])
            out[n] = o
        res["rows"][f"{m}/{h}"] = out
    f.write_text(json.dumps(res, indent=1, default=float))
    return res


# -------------------------------------------------------------------------------------------------------- DOTA

D_MODELS = ("yolov10s", "yolo11s", "fcos")  # FCOS (2048) added by the user 2026-11-08 (DECISIONS, before any number)
_D: dict = {}
FCOS_WORKERS = int(os.environ.get("SCAHI_WBF_FCOS_WORKERS", "8"))


def _d_limit(model: str, split: str) -> int:
    """Per-pass limit of the frozen box-fusion row: calib = the tuning limit (``pipeline.pipe``), val = the frozen one."""
    from scahi.dota import final as F
    from scahi.dota import pipeline as P

    if split == "calib":
        return P.pipe(model)
    fz = F.FROZEN_FCOS if model == P.T.FCOS else (None if model in P.T.MODELS else F.FROZEN_NEW)
    return (F.frozen(fz) if fz else F.frozen())["pipeline_limit"]


def _d_fcos_image(args: tuple[str, str, str, list[tuple[int, dict]], str]) -> tuple[str, list[float]]:
    """FCOS (caches too large to hold a split in memory): one image, every config -> <parts>/<i>/<stem>.parquet."""
    import torch

    from scahi.dota import pipeline as P

    _slot()
    torch.set_num_threads(1)
    split, arm, stem, cfgs, parts = args
    lim = _d_limit("fcos", split)
    data = P.load("fcos", SZ, split, "o2m", "C", [stem])
    t0 = time.perf_counter()
    raw = {s: weighted_raw(lv, W, H, "o2m", lim, P.NMS_KW) for s, (lv, W, H) in data.items()}
    t_raw = time.perf_counter() - t0
    ts = []
    for i, cfg in cfgs:
        t1 = time.perf_counter()
        d = _d_fuse(data, raw, arm, cfg, lim)
        ts.append(t_raw + time.perf_counter() - t1)  # weighting + this config's merge
        f = Path(parts) / str(i) / f"{stem}.parquet"
        f.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(f)
    return stem, ts


def _d_fcos_run(split: str, arm: str, cfgs: list[dict], parts: Path) -> list[float]:
    """Fused detections of every config on ``split`` -> parts (per image, FCOS_WORKERS processes; each image loaded
    once for all configs); returns the fusion s per image of each config (weighting + merge; mean over images)."""
    from scahi.dota import pipeline as P

    stems = P.split_stems(split)
    todo = [(split, arm, s, list(enumerate(cfgs)), str(parts)) for s in stems
            if not all((parts / str(i) / f"{s}.parquet").exists() for i in range(len(cfgs)))]
    with ProcessPoolExecutor(FCOS_WORKERS, mp_context=_spawn()) as ex:
        ts = [t for _, t in ex.map(_d_fcos_image, todo)]
    tf = parts / "fuse_s_per_image.json"
    if ts:
        tf.write_text(json.dumps(np.mean(ts, axis=0).tolist()))
    return json.loads(tf.read_text()) if tf.exists() else [float("nan")] * len(cfgs)


def _d_fcos_read(split: str, parts: Path, i: int) -> pd.DataFrame:
    from scahi.dota import pipeline as P

    return pd.concat([pd.read_parquet(parts / str(i) / f"{s}.parquet") for s in P.split_stems(split)], ignore_index=True)


def _d_fuse(data: dict, raw: dict, arm: str, cfg: dict, limit: int) -> pd.DataFrame:
    """As ``scahi.dota.pipeline.fuse`` (box-fusion branch), with the (weighted) per-pass detections given."""
    from sahi.postprocess.backends import get_postprocess_backend, set_postprocess_backend

    from scahi.dota import evaluate as EV
    from scahi.dota import pipeline as P
    from scahi.fusion import run as R

    if get_postprocess_backend() != P.SAHI_BACKEND:
        set_postprocess_backend(P.SAHI_BACKEND)
    out = []
    for stem, (levels, W, H) in data.items():
        d = R.f0_image(levels, W, H, "o2m", VARIANT[arm], cfg, raw=raw[stem], max_det=limit, nms_kw=P.NMS_KW)
        out.append(P._to_dota(R.clip_xywh(d, W, H), stem))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=EV.DET_COLS)


def _d_init(model: str, split: str) -> None:
    from scahi.dota import pipeline as P

    _slot()
    data = P.load(model, SZ, split, "o2m", "C")
    raw = {s: weighted_raw(lv, W, H, "o2m", P.PIPE, P.NMS_KW) for s, (lv, W, H) in data.items()}
    _D.update(model=model, data=data, raw=raw)


def _d_calib(args: tuple[str, list[dict]]) -> list[dict]:
    from scahi.dota import pipeline as P

    arm, cfgs = args
    out = []
    for cfg in cfgs:
        t0 = time.perf_counter()
        d = _d_fuse(_D["data"], _D["raw"], arm, cfg, P.PIPE)
        out.append({**cfg, **P._score(d, P.EV.gt_arrays("train", P.T.split_stems("calib"))),
                    "fuse_s_per_image": (time.perf_counter() - t0) / len(_D["data"])})
    return out


def _d_val(args: tuple[str, str, dict]) -> tuple[str, dict]:
    import torch

    from scahi.dota import final as F
    from scahi.dota import pipeline as P

    _slot()
    torch.set_num_threads(1)
    model, arm, cfg = args
    rid = f"w{'NMS' if arm == 'a' else 'WBF'}_{model}_{SZ}"
    d = OUT / "dota"
    if not (d / f"{rid}.json").exists():
        lim = F.frozen()["pipeline_limit"] if model in P.T.MODELS else F.frozen(F.FROZEN_NEW)["pipeline_limit"]
        data = P.load(model, SZ, "val", "o2m", "C")
        t0 = time.perf_counter()
        raw = {s: weighted_raw(lv, W, H, "o2m", lim, P.NMS_KW) for s, (lv, W, H) in data.items()}
        dets = _d_fuse(data, raw, arm, cfg, lim)
        t = (time.perf_counter() - t0) / len(data)
        o, vec = F.evaluate_row(rid, dets=dets)
        o.update(images_at_limit=P.at_limit(dets, lim), cfg=cfg, fuse_s_per_image=t, pipeline_limit=lim, label=LABEL)
        np.savez(d / f"{rid}_boot.npz", **vec)
        (d / f"{rid}.json").write_text(json.dumps(o, indent=1, default=float))
    return rid, json.loads((d / f"{rid}.json").read_text())


def _d_val_fcos(model: str, arm: str, cfg: dict) -> tuple[str, dict]:
    """As ``_d_val`` for FCOS (per image, parts on disk; evaluation in this process)."""
    import torch

    from scahi.dota import final as F
    from scahi.dota import pipeline as P

    torch.set_num_threads(1)
    rid = f"w{'NMS' if arm == 'a' else 'WBF'}_{model}_{SZ}"
    d = OUT / "dota"
    if not (d / f"{rid}.json").exists():
        lim = _d_limit(model, "val")
        (t,) = _d_fcos_run("val", arm, [cfg], d / f"parts_val_{rid}")
        dets = _d_fcos_read("val", d / f"parts_val_{rid}", 0)
        o, vec = F.evaluate_row(rid, dets=dets)
        o.update(images_at_limit=P.at_limit(dets, lim), cfg=cfg, fuse_s_per_image=t, pipeline_limit=lim, label=LABEL,
                 note="per-pass limit; merged output not capped (frozen box-fusion path)")
        np.savez(d / f"{rid}_boot.npz", **vec)
        (d / f"{rid}.json").write_text(json.dumps(o, indent=1, default=float))
    return rid, json.loads((d / f"{rid}.json").read_text())


def dota(arms: tuple[str, ...], workers: int, models: tuple[str, ...] = D_MODELS) -> dict:
    from scahi.dota import final as F
    from scahi.dota import pipeline as P
    from scahi.fusion import run as R

    _slot()
    (OUT / "dota").mkdir(parents=True, exist_ok=True)
    f = OUT / "dota.json"
    res = json.loads(f.read_text()) if f.exists() else {"label": LABEL + "; DOTA-val already evaluated once for the "
                                                                        "frozen rows (post-hoc analysis)", "calib": {}}
    for m in models:
        for a in arms:
            k = f"{m}/{a}"
            if k in res["calib"]:
                continue
            grid = P.NMS_GRID if a == "a" else R.grid("F0-WBF")
            if m == P.T.FCOS:
                gt, lim = P.EV.gt_arrays("train", P.T.split_stems("calib")), P.pipe(m)
                parts = OUT / "dota" / f"parts_calib_{m}_{a}"
                t = _d_fcos_run("calib", a, grid, parts)
                res["calib"][k] = []
                for i, cfg in enumerate(grid):  # one config at a time in memory
                    d = _d_fcos_read("calib", parts, i)
                    res["calib"][k].append({**cfg, **P._score(d, gt, lim), "fuse_s_per_image": t[i]})
                    del d
                f.write_text(json.dumps(res, indent=1, default=float))
                continue
            n = min(workers, len(grid))
            with ProcessPoolExecutor(n, mp_context=_spawn(), initializer=_d_init, initargs=(m, "calib")) as ex:
                res["calib"][k] = [r for part in ex.map(_d_calib, [(a, grid[i::n]) for i in range(n)]) for r in part]
            f.write_text(json.dumps(res, indent=1, default=float))
    sel = {f"{m}/{a}": P.select(res["calib"][f"{m}/{a}"]) for m in models for a in arms}
    cfgs = {k: {c: v for c, v in r.items() if c in ("metric", "thr", "conf")} for k, r in sel.items()}
    jobs = [(*k.split("/"), c) for k, c in cfgs.items()]
    with ProcessPoolExecutor(max(1, min(workers, len(jobs))), mp_context=_spawn()) as ex:
        out = dict(ex.map(_d_val, [j for j in jobs if j[0] != P.T.FCOS]))
    out.update(_d_val_fcos(*j) for j in jobs if j[0] == P.T.FCOS)
    res.setdefault("selected", {}).update({k: {"cfg": cfgs[k], "calib_mAP50": sel[k]["mAP50"]} for k in cfgs})
    keys = ("mAP50", "AP", "R50_crowd", "R50_iso")
    res.setdefault("rows", {})  # other models of earlier runs kept
    for m in models:
        rows = {"plain": f"NMS-o2m_{m}_{SZ}", "F2": f"F2_{m}_{SZ}", "RG": f"RG_{m}_{SZ}"}
        src = {n: (json.loads((F.VAL_DIR / "eval" / f"{r}.json").read_text()), dict(np.load(F.VAL_DIR / "eval" / f"{r}_boot.npz")))
               for n, r in rows.items()}
        for a in arms:
            rid = f"w{'NMS' if a == 'a' else 'WBF'}_{m}_{SZ}"
            src[rid.split("_")[0]] = (out[rid], dict(np.load(OUT / "dota" / f"{rid}_boot.npz")))
        val = lambda o, k: o[k] if k in o else o["primary"][k]  # noqa: E731
        r_out = {}
        for n, (o, v) in src.items():
            x = {k: val(o, k) for k in keys}
            if n.startswith("w"):
                x.update(cfg=o["cfg"], images_at_limit=o["images_at_limit"], fuse_s_per_image=o["fuse_s_per_image"])
                for ref in ("plain", "F2", "RG"):
                    ro, rv = src[ref]
                    x[f"vs_{ref}"] = {k: _ci(val(o, k), val(ro, k), v[k], rv[k]) for k in keys}
            r_out[n] = x
        res["rows"][m] = r_out
    f.write_text(json.dumps(res, indent=1, default=float))
    return res


def _probe_one(args: tuple[str, str, str, dict]) -> float:
    from scahi.dota import pipeline as P

    _slot()
    m, stem, arm, cfg = args
    lim = P.pipe(m)
    data = P.load(m, SZ, "calib", "o2m", "C", [stem])
    raw = {s: weighted_raw(lv, W, H, "o2m", lim, P.NMS_KW) for s, (lv, W, H) in data.items()}
    t0 = time.perf_counter()
    _d_fuse(data, raw, arm, cfg, lim)
    return time.perf_counter() - t0


def wbf_probe(n_images: int = 3, timeout_s: float = 1800.0, models: tuple[str, ...] = D_MODELS) -> dict:
    """Time of (b) against (a) on the densest DOTA calib images, one configuration each (decides the order); each
    image in its own process, (b) stopped after ``timeout_s`` (recorded as a lower bound)."""
    import multiprocessing as mp

    from scahi.dota import pipeline as P
    from scahi.fusion import run as R

    _slot()
    out = {}
    for m in models:
        stems = sorted(P.T.split_stems("calib"), key=lambda st: -P._cache_bytes(m, SZ, "C", st))[:n_images]
        t: dict = {"a": [], "b": []}
        for st in stems:
            for a, cfg in (("a", P.NMS_GRID[0]), ("b", R.grid("F0-WBF")[0])):
                with ProcessPoolExecutor(1, mp_context=_spawn()) as ex:
                    fut = ex.submit(_probe_one, (m, st, a, cfg))
                    try:
                        t[a].append(fut.result(timeout=timeout_s))
                    except TimeoutError:
                        t[a].append(f">{timeout_s:.0f}")
                        for pr in mp.active_children():
                            pr.kill()
        n_raw = {}
        for st in stems:
            lv, W, H = P.load(m, SZ, "calib", "o2m", "C", [st])[st]
            n_raw[st] = len(weighted_raw(lv, W, H, "o2m", P.pipe(m), P.NMS_KW, weighted=False))
        out[m] = {"stems": stems, "s_per_image": t, "n_raw": n_raw}
        print(f"[wbf-probe] {m}: {t}", flush=True)
    (OUT / "dota").mkdir(parents=True, exist_ok=True)
    (OUT / "dota" / "wbf_probe.json").write_text(json.dumps(out, indent=1, default=float))
    return out



# ------------------------------------------------------------------------------------------------------- report

NAME = {"yolov10s": "YOLOv10s", "yolo11s": "YOLOv11s", "fcos": "FCOS"}
ROW_LABEL = {"plain": "Box-Fusion (eingefroren)", "wNMS": "(a) gewichtetes NMS", "wWBF": "(b) gewichtetes WBF",
             "F2": "SCAHI vorab festgelegt (F2)", "RG": "SCAHI verfeinertes Raster (RG)"}
ORDER = ("plain", "wNMS", "wWBF", "F2", "RG")


def _f(x: float, nd: int = 2) -> str:
    return f"{x:+.{nd}f}".replace(".", ",") if x is not None else "–"


def _v(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def _cell(c: dict | None) -> str:
    return "–" if c is None else f"{_f(c['delta'])} [{_f(c['ci95'][0])}; {_f(c['ci95'][1])}]"


def report() -> Path:
    """reports/ANALYSIS_WEIGHTED_BOX_FUSION.{md,csv} from runs/analysis_wbf/{visdrone,dota}.json (what exists)."""
    vis = json.loads((OUT / "visdrone.json").read_text()) if (OUT / "visdrone.json").exists() else {"rows": {}}
    dot = json.loads((OUT / "dota.json").read_text()) if (OUT / "dota.json").exists() else {"rows": {}}
    sets = [("VisDrone val", "AP", "AP50", {k: v for k, v in vis.get("rows", {}).items()},
             {k: v for k, v in vis.get("val", {}).items()}),
            ("DOTA-val", "mAP50", "AP", {f"{m}/o2m": v for m, v in dot.get("rows", {}).items()},
             dot.get("selected", {}))]
    csv, md = [], [
        "# Randgewichtete Box-Fusion (explorativ, post hoc)", "",
        "Nutzerauftrag 2026-11-07 (Spezifikation `docs/ANALYSIS_WEIGHTED_BOX_FUSION.md`, Einordnung in `docs/DECISIONS.md`, "
        "beides vor jeder Zahl); FCOS auf DOTA ergänzt 2026-11-08. **Nie Grundlage eines Urteils**; Verwendung im Paper nur "
        "nach Entscheidung der Autoren. Nur CPU, aus den vorhandenen Caches (`scripts/weighted_box_fusion.py`).", "",
        "Frage: Kommt der Gewinn von SCAHI gegenüber Box-Fusion aus der abdeckungsbewussten Gewichtung oder daraus, dass "
        "die dichten Ausgaben zusammengeführt werden, bevor Boxen entschieden sind?", "",
        "- Dieselben Pässe wie die eingefrorenen Box-Fusion-Zeilen bei 2048 px (Einzelpass bei der Trainingsgröße + "
        "Kacheln), derselbe Kopf, die eigenen Detektionen des Detektors pro Pass.",
        "- (a) gewichtetes NMS: Score jeder Detektion pro Pass × SCAHI-Gewicht (Rand-Taper b_t der eigenen Kachel, "
        "Einzelpass 1, kein Validity-Fenster) am Boxmittelpunkt, dann die Box-Fusion der eingefrorenen Zeile (NMS mit "
        "Schnitt über der kleineren Box). (b) gewichtetes WBF: dieselben Scores, Weighted Boxes Fusion (Raster der "
        "Hauptstudie).",
        "- Nur die Box-Fusion-Parameter auf calib gewählt, mit Raster und Regel der eingefrorenen Zeile.",
        "- Vergleiche nur zwischen Zeilen auf demselben Kopf. Gepaarter Bootstrap über Bilder (1000 Resamples, Seed 0, "
        "dieselben Resamples wie die Hauptauswertung), Perzentil-95-%-KI. R50: Recall bei IoU 0,5 dicht stehender bzw. "
        "vereinzelter Objekte (Definitionen wie bisher).",
        "- VisDrone YOLOv10s o2m: die Box-Fusion auf o2m ist neu (post hoc, abgestimmt wie X1); auf o2o gibt es keine "
        "vorab festgelegte SCAHI-Zeile.", ""]
    summary = {}
    for ds, pm, sm, rows, sel in sets:
        md += [f"## {ds}", "", f"Primärmetrik {pm} (zusätzlich {sm}). Δ = Zeile − Referenz.", "",
               f"| Detektor | Kopf | Zeile | {pm} | {sm} | R50 dicht | R50 vereinzelt | Δ {pm} gg. Box-Fusion [KI] "
               f"| Δ {pm} gg. RG [KI] | Δ {sm} gg. Box-Fusion [KI] | Δ {sm} gg. RG [KI] |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        a_p, r_w = {pm: [], sm: []}, {pm: [], sm: []}
        for key in sorted(rows, key=lambda k: (list(NAME).index(k.split("/")[0]), k.split("/")[1] != "o2o")):
            m, head = key.split("/")
            r = rows[key]
            for n in ORDER:
                if n not in r:
                    continue
                x = r[n]
                vp, vr = x.get("vs_plain"), x.get("vs_RG")
                lab = ("Box-Fusion (neu, post hoc, abgestimmt wie X1)" if n == "plain" and ds.startswith("VisDrone")
                       and V_SETS.get((m, head), {}).get("plain", "") is None else ROW_LABEL[n])
                md.append(f"| {NAME[m]} | {head} | {lab} | {_v(x[pm])} | {_v(x[sm])} | {_v(x['R50_crowd'], 1)} | "
                          f"{_v(x['R50_iso'], 1)} | {_cell(vp and vp[pm]) if n.startswith('w') else ''} | "
                          f"{_cell(vr and vr[pm]) if n.startswith('w') else ''} | "
                          f"{_cell(vp and vp[sm]) if n.startswith('w') else ''} | "
                          f"{_cell(vr and vr[sm]) if n.startswith('w') else ''} |")
                csv.append({"dataset": ds, "detector": NAME[m], "head": head, "row": n, pm: x[pm], sm: x[sm],
                            "R50_crowd": x["R50_crowd"], "R50_iso": x["R50_iso"],
                            **{f"{ref}_{k}_{q}": (x[f"vs_{ref}"][k]["delta"] if q == "delta" else
                                                  x[f"vs_{ref}"][k]["ci95"][0 if q == "lo" else 1])
                               for ref in ("plain", "RG") if f"vs_{ref}" in x and n.startswith("w")
                               for k in (pm, sm, "R50_crowd", "R50_iso") for q in ("delta", "lo", "hi")}})
            if "wNMS" in r:
                for k in (pm, sm):
                    a_p[k].append(r["wNMS"]["vs_plain"][k]["delta"])
                    r_w[k].append(-r["wNMS"]["vs_RG"][k]["delta"])
        md.append("")
        if a_p[pm]:
            rng = lambda v: f"{_f(min(v))} bis {_f(max(v))}" if len(v) > 1 else _f(v[0])  # noqa: E731
            line = (f"**Zusammenfassung {ds}** ({len(a_p[pm])} Detektor-/Kopf-Paare): (a) gewichtetes NMS − Box-Fusion "
                    f"{pm} {rng(a_p[pm])} ({sm} {rng(a_p[sm])}); SCAHI (RG) − gewichtetes NMS {pm} {rng(r_w[pm])} "
                    f"({sm} {rng(r_w[sm])}).")
            summary[ds] = line
            md += [line, ""]
        md += ["Auf calib gewählte Box-Fusion-Parameter der gewichteten Varianten: "
               + "; ".join(f"{k}: {json.dumps(v.get('cfg', v))}" for k, v in sel.items()) + ".", ""]
    have_fcos = "fcos" in dot.get("rows", {})
    md += ["## Hinweise", "",
           "- DOTA FCOS (2048): " + ("Variante (a) ausgewertet; " if have_fcos else "**ausstehend** (läuft nach der "
                                                                                    "DOTA-val-Grenzprüfung); ")
           + "Variante (b) nicht gerechnet (Nutzerentscheidung 2026-11-08): Kosten (bis zu 1,4 M Detektionen pro Bild "
             "nach der Fusion auf calib; WBF ist bei dieser Menge nicht praktikabel), und (b) lag schon auf VisDrone und "
             "auf DOTA für YOLO unter der Box-Fusion.",
           "- DOTA: Grenze 200 000 bzw. 20 000 pro Pass wie die eingefrorene Box-Fusion-Zeile; die zusammengeführte "
           "Ausgabe ist nicht gekürzt (eingefrorener Pfad).",
           "- Fusionszeit pro Bild (informativ, nicht exklusiv gemessen): in `runs/analysis_wbf/*.json`.",
           "- CSV mit allen Werten und KI-Grenzen: `reports/ANALYSIS_WEIGHTED_BOX_FUSION.csv`.", ""]
    f = REPO / "reports/ANALYSIS_WEIGHTED_BOX_FUSION.md"
    f.write_text("\n".join(md))
    pd.DataFrame(csv).to_csv(f.with_suffix(".csv"), index=False)
    for line in summary.values():
        print(line)
    return f

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("visdrone", "dota", "wbf_probe", "report"))
    ap.add_argument("--variants", default="a")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--models", default=",".join(D_MODELS), help="DOTA: subset of the models (resumable per model)")
    a = ap.parse_args()
    _slot()
    arms = tuple(a.variants.split(","))
    if a.cmd == "report":
        print(report())
    elif a.cmd == "wbf_probe":
        print(json.dumps(wbf_probe(models=tuple(a.models.split(","))), indent=1, default=float))
    else:
        visdrone(arms, a.workers) if a.cmd == "visdrone" else dota(arms, a.workers, tuple(a.models.split(",")))


if __name__ == "__main__":
    main()
