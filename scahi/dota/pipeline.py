# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase D3-D4: caches, fusion and calib tuning on DOTA (docs/PREREGISTRATION_DOTA.md incl. addendum 1).

Stacks: ``dota<s>`` = whole-image pass at the training size s (``w<s>``) + native-resolution tiles (``t1.0n1024``: 1024 px,
overlap 0.3), the same raw dense-head caches as VisDrone (scahi.infer.cache_mz; both heads). Every fusion uses the
VisDrone code (scahi.fusion.run.f0_image, scahi.fusion.scalespace.fuse_image) with the DOTA pipeline limit of
20 000 detections per pass / image (addendum 1: the primary metric has no cap). Selection metric everywhere: official
mAP50 without cap on calib (scahi.dota.evaluate). DOTA-val caches are built, but nothing is evaluated on DOTA-val
before the freeze.

Rows per model and s (Section 3): A_s (single pass, o2o, conf grid), NMS_s (box fusion IoS, 12 configs, o2o and o2m),
F2_s (SCAHI pre-specified, VisDrone F2 grid, o2m), RG_s (refined grid: 108 variants + the calib-selected F2 setting
as reference, fusion time <= 2x the reference measured in the same process; ties -> faster), B-F2_s (whole-image pass
only, parameters of F2_s), B-RG_s (whole-image pass only, own refined-grid selection).
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scahi.dota import data as D
from scahi.dota import evaluate as EV
from scahi.dota import train as T
from scahi.machine import REPO_ROOT

OVERLAP = 0.3
STACK = {1024: "dota1024", 2048: "dota2048"}
WHOLE = {1024: "w1024", 2048: "w2048"}
TILES = "t1.0n1024"
PIPE = T.PIPE_MAX_DET
CONF_GRID = (0.001, 0.01, 0.05, 0.1)
NMS_GRID = [{"metric": "IOS", "thr": t, "conf": c} for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8) for c in (0.01, 0.1)]
TIME_CAP = 2.0
# Ultralytics NMS on global DOTA coordinates: class offset above the largest image side, no wall-clock limit (its default
# limit of ~2 s per call would make dense images non-deterministic)
NMS_KW = {"max_wh": 1 << 15, "max_time_img": 1e6}
OUT = REPO_ROOT / "runs/dota/fusion"


def f2_grid() -> list[dict[str, Any]]:
    from scahi.fusion import run as R

    return [{"iou_cluster": 0.5, **g} for g in R.grid("F2")]


def refined_grid() -> list[dict[str, Any]]:
    """The complete refined grid for rows B and C (addendum: 'complete grid for rows B and C' = frozen row-C grid)."""
    g = json.loads((REPO_ROOT / "configs/ext/xgrid_frozen.json").read_text())["grid"]["C"]
    return [{"iou_cluster": 0.5, "tau_splat": a, "tau_peak": b, "cell_factor": c, "bins_per_oct": d}
            for a, b, c, d in itertools.product(g["tau_splat"], g["tau_peak"], g["cell_factor"], g["bins_per_oct"])]


def split_stems(split: str) -> list[str]:
    return T.split_stems("calib") if split == "calib" else D.stems("val")


def images(split: str) -> pd.DataFrame:
    src = "train" if split == "calib" else "val"
    return pd.DataFrame([(s, str(D.image_path(src, s))) for s in split_stems(split)], columns=["stem", "path"])


def caches(model: str, s: int, splits: tuple[str, ...] = ("calib", "val")) -> None:
    """GPU: raw caches of both levels of ``dota<s>`` for the given splits (resumable); FCOS: scahi.dota.fcos_train."""
    if model == T.FCOS:
        from scahi.dota import fcos_train

        return fcos_train.caches(s, splits)
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm
    from scahi.infer.raw_head import load_unfused

    w = T.weights(model, s)
    m = load_unfused(w)
    for split in splits:
        imgs = images(split)
        for lk in (WHOLE[s], TILES):
            t0 = time.time()
            cm.cache_level(m, w, imgs, lk, OVERLAP, split, CACHE_ROOT)
            print(f"[dota] cache {model} {s} {lk} {split}: {time.time() - t0:.0f} s", flush=True)


def load(model: str, s: int, split: str, branch: str, row: str, stems: list[str] | None = None,
         keep_dets: bool = True) -> dict:
    """{stem: (levels, W, H)}; row 'C' = stack (whole + tiles), 'B' = whole-image pass only. ``stems``: a subset
    (default: the whole split); ``keep_dets=False`` drops the per-pass detections of the FCOS caches (only box fusion
    reads them; memory)."""
    from scahi.fusion import scalespace as ss
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm

    w = T.weights(model, s)
    lks = (WHOLE[s], TILES) if row == "C" else (WHOLE[s],)
    dirs = [(k, cm.cache_dir(CACHE_ROOT, w, k, OVERLAP, split)) for k in lks]
    return {st: ss.load_levels(dirs, st, branch, OVERLAP, dets=keep_dets) for st in (stems or split_stems(split))}


def _to_dota(d: pd.DataFrame, stem: str) -> pd.DataFrame:
    return pd.DataFrame({"image": stem, "cls": d["category_id"].to_numpy(int) - 1, "score": d["score"].to_numpy(float),
                         "x1": d["x"].to_numpy(float), "y1": d["y"].to_numpy(float),
                         "x2": (d["x"] + d["w"]).to_numpy(float), "y2": (d["y"] + d["h"]).to_numpy(float)})


SAHI_BACKEND = "numba"  # box fusion (SAHI batched NMS): CPU, no N x N matrix, independent of the GPU (DOTA; the
#                        VisDrone runs used SAHI's "auto" = torchvision on the GPU; same greedy algorithm)


def fuse(data: dict, branch: str, variant: str, cfg: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    """Fused detections (scahi.dota.evaluate.DET_COLS) of every image; DOTA pipeline limit ``limit`` (default PIPE)."""
    from sahi.postprocess.backends import get_postprocess_backend, set_postprocess_backend

    from scahi.fusion import run as R
    from scahi.fusion import scalespace as ss

    if get_postprocess_backend() != SAHI_BACKEND:
        set_postprocess_backend(SAHI_BACKEND)

    out = []
    for stem, (levels, W, H) in data.items():
        if variant == "F0-NMS":
            d = R.f0_image(levels, W, H, branch, variant, cfg, max_det=limit or PIPE, nms_kw=NMS_KW)
        else:
            d = ss.fuse_image(levels, W, H, variant, ss.FusionParams(**{**cfg, "top_k": limit or PIPE}),
                              nc=len(D.CLASSES))  # variant: F2 (all frozen rows); F1 / F3 / F2n only in the ablation
        d = R.clip_xywh(d, W, H)
        out.append(_to_dota(d, stem))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=EV.DET_COLS)


def at_limit(dets: pd.DataFrame, limit: int | None = None) -> int:
    """Images whose detection count reaches the pipeline limit (must be 0; addendum 1)."""
    return int((dets.groupby("image").size() >= (limit or PIPE)).sum()) if len(dets) else 0


_W: dict = {}


def pipe(model: str) -> int:
    """Pipeline limit of the calib tuning: 20 000 for the YOLO models (as selected), 200 000 for FCOS (addendum 3)."""
    return 200_000 if model == T.FCOS else PIPE


def _init(model: str, s: int, branch: str, row: str, stems: list[str] | None = None, keep_dets: bool = True) -> None:
    _W.update(data=load(model, s, "calib", branch, row, stems, keep_dets), gt=EV.gt_arrays("train", T.split_stems("calib")),
              branch=branch, limit=pipe(model) if model == T.FCOS else None)


def _score(dets: pd.DataFrame, gt=None, limit: int | None = None) -> dict[str, float]:
    gt = _W["gt"] if gt is None else gt
    limit = _W.get("limit") if gt is _W.get("gt") else limit
    c = EV.build(gt, dets, None, sizes=False)
    c4 = EV.build(gt, dets, T.CAP, sizes=False)
    return {"mAP50": EV.map50(c), "mAP50_cap4000": EV.map50(c4), "images_at_limit": at_limit(dets, limit),
            "dets_per_image_max": int(dets.groupby("image").size().max()) if len(dets) else 0}


def _run_cfgs(args: tuple[str, list[dict], dict | None]) -> list[dict]:
    """Worker: evaluate configs on calib; with a reference, time it first in this process (refined-grid rule)."""
    variant, cfgs, ref = args
    t_ref = None
    if ref is not None:
        t0 = time.perf_counter()
        fuse(_W["data"], _W["branch"], variant, ref, limit=_W.get("limit"))
        t_ref = (time.perf_counter() - t0) / len(_W["data"])
    out = []
    for cfg in cfgs:
        t0 = time.perf_counter()
        d = fuse(_W["data"], _W["branch"], variant, cfg, limit=_W.get("limit"))
        t = (time.perf_counter() - t0) / len(_W["data"])
        out.append({**cfg, **_score(d), "fuse_s_per_image": t,
                    **({"time_ratio_vs_ref": t / t_ref, "ref_s_per_image": t_ref} if t_ref else {})})
    return out


def _run_task(args: tuple) -> tuple[pd.DataFrame, float]:
    """Worker (task split): fused detections and fusion time of one config on one image."""
    model, s, branch, row, stem, variant, cfg = args
    _pin()
    data = load(model, s, "calib", branch, row, [stem], keep_dets=variant == "F0-NMS")
    t0 = time.perf_counter()
    d = fuse(data, branch, variant, cfg, limit=pipe(model) if model == T.FCOS else None)
    return d, time.perf_counter() - t0


def _cache_bytes(model: str, s: int, row: str, stem: str, split: str = "calib") -> int:
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm

    lks = (WHOLE[s], TILES) if row == "C" else (WHOLE[s],)
    return sum((cm.cache_dir(CACHE_ROOT, T.weights(model, s), k, OVERLAP, split) / f"{stem}.npz").stat().st_size
               for k in lks)


def tune(model: str, s: int, branch: str, row: str, variant: str, cfgs: list[dict], ref: dict | None = None,
         workers: int = 7, split_images: bool | None = None) -> list[dict]:
    """Evaluate ``cfgs`` on calib in ``workers`` processes (cores ``FUSE_CORES``, nice). Config split (default): each
    worker loads all images and evaluates its share of the configs. Task split (fewer configs than workers and no
    timing reference, or ``split_images``): one task per (image, config), densest images first, each task loads only
    its image; the detections are put back into the image order of the config split before scoring, so the results are
    identical (parallelisation only; fuse_s_per_image = summed fusion time / images, as before)."""
    from concurrent.futures import ProcessPoolExecutor

    from scahi.common import spawn_context as _spawn

    keep = variant == "F0-NMS"
    if split_images if split_images is not None else (ref is None and len(cfgs) < workers):
        stems = T.split_stems("calib")
        rank = {st: i for i, st in enumerate(stems)}
        dense = sorted(stems, key=lambda st: -_cache_bytes(model, s, row, st))  # longest tasks first
        tasks = [(model, s, branch, row, st, variant, cfg) for st in dense for cfg in cfgs]
        res = _run_budgeted(tasks, workers, variant, model, s, row)
        gt = EV.gt_arrays("train", stems)
        out = []
        for k, cfg in enumerate(cfgs):
            parts = [res[i] for i in range(k, len(res), len(cfgs))]
            d = pd.concat([x[0] for x in parts], ignore_index=True)
            d = d.iloc[np.argsort(d.image.map(rank).to_numpy(), kind="stable")].reset_index(drop=True)
            t = sum(x[1] for x in parts) / len(stems)
            out.append({**cfg, **_score(d, gt, pipe(model) if model == T.FCOS else None), "fuse_s_per_image": t})
        return out
    chunks = [cfgs[i::workers] for i in range(min(workers, len(cfgs)))]
    with ProcessPoolExecutor(len(chunks), mp_context=_spawn(), initializer=_init_pinned,
                             initargs=(model, s, branch, row, None, keep)) as ex:
        return [r for part in ex.map(_run_cfgs, [(variant, c, ref) for c in chunks]) for r in part]


# memory per task ~ factor x cache size of the image (measured 2026-11-02 on the densest FCOS calib images: F2 14.8 GB
# for 356 MB, 10.2 GB for 263 MB; box fusion 2.9 GB for 356 MB) - conservative factors + 1 GB
MEM_FACTOR = {"F0-NMS": 15.0, "F2": 45.0}


def _run_budgeted(tasks: list[tuple], workers: int, variant: str, model: str, s: int, row: str) -> list:
    """Run the tasks in ``workers`` processes; a task starts only while the summed memory estimate of the running
    tasks stays within env SCAHI_FUSE_BUDGET_GB (default unlimited). Scheduling only; results in task order."""
    import os
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    from scahi.common import spawn_context as _spawn

    budget = float(os.environ.get("SCAHI_FUSE_BUDGET_GB", "inf"))
    est = [1.0 + MEM_FACTOR.get(variant, 45.0) * _cache_bytes(model, s, row, x[4]) / 2**30 for x in tasks]
    res: list = [None] * len(tasks)
    pending, running, used = list(range(len(tasks))), {}, 0.0
    with ProcessPoolExecutor(workers, mp_context=_spawn()) as ex:
        while pending or running:
            for i in list(pending):
                if len(running) >= workers:
                    break
                if not running or used + est[i] <= budget:
                    running[ex.submit(_run_task, tasks[i])] = i
                    used += est[i]
                    pending.remove(i)
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                i = running.pop(f)
                used -= est[i]
                res[i] = f.result()
    return res


# Refined grid as task split with the reference timed in the same process (user decision 2026-11-02, option 3; FCOS):
# one task per (image, config) in a fresh process, densest images first; each task also fuses the reference on its
# image (reference first on even, variant first on odd tasks); time_ratio_vs_ref = sum of the variant's times / sum of
# the reference times over its tasks. Detections go to parquet parts and are reassembled in image order for scoring.
RG_SAFETY = 1.25  # memory estimate: measured task peak on the densest image, scaled by cache size, times this factor
RG_BASE_GIB = 1.0  # interpreter + modules of a task process (measured 0.8 GB)


def _rg_key(cfg: dict) -> str:
    return "_".join(f"{cfg[k]:g}" for k in ("tau_splat", "tau_peak", "cell_factor", "bins_per_oct"))


def _run_ref_task(args: tuple) -> dict:
    """Worker (fresh process): fuse ``cfg`` and the reference ``ref`` on one image, alternating order by task index
    ``k``; detections of ``cfg`` -> ``part`` (parquet, if given); returns the times and the peak RSS of the process."""
    model, s, branch, row, stem, cfg, ref, k, part = args
    _pin()
    data = load(model, s, "calib", branch, row, [stem], keep_dets=False)
    lim = pipe(model) if model == T.FCOS else None

    def run(c: dict) -> tuple[pd.DataFrame, float]:
        t0 = time.perf_counter()
        d = fuse(data, branch, "F2", c, limit=lim)
        return d, time.perf_counter() - t0

    if k % 2 == 0:
        _, t_ref = run(ref)
        d, t = run(cfg)
    else:
        d, t = run(cfg)
        _, t_ref = run(ref)
    out = {"t": t, "t_ref": t_ref, "n": len(d), "peak_gib": _own_peak_gib(), "peak_src": "VmHWM"}
    if part is not None:
        d.to_parquet(part.with_suffix(".parquet"))
        part.with_suffix(".json").write_text(json.dumps(out))
    return out


def _own_peak_gib() -> float:
    """Peak RSS of this process's own address space (VmHWM, reset at exec). Not ru_maxrss: a spawned worker keeps the
    parent's high-water mark across exec (FCOS 2048 RGC: every small task read 13.5 GiB; user, 2026-11-05)."""
    for ln in Path("/proc/self/status").read_text().splitlines():
        if ln.startswith("VmHWM:"):
            return int(ln.split()[1]) / 2**20
    raise RuntimeError("no VmHWM in /proc/self/status")


def _rg_peak_ok(prof: dict, recs: list[dict]):
    """Filter for measured peaks usable in the tight fit: records without ``peak_src`` (ru_maxrss, before 2026-11-05)
    are at least the parent's high-water mark; those at that floor (the smallest legacy value, incl. the profile
    points) carry no information about the task and are left out. Larger legacy values are the task's own peak."""
    legacy = [r["peak_gib"] for r in recs if "peak_src" not in r] + [v["peak_gib"] for v in prof["points"].values()]
    floor = min(legacy) if legacy else float("-inf")
    return lambda r: "peak_src" in r or r["peak_gib"] > floor + 0.05


def _fresh_pool(n: int):
    from concurrent.futures import ProcessPoolExecutor

    from scahi.common import spawn_context as _spawn

    return ProcessPoolExecutor(n, mp_context=_spawn(), max_tasks_per_child=1)


def rg_mem_profile(model: str, s: int, branch: str, row: str, ref: dict) -> dict:
    """Measured task peaks (reference + variant in one process) on the densest calib image for the corners of the
    refined grid at the smallest tau_peak, the reference, interior points (monotonicity check) and the heaviest corner
    on a second, smaller image (cache-size scaling check). Serial, one fresh process per point. Resumable:
    runs/dota/fusion/<model>_<s>/rg_mem_<row>.json."""
    f = OUT / f"{model}_{s}" / f"rg_mem_{row}.json"
    prof = json.loads(f.read_text()) if f.exists() else {"points": {}}
    g = json.loads((REPO_ROOT / "configs/ext/xgrid_frozen.json").read_text())["grid"]["C"]
    stems = sorted(T.split_stems("calib"), key=lambda st: -_cache_bytes(model, s, row, st))
    prof.update(stem=stems[0], bytes=_cache_bytes(model, s, row, stems[0]), grid=g)

    def c(a, b, cf, bi):
        return {"iou_cluster": 0.5, "tau_splat": a, "tau_peak": b, "cell_factor": cf, "bins_per_oct": bi}

    tp = min(g["tau_peak"])
    corners = [c(a, tp, cf, bi) for a in (min(g["tau_splat"]), max(g["tau_splat"]))
               for cf in (min(g["cell_factor"]), max(g["cell_factor"])) for bi in (min(g["bins_per_oct"]), max(g["bins_per_oct"]))]
    mid = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    interior = [c(mid(g["tau_splat"]), tp, mid(g["cell_factor"]), mid(g["bins_per_oct"])),
                c(min(g["tau_splat"]), max(g["tau_peak"]), max(g["cell_factor"]), max(g["bins_per_oct"]))]
    pts = [("ref", stems[0], ref)] + [(_rg_key(x), stems[0], x) for x in corners + interior]
    heavy = c(min(g["tau_splat"]), tp, max(g["cell_factor"]), max(g["bins_per_oct"]))
    second = stems[min(9, len(stems) - 1)]
    pts.append((f"{_rg_key(heavy)}@{second}", second, heavy))
    for key, st, cfg in pts:
        if key in prof["points"]:
            continue
        with _fresh_pool(1) as ex:
            r = ex.submit(_run_ref_task, (model, s, branch, row, st, cfg, ref, 0, None)).result()
        prof["points"][key] = {**r, "stem": st, "bytes": _cache_bytes(model, s, row, st), "cfg": cfg}
        f.write_text(json.dumps(prof, indent=1))
        print(f"[rg-mem] {model} {s} {row} {key} on {st}: peak {r['peak_gib']:.1f} GiB, {r['t'] + r['t_ref']:.0f} s",
              flush=True)
    prof["corners"] = [_rg_key(x) for x in corners]
    prof["interior"] = [_rg_key(x) for x in interior]
    prof["scaling_check"] = f"{_rg_key(heavy)}@{second}"
    f.write_text(json.dumps(prof, indent=1))
    return prof


def _rg_peak(prof: dict, cfg: dict, ref: dict) -> float:
    """Task peak on the densest image: the dominating corner (smaller tau_splat, larger cell factor, more bins are
    assumed heavier); if the interior points contradict that ordering, the largest measured peak."""
    P_ = prof["points"]
    g = prof["grid"]
    if cfg == ref:
        return P_["ref"]["peak_gib"]

    def corner(x: dict) -> float:
        k = {"iou_cluster": 0.5, "tau_peak": min(g["tau_peak"]),
             "tau_splat": max(g["tau_splat"]) if x["tau_splat"] >= max(g["tau_splat"]) else min(g["tau_splat"]),
             "cell_factor": min(g["cell_factor"]) if x["cell_factor"] <= min(g["cell_factor"]) else max(g["cell_factor"]),
             "bins_per_oct": min(g["bins_per_oct"]) if x["bins_per_oct"] <= min(g["bins_per_oct"]) else max(g["bins_per_oct"])}
        return P_[_rg_key(k)]["peak_gib"]

    worst = max(v["peak_gib"] for k, v in P_.items() if "@" not in k)
    ok = all(P_[k]["peak_gib"] <= corner(P_[k]["cfg"]) for k in prof["interior"])
    return max(corner(cfg), P_["ref"]["peak_gib"]) if ok else worst


def _rg_est(prof: dict, cfg: dict, ref: dict, nbytes: int) -> float:
    # never below the base: where the fusion needs almost nothing (FCOS row B: every measured peak 0.78 GiB, below the
    # assumed 1.0 GiB base), peak - base < 0 would otherwise push the estimate under the measured peak (2026-11-03)
    return max(RG_BASE_GIB, RG_BASE_GIB + RG_SAFETY * (_rg_peak(prof, cfg, ref) - RG_BASE_GIB) * nbytes / prof["bytes"])


# Tighter memory model (user, 2026-11-02; scheduling only): refitted on the measured peaks of the stage's finished
# tasks (+ its profile points). y = (peak - base) / (dense peak of the config - base) against x = cache size / cache
# size of the densest image: y ~ Q x^alpha (alpha by least squares on log y for x >= 0.2, Q = 99.9th percentile of
# y / x^alpha there, raised if needed so that the estimate covers every measured peak); estimate = base + 1.25 Q
# (dense peak - base) x^alpha, never below a measured peak of the same config on an image that is not larger. Used once a stage has >= RG_TIGHT_MIN finished tasks; after one OOM kill of a run that used it
# (marker tight_model.json left in the task dir + a record in SCAHI_OOM_LOG), the rest of the stage falls back to _rg_est.
RG_TIGHT_SAFETY = 1.25
RG_TIGHT_XMIN = 0.2  # fit range in x
RG_TIGHT_MIN = (1000, 20)  # finished tasks of the stage, of which on images with x >= RG_TIGHT_XMIN


def _rg_tight_fit(prof: dict, cfgs: list[dict], ref: dict, done: list[tuple[int, float, float]]) -> dict | None:
    """``done``: (config index, x, peak GiB) of the finished tasks. None while there are too few."""
    if len(done) < RG_TIGHT_MIN[0] or sum(x >= RG_TIGHT_XMIN for _, x, _ in done) < RG_TIGHT_MIN[1]:
        return None
    pts = [(_rg_peak(prof, cfgs[ci], ref), x, pk) for ci, x, pk in done]
    pts += [(_rg_peak(prof, v["cfg"], ref) if k != "ref" else prof["points"]["ref"]["peak_gib"], v["bytes"] / prof["bytes"],
             v["peak_gib"]) for k, v in prof["points"].items()]
    pd_, x, pk = map(np.asarray, zip(*pts))
    y = (pk - RG_BASE_GIB) / (pd_ - RG_BASE_GIB)
    m = (x >= RG_TIGHT_XMIN) & (y > 0)  # fitted where memory decides the scheduling (small images: base overhead)
    if m.sum() < RG_TIGHT_MIN[1]:
        return None
    alpha = float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])
    q999 = float(np.quantile(y[m] / x[m] ** alpha, 0.999))
    # never below any measured peak (all points, all x): 1.25 q x^alpha >= y everywhere
    q = max(q999, float(np.max(y[y > 0] / x[y > 0] ** alpha)) / RG_TIGHT_SAFETY)
    return {"alpha": alpha, "q999_fit": q999, "q999": q, "n": int(len(done)), "n_fit": int(m.sum())}


def _rg_est_tight(fit: dict, prof: dict, cfg: dict, ref: dict, x: float, env: dict) -> float:
    pd_ = prof["points"]["ref"]["peak_gib"] if cfg == ref else _rg_peak(prof, cfg, ref)
    e = max(RG_BASE_GIB, RG_BASE_GIB + RG_TIGHT_SAFETY * fit["q999"] * (pd_ - RG_BASE_GIB) * x ** fit["alpha"])
    return max([e, *(pk for xx, pk in env.get(_rg_key(cfg), ()) if xx <= x)])


def _score_parts(args: tuple) -> dict:
    """Worker: detections of one config from its parts in image order -> _score (as the config-split path)."""
    model, d, stems = args
    det = pd.concat([pd.read_parquet(d / f"{st}.parquet") for st in stems], ignore_index=True)
    return _score(det, EV.gt_arrays("train", stems), pipe(model) if model == T.FCOS else None)


def tune_rg(model: str, s: int, branch: str, row: str, cfgs: list[dict], ref: dict, workers: int,
            task_dir: Path | None = None) -> list[dict]:
    """Refined grid, option 3 (see above). Resumable per task (parts + timings in ``task_dir``, default
    runs/dota/fusion/<model>_<s>/rg_tasks/RG<row>_<branch>). Memory: per-task estimate from rg_mem_profile; a task
    starts only while the summed estimates of the running tasks stay within env SCAHI_FUSE_BUDGET_GB (or when nothing
    else runs). Scheduling only; results in config order."""
    import os
    from concurrent.futures import FIRST_COMPLETED, wait

    prof = rg_mem_profile(model, s, branch, row, ref)
    sc = prof["points"][prof["scaling_check"]]
    est_sc = _rg_est(prof, sc["cfg"], ref, sc["bytes"])
    assert sc["peak_gib"] <= est_sc, f"cache-size scaling underestimates: {sc['peak_gib']:.1f} > {est_sc:.1f} GiB"
    task_dir = task_dir or OUT / f"{model}_{s}" / "rg_tasks" / f"RG{row}_{branch}"
    stems = T.split_stems("calib")
    nb = {st: _cache_bytes(model, s, row, st) for st in stems}
    dense = sorted(stems, key=lambda st: -nb[st])
    tasks, est = [], []
    for st in dense:
        for ci, cfg in enumerate(cfgs):
            (task_dir / f"{ci:03d}").mkdir(parents=True, exist_ok=True)
            tasks.append((model, s, branch, row, st, cfg, ref, len(tasks), task_dir / f"{ci:03d}" / st))
            est.append(_rg_est(prof, cfg, ref, nb[st]))
    todo = [i for i, t in enumerate(tasks) if not t[-1].with_suffix(".json").exists()]
    budget = float(os.environ.get("SCAHI_FUSE_BUDGET_GB", "inf"))
    print(f"[rg] {model} {s} RG{row}: {len(tasks)} tasks, {len(tasks) - len(todo)} done, max est {max(est):.1f} GiB, "
          f"budget {budget} GiB, {workers} workers", flush=True)
    marker = task_dir / "tight_model.json"
    # fallback only after an actual OOM kill of the scope since the tight model started in this stage (user, 2026-11-03):
    # a planned stop leaves the marker but no record in SCAHI_OOM_LOG (written by fcos_tuning_loop.sh); without that
    # variable every marker counts as a kill
    oom_log = os.environ.get("SCAHI_OOM_LOG")
    fallback = marker.exists() and (oom_log is None or any(
        ln[:19] >= json.loads(marker.read_text())["started"]
        for ln in (Path(oom_log).read_text().splitlines() if Path(oom_log).exists() else [])))
    if fallback:
        print(f"[rg] {marker} exists and the scope was OOM-killed since -> conservative estimate for the rest of the "
              "stage", flush=True)
    elif marker.exists():
        print(f"[rg] {marker} exists, no OOM kill recorded since (planned stop) -> tight model", flush=True)
    x_of = {st: nb[st] / prof["bytes"] for st in stems}
    est0 = list(est)

    def refit() -> None:
        idx = [i for i in range(len(tasks)) if i not in todo_set and tasks[i][-1].with_suffix(".json").exists()]
        rec = {i: json.loads(tasks[i][-1].with_suffix(".json").read_text()) for i in idx}
        ok = _rg_peak_ok(prof, list(rec.values()))  # without the inherited-floor artefact (user, 2026-11-05)
        idx = [i for i in idx if ok(rec[i])]
        pk = {i: rec[i]["peak_gib"] for i in idx}
        pf = {**prof, "points": {k: v for k, v in prof["points"].items() if ok(v)}}
        done = [(i % len(cfgs), x_of[tasks[i][4]], pk[i]) for i in idx]
        fit = _rg_tight_fit(pf, cfgs, ref, done)
        if fit is None:
            return
        env: dict = {}
        for ci, x, p in done:
            env.setdefault(_rg_key(cfgs[ci]), []).append((x, p))
        for k, v in pf["points"].items():
            key = _rg_key(ref) if k == "ref" else _rg_key(v["cfg"])
            env.setdefault(key, []).append((v["bytes"] / prof["bytes"], v["peak_gib"]))
        new = [_rg_est_tight(fit, prof, tasks[i][5], ref, x_of[tasks[i][4]], env) for i in range(len(tasks))]
        cover = max(pk[i] / _rg_est_tight(fit, prof, tasks[i][5], ref, x_of[tasks[i][4]], {}) for i in idx)  # fit only
        for i in todo:  # never above the conservative estimate (user, 2026-11-05)
            est[i] = min(new[i], est0[i])
        if not marker.exists():
            marker.write_text(json.dumps({**fit, "started": time.strftime("%F %T")}))
        print(f"[rg] tight memory model: alpha {fit['alpha']:.3f}, q999 {fit['q999']:.3f} ({fit['n_fit']} points), "
              f"max measured peak / fitted estimate {cover:.2f}, open tasks: max est {max([est[i] for i in todo] or [0]):.1f} GiB "
              f"(conservative {max([est0[i] for i in todo] or [0]):.1f})", flush=True)

    todo_set = set(todo)
    n_fit = -1
    running, used, t0, n0 = {}, 0.0, time.time(), len(todo)
    with _fresh_pool(workers) as ex:
        while todo or running:
            if not fallback and len(tasks) - len(todo_set) - n_fit >= 500:  # refit every 500 finished tasks
                n_fit = len(tasks) - len(todo_set)
                refit()
            for i in list(todo):
                if len(running) >= workers:
                    break
                if not running or used + est[i] <= budget:
                    running[ex.submit(_run_ref_task, tasks[i])] = i
                    used += est[i]
                    todo.remove(i)
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                i = running.pop(f)
                used -= est[i]  # the estimate the task was started with (est[i] of a running task is not refitted)
                todo_set.discard(i)
                r = f.result()
                if r["peak_gib"] > est[i]:
                    print(f"[rg] WARNING task {tasks[i][4]} {_rg_key(tasks[i][5])}: peak {r['peak_gib']:.1f} > "
                          f"est {est[i]:.1f} GiB", flush=True)
            if done and (n0 - len(todo)) % 500 < len(done):
                print(f"[rg] {n0 - len(todo) - len(running)}/{n0} tasks, {time.time() - t0:.0f} s", flush=True)
    t = {ci: [json.loads((task_dir / f"{ci:03d}" / f"{st}.json").read_text()) for st in stems] for ci in range(len(cfgs))}
    with _fresh_pool(min(workers, 8)) as ex:
        scores = list(ex.map(_score_parts, [(model, task_dir / f"{ci:03d}", stems) for ci in range(len(cfgs))]))
    out = []
    for ci, cfg in enumerate(cfgs):
        tv, tr = sum(x["t"] for x in t[ci]), sum(x["t_ref"] for x in t[ci])
        out.append({**cfg, **scores[ci], "fuse_s_per_image": tv / len(stems), "time_ratio_vs_ref": tv / tr,
                    "ref_s_per_image": tr / len(stems), "peak_gib_max": max(x["peak_gib"] for x in t[ci]),
                    "timing": "option 3: task split, reference in the same process per image"})
    return out


def fuse_cores() -> set[int]:
    """CPU cores of the calib tuning: env SCAHI_FUSE_CORES ("a-b"), default 16-23 (cores 0-15 for the GPU jobs)."""
    import os

    a, b = os.environ.get("SCAHI_FUSE_CORES", "16-23").split("-")
    return set(range(int(a), int(b) + 1))


def _pin() -> None:
    import os

    try:
        os.sched_setaffinity(0, fuse_cores())
        os.nice(19)
    except OSError:
        pass
    import torch

    torch.set_num_threads(1)  # one thread per worker (7 workers on 8 cores; as the VisDrone fusion, OMP_NUM_THREADS=1)


def _init_pinned(model: str, s: int, branch: str, row: str, stems: list[str] | None = None,
                 keep_dets: bool = True) -> None:
    _pin()
    _init(model, s, branch, row, stems, keep_dets)


def select(rows: list[dict], capped: bool = False) -> dict:
    """Max calib mAP50; refined grid: only variants with time <= 2x the reference; ties -> faster."""
    ok = [r for r in rows if not capped or r["time_ratio_vs_ref"] <= TIME_CAP] or rows
    return max(ok, key=lambda r: (r["mAP50"], -r["fuse_s_per_image"]))


def single_calib(model: str, s: int) -> dict:
    """GPU: single pass (o2o) of the s weights on calib: conf grid for A_s; for s = 2048 also the pre-specified
    inference crossing at 2560 and 3072 (addendum 1, item 4; best size selected on calib). Detections saved for the
    later freeze; calib numbers only."""
    d = OUT / f"{model}_{s}"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "single_calib.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    gt = EV.gt_arrays("train", T.split_stems("calib"))
    br = T.heads(model)[0]  # o2o for YOLOv10; the only head (+ NMS) for single-head models (addendum 2)
    for isz in ((s, 2560, 3072) if s == 2048 else (s,)):
        if str(isz) in res:
            continue
        dets, info = T.predict(T.weights(model, s), "train", T.split_stems("calib"), isz, br)
        dets.to_parquet(d / f"single_calib_infer{isz}_{br}.parquet")
        grid = {c: EV.map50(EV.build(gt, dets[dets.score >= c], None, sizes=False)) for c in CONF_GRID}
        res[str(isz)] = {"conf_grid": grid, "conf": max(grid, key=grid.get), "mAP50": max(grid.values()),
                         "images_at_limit": info["images_at_max_det"], "peak_mem_mb": info["peak_mem_mb"]}
        f.write_text(json.dumps(res, indent=1))
        print(f"[dota] single {model} {s} infer {isz}: {res[str(isz)]}", flush=True)
    return res


def fuse_job(model: str, s: int, workers: int = 7, workers_split: int | None = None) -> dict:
    """CPU: all calib tunings of one weight set (resumable per stage) -> runs/dota/fusion/<model>_<s>/tuning.json.
    Waits for the caches of the GPU job (d_gpu_<model>_<s>); CPU only (CUDA hidden from the workers)."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm

    def ready() -> bool:
        w = T.weights(model, s)
        return all((cm.cache_dir(CACHE_ROOT, w, k, OVERLAP, "calib") / f"{st}.npz").exists()
                   for k in (WHOLE[s], TILES) for st in split_stems("calib")) if w.exists() else False

    while not ready():
        time.sleep(120)
    d = OUT / f"{model}_{s}"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "tuning.json"
    res = json.loads(f.read_text()) if f.exists() else {}

    def stage(key: str, fn) -> None:
        if key in res:
            return
        t0 = time.time()
        rows = fn()
        res[key] = {"rows": rows, "wall_s": time.time() - t0}
        f.write_text(json.dumps(res, indent=1))
        print(f"[dota] {model} {s} {key}: {len(rows)} configs, {time.time() - t0:.0f} s", flush=True)

    for br in T.heads(model):
        stage(f"NMS_{br}", lambda br=br: tune(model, s, br, "C", "F0-NMS", NMS_GRID, workers=workers_split or workers,
                                              split_images=True if workers_split else None))
    stage("F2_o2m", lambda: tune(model, s, "o2m", "C", "F2", f2_grid(), workers=workers_split or workers,
                                 split_images=True if workers_split else None))
    ref = {k: v for k, v in select(res["F2_o2m"]["rows"]).items()
           if k in ("iou_cluster", "tau_splat", "tau_peak", "cell_factor", "bins_per_oct")}
    stage("B-F2_o2m", lambda: tune(model, s, "o2m", "B", "F2", [ref], workers=1))
    for row in ("C", "B"):  # refined grid: 108 variants + the reference, time measured against it in each worker
        if workers_split:  # FCOS: task split, reference timed per task (user decision 2026-11-02, option 3)
            stage(f"RG{row}_o2m", lambda row=row: tune_rg(model, s, "o2m", row, [ref, *refined_grid()], ref,
                                                          workers_split))
            continue
        stage(f"RG{row}_o2m", lambda row=row: tune(model, s, "o2m", row, "F2", [ref, *refined_grid()], ref=ref,
                                                   workers=workers))
    res["selected"] = {**{f"NMS_{br}": select(res[f"NMS_{br}"]["rows"]) for br in T.heads(model)},
                       "F2_o2m": select(res["F2_o2m"]["rows"]), "B-F2_o2m": res["B-F2_o2m"]["rows"][0],
                       "RGC_o2m": select(res["RGC_o2m"]["rows"], capped=True),
                       "RGB_o2m": select(res["RGB_o2m"]["rows"], capped=True), "reference_F2": ref}
    f.write_text(json.dumps(res, indent=1))
    return res


GATE_AP_TOL = 0.01  # |mAP50 own - Ultralytics| on calib (VisDrone gate rule, 2026-10-28)


def gate(model: str, s: int, n: int = 3) -> dict:
    """Parity gate (PREREGISTRATION_DOTA §5, as VisDrone G-E1): (1) own DFL decoding vs. Ultralytics' decoding of the
    same forward at every input geometry used (whole image at s, 1024 px tiles), per coordinate <= max(1e-4 px, 4 float32
    ulp), probabilities < 1e-4, TF32 off; o2o: own top-k vs. Ultralytics' end-to-end output; (2) calib mAP50 (primary,
    no cap) of the whole-image pass, own vs. Ultralytics decoding, identical post-processing, both heads, |Δ| <= 0.01.
    Writes runs/dota/fusion/<model>_<s>/gate.json; raises if it fails (no caches / fusion on a failed gate).
    FCOS: the FCOS version of the gate (addendum 3 item 8, scahi.dota.fcos_train.gate)."""
    if model == T.FCOS:
        from scahi.dota import fcos_train

        return fcos_train.gate(s, n)
    import cv2
    import torch
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.tal import dist2bbox, make_anchors

    from scahi.common import ulp_cmp as _ulp_cmp
    from scahi.infer.dfl_decode import decode, topk_like_ultralytics
    from scahi.infer.preprocess import full_image_shape, input_to_source, letterbox, to_tensor
    from scahi.infer.raw_head import forward_raw, load_unfused

    w = T.weights(model, s)
    imgs = images("calib")
    out: dict = {"gate": "D-parity", "model": model, "s": s, "rows": []}
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    m = load_unfused(w, "cuda", end2end=False)
    head = m.model[-1]
    for kind in ("whole", "tile"):
        for p in imgs.path.iloc[:n]:
            im = cv2.imread(p)
            x = (letterbox(im[:1024, :1024], (1024, 1024))[0] if kind == "tile" else
                 letterbox(im, full_image_shape(im.shape[0], im.shape[1], s))[0])
            xt = to_tensor([x], "cuda")
            raw = forward_raw(m, xt)
            d = decode(raw.boxes["o2m"], raw.scores["o2m"], raw.anchors, raw.strides, raw.reg_max)
            _, preds = m(xt)
            pp = preds if "boxes" in preds else preds["one2many"]
            anc, st = (a.transpose(0, 1) for a in make_anchors(pp["feats"], head.stride, 0.5))
            ref = (dist2bbox(head.dfl(pp["boxes"]), anc.unsqueeze(0), xywh=False, dim=1) * st).transpose(1, 2)
            c = _ulp_cmp(d.xyxy, ref)
            c["max_abs_prob"] = float((d.probs - pp["scores"].sigmoid().transpose(1, 2)).abs().max())
            out["rows"].append({"input": kind, "branch": "o2m", "shape": list(x.shape[:2]), **c})
            if "o2o" not in raw.boxes:  # single-head model (addendum 2): no end-to-end branch
                continue
            m.end2end, head.max_det = True, PIPE
            y2 = m(xt)[0]
            m.end2end, head.max_det = False, 300
            d1 = decode(raw.boxes["o2o"], raw.scores["o2o"], raw.anchors, raw.strides, raw.reg_max)
            ours = topk_like_ultralytics(d1.xyxy, d1.probs, PIPE)
            c1 = _ulp_cmp(ours[..., :4], y2[..., :4])
            c1["max_abs_prob"] = float((ours[..., 4] - y2[..., 4]).abs().max())
            c1["class_agreement"] = float((ours[..., 5] == y2[..., 5]).float().mean())
            out["rows"].append({"input": kind, "branch": "o2o", "shape": list(x.shape[:2]), **c1})
    # mAP50 identity on calib (whole image at s)
    gt = EV.gt_arrays("train", T.split_stems("calib"))
    m.end2end = False
    hs = T.heads(model)
    dets: dict = {(br, k): [] for br in hs for k in ("own", "ultralytics")}
    for stem, p in zip(imgs.stem, imgs.path):
        im = cv2.imread(p)
        x, lb = letterbox(im, full_image_shape(im.shape[0], im.shape[1], s))
        xt = to_tensor([x], "cuda")
        raw = forward_raw(m, xt)
        for br in hs:
            d = decode(raw.boxes[br], raw.scores[br], raw.anchors, raw.strides, raw.reg_max)
            if br == "o2o":
                own = topk_like_ultralytics(d.xyxy, d.probs, PIPE)[0]
                m.end2end, head.max_det = True, PIPE
                ul = m(xt)[0][0]
                m.end2end, head.max_det = False, 300
            else:
                o = torch.cat([(d.xyxy[..., :2] + d.xyxy[..., 2:]) / 2, d.xyxy[..., 2:] - d.xyxy[..., :2], d.probs], -1)
                own = non_max_suppression(o.transpose(1, 2), conf_thres=0.001, iou_thres=0.7, max_det=PIPE,
                                          max_nms=PIPE * 2, max_time_img=1e6)[0]
                ul = non_max_suppression(raw.y_ultralytics, conf_thres=0.001, iou_thres=0.7, max_det=PIPE,
                                         max_nms=PIPE * 2, max_time_img=1e6)[0]
            for k, r in (("own", own), ("ultralytics", ul)):
                r = r[r[:, 4] > 0.001].double().cpu().numpy()
                b = input_to_source(r[:, :4], lb)
                dets[(br, k)].append(pd.DataFrame({"image": stem, "cls": r[:, 5].astype(int), "score": r[:, 4],
                                                   "x1": b[:, 0], "y1": b[:, 1], "x2": b[:, 2], "y2": b[:, 3]}))
    torch.backends.cudnn.allow_tf32 = tf32
    out["map_identity"] = {}
    for br in hs:
        a = {k: EV.map50(EV.build(gt, pd.concat(dets[(br, k)], ignore_index=True), None, sizes=False))
             for k in ("own", "ultralytics")}
        out["map_identity"][br] = {**a, "abs_delta": abs(a["own"] - a["ultralytics"])}
    out["passed_decode"] = all(r["max_diff_over_tol"] <= 1.0 and r["max_abs_prob"] < 1e-4 for r in out["rows"])
    out["passed_o2o_class"] = all(r.get("class_agreement", 1.0) > 0.999 for r in out["rows"])
    out["passed_map"] = all(v["abs_delta"] <= GATE_AP_TOL for v in out["map_identity"].values())
    out["passed"] = out["passed_decode"] and out["passed_o2o_class"] and out["passed_map"]
    d = OUT / f"{model}_{s}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "gate.json").write_text(json.dumps(out, indent=1))
    del m
    torch.cuda.empty_cache()
    print(f"[dota] gate {model} {s}: passed={out['passed']} map {out['map_identity']}", flush=True)
    if not out["passed"]:
        raise RuntimeError(f"[dota] parity gate failed for {model} {s} (runs/dota/fusion/{model}_{s}/gate.json)")
    return out


def backend_compare(model: str = "yolov10s", s: int = 1024, n: int = 40) -> dict:
    """GPU (queue, GPU otherwise idle): box fusion (F0-NMS, the 12 NMS_GRID configs, both heads) with SAHI's numba backend
    (DOTA) vs the torchvision backend (what SAHI's "auto" resolved to in the VisDrone runs) on the first ``n`` calib
    images; same per-pass detections for both. Reports per config identical kept sets (images), |delta| of the kept
    counts and the mAP50 of both on these images -> runs/dota/fusion/backend_compare.json. Note: SAHI's torchvision
    backend computes IoS in float32 on the GPU only for < 2000 boxes per class and otherwise takes the numpy sparse
    path (CPU); numba computes in float64."""
    import torch
    from sahi.postprocess.backends import set_postprocess_backend
    from sahi.postprocess.combine import batched_nms

    from scahi.fusion import run as R

    stems = split_stems("calib")[:n]
    gt = EV.gt_arrays("train", stems)
    cuda = torch.cuda.is_available()
    out: dict[str, Any] = {"model": model, "s": s, "images": stems, "torch_device": "cuda" if cuda else "cpu", "rows": []}
    for br in ("o2o", "o2m"):
        data = {st: v for st, v in load(model, s, "calib", br, "C").items() if st in gt}
        raw = {st: pd.concat([R.pass_dets(lv, br, 0.001, W, H, PIPE, NMS_KW) for lv in lvs], ignore_index=True)
               for st, (lvs, W, H) in data.items()}
        for cfg in NMS_GRID:
            res: dict[str, Any] = {"branch": br, **cfg}
            keeps, dets = {}, {}
            for be in ("numba", "torchvision"):
                set_postprocess_backend(be)
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                t0, kk, dd = time.perf_counter(), {}, []
                for st, r in raw.items():
                    W, H = data[st][1:]
                    a = r[r.score >= cfg["conf"]][["x1", "y1", "x2", "y2", "score", "cls"]].to_numpy(np.float64)
                    k = np.asarray(batched_nms(a, match_metric=cfg["metric"], match_threshold=cfg["thr"])
                                   if len(a) > 1 else list(range(len(a))), int)
                    kk[st] = set(k.tolist())
                    b = a[k]
                    dd.append(_to_dota(R.clip_xywh(pd.DataFrame({
                        "x": b[:, 0], "y": b[:, 1], "w": b[:, 2] - b[:, 0], "h": b[:, 3] - b[:, 1], "score": b[:, 4],
                        "category_id": b[:, 5].astype(int) + 1}), W, H), st))
                keeps[be], dets[be] = kk, pd.concat(dd, ignore_index=True)
                res[f"{be}_s"] = time.perf_counter() - t0
                res[f"{be}_mAP50"] = EV.map50(EV.build(gt, dets[be], None, sizes=False))
                res[f"{be}_peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20 if cuda else None
            res["images_identical"] = sum(keeps["numba"][st] == keeps["torchvision"][st] for st in raw)
            res["kept_numba"] = sum(map(len, keeps["numba"].values()))
            res["kept_torchvision"] = sum(map(len, keeps["torchvision"].values()))
            res["kept_symmetric_difference"] = sum(len(keeps["numba"][st] ^ keeps["torchvision"][st]) for st in raw)
            res["abs_delta_mAP50"] = abs(res["numba_mAP50"] - res["torchvision_mAP50"])
            out["rows"].append(res)
            print(f"[dota] backend {br} {cfg}: identical {res['images_identical']}/{len(raw)}, "
                  f"|dmAP50| {res['abs_delta_mAP50']:.4f}", flush=True)
    set_postprocess_backend(SAHI_BACKEND)
    out["all_identical"] = all(r["images_identical"] == len(stems) for r in out["rows"])
    out["max_abs_delta_mAP50"] = max(r["abs_delta_mAP50"] for r in out["rows"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "backend_compare.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------- SAHI (Section 3)
# Standard SAHI on the 1024 weights (o2o), VisDrone grid v2 (scahi.infer.sahi_baseline: slices 640, overlaps
# 0.1 / 0.2 / 0.3, merges GREEDYNMM / NMM / NMS x IOU / IOS x 6 thresholds x conf {0.01, 0.1}, WBF x 6 thresholds x
# 4 skip thresholds; 72 configs per merge) tuned on calib with the primary metric. DOTA differences: SAHI's per-slice
# max_det 300 raised to the pipeline limit, no 500 cap after merging.

def sahi_raw(model: str, split: str, overlap: float, limit: int | None = None) -> Path:
    """GPU: raw (un-merged) SAHI predictions of every image (slices + standard pass) -> pickle of {stem: [n, 6]};
    ``limit``: predictor max_det per slice (default PIPE; a 640 slice has at most 8400 anchors)."""
    import pickle

    from sahi.predict import get_sliced_prediction

    from scahi.infer import sahi_baseline as sb

    d = OUT / f"{model}_1024"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"sahi_raw_{split}_o{int(overlap * 100):02d}.pkl"
    if f.exists():
        return f
    sb._register_identity()
    dm = sb.load_model(T.weights(model, 1024), min(sb.GRID2_WBF_SKIP), "o2o")
    dm.model.overrides["max_det"] = limit or PIPE
    raw = {}
    t0 = time.time()
    for stem, path in zip(images(split).stem, images(split).path):
        r = get_sliced_prediction(path, dm, slice_height=sb.SLICE, slice_width=sb.SLICE, overlap_height_ratio=overlap,
                                  overlap_width_ratio=overlap, perform_standard_pred=True, postprocess_type="IDENTITY",
                                  postprocess_match_threshold=0.5, postprocess_match_metric="IOS",
                                  force_postprocess_type=True, verbose=0)
        raw[stem] = np.array([[*p.bbox.to_xyxy(), p.score.value, p.category.id] for p in r.object_prediction_list],
                             np.float64).reshape(-1, 6)
    f.write_bytes(pickle.dumps(raw))
    print(f"[dota] SAHI raw {model} {split} overlap {overlap}: {time.time() - t0:.0f} s, "
          f"max {max(len(v) for v in raw.values())} raw predictions per image", flush=True)
    return f


def sahi_merge(raw: dict, merge: str, metric: str, thr: float, conf: float) -> pd.DataFrame:
    """scahi.infer.sahi_baseline.merge_offline on arrays, DOTA output, no cap."""
    from sahi.postprocess.backends import get_postprocess_backend, set_postprocess_backend
    from sahi.prediction import ObjectPrediction

    if get_postprocess_backend() != SAHI_BACKEND:
        set_postprocess_backend(SAHI_BACKEND)
    out = []
    sizes = _sizes()
    for stem, a in raw.items():
        a = a[a[:, 4] >= conf]
        if not len(a):
            continue
        W, H = sizes[stem]
        if merge in ("GREEDYNMM", "NMM", "NMS"):
            from sahi.predict import POSTPROCESS_NAME_TO_CLASS

            post = POSTPROCESS_NAME_TO_CLASS[merge](match_threshold=thr, match_metric=metric, class_agnostic=False)
            preds = [ObjectPrediction(bbox=list(map(float, r[:4])), category_id=int(r[5]), score=float(r[4]),
                                      full_shape=[H, W]) for r in a]
            k = post(preds) if len(preds) > 1 else preds
            m = np.array([[*p.bbox.to_xyxy(), p.score.value, p.category.id] for p in k], np.float64).reshape(-1, 6)
        else:
            from ensemble_boxes import weighted_boxes_fusion

            b = np.clip(a[:, :4] / [W, H, W, H], 0, 1)
            fb, fs, fl = weighted_boxes_fusion([b.tolist()], [a[:, 4].tolist()], [a[:, 5].tolist()], iou_thr=thr,
                                               skip_box_thr=conf)
            m = (np.column_stack([np.asarray(fb).reshape(-1, 4) * [W, H, W, H], fs, fl]) if len(fs)
                 else np.zeros((0, 6)))
        out.append(pd.DataFrame({"image": stem, "cls": m[:, 5].astype(int), "score": m[:, 4], "x1": m[:, 0],
                                 "y1": m[:, 1], "x2": m[:, 2], "y2": m[:, 3]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=EV.DET_COLS)


_SIZES: dict = {}


def _sizes() -> dict:
    if not _SIZES:
        from PIL import Image

        for split in ("calib", "val"):
            for stem, path in zip(images(split).stem, images(split).path):
                with Image.open(path) as im:
                    _SIZES[stem] = im.size
    return _SIZES


def _sahi_init(paths: list[str]) -> None:
    import pickle

    try:
        import os

        os.sched_setaffinity(0, set(range(16, 24)))
        os.nice(19)
    except OSError:
        pass
    import torch

    torch.set_num_threads(1)
    _W["sahi"] ={p: pickle.loads(Path(p).read_bytes()) for p in paths}
    _W["gt"] = EV.gt_arrays("train", T.split_stems("calib"))


def _sahi_eval(args: tuple[str, str, str, float, float]) -> dict:
    p, merge, metric, thr, conf = args
    t0 = time.perf_counter()
    d = sahi_merge(_W["sahi"][p], merge, metric, thr, conf)
    return {"raw": Path(p).name, "merge": merge, "metric": metric, "match_thr": thr, "conf": conf, **_score(d),
            "merge_s_per_image": (time.perf_counter() - t0) / len(_W["sahi"][p])}


def sahi_tune(model: str, workers: int = 7) -> dict:
    """CPU: grid v2 on calib for every overlap (raw pickles from the GPU job) -> sahi_tuning.json."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    from scahi.common import spawn_context as _spawn
    from scahi.infer import sahi_baseline as sb

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    d = OUT / f"{model}_1024"
    paths = [str(d / f"sahi_raw_calib_o{int(ov * 100):02d}.pkl") for ov in sb.OVERLAPS]
    while not all(Path(p).exists() for p in paths):
        time.sleep(120)
    cfgs = [(p, m, met, t, c) for p in paths for m in ("GREEDYNMM", "NMM", "NMS") for met in sb.GRID2_METRICS
            for t in sb.GRID2_THRS for c in sb.GRID2_CONFS]
    cfgs += [(p, "WBF", "IOU", t, c) for p in paths for t in sb.GRID2_THRS for c in sb.GRID2_WBF_SKIP]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=_spawn(), initializer=_sahi_init, initargs=(paths,)) as ex:
        rows = list(ex.map(_sahi_eval, cfgs))
    for r in rows:
        r["overlap"] = sb.OVERLAPS[[Path(p).name for p in paths].index(r["raw"])]
    best = {m: max((r for r in rows if r["merge"] == m), key=lambda r: r["mAP50"]) for m in sb.GRID2_MERGES}
    res = {"rows": rows, "best_per_merge": best, "selected": max(best.values(), key=lambda r: r["mAP50"]),
           "wall_s": time.time() - t0}
    (d / "sahi_tuning.json").write_text(json.dumps(res, indent=1))
    print(f"[dota] SAHI tuning {model}: {len(rows)} configs, {time.time() - t0:.0f} s, selected {res['selected']}",
          flush=True)
    return res
