# SPDX-License-Identifier: AGPL-3.0-or-later
"""Latency of the multi-zoom arm on DOTA (user 2026-11-13; supplement Tab. s-multizoom), ONE exclusive run on host-b with
the protocol of the DOTA latency table (scripts/latency_dota_host_b.py).

    python scripts/latency_multizoom_host_b.py parity    # GPU path vs. reference (caches + SCAHI on the CPU), every image of F
    python scripts/latency_multizoom_host_b.py measure   # the exclusive run -> runs/latency_multizoom_host_b/latency.json
    python scripts/latency_multizoom_host_b.py report    # reports/dota/tab_dota_latency_multizoom.{md,csv}

Rows per detector (YOLOv10s, YOLO11s; frozen 2048 weights) on the images where the native pass fits (subset F of the
native-pass analysis): multi-zoom = native whole-image pass (z = 1, padding only) + 2048 px pass + SCAHI (refined-grid
parameters of RG_2048) on the GPU in the optimised path (unfused model, one-to-many head only, decoding and SCAHI on the
GPU; image decoding counted in the native pass), timed as native pass / 2048 px pass / fusion / total, with the peak GPU
memory per image; for comparison in the same run and on the same images: tiles + SCAHI (optimised path, ``opt`` of
latency_dota_host_b) and the 2048 px single pass (``single`` of latency_host_a_dota). Exclusivity as the DOTA table: CPU
queue paused, 575 W, no foreign compute process on the GPU, foreign CPU load < 1 core; every row "final", otherwise the
whole run is repeated (at most 3 runs); nvidia-smi lists at the start and end of each run. Allocator: every row of this
run uses PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (the native passes of the largest images otherwise fail on
fragmentation of the caching allocator; allocator setting only, same computation).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch is imported (see docstring)

import datetime as dt
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import dota_native_more as NM  # noqa: E402
import dota_native_pass as NP  # noqa: E402
import latency_dota_host_b as LM  # noqa: E402
import latency_host_a_dota as KD  # noqa: E402

OUTD = REPO / "runs/latency_multizoom_host_b"
MAX_RUNS = 3
MZ_STACK = "mz_w2048"  # runtime-only stack key: the w2048 level alone (source of cache_mz unchanged)


def _setup() -> list[dict]:
    return [d for d in KD.setup() if d["model"] in NP.MODELS]


def _images(m: str) -> pd.DataFrame:
    from scahi.dota import pipeline as P

    imgs = P.images("val")
    return imgs[imgs.stem.isin(set(NP._subset(m)))].reset_index(drop=True)


def _native_level(model, img: np.ndarray, nc: int):
    """Native whole-image level (z = 1, padding only) on the GPU, CachedLevel as ``pipeline_gpu.gpu_levels``; values as
    ``dota_native_more.native_level`` (the cache of the accuracy analysis)."""
    import cv2
    import torch

    from scahi.fusion import scalespace as ss
    from scahi.infer.cache import SCORE_MIN
    from scahi.infer.dfl_decode import decode
    from scahi.infer.preprocess import to_tensor
    from scahi.infer.raw_head import forward_raw_branch

    H, W = img.shape[:2]
    Hp, Wp = -(-H // 32) * 32, -(-W // 32) * 32
    x = cv2.copyMakeBorder(img, 0, Hp - H, 0, Wp - W, cv2.BORDER_CONSTANT, value=(NM.PAD, NM.PAD, NM.PAD))
    raw = forward_raw_branch(model, to_tensor([x], "cuda"), "o2m")
    d = decode(raw.boxes["o2m"], raw.scores["o2m"], raw.anchors, raw.strides, raw.reg_max)
    k = d.probs[0].max(-1)[0] >= SCORE_MIN
    xy, pr = d.xyxy[0][k], d.probs[0][k]
    if pr.shape[0] == 0:
        pr = torch.zeros((0, nc), device="cuda")
    return ss.CachedLevel("w1.0", xy, pr, None, torch.zeros(int(k.sum()), dtype=torch.long, device="cuda"),
                          np.array([[0, 0, W, H]], np.int32), np.array([1.0]), True, LM.KD_OVERLAP(), ss.LevelCal())


def _mz_levels(model, img: np.ndarray, nc: int, t: list | None = None):
    import torch

    from scahi.infer import cache_mz as cm

    cm.STACK_LEVELS.setdefault(MZ_STACK, ("w2048",))
    nat = _native_level(model, img, nc)
    torch.cuda.synchronize()
    if t is not None:
        t.append(time.perf_counter())
    lv, W, H = LM._opt_levels(model, img, MZ_STACK, "o2m", nc)
    return [nat, *lv], W, H


def parity() -> dict:
    """Multi-zoom on the GPU (as timed) vs. the reference of the accuracy analysis (native cache + frozen w2048 cache,
    SCAHI on the CPU), one-to-one matched per image and class (scripts/parity_opt_matched.py)."""
    import cv2
    import torch

    import parity_opt_matched as PM
    from scahi.dota import data as D
    from scahi.dota import train as DT
    from scahi.fusion import scalespace as ss
    from scahi.fusion.run import clip_xywh
    from scahi.fusion.scalespace_torch import fuse_image_f2
    from scahi.infer.raw_head import load_unfused

    nc, out = len(D.CLASSES), {}
    for d in _setup():
        m, rg = d["model"], d["RG"]
        prm = ss.FusionParams(**{**rg["cfg"], "top_k": d["limit"]})
        model = load_unfused(DT.weights(m, d["s"]))
        imgs = _images(m)
        ref = NM._mz_data(m, list(imgs.stem))
        res = PM._new()
        with torch.no_grad():
            for st, pa in zip(imgs.stem, imgs.path):
                lv, W, H = ref[st]
                a = clip_xywh(ss.fuse_image(lv, W, H, "F2", prm, nc=nc), W, H)
                levels, W2, H2 = _mz_levels(model, cv2.imread(pa), nc)
                b = clip_xywh(fuse_image_f2(levels, W2, H2, prm, nc=nc), W2, H2)
                PM.compare(a, b, res, st)
        out[m] = res
        print("[mz-parity]", m, PM._summary(res), flush=True)
        del model
        torch.cuda.empty_cache()
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "parity.json").write_text(json.dumps({"tolerance_relative": PM.TOL, "rows": out}, indent=1, default=float))
    return out


def mz(d: dict, imgs: pd.DataFrame, n_warmup: int = 2) -> dict:
    import cv2
    import torch

    from scahi.dota import data as D
    from scahi.dota import train as DT
    from scahi.fusion import scalespace as ss
    from scahi.fusion.run import clip_xywh
    from scahi.fusion.scalespace_torch import fuse_image_f2
    from scahi.infer.raw_head import load_unfused
    from scahi.machine import check_exclusive

    rg, nc = d["RG"], len(D.CLASSES)
    model = load_unfused(DT.weights(d["model"], d["s"]))
    prm = ss.FusionParams(**{**rg["cfg"], "top_k": d["limit"]})

    def one(path: str) -> tuple[float, float, float, float]:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = [time.perf_counter()]
        img = cv2.imread(path)
        levels, W, H = _mz_levels(model, img, nc, t)
        torch.cuda.synchronize()
        t.append(time.perf_counter())
        clip_xywh(fuse_image_f2(levels, W, H, prm, nc=nc), W, H)
        torch.cuda.synchronize()
        t.append(time.perf_counter())
        return (1000 * (t[1] - t[0]), 1000 * (t[2] - t[1]), 1000 * (t[3] - t[2]),
                torch.cuda.max_memory_allocated() / 2**20)

    with torch.no_grad():
        for pa in imgs.path[:n_warmup]:
            one(pa)
        ts = []
        for n, pa in enumerate(imgs.path):
            if n % 100 == 0:
                check_exclusive()
            ts.append(one(pa))
        check_exclusive()
    t = np.array(ts)
    del model
    torch.cuda.empty_cache()
    return {"native_pass_ms": KD._stats(t[:, 0]), "pass_2048_ms": KD._stats(t[:, 1]), "scahi_gpu_ms": KD._stats(t[:, 2]),
            "total_ms": KD._stats(t[:, :3].sum(1)), "peak_mem_mb": {"median": float(np.median(t[:, 3])),
                                                                    "max": float(t[:, 3].max())},
            "n_images": len(t), "path": "optimised (unfused model, one-to-many head only, decoding + SCAHI on the GPU)"}


def _with_peak(fn):
    def run(d, imgs):
        import torch

        torch.cuda.reset_peak_memory_stats()
        r = fn(d, imgs)
        return {**r, "peak_mem_mb_row": torch.cuda.max_memory_allocated() / 2**20}
    return run


def _jobs() -> list[tuple[str, dict, object]]:
    jobs = []
    for d in _setup():
        jobs += [(f"{d['model']}_single", d, _with_peak(KD.single)), (f"{d['model']}_tiles_opt", d, _with_peak(LM.opt)),
                 (f"{d['model']}_mz", d, mz)]
    return jobs


def run(n: int | None, tag: str | None) -> dict:
    from scahi.machine import CPU_SHARING, GPU_SHARING, latency_validity, machine_info

    if tag:
        LM.OUTD = OUTD
        LM._smi(f"{tag}_start")
    res = {"started": dt.datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
           "machine": machine_info(), "rows": {}}
    for key, d, fn in _jobs():
        imgs = _images(d["model"])
        if n:
            imgs = imgs.head(n)
        GPU_SHARING.update(checks=0, other_pids=[])
        CPU_SHARING.update(checks=0, cpu_queue_busy=False, other_load_cores=0.0)
        info = fn(d, imgs)
        v = latency_validity()
        res["rows"][key] = {"model": d["model"], "kind": key.split("_", 1)[1], **info, **v}
        if tag:
            (OUTD / "latency_partial.json").write_text(json.dumps(res, indent=1, default=str))
        print(f"[lat-mz] {key}: {v['latency_status']}", flush=True)
    if tag:
        LM._smi(f"{tag}_end")
    res["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    res["all_final"] = all(x["latency_status"] == "final" for x in res["rows"].values())
    return res


def measure() -> None:
    from scahi import ext_latency as EL
    from scahi.machine import require_default_power

    OUTD.mkdir(parents=True, exist_ok=True)
    par = json.loads((OUTD / "parity.json").read_text())
    require_default_power("latency_multizoom_host_b")
    EL._resume_cpu_queue = lambda: (EL.QDIR / "STOP_CPU").unlink(missing_ok=True)  # no CPU queue to restart here
    EL._pause_cpu_queue()
    try:
        run(3, None)  # smoke
        print("[lat-mz] smoke passed", flush=True)
        runs = []
        for k in range(1, MAX_RUNS + 1):
            res = {**run(None, f"run{k}"), "run": k, "parity": par}
            runs.append(res)
            if res["all_final"]:
                break
            print(f"[lat-mz] run {k}: not every row final, measuring the whole run again", flush=True)
        out = {"runs": runs, "used_run": runs[-1]["run"] if runs[-1]["all_final"] else None}
        (OUTD / "latency.json").write_text(json.dumps(out, indent=1, default=str))
        (OUTD / "latency_partial.json").unlink(missing_ok=True)
        print(f"[lat-mz] done, used run: {out['used_run']}", flush=True)
    finally:
        EL._resume_cpu_queue()


def report() -> None:
    lat = json.loads((OUTD / "latency.json").read_text())
    if lat["used_run"] is None:
        sys.exit("no run with every row final - table not written")
    res = next(r for r in lat["runs"] if r["run"] == lat["used_run"])
    R = res["rows"]
    f = lambda v: "–" if v is None or pd.isna(v) else f"{v:.1f}"  # noqa: E731
    m = lambda x, k: (x.get(k) or {}).get("mean") if x else None  # noqa: E731
    recs = []
    for d in _setup():
        mm, sz = d["model"], KD.NAME[d["model"]]
        sp, ti, z = R[f"{mm}_single"], R[f"{mm}_tiles_opt"], R[f"{mm}_mz"]
        recs += [
            {"detector": sz, "method": "single pass 2048", "images": sp["n_images"], "native_ms": None, "pass_2048_ms": None,
             "passes_ms": None, "fusion_ms": None, "total_ms": m(sp, "total_ms"), "peak_mem_gb": sp["peak_mem_mb_row"] / 1024},
            {"detector": sz, "method": "2048 + tiles, SCAHI refined grid (GPU, optimised path)", "images": ti["n_images"],
             "native_ms": None, "pass_2048_ms": None, "passes_ms": m(ti, "passes_ms"), "fusion_ms": m(ti, "scahi_gpu_ms"),
             "total_ms": m(ti, "total_ms"), "peak_mem_gb": ti["peak_mem_mb_row"] / 1024},
            {"detector": sz, "method": "native + 2048 pass, SCAHI refined grid (GPU, optimised path)", "images": z["n_images"],
             "native_ms": m(z, "native_pass_ms"), "pass_2048_ms": m(z, "pass_2048_ms"),
             "passes_ms": m(z, "native_pass_ms") + m(z, "pass_2048_ms"), "fusion_ms": m(z, "scahi_gpu_ms"),
             "total_ms": m(z, "total_ms"), "peak_mem_gb": z["peak_mem_mb"]["max"] / 1024,
             "peak_mem_gb_median": z["peak_mem_mb"]["median"] / 1024}]
    df = pd.DataFrame(recs)
    out = REPO / "reports/dota/tab_dota_latency_multizoom"
    df.to_csv(out.with_suffix(".csv"), index=False, float_format="%.2f")
    par = res["parity"]["rows"]
    head = (f"One exclusive run on {res['machine'].get('name', res['host'])}, {res['started']} – {res['finished']} (run "
            f"{res['run']} of {len(lat['runs'])}); every row final: {res['all_final']}; nvidia-smi lists: "
            f"runs/latency_multizoom_host_b/nvidia_smi_run{res['run']}_{{start,end}}.txt. ms per image, mean over the images "
            "where the native pass fits (subset F of the native-pass analysis), batch 1, RTX 5090 at 575 W, same protocol as "
            "the DOTA latency table. Peak memory: torch peak allocated (max over the images of the row; multi-zoom also the "
            "median). Parity of the timed multi-zoom path with the accuracy analysis (caches + SCAHI on the CPU), one-to-one "
            "matched: " + "; ".join(f"{k}: {v['n_opt']} / {v['n_ref']} detections, unmatched {v['unmatched']}, pairs beyond "
                                    f"tolerance {v['pairs_beyond_tol']}, max |Δbox| {v['max_box_px']:.2g} px"
                                    for k, v in par.items()) + ".")
    L = ["# DOTA-val latency of the multi-zoom arm (exploratory, post hoc)", "", head, "",
         "| detector | method | images | native pass | 2048 pass | passes | fusion | total | peak GPU memory (GB) |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    L += [f"| {r.detector} | {r.method} | {r.images} | {f(r.native_ms)} | {f(r.pass_2048_ms)} | {f(r.passes_ms)} | "
          f"{f(r.fusion_ms)} | {f(r.total_ms)} | {r.peak_mem_gb:.1f}"
          + (f" (median {r.peak_mem_gb_median:.2f})" if not pd.isna(getattr(r, "peak_mem_gb_median", np.nan)) else "")
          + " |" for r in df.itertuples()]
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"parity": parity, "measure": measure, "report": report,
     "smoke": lambda: print(json.dumps(run(3, None), indent=1, default=str)[:4000])}[sys.argv[1]]()
