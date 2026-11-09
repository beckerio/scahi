# SPDX-License-Identifier: AGPL-3.0-or-later
"""DOTA latency table measured on host-a (host-a brief 2026-11-01, task 4): one run, exclusive GPU and CPU, default
power limit, batch 1, every DOTA-val image (timing only; no detection is evaluated).

    python scripts/latency_host_a_dota.py smoke     # 3 images per row (not reported)
    python scripts/latency_host_a_dota.py measure   # the run -> runs/latency_host_a/dota_latency.json
    python scripts/latency_host_a_dota.py report    # reports/dota/tab_dota_latency_host_a.{md,csv}

Per detector at its S_max (YOLOv10s, YOLOv10m, YOLO11s 2048; YOLO12s 1024), frozen configurations of the DOTA freezes:
* single pass (row A): Ultralytics predictor (fused, the row's head), as the frozen DOTA-val path;
* SAHI (1024 weights, rows SAHI_<m>, YOLOv10s / YOLOv10m only): library slices + standard pass with the identity
  post-process (= the frozen raw path), then the frozen merge on that image; passes and merge timed separately;
* tiles: the passes of the stack (whole image at S + native 1024 tiles, reference path: unfused model, outputs to the
  CPU) once per image, then on the same pass outputs (a) box fusion (row NMS-o2m: per-pass NMS + cross-pass NMS), (b)
  SCAHI refined grid (row RG) on the CPU (reference) and (c) the same on the GPU (blocked ``scalespace_torch``, incl.
  host->GPU copy; parity: runs/gpu_fusion/parity_dota.json). Totals = passes + merge step. Peak GPU memory of the GPU
  fusion per image (max over images).
All values measured on host-a in this one run; never combined with the host-b tables.
"""

from __future__ import annotations

import datetime as dt
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUTD = REPO / "runs/latency_host_a"
NAME = {"yolov10s": "YOLOv10s", "yolov10m": "YOLOv10m", "yolo11s": "YOLOv11s", "yolo12s": "YOLOv12s"}


def setup() -> list[dict[str, Any]]:
    from scahi.dota import final as F
    from scahi.dota import train as DT

    out = []
    for fz in (F.frozen(), F.frozen(F.FROZEN_NEW)):
        fr = {r["id"]: r for r in fz["rows"]}
        for m in (x for x in DT.ALL_MODELS if f"RG_{x}_{DT.S_MAX[x]}" in fr):
            s = DT.S_MAX[m]
            out.append({"model": m, "s": s, "limit": fz["pipeline_limit"], "A": fr[f"A_{m}_{s}"],
                        "SAHI": fr.get(f"SAHI_{m}"), "NMS": fr[f"NMS-o2m_{m}_{s}"], "RG": fr[f"RG_{m}_{s}"]})
    return out


def _stats(v: list[float]) -> dict[str, float]:
    a = np.asarray(v, float)
    return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95))}


def single(d: dict, imgs: pd.DataFrame) -> dict:
    from scahi.dota import train as DT
    from scahi.machine import check_exclusive

    r = d["A"]
    check_exclusive()
    info = DT.predict(DT.weights(d["model"], d["s"]), "val", list(imgs.stem), r["imgsz"], r["branch"], conf=r["conf"],
                      max_det=d["limit"], measure_latency=True)[1]
    check_exclusive()
    return {"total_ms": {"mean": info["latency_ms_mean"], "p95": info["latency_ms_p95"]}, "n_images": len(imgs),
            "peak_mem_mb_predictor": info.get("peak_mem_mb")}


def sahi(d: dict, imgs: pd.DataFrame, n_warmup: int = 2) -> dict:
    import torch
    from sahi.predict import get_sliced_prediction

    from scahi.dota import pipeline as P
    from scahi.dota import train as DT
    from scahi.infer import sahi_baseline as sb
    from scahi.machine import check_exclusive

    c = d["SAHI"]["cfg"]
    sb._register_identity()
    dm = sb.load_model(DT.weights(d["model"], 1024), min(sb.GRID2_WBF_SKIP), "o2o")
    dm.model.overrides["max_det"] = d["limit"]

    def one(stem: str, path: str) -> tuple[float, float]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = get_sliced_prediction(path, dm, slice_height=sb.SLICE, slice_width=sb.SLICE, overlap_height_ratio=c["overlap"],
                                  overlap_width_ratio=c["overlap"], perform_standard_pred=True, postprocess_type="IDENTITY",
                                  postprocess_match_threshold=0.5, postprocess_match_metric="IOS",
                                  force_postprocess_type=True, verbose=0)
        raw = {stem: np.array([[*p.bbox.to_xyxy(), p.score.value, p.category.id] for p in r.object_prediction_list],
                              np.float64).reshape(-1, 6)}
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        P.sahi_merge(raw, c["merge"], c["metric"], c["match_thr"], c["conf"])
        return 1000 * (t1 - t0), 1000 * (time.perf_counter() - t1)

    for st, pa in zip(imgs.stem[:n_warmup], imgs.path[:n_warmup]):
        one(st, pa)
    ts = []
    for n, (st, pa) in enumerate(zip(imgs.stem, imgs.path)):
        if n % 100 == 0:
            check_exclusive()
        ts.append(one(st, pa))
    check_exclusive()
    t = np.array(ts)
    del dm
    torch.cuda.empty_cache()
    return {"passes_ms": _stats(t[:, 0]), "merge_ms": _stats(t[:, 1]), "total_ms": _stats(t.sum(1)), "n_images": len(t),
            "merge": f"{c['merge']} ({c['metric']} {c['match_thr']}, conf {c['conf']}), overlap {c['overlap']}"}


def tiles(d: dict, imgs: pd.DataFrame, n_warmup: int = 2) -> dict:
    import cv2
    import torch
    from sahi.postprocess.backends import set_postprocess_backend

    from scahi.dota import data as D
    from scahi.dota import pipeline as P
    from scahi.dota import train as DT
    from scahi.fusion import run as R
    from scahi.fusion import scalespace as ss
    from scahi.fusion.scalespace_torch import fuse_image_f2
    from scahi.infer import cache_mz as cm
    from scahi.infer.raw_head import load_unfused
    from scahi.machine import check_exclusive

    set_postprocess_backend(P.SAHI_BACKEND)
    model = load_unfused(DT.weights(d["model"], d["s"]))
    stack = P.STACK[d["s"]]
    nms, rg = d["NMS"], d["RG"]
    prm = ss.FusionParams(**{**rg["cfg"], "top_k": d["limit"]})
    nc = len(D.CLASSES)

    def one(path: str) -> tuple[float, ...]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = cv2.imread(path)
        H, W = img.shape[:2]
        lv = {}
        for k in cm.STACK_LEVELS[stack]:
            z = cm.raw_predict_level(model, img, cm.LEVELS[k], P.OVERLAP)
            lv[k] = {br: ss.CachedLevel(k, z[f"{br}_xyxy"], z[f"{br}_probs"], z[f"{br}_std"], z[f"{br}_tile"], z["tiles"],
                                        z["gains"], cm.LEVELS[k].n_input is None, P.OVERLAP, ss.LevelCal())
                     for br in {nms["branch"], rg["branch"]}}
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        R.f0_image([v[nms["branch"]] for v in lv.values()], W, H, nms["branch"], nms["variant"], nms["cfg"],
                   max_det=d["limit"], nms_kw=P.NMS_KW)
        t2 = time.perf_counter()
        levels = [v[rg["branch"]] for v in lv.values()]
        ss.fuse_image(levels, W, H, "F2", prm, nc=nc)
        t3 = time.perf_counter()
        st: dict = {}
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        fuse_image_f2(levels, W, H, prm, nc=nc, stats=st)
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        return (1000 * (t1 - t0), 1000 * (t2 - t1), 1000 * (t3 - t2), 1000 * (t5 - t4), st.get("peak_mem_bytes", 0) / 2**20)

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
    return {"passes_ms": _stats(t[:, 0]), "box_fusion_ms": _stats(t[:, 1]), "scahi_cpu_ms": _stats(t[:, 2]),
            "scahi_gpu_ms": _stats(t[:, 3]), "total_box_fusion_ms": _stats(t[:, 0] + t[:, 1]),
            "total_scahi_cpu_ms": _stats(t[:, 0] + t[:, 2]), "total_scahi_gpu_ms": _stats(t[:, 0] + t[:, 3]),
            "scahi_gpu_peak_mem_mb_max": float(t[:, 4].max()), "scahi_gpu_peak_mem_mb_median": float(np.median(t[:, 4])),
            "n_images": len(t), "path": "reference passes (unfused model, both heads, outputs to CPU)"}


def run(n: int | None, out: Path | None) -> dict:
    from scahi.dota import pipeline as P
    from scahi.machine import CPU_SHARING, GPU_SHARING, latency_validity, machine_info, require_default_power

    require_default_power("latency_host_a_dota")
    imgs = P.images("val")
    if n:
        imgs = imgs.head(n)
    res = {"started": dt.datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
           "machine": machine_info(), "rows": {}}
    for d in setup():
        jobs = [("single", single), ("tiles", tiles)] + ([("sahi", sahi)] if d["SAHI"] else [])
        for kind, fn in jobs:
            key = f"{d['model']}_{kind}"
            for attempt in range(1, 4):
                GPU_SHARING.update(checks=0, other_pids=[])
                CPU_SHARING.update(checks=0, cpu_queue_busy=False, other_load_cores=0.0)
                info = fn(d, imgs)
                v = latency_validity()
                if v["latency_status"] == "final" or attempt == 3 or n:
                    break
                print(f"[lat] {key}: attempt {attempt} not exclusive ({v}), repeating", flush=True)
            res["rows"][key] = {"model": d["model"], "s": d["s"], "kind": kind,
                                "rows": {k: d[k]["id"] for k in ("A", "SAHI", "NMS", "RG") if d[k]}, **info, **v,
                                "latency_attempts": attempt}
            if out:
                out.write_text(json.dumps(res, indent=1, default=str))
            print(f"[lat] {key}: {json.dumps({k: v for k, v in info.items() if k.endswith('_ms')})} "
                  f"({v['latency_status']})", flush=True)
    res["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    res["all_final"] = all(x["latency_status"] == "final" for x in res["rows"].values())
    return res


def smoke() -> None:
    run(3, None)
    print("[lat] smoke passed", flush=True)


def measure() -> None:
    OUTD.mkdir(parents=True, exist_ok=True)
    part = OUTD / "dota_latency_partial.json"
    if part.exists():
        part.rename(OUTD / f"dota_latency_aborted_{dt.datetime.now():%Y%m%dT%H%M}.json")
    smoke()
    res = run(None, part)
    part.write_text(json.dumps(res, indent=1, default=str))
    part.rename(OUTD / "dota_latency.json")
    print(f"[lat] done, all final: {res['all_final']}", flush=True)


def report() -> None:
    res = json.loads((OUTD / "dota_latency.json").read_text())
    par = json.loads((REPO / "runs/gpu_fusion/parity_dota.json").read_text())
    bad = {k: x["latency_status"] for k, x in res["rows"].items() if x["latency_status"] != "final"}
    if bad:  # never a table from non-exclusive measurements, never mixed (user 2026-11-01)
        raise SystemExit(f"not all rows final, no table written: {bad}")
    gpu_ok = bool(par["passed"])  # GPU fusion times only with a passed parity gate
    recs = []
    for x in res["rows"].values():
        det, s = NAME[x["model"]], x["s"]
        base = {"detector": det, "s": s, "latency_status": x["latency_status"]}
        if x["kind"] == "single":
            recs.append({**base, "method": f"single pass {s}", "row_id": x["rows"]["A"], "passes_ms": x["total_ms"]["mean"],
                         "merge_ms": None, "total_ms": x["total_ms"]["mean"], "total_p95_ms": x["total_ms"]["p95"],
                         "fusion_peak_gpu_mem_mb": None})
        elif x["kind"] == "sahi":
            recs.append({**base, "method": "SAHI (1024 weights)", "row_id": x["rows"]["SAHI"],
                         "passes_ms": x["passes_ms"]["mean"], "merge_ms": x["merge_ms"]["mean"],
                         "total_ms": x["total_ms"]["mean"], "total_p95_ms": x["total_ms"]["p95"],
                         "fusion_peak_gpu_mem_mb": None})
        else:
            p = x["passes_ms"]["mean"]
            recs.append({**base, "method": f"{s} + tiles, box fusion (NMS)", "row_id": x["rows"]["NMS"], "passes_ms": p,
                         "merge_ms": x["box_fusion_ms"]["mean"], "total_ms": x["total_box_fusion_ms"]["mean"],
                         "total_p95_ms": x["total_box_fusion_ms"]["p95"], "fusion_peak_gpu_mem_mb": None})
            recs.append({**base, "method": f"{s} + tiles, SCAHI, refined grid (fusion on CPU)", "row_id": x["rows"]["RG"],
                         "passes_ms": p, "merge_ms": x["scahi_cpu_ms"]["mean"], "total_ms": x["total_scahi_cpu_ms"]["mean"],
                         "total_p95_ms": x["total_scahi_cpu_ms"]["p95"], "fusion_peak_gpu_mem_mb": None})
            if not gpu_ok:
                continue
            recs.append({**base, "method": f"{s} + tiles, SCAHI, refined grid (fusion on GPU)", "row_id": x["rows"]["RG"],
                         "passes_ms": p, "merge_ms": x["scahi_gpu_ms"]["mean"], "total_ms": x["total_scahi_gpu_ms"]["mean"],
                         "total_p95_ms": x["total_scahi_gpu_ms"]["p95"],
                         "fusion_peak_gpu_mem_mb": x["scahi_gpu_peak_mem_mb_max"]})
    df = pd.DataFrame(recs)
    df.insert(0, "measured_on", "host-a")
    out = REPO / "reports/dota/tab_dota_latency_host_a"
    df.to_csv(out.with_suffix(".csv"), index=False, float_format="%.2f")
    f = lambda v: "–" if v is None or pd.isna(v) else f"{v:.1f}"  # noqa: E731
    m = res["machine"]
    gpu_rows = [k for k, v in par["rows"].items() if k.startswith("RG_")]
    L = ["# DOTA-val latency – measured on host-a", "",
         "**Superseded (2026-11-09):** all DOTA latencies of the paper come from the single host-b run (`reports/dota/tab_dota_latency_host_b.md`, `reports/dota/tab_dota_latency_host_b_dh5.md`); this host-a table is kept for comparison only.", "",
         f"ms per image (mean over {next(iter(res['rows'].values()))['n_images']} DOTA-val images; p95 in the CSV); one run on "
         f"host-a ({m.get('gpu')}, driver {m.get('driver')}, default power limit), {res['started']} – {res['finished']}; "
         f"GPU and CPU exclusive, batch 1; all rows final: {res['all_final']}. **Not comparable with the host-b tables "
         "(`reports/tab_latency_breakdown.md`); never mix the two.** For the paper, DOTA latencies come only from this "
         "table and VisDrone latencies only from the host-b table; both machines have the same GPU model (RTX 5090, "
         "32 GB) at the same default power limit (575 W).", "",
         "**How to read the merge columns.** Box fusion (SAHI's NMS, numba backend) runs on the CPU; SCAHI in the GPU rows "
         "runs on the GPU. The table is therefore **not** a speed comparison of SCAHI against box fusion (box fusion on "
         "the GPU would also be faster). The valid statement: the tile passes dominate the cost (91–94 % of the total "
         "with GPU fusion) and SCAHI on the GPU adds about 12 ms per image.", "",
         "Passes of the tiled rows (whole image at S + native 1024 tiles, overlap 0.3) are measured once per image in the "
         "reference path (unfused model, outputs to the CPU); box fusion and SCAHI then run on the same pass outputs, so "
         "the pass time is shared by every tiled method. Merge step = box fusion (per-pass NMS + cross-pass NMS on the "
         "o2m head, frozen NMS-o2m row) or SCAHI refined grid (frozen RG row) on the CPU (reference) or the GPU (blocked "
         "implementation, incl. host→GPU copy). GPU fusion parity: "
         f"{sum(par['rows'][k]['passed'] for k in gpu_rows)}/{len(gpu_rows)} RG rows identical on every DOTA-val image "
         "(`runs/gpu_fusion/parity_dota.json`). Single pass: Ultralytics predictor (fused, the row's head). SAHI: library "
         "slices (1024 weights) + standard pass, then the frozen merge; rows exist for YOLOv10s / YOLOv10m only. YOLOv12s at "
         "1024 px (largest trainable size).", ""]
    setup = OUTD / "setup.json"
    if setup.exists():
        su = json.loads(setup.read_text())
        L += ["**Measurement setup.** " + "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in su.items()) + ".", ""]
    L += [
         "| detector | method | passes | merge step | total | peak GPU mem. fusion (MB) |", "|---|---|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        L.append(f"| {r.detector} | {r.method} | {f(r.passes_ms)} | {f(r.merge_ms)} | {f(r.total_ms)} | "
                 f"{f(r.fusion_peak_gpu_mem_mb)} |")
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"smoke": smoke, "measure": measure, "report": report}[sys.argv[1]]()
