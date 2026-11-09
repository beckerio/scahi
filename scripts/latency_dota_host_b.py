# SPDX-License-Identifier: AGPL-3.0-or-later
"""All DOTA latencies on host-b in ONE exclusive run (user 2026-11-09), replacing the host-a tables, so that every latency of
the paper comes from one machine.

    python scripts/latency_dota_host_b.py parity_opt   # optimised SCAHI path vs. reference (caches + CPU) on every DOTA-val image
    python scripts/latency_dota_host_b.py measure      # the exclusive run -> runs/latency_dota_host_b/latency.json
    python scripts/latency_dota_host_b.py report       # reports/dota/tab_dota_latency_host_b{,_dh5}.{md,csv}

Rows (same protocol and code as scripts/latency_host_a_dota.py and scripts/latency_host_a_dh5.py, whose measurement
functions are reused unchanged): per detector at S_max (YOLOv10s, YOLOv10m, YOLOv11s 2048; YOLOv12s 1024) the single pass,
the tile passes (reference path) with box fusion on the CPU and SCAHI refined grid on the CPU and on the GPU, SAHI (frozen
configuration, YOLOv10s / YOLOv10m); D-H5 (YOLOv10s / YOLOv10m): SAHI passes + SCAHI pre-specified and refined, CPU and GPU.
New: the SCAHI rows of the DOTA table also in the optimised path of Table 2 (``pipeline_gpu``: unfused model, one-to-many
head only, decoding and SCAHI on the GPU), with the DOTA class count; parity against the reference path is checked on every
DOTA-val image before timing. SAHI is measured once and shown in both tables. Exclusivity: CPU queue paused and waited for,
575 W, no foreign compute process on the GPU, foreign CPU load < 1 core; every row must be "final", otherwise the whole run
is repeated (at most 3 runs); nvidia-smi process lists at the start and end of each run.
"""

from __future__ import annotations

import datetime as dt
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import latency_host_a_dh5 as KH  # noqa: E402
import latency_host_a_dota as KD  # noqa: E402

OUTD = REPO / "runs/latency_dota_host_b"
OLD_RUNS = REPO / "runs/latency_host_a"
MAX_RUNS = 3


def _opt_levels(model, img, stack: str, branch: str, nc: int):
    """``pipeline_gpu.gpu_levels`` with the DOTA class count for levels without predictions."""
    import torch

    from scahi.fusion import pipeline_gpu as PG

    levels, W, H = PG.gpu_levels(model, img, stack, KD_OVERLAP(), branch)
    for lv in levels:
        if lv.probs.shape[0] == 0 and lv.probs.shape[1] != nc:
            lv.probs = torch.zeros((0, nc), device=lv.probs.device)
    return levels, W, H


def KD_OVERLAP() -> float:  # noqa: N802
    from scahi.dota import pipeline as P

    return P.OVERLAP


def opt(d: dict, imgs: pd.DataFrame, n_warmup: int = 2) -> dict:
    """SCAHI refined grid (row RG) in the optimised path: passes (incl. image decoding) and SCAHI on the GPU."""
    import cv2
    import torch

    from scahi.dota import data as D
    from scahi.dota import pipeline as P
    from scahi.dota import train as DT
    from scahi.fusion import scalespace as ss
    from scahi.fusion.run import clip_xywh
    from scahi.fusion.scalespace_torch import fuse_image_f2
    from scahi.infer.raw_head import load_unfused
    from scahi.machine import check_exclusive

    rg, nc = d["RG"], len(D.CLASSES)
    model = load_unfused(DT.weights(d["model"], d["s"]))
    prm = ss.FusionParams(**{**rg["cfg"], "top_k": d["limit"]})
    stack = P.STACK[d["s"]]

    def one(path: str) -> tuple[float, float]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = cv2.imread(path)
        levels, W, H = _opt_levels(model, img, stack, rg["branch"], nc)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        clip_xywh(fuse_image_f2(levels, W, H, prm, nc=nc), W, H)
        torch.cuda.synchronize()
        return 1000 * (t1 - t0), 1000 * (time.perf_counter() - t1)

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
    return {"passes_ms": KD._stats(t[:, 0]), "scahi_gpu_ms": KD._stats(t[:, 1]), "total_ms": KD._stats(t.sum(1)),
            "n_images": len(t), "path": "optimised (unfused model, one-to-many head only, decoding + SCAHI on the GPU)"}


def parity_opt() -> dict:
    """Optimised path vs. reference (stored dense-head caches + SCAHI on the CPU) on every DOTA-val image, RG rows."""
    import cv2
    import torch

    sys.path.insert(0, str(REPO / "scripts"))
    import latency_breakdown as LB

    from scahi.dota import data as D
    from scahi.dota import pipeline as P
    from scahi.dota import train as DT
    from scahi.fusion import scalespace as ss
    from scahi.fusion.run import clip_xywh
    from scahi.fusion.scalespace_torch import fuse_image_f2
    from scahi.infer.raw_head import load_unfused

    nc, out = len(D.CLASSES), {}
    imgs = P.images("val")
    for d in KD.setup():
        rg = d["RG"]
        prm = ss.FusionParams(**{**rg["cfg"], "top_k": d["limit"]})
        model = load_unfused(DT.weights(d["model"], d["s"]))
        res = {"n_mismatch": 0, "max_box": 0.0, "max_score": 0.0, "class_mismatch": 0, "n_dets": 0, "n_det_diff": 0,
               "n_img_diff": 0}
        with torch.no_grad():
            for st, pa in zip(imgs.stem, imgs.path):
                lv, W, H = P.load(d["model"], d["s"], "val", rg["branch"], "C", [st])[st]
                a = clip_xywh(ss.fuse_image(lv, W, H, "F2", prm, nc=nc), W, H)
                levels, W2, H2 = _opt_levels(model, cv2.imread(pa), P.STACK[d["s"]], rg["branch"], nc)
                b = clip_xywh(fuse_image_f2(levels, W2, H2, prm, nc=nc), W2, H2)
                res["n_dets"] += len(a)
                LB._compare(a.reset_index(drop=True), b.reset_index(drop=True), res)
        res.update(images=len(imgs), row=rg["id"], passed=res["n_mismatch"] == 0 and res["max_box"] <= LB.TOL
                   and res["max_score"] <= LB.TOL and res["class_mismatch"] == 0)
        out[rg["id"]] = res
        print("[parity-opt]", rg["id"], res, flush=True)
        del model
        torch.cuda.empty_cache()
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "parity_opt.json").write_text(json.dumps(out, indent=1))
    return out


def _smi(tag: str) -> None:
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                       capture_output=True, text=True).stdout
    full = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    (OUTD / f"nvidia_smi_{tag}.txt").write_text(f"{dt.datetime.now().isoformat(timespec='seconds')}\n\ncompute apps:\n{q}\n"
                                                f"{full}")


def _jobs() -> list[tuple[str, dict, object]]:
    jobs = []
    for d in KD.setup():
        jobs += [(f"{d['model']}_single", d, KD.single), (f"{d['model']}_tiles", d, KD.tiles),
                 (f"{d['model']}_opt", d, opt)]
        if d["SAHI"]:
            jobs.append((f"{d['model']}_sahi", d, KD.sahi))
    jobs += [(f"{d['model']}_sahi_scahi", d, KH.sahi_scahi) for d in KH.setup()]
    return jobs


def run(n: int | None, tag: str | None) -> dict:
    from scahi.dota import pipeline as P
    from scahi.machine import CPU_SHARING, GPU_SHARING, latency_validity, machine_info

    imgs = P.images("val")
    if n:
        imgs = imgs.head(n)
    if tag:
        _smi(f"{tag}_start")
    res = {"started": dt.datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
           "machine": machine_info(), "rows": {}}
    for key, d, fn in _jobs():
        GPU_SHARING.update(checks=0, other_pids=[])
        CPU_SHARING.update(checks=0, cpu_queue_busy=False, other_load_cores=0.0)
        info = fn(d, imgs)
        v = latency_validity()
        res["rows"][key] = {"model": d["model"], "kind": key.split("_", 1)[1], **info, **v}
        if tag:
            (OUTD / "latency_partial.json").write_text(json.dumps(res, indent=1, default=str))
        print(f"[lat] {key}: {v['latency_status']}", flush=True)
    if tag:
        _smi(f"{tag}_end")
    res["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    res["all_final"] = all(x["latency_status"] == "final" for x in res["rows"].values())
    return res


def measure() -> None:
    from scahi import ext_latency as EL
    from scahi.machine import require_default_power

    OUTD.mkdir(parents=True, exist_ok=True)
    par = json.loads((OUTD / "parity_opt.json").read_text())
    require_default_power("latency_dota_host_b")
    EL._resume_cpu_queue = lambda: (EL.QDIR / "STOP_CPU").unlink(missing_ok=True)  # no CPU queue to restart here
    EL._pause_cpu_queue()
    try:
        r = run(3, None)  # smoke
        bad = [k for k, x in r["rows"].items() if not all(x.get("parity_live_vs_cache_calib_3x2", [True]))]
        if bad:
            raise SystemExit(f"[lat] D-H5 live passes differ from the cached path: {bad}")
        print("[lat] smoke passed", flush=True)
        runs = []
        for k in range(1, MAX_RUNS + 1):
            res = {**run(None, f"run{k}"), "run": k, "parity_opt": par}
            runs.append(res)
            if res["all_final"]:
                break
            print(f"[lat] run {k}: not every row final, measuring the whole run again", flush=True)
        out = {"runs": runs, "used_run": runs[-1]["run"] if runs[-1]["all_final"] else None}
        (OUTD / "latency.json").write_text(json.dumps(out, indent=1, default=str))
        (OUTD / "latency_partial.json").unlink(missing_ok=True)
        print(f"[lat] done, used run: {out['used_run']}", flush=True)
    finally:
        EL._resume_cpu_queue()


def matched_parity_text(part: str) -> str:
    """Parity of the optimised GPU path, one-to-one matched per image and class (scripts/parity_opt_matched.py)."""
    p = json.loads((REPO / f"runs/parity_opt_matched/{part}.json").read_text())
    return (f"one-to-one matching per image and class, runs/parity_opt_matched/{part}.json: " + "; ".join(
        f"{k}: {v['n_opt']} of {v['n_ref']} detections matched in {v['images']} images, count mismatches "
        f"{len(v['count_mismatch_images'])}, unmatched {v['unmatched']}, pairs beyond {p['tolerance_relative']:g} relative "
        f"{v['pairs_beyond_tol']}, max |Δbox| {v['max_box_px']:.2g} px, max |Δscore| {v['max_score']:.2g}"
        for k, v in p["rows"].items()))


def _host_a() -> tuple[dict, dict]:
    return (json.loads((OLD_RUNS / "dota_latency.json").read_text())["rows"],
            json.loads((OLD_RUNS / "dh5_latency.json").read_text())["rows"])


def report() -> None:
    lat = json.loads((OUTD / "latency.json").read_text())
    if lat["used_run"] is None:
        sys.exit("no run with every row final - tables not written")
    res = next(r for r in lat["runs"] if r["run"] == lat["used_run"])
    R, (K, KH5) = res["rows"], _host_a()
    f = lambda v: "–" if v is None or pd.isna(v) else f"{v:.1f}"  # noqa: E731
    m = lambda x, k: (x.get(k) or {}).get("mean") if x else None  # noqa: E731
    recs = []
    for d in KD.setup():
        mm, s, sz = d["model"], d["s"], KD.NAME[d["model"]]
        sp, ti, op, sa = (R.get(f"{mm}_{k}") for k in ("single", "tiles", "opt", "sahi"))
        ksp, kti, ksa = (K.get(f"{mm}_{k}") for k in ("single", "tiles", "sahi"))
        rows = [("single pass", f"single pass {s}", m(sp, "total_ms"), None, m(sp, "total_ms"), m(ksp, "total_ms"), None,
                 m(ksp, "total_ms")),
                ("box fusion", f"{s} + tiles, box fusion (CPU)", m(ti, "passes_ms"), m(ti, "box_fusion_ms"),
                 m(ti, "total_box_fusion_ms"), m(kti, "passes_ms"), m(kti, "box_fusion_ms"), m(kti, "total_box_fusion_ms")),
                ("scahi cpu", f"{s} + tiles, SCAHI refined grid (CPU)", m(ti, "passes_ms"), m(ti, "scahi_cpu_ms"),
                 m(ti, "total_scahi_cpu_ms"), m(kti, "passes_ms"), m(kti, "scahi_cpu_ms"), m(kti, "total_scahi_cpu_ms")),
                ("scahi gpu", f"{s} + tiles, SCAHI refined grid (GPU, reference passes)", m(ti, "passes_ms"),
                 m(ti, "scahi_gpu_ms"), m(ti, "total_scahi_gpu_ms"), m(kti, "passes_ms"), m(kti, "scahi_gpu_ms"),
                 m(kti, "total_scahi_gpu_ms")),
                ("scahi opt", f"{s} + tiles, SCAHI refined grid (GPU, optimised path)", m(op, "passes_ms"),
                 m(op, "scahi_gpu_ms"), m(op, "total_ms"), None, None, None)]
        if sa:
            rows.append(("sahi", "SAHI (1024 weights, frozen)", m(sa, "passes_ms"), m(sa, "merge_ms"), m(sa, "total_ms"),
                         m(ksa, "passes_ms"), m(ksa, "merge_ms"), m(ksa, "total_ms")))
        for key, meth, p, g, t, kp, kg, kt in rows:
            recs.append({"table": "dota", "detector": sz, "key": key, "method": meth, "passes_ms": p, "merge_ms": g,
                         "total_ms": t, "host_a_passes_ms": kp, "host_a_merge_ms": kg, "host_a_total_ms": kt})
    for d in KH.setup():
        mm, sz = d["model"], KD.NAME[d["model"]]
        ss_, sa = R.get(f"{mm}_sahi_scahi"), R.get(f"{mm}_sahi")
        kss, ksa = KH5.get(f"{mm}_sahi_scahi"), KH5.get(f"{mm}_sahi")
        recs.append({"table": "dh5", "detector": sz, "key": "sahi", "method": "SAHI (frozen)", "passes_ms": m(sa, "passes_ms"),
                     "merge_ms": m(sa, "merge_ms"), "total_ms": m(sa, "total_ms"), "host_a_passes_ms": m(ksa, "passes_ms"),
                     "host_a_merge_ms": m(ksa, "merge_ms"), "host_a_total_ms": m(ksa, "total_ms")})
        for x in KH.ROWS:
            for dev in ("cpu", "gpu"):
                recs.append({"table": "dh5", "detector": sz, "key": f"{x} {dev}",
                             "method": f"SAHI passes + SCAHI, {x} (fusion on {dev.upper()})",
                             "passes_ms": m(ss_, "passes_ms"), "merge_ms": m(ss_[x], f"scahi_{dev}_ms"),
                             "total_ms": m(ss_[x], f"total_scahi_{dev}_ms"), "host_a_passes_ms": m(kss, "passes_ms"),
                             "host_a_merge_ms": m(kss[x], f"scahi_{dev}_ms"), "host_a_total_ms": m(kss[x], f"total_scahi_{dev}_ms")})
    df = pd.DataFrame(recs)
    head = (f"One exclusive run on {res['machine'].get('name', res['host'])}, {res['started']} – {res['finished']} (run "
            f"{res['run']} of {len(lat['runs'])}); every row final: {res['all_final']}; nvidia-smi lists: "
            f"runs/latency_dota_host_b/nvidia_smi_run{res['run']}_{{start,end}}.txt. ms per image, mean over the DOTA-val "
            "images, batch 1, RTX 5090 at 575 W. Replaces the host-a tables (columns \"host-a\" for comparison only). Same "
            "protocol and code as the host-a run; the SCAHI rows additionally in the optimised path of Table 2 (unfused model, "
            "one-to-many head only, decoding and SCAHI on the GPU); parity with the reference path (stored caches + SCAHI on "
            f"the CPU) on every DOTA-val image, {matched_parity_text('dota')}. Box differences of 2^-11 px are one float32 "
            "step at coordinates of 4096-8192 px.")
    for tab, name, title in (("dota", "tab_dota_latency_host_b", "DOTA-val latency – one run on host-b"),
                             ("dh5", "tab_dota_latency_host_b_dh5", "D-H5 latency (SAHI vs. SAHI passes + SCAHI) – same run")):
        t = df[df.table == tab].drop(columns="table")
        out = REPO / "reports/dota" / name
        t.to_csv(out.with_suffix(".csv"), index=False, float_format="%.2f")
        L = [f"# {title}", "", head, "", "| detector | method | passes | merge | total | host-a passes | host-a merge | "
             "host-a total |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        L += [f"| {r.detector} | {r.method} | {f(r.passes_ms)} | {f(r.merge_ms)} | {f(r.total_ms)} | "
              f"{f(r.host_a_passes_ms)} | {f(r.host_a_merge_ms)} | {f(r.host_a_total_ms)} |" for r in t.itertuples()]
        out.with_suffix(".md").write_text("\n".join(L) + "\n")
        print("\n".join(L))


if __name__ == "__main__":
    {"parity_opt": parity_opt, "measure": measure, "report": report,
     "smoke": lambda: print(json.dumps(run(3, None), indent=1, default=str)[:3000])}[sys.argv[1]]()
