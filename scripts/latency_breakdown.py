# SPDX-License-Identifier: AGPL-3.0-or-later
"""Latency breakdown for the supplement (user 2026-10-31; one exclusive run on host-b, same procedure as E2).

    python scripts/latency_breakdown.py parity    # GPU vs. CPU fusion on every val image of the dense rows (cached levels)
    python scripts/latency_breakdown.py smoke     # functional check, 3 images per row, no timing value
    python scripts/latency_breakdown.py measure   # exclusive run (CPU queue paused, 575 W, smoke first, <= 3 attempts)
    python scripts/latency_breakdown.py report    # table -> reports/tab_latency_breakdown.{csv,md}

Rows: VisDrone val (548 images) YOLOv10s single pass 2048 (R2), single pass 2560 (R3), tiles + SCAHI pre-specified (X2)
and refined grid (XG_yolov10s_2048_o2m_C); DOTA-val (458 images) single pass 2048 (A) and tiles + SCAHI refined (RG),
YOLOv10s and YOLOv10m (the D-H3 rows). Single passes: Ultralytics predictor (fused weights, one head; passes = total, no
fusion). Dense rows, per image on the same pass outputs: (1) image decoding + all passes of the stack (reference path
``cache_mz.raw_predict_level``: unfused model, DFL decoding, anchors >= 0.001 to the CPU), (2) SCAHI on the CPU
(``scalespace.fuse_image``, reference), (3) SCAHI on the GPU (``scalespace_torch.fuse_image_f2``, incl. the host->GPU
copy of the pass outputs). Totals per image: (1) + (2) and (1) + (3). GPU timings only for rows whose parity check
(same detections as the CPU fusion on every val image, |Δ| <= 1e-4 px / score, same classes) passed.
Output: runs/latency_breakdown/{parity,latency}.json.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUTD = REPO / "runs/latency_breakdown"
TOL = 1e-4
VD = ("R2", "E_yolov10s_full2560_o2o", "X2", "XG_yolov10s_2048_o2m_C")
TERM = {"R2": "single pass 2048", "E_yolov10s_full2560_o2o": "single pass 2560", "X2": "tiles + SCAHI (pre-specified)",
        "XG_yolov10s_2048_o2m_C": "tiles + SCAHI, refined grid", "A": "single pass 2048", "RG": "tiles + SCAHI, refined grid"}


ANCHORS = ("A_yolov10s_2048", "RG_yolov10s_2048")  # measured again in the supplementary run (drift check)
NAME = {"yolov10s": "YOLOv10s", "yolov10m": "YOLOv10m", "yolo11s": "YOLOv11s", "yolo12s": "YOLOv12s"}


def rows(new: bool = False) -> list[dict[str, Any]]:
    """Every row with what it needs: dataset, kind, weights, images, stack / cfg / nc / top_k for dense rows.
    ``new``: supplementary run (addendum 2) - the D-H3 rows of YOLO11s / YOLO12s at S_max plus the anchor rows."""
    from scahi import ext as E
    from scahi import ext_latency as EL
    from scahi.dota import data as D
    from scahi.dota import final as F
    from scahi.dota import pipeline as P
    from scahi.dota import train as DT

    EL._single_levels()
    _, vimg = E._split_images("val")
    er = {r["id"]: r for r in EL.rows()}
    out = []
    for i in ([] if new else VD):
        r = er[i]
        base = {"id": i, "dataset": "VisDrone val", "detector": "YOLOv10s", "method": TERM[i], "weights": r["weights"],
                "images": vimg}
        if r["kind"] == "single":
            out.append({**base, "kind": "single", "imgsz": r["imgsz"], "branch": r["branch"], "conf": r["conf"]})
        else:
            out.append({**base, "kind": "dense", "stack": r["stack"], "branch": r["branch"], "overlap": r["overlap"],
                        "cfg": dict(r["cfg"]), "nc": 10, "split_dirs": None})
    fz = F.frozen()
    fr = {r["id"]: r for r in fz["rows"]}
    sets = [(fz, m, 2048, False) for m in DT.MODELS]
    if new:
        fn = F.frozen(F.FROZEN_NEW)
        sets = [(fz, "yolov10s", 2048, True)] + [(fn, m, DT.S_MAX[m], False) for m in DT.NEW_MODELS]
    dimg = P.images("val")
    for z, m, s, anchor in sets:
        fr = {r["id"]: r for r in z["rows"]}
        base = {"dataset": "DOTA-val", "detector": NAME[m], "weights": str(DT.weights(m, s)), "images": dimg,
                "anchor": anchor}
        a = fr[f"A_{m}_{s}"]
        out.append({**base, "id": a["id"], "model": m, "method": f"single pass {s}", "kind": "dota_single",
                    "imgsz": s, "conf": a["conf"], "max_det": z["pipeline_limit"]})
        rg = fr[f"RG_{m}_{s}"]
        out.append({**base, "id": rg["id"], "method": f"tiles + SCAHI, refined grid ({s})", "kind": "dense",
                    "stack": P.STACK[s], "branch": "o2m", "overlap": P.OVERLAP,
                    "cfg": {**rg["cfg"], "top_k": z["pipeline_limit"]}, "nc": len(D.CLASSES)})
    return out


def _cache_levels(r: dict, stem: str):
    from scahi.fusion import scalespace as ss
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm

    dirs = [(k, cm.cache_dir(CACHE_ROOT, Path(r["weights"]), k, r["overlap"], "val")) for k in cm.STACK_LEVELS[r["stack"]]]
    return ss.load_levels(dirs, stem, r["branch"], r["overlap"])


def _compare(a: pd.DataFrame, b: pd.DataFrame, res: dict) -> None:
    if len(a) != len(b):
        res["n_mismatch"] += 1
        return
    if not len(a):
        return
    ka = a.sort_values(["score", "x", "y"], ascending=[False, True, True]).to_numpy(float)
    kb = b.sort_values(["score", "x", "y"], ascending=[False, True, True]).to_numpy(float)
    res["max_box"] = max(res["max_box"], float(np.abs(ka[:, :4] - kb[:, :4]).max()))
    res["max_score"] = max(res["max_score"], float(np.abs(ka[:, 4] - kb[:, 4]).max()))
    res["class_mismatch"] += int((ka[:, 5] != kb[:, 5]).sum())
    nd = int((np.abs(ka[:, :5] - kb[:, :5]).max(1) > TOL).sum())
    res["n_det_diff"] += nd
    res["n_img_diff"] += int(nd > 0)


def parity() -> dict:
    """CPU vs. GPU fusion on the cached levels of every val image of each dense row (no timing)."""
    from scahi.fusion import scalespace as ss
    from scahi.fusion.scalespace_torch import fuse_image_f2

    out = {}
    for r in rows():
        if r["kind"] != "dense":
            continue
        p = ss.FusionParams(**r["cfg"])
        res = {"n_mismatch": 0, "max_box": 0.0, "max_score": 0.0, "class_mismatch": 0, "n_dets": 0, "n_det_diff": 0,
               "n_img_diff": 0}
        import torch

        res["available"] = True
        for st in r["images"].stem:
            lv, W, H = _cache_levels(r, st)
            a = ss.fuse_image(lv, W, H, "F2", p, nc=r["nc"])
            try:
                b = fuse_image_f2(lv, W, H, p, nc=r["nc"])
            except torch.OutOfMemoryError as e:  # dense peak x prediction membership matrix (DOTA: top_k 200 000)
                res.update(available=False, reason=f"GPU out of memory on {st}: {str(e).splitlines()[0]}")
                torch.cuda.empty_cache()
                break
            res["n_dets"] += len(a)
            _compare(a, b, res)
        res.update(images=len(r["images"]), passed=res["available"] and res["n_mismatch"] == 0 and res["max_box"] <= TOL
                   and res["max_score"] <= TOL and res["class_mismatch"] == 0)
        out[r["id"]] = res
        print("[parity]", r["id"], res, flush=True)
    out["tolerance"] = TOL
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "parity.json").write_text(json.dumps(out, indent=1))
    return out


def _dense(r: dict, imgs: pd.DataFrame, gpu: bool, n_warmup: int = 3) -> dict:
    import cv2
    import torch

    from scahi.fusion import scalespace as ss
    from scahi.fusion.scalespace_torch import fuse_image_f2
    from scahi.infer import cache_mz as cm
    from scahi.infer.raw_head import load_unfused
    from scahi.machine import check_exclusive

    model = load_unfused(r["weights"])
    p = ss.FusionParams(**r["cfg"])
    br = r["branch"]

    def one(path: str) -> tuple[float, float, float]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = cv2.imread(path)
        H, W = img.shape[:2]
        lv = []
        for k in cm.STACK_LEVELS[r["stack"]]:
            z = cm.raw_predict_level(model, img, cm.LEVELS[k], r["overlap"])
            lv.append(ss.CachedLevel(k, z[f"{br}_xyxy"], z[f"{br}_probs"], z[f"{br}_std"], z[f"{br}_tile"], z["tiles"],
                                     z["gains"], cm.LEVELS[k].n_input is None, r["overlap"], ss.LevelCal()))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ss.fuse_image(lv, W, H, "F2", p, nc=r["nc"])
        t2 = time.perf_counter()
        tg = np.nan
        if gpu:
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            fuse_image_f2(lv, W, H, p, nc=r["nc"])
            torch.cuda.synchronize()
            tg = 1000 * (time.perf_counter() - t3)
        return 1000 * (t1 - t0), 1000 * (t2 - t1), tg

    for path in imgs.path[:n_warmup]:
        one(path)
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for n, path in enumerate(imgs.path):
        if n % 100 == 0:
            check_exclusive()
        ts.append(one(path))
    check_exclusive()
    t = np.array(ts)
    mem = torch.cuda.max_memory_allocated() / 2**20
    del model
    torch.cuda.empty_cache()
    pc = lambda v: float(np.percentile(v, 95))  # noqa: E731
    out = {"n_images": len(t), "passes_ms_mean": float(t[:, 0].mean()), "passes_ms_p95": pc(t[:, 0]),
           "fusion_cpu_ms_mean": float(t[:, 1].mean()), "fusion_cpu_ms_p95": pc(t[:, 1]),
           "total_cpu_ms_mean": float((t[:, 0] + t[:, 1]).mean()), "total_cpu_ms_p95": pc(t[:, 0] + t[:, 1]),
           "peak_mem_mb": mem, "path": "reference passes (unfused model, both heads, outputs to CPU)"}
    if gpu:
        out |= {"fusion_gpu_ms_mean": float(t[:, 2].mean()), "fusion_gpu_ms_p95": pc(t[:, 2]),
                "total_gpu_ms_mean": float((t[:, 0] + t[:, 2]).mean()), "total_gpu_ms_p95": pc(t[:, 0] + t[:, 2])}
    out["latency_ms_mean"] = out["total_cpu_ms_mean"]
    return out


def _fn(r: dict, imgs: pd.DataFrame, gpu_ok: dict[str, bool]):
    if r["kind"] == "single":
        from scahi.infer.full import predict_full

        return lambda: predict_full(r["weights"], imgs, r["imgsz"], conf=r["conf"], measure_latency=True,
                                    branch=r["branch"])[1]
    if r["kind"] == "dota_single":
        from scahi.dota import train as DT

        from scahi.machine import check_exclusive

        def run() -> dict:  # exclusivity checked around the predictor loop (as scahi.dota.final.val_latency)
            check_exclusive()
            info = DT.predict(r["weights"], "val", list(imgs.stem), r["imgsz"], DT.heads(r["model"])[0], conf=r["conf"],
                              max_det=r["max_det"], measure_latency=True)[1]
            check_exclusive()
            return info

        return run
    return lambda: _dense(r, imgs, gpu_ok.get(r["id"], False))


def smoke() -> None:
    par = json.loads((OUTD / "parity.json").read_text())
    gpu_ok = {k: v["available"] for k, v in par.items() if isinstance(v, dict)}
    for r in rows():
        print("[smoke]", r["id"], {k: v for k, v in _fn(r, r["images"].head(3), gpu_ok)().items()
                                   if k.endswith("ms_mean")}, flush=True)


def measure(new: bool = False) -> None:
    from scahi import ext_latency as EL
    from scahi.machine import CPU_SHARING, GPU_SHARING, latency_validity, machine_info, require_default_power

    par = json.loads((OUTD / "parity.json").read_text())
    gpu_ok = {k: v["available"] for k, v in par.items() if isinstance(v, dict)}  # timed where it runs; parity reported
    rs = rows(new)
    name = "latency_new" if new else "latency"
    require_default_power("latency_breakdown")
    part = OUTD / f"{name}_partial.json"
    if part.exists():  # an interrupted earlier run: never mixed with this one
        part.rename(OUTD / f"{name}_aborted_{dt.datetime.now():%Y%m%dT%H%M}.json")
    EL._resume_cpu_queue = lambda: (EL.QDIR / "STOP_CPU").unlink(missing_ok=True)  # no CPU queue to restart here
    EL._pause_cpu_queue()
    try:
        for r in rs:  # smoke first
            _fn(r, r["images"].head(3), gpu_ok)()
        print("[lat] smoke passed for all rows", flush=True)
        res = {"started": dt.datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
               "machine": machine_info(), "gpu_fusion_parity": gpu_ok, "rows": {}}
        for r in rs:
            meta = {k: v for k, v in r.items() if k != "images"}
            for attempt in range(1, 4):
                GPU_SHARING.update(checks=0, other_pids=[])
                CPU_SHARING.update(checks=0, cpu_queue_busy=False, other_load_cores=0.0)
                info = _fn(r, r["images"], gpu_ok)()
                v = latency_validity()
                if v["latency_status"] == "final" or attempt == 3:
                    break
                print(f"[lat] {r['id']}: attempt {attempt} not exclusive ({v}), repeating", flush=True)
            res["rows"][r["id"]] = {"meta": meta, **info, **v, "latency_attempts": attempt}
            part.write_text(json.dumps(res, indent=1, default=str))
            print(f"[lat] {r['id']}: {info['latency_ms_mean']:.1f} ms ({v['latency_status']})", flush=True)
        res["finished"] = dt.datetime.now().isoformat(timespec="seconds")
        res["all_final"] = all(x["latency_status"] == "final" for x in res["rows"].values())
        part.write_text(json.dumps(res, indent=1, default=str))
        part.rename(OUTD / f"{name}.json")
        print(f"[lat] done, all final: {res['all_final']}", flush=True)
    finally:
        EL._resume_cpu_queue()


def _parity_note(i: str, v: dict) -> str:
    if not v["available"]:
        return f"{i}: GPU fusion not available (out of GPU memory: dense peak × prediction membership matrix)"
    if v["passed"]:
        return f"{i}: identical ({v['n_dets']} detections)"
    return (f"{i}: {v['n_det_diff']} of {v['n_dets']} detections differ in {v['n_img_diff']} image(s) (box up to "
            f"{v['max_box']:.2f} px, same score and class)")


def _new_note(res: dict, new: dict) -> str:
    """Supplementary run (addendum 2): new D-H3 rows; GPU fusion not used on DOTA; anchor drift vs. the main run."""
    dr = [f"{i} {x['latency_ms_mean']:.1f} ms (main run {res['rows'][i]['latency_ms_mean']:.1f})"
          for i, x in new["rows"].items() if x["meta"].get("anchor")]
    return (f" YOLOv11s / YOLOv12s rows (DOTA addendum 2, at S_max): supplementary run on {new['machine'].get('name')}, "
            f"{new['started']} – {new['finished']}, same procedure, all rows final: {new['all_final']}; anchors measured "
            f"again in that run: {'; '.join(dr)}. GPU fusion not used on DOTA (see parity).")


def display(name: str) -> str:
    """Displayed model name (paper decision 2026-11-07): YOLOv11 / YOLOv12; stored data keep Ultralytics' names."""
    return re.sub(r"\bYOLO(1[12])", r"YOLOv\1", name)


def report() -> None:
    res = json.loads((OUTD / "latency.json").read_text())
    par = json.loads((OUTD / "parity.json").read_text())
    new = json.loads((OUTD / "latency_new.json").read_text()) if (OUTD / "latency_new.json").exists() else None
    recs = []
    items = [(i, x, "main") for i, x in res["rows"].items()]
    items += [(i, x, "supplementary") for i, x in (new or {"rows": {}})["rows"].items() if not x["meta"].get("anchor")]
    for i, x, run in items:
        m = x["meta"]
        single = m["kind"] != "dense"
        recs.append({"dataset": m["dataset"], "detector": display(m["detector"]), "method": m["method"], "row_id": i,
                     "passes_ms": x["latency_ms_mean"] if single else x["passes_ms_mean"],
                     "fusion_cpu_ms": None if single else x["fusion_cpu_ms_mean"],
                     "fusion_gpu_ms": None if single else x.get("fusion_gpu_ms_mean"),
                     "total_cpu_fusion_ms": x["latency_ms_mean"] if single else x["total_cpu_ms_mean"],
                     "total_gpu_fusion_ms": x["latency_ms_mean"] if single else x.get("total_gpu_ms_mean"),
                     "total_p95_ms": x["latency_ms_p95"] if single else x["total_cpu_ms_p95"],
                     "latency_status": x["latency_status"], "attempts": x["latency_attempts"], "run": run})
    df = pd.DataFrame(recs)
    out = REPO / "reports/tab_latency_breakdown"
    df.to_csv(out.with_suffix(".csv"), index=False, float_format="%.2f")
    f = lambda v: "–" if v is None or pd.isna(v) else f"{v:.1f}"  # noqa: E731
    L = ["# Latency breakdown (ms per image, mean; exclusive GPU and CPU, RTX 5090 at 575 W, batch 1)", "",
         f"One run on {res['machine'].get('name', res['host'])}, {res['started']} – {res['finished']}; all rows final: "
         f"{res['all_final']}. Single passes: Ultralytics predictor (fused, one head). Tiles + SCAHI: passes of the stack "
         "in the reference path (unfused model, outputs to the CPU), then SCAHI on the CPU (reference) or on the GPU "
         "(scalespace_torch, incl. host→GPU copy) on the same pass outputs; totals = passes + fusion. "
         "Parity of GPU vs. CPU fusion on every val image before timing (runs/latency_breakdown/parity.json): "
         + "; ".join(_parity_note(i, v) for i, v in par.items() if isinstance(v, dict)) + "."
         + (_new_note(res, new) if new else ""), "",
         "| dataset | detector | method | passes | fusion CPU | fusion GPU | total (CPU fusion) | total (GPU fusion) |",
         "|---|---|---|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        L.append(f"| {r.dataset} | {r.detector} | {r.method} | {f(r.passes_ms)} | {f(r.fusion_cpu_ms)} | "
                 f"{f(r.fusion_gpu_ms)} | {f(r.total_cpu_fusion_ms)} | {f(r.total_gpu_fusion_ms)} |")
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"parity": parity, "smoke": smoke, "measure": measure, "measure_new": lambda: measure(True),
     "report": report}[sys.argv[1]]()
