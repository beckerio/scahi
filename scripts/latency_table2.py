# SPDX-License-Identifier: AGPL-3.0-or-later
"""Table 2 (VisDrone compute control, YOLOv10s): every latency of the table from ONE new exclusive run on host-b
(user 2026-11-08, DECISIONS 21:16; same procedure as E2 / scripts/latency_breakdown.py).

    python scripts/latency_table2.py smoke     # functional check, 3 images per row, no timing value; X1 equality
    python scripts/latency_table2.py measure   # exclusive run (575 W, CPU queue paused, smoke first, <= 3 table runs)
    python scripts/latency_table2.py parity_opt    # optimised vs. reference SCAHI path on every val image
    python scripts/latency_table2.py measure_opt   # the Table 2 run: SCAHI rows in the optimised path (user 2026-11-09)
    python scripts/latency_table2.py report    # reports/tab_compute_control.{md,csv} (paper table from measure_opt;
                                               # the reference-path run only as a report section)

Rows (VisDrone val, 548 images): single pass 2048 (R2), single pass 2560 (R3), 2560 weights at 3072 px (crossing row),
2048 + tiles box fusion (X1, frozen: NMS with intersection over the smaller box), 2048 + tiles SCAHI (X2) and SCAHI
refined grid (XG_yolov10s_2048_o2m_C). Single passes: Ultralytics predictor (fused weights, one head, batch 1, end to
end; no merge step). X1: the path of its frozen val run (``multizoom.predict_fused``), split per image into the passes
(image decoding + all passes, ``predict_raw``) and the box fusion on the CPU (``multizoom.fuse``); its detection counts
are compared with the frozen dets_val.parquet (informational: GPU inference is not bit-exact). X2 / XG:
``latency_breakdown._dense`` (passes in the reference path, SCAHI on the CPU and, on the same pass outputs, on the GPU; parity runs/latency_breakdown/parity.json).
Every row must be "final" (no foreign compute process on the GPU, foreign CPU load < 1 core); otherwise the whole table
is measured again (at most 3 runs). nvidia-smi process lists are saved at the start and end of each run.
Output: runs/latency_table2/{latency.json, nvidia_smi_<run>_{start,end}.txt}.
"""

from __future__ import annotations

import datetime as dt
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import latency_breakdown as LB  # noqa: E402

OUTD = REPO / "runs/latency_table2"
CROSS = REPO / "runs/ext/crossing/X_v10s_train2560_infer3072_o2o"
X1_RUN = REPO / "runs/p5/20261023_v10s_BFULL2048_native2048_F0-NMS_o2o"
X1_TESTDEV = REPO / "runs/testdev/20261023_v10s_BFULL2048_native2048_F0-NMS_o2o"
GFLOPS = REPO / "runs/gflops_unified.json"
METHOD = {"R2": "single pass 2048", "E_yolov10s_full2560_o2o": "single pass 2560",
          "X_v10s_train2560_infer3072_o2o": "2560 weights at 3072 px", "X1": "2048 + tiles, box fusion",
          "X2": "2048 + tiles, SCAHI", "XG_yolov10s_2048_o2m_C": "2048 + tiles, SCAHI, refined grid"}
MAX_RUNS = 3


def rows(opt: bool = False) -> list[dict[str, Any]]:
    from scahi import ext as E

    lb = {r["id"]: r for r in LB.rows()}
    _, vimg = E._split_images("val")
    fz = json.loads((REPO / "configs/final/frozen_rows.json").read_text())
    x1 = next(r for r in (fz["rows"] if isinstance(fz, dict) else fz) if r["id"] == "X1")
    cm = json.loads((CROSS / "metrics.json").read_text())
    out = [lb["R2"], lb["E_yolov10s_full2560_o2o"],
           {"id": CROSS.name, "kind": "single", "weights": str(REPO / cm["weights"]), "imgsz": cm["inference"]["imgsz"],
            "branch": cm["inference"]["branch"], "conf": cm["inference"]["conf"], "images": vimg},
           {"id": "X1", "kind": "boxfusion", "weights": str(REPO / x1["weights"]), "stack": x1["inference"]["stack"],
            "overlap": x1["inference"]["overlap"], "branch": x1["inference"]["branch"], **x1["inference"]["params"],
            "images": vimg},
           lb["X2"], lb["XG_yolov10s_2048_o2m_C"]]
    for r in out:
        r["method"] = METHOD[r["id"]]
        if opt and r["kind"] == "dense":  # user 2026-11-09: SCAHI rows in the optimised path of the E2 run
            r["kind"] = "dense_opt"
    return out


def _dense_opt(r: dict, imgs: pd.DataFrame) -> dict:
    """SCAHI in the optimised path (``pipeline_gpu.measure``, the "optimised" column of the E2 run): unfused model (fusing
    removes the one-to-many branch), only the one-to-many head, DFL decoding and SCAHI on the GPU; the dense outputs stay
    on the GPU, only the final detections go to the host."""
    from scahi.fusion import pipeline_gpu as PG

    i = PG.measure(r["weights"], imgs, r["stack"], r["overlap"], r["branch"], r["cfg"])
    return {"n_images": i["n_images"], "passes_ms_mean": i["passes_ms_mean"], "fusion_gpu_ms_mean": i["fusion_ms_mean"],
            "total_gpu_ms_mean": i["latency_ms_mean"], "total_gpu_ms_p95": i["latency_ms_p95"],
            "latency_ms_mean": i["latency_ms_mean"], "latency_ms_p95": i["latency_ms_p95"], "peak_mem_mb": i["peak_mem_mb"],
            "path": i["path"]}


def parity_opt() -> dict:
    """Optimised path vs. reference path (cached pass outputs + SCAHI on the CPU) on every val image of the SCAHI rows,
    as scripts/e2_pipeline_parity.py (which covered X2 on 25 images only)."""
    import cv2

    from scahi.fusion import pipeline_gpu as PG
    from scahi.fusion import scalespace as ss
    from scahi.fusion.run import clip_xywh
    from scahi.infer.raw_head import load_unfused

    out = {}
    for r in rows():
        if r["kind"] != "dense":
            continue
        p = ss.FusionParams(**r["cfg"])
        m = load_unfused(r["weights"])
        res = {"n_mismatch": 0, "max_box": 0.0, "max_score": 0.0, "class_mismatch": 0, "n_dets": 0, "n_det_diff": 0,
               "n_img_diff": 0}
        for st, path in zip(r["images"].stem, r["images"].path):
            lv, W, H = LB._cache_levels(r, st)
            a = clip_xywh(ss.fuse_image(lv, W, H, "F2", p, nc=r["nc"]), W, H)
            b, _, _ = PG.predict_image(m, cv2.imread(path), r["stack"], r["overlap"], r["branch"], r["cfg"])
            res["n_dets"] += len(a)
            LB._compare(a, b, res)
        res.update(images=len(r["images"]), passed=res["n_mismatch"] == 0 and res["max_box"] <= LB.TOL
                   and res["max_score"] <= LB.TOL and res["class_mismatch"] == 0)
        out[r["id"]] = res
        print("[parity-opt]", r["id"], res, flush=True)
        del m
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "parity_opt.json").write_text(json.dumps(out, indent=1))
    return out


def _boxfusion(r: dict, imgs: pd.DataFrame, n_warmup: int = 3) -> dict:
    """X1 as its frozen val run (predict_fused), timed per image: passes, then box fusion on the CPU."""
    import torch
    from ultralytics import YOLO

    from scahi.infer import multizoom as mz
    from scahi.eval.detections import finalize_detections
    from scahi.machine import check_exclusive

    model = YOLO(r["weights"])
    st = mz.STACKS[r["stack"]]

    def one(n: int) -> tuple[float, float, pd.DataFrame]:
        im = imgs.iloc[[n]]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        raw, _ = mz.predict_raw(model, im, st, r["overlap"], r["branch"], r["conf"])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        d = mz.fuse(raw, r["thr"], r["conf"], r["metric"])
        return 1000 * (t1 - t0), 1000 * (time.perf_counter() - t1), d

    for n in range(min(n_warmup, len(imgs))):
        one(n)
    torch.cuda.reset_peak_memory_stats()
    ts, dets = [], []
    for n in range(len(imgs)):
        if n % 100 == 0:
            check_exclusive()
        a, b, d = one(n)
        ts.append((a, b))
        dets.append(d)
    check_exclusive()
    t = np.array(ts)
    pc = lambda v: float(np.percentile(v, 95))  # noqa: E731
    out = {"n_images": len(t), "passes_ms_mean": float(t[:, 0].mean()), "passes_ms_p95": pc(t[:, 0]),
           "fusion_cpu_ms_mean": float(t[:, 1].mean()), "fusion_cpu_ms_p95": pc(t[:, 1]),
           "total_cpu_ms_mean": float(t.sum(1).mean()), "total_cpu_ms_p95": pc(t.sum(1)),
           "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
           "path": "frozen X1 val path (multizoom.predict_raw + multizoom.fuse, Ultralytics predictor, fused weights)"}
    out["latency_ms_mean"] = out["total_cpu_ms_mean"]
    df = finalize_detections(pd.concat(dets, ignore_index=True))
    out["equal_to_frozen_dets"] = _equal_frozen(df, set(imgs.image_id))
    return out


def _equal_frozen(df: pd.DataFrame, ids: set) -> dict:
    ref = pd.read_parquet(X1_RUN / "dets_val.parquet")
    ref = ref[ref.image_id.isin(ids)]
    """Detection counts against the frozen file (same machine, code and environment; GPU inference is not bit-exact:
    the frozen path itself gives a few detections more or less near conf, smoke 2026-11-08: 5310 vs. 5307 on 20 images).
    Informational; accuracy values stay the frozen ones."""
    a, b = df.groupby("image_id").size(), ref.groupby("image_id").size()
    d = a.sub(b, fill_value=0)
    return {"n": len(df), "n_frozen": len(ref), "images_with_other_count": int((d != 0).sum()),
            "max_count_diff_per_image": int(d.abs().max()) if len(d) else 0}


def _fn(r: dict, imgs: pd.DataFrame, gpu_ok: dict[str, bool]):
    if r["kind"] == "boxfusion":
        return lambda: _boxfusion(r, imgs)
    if r["kind"] == "dense_opt":
        return lambda: _dense_opt(r, imgs)
    return LB._fn(r, imgs, gpu_ok)


def _smi(tag: str) -> None:
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                       capture_output=True, text=True).stdout
    full = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    (OUTD / f"nvidia_smi_{tag}.txt").write_text(f"{dt.datetime.now().isoformat(timespec='seconds')}\n\n"
                                                f"compute apps:\n{q}\n{full}")


def _gpu_ok() -> dict[str, bool]:
    par = json.loads((LB.OUTD / "parity.json").read_text())
    return {k: v["available"] for k, v in par.items() if isinstance(v, dict)}


def smoke(opt: bool = False) -> None:
    gpu_ok = _gpu_ok()
    for r in rows(opt):
        info = _fn(r, r["images"].head(3), gpu_ok)()
        print("[smoke]", r["id"], {k: round(v, 1) for k, v in info.items() if k.endswith("ms_mean")},
              info.get("equal_to_frozen_dets", ""), flush=True)


def measure(opt: bool = False) -> None:
    """``opt``: SCAHI rows in the optimised path -> latency_opt.json (the Table 2 run); else the reference path ->
    latency.json (run of 2026-11-09 08:00, kept for the report)."""
    from scahi import ext_latency as EL
    from scahi.machine import CPU_SHARING, GPU_SHARING, latency_validity, machine_info, require_default_power

    OUTD.mkdir(parents=True, exist_ok=True)
    gpu_ok = _gpu_ok()
    rs = rows(opt)
    name = "latency_opt" if opt else "latency"
    require_default_power("latency_table2")
    part = OUTD / f"{name}_partial.json"
    if part.exists():  # an interrupted earlier run: never mixed with this one
        part.rename(OUTD / f"{name}_aborted_{dt.datetime.now():%Y%m%dT%H%M}.json")
    EL._resume_cpu_queue = lambda: (EL.QDIR / "STOP_CPU").unlink(missing_ok=True)  # no CPU queue to restart here
    EL._pause_cpu_queue()
    try:
        for r in rs:  # smoke first
            _fn(r, r["images"].head(3), gpu_ok)()
        print("[t2] smoke passed for all rows", flush=True)
        runs = []
        for k in range(1, MAX_RUNS + 1):
            tag = f"{'opt_' if opt else ''}run{k}"
            _smi(f"{tag}_start")
            res = {"run": k, "started": dt.datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
                   "machine": machine_info(), "gpu_fusion_parity": gpu_ok, "rows": {}}
            for r in rs:
                GPU_SHARING.update(checks=0, other_pids=[])
                CPU_SHARING.update(checks=0, cpu_queue_busy=False, other_load_cores=0.0)
                info = _fn(r, r["images"], gpu_ok)()
                v = latency_validity()
                res["rows"][r["id"]] = {"meta": {x: y for x, y in r.items() if x not in ("images", "_cals")}, **info,
                                        **v}
                part.write_text(json.dumps({"runs": runs + [res]}, indent=1, default=str))
                print(f"[t2] {tag} {r['id']}: {info['latency_ms_mean']:.1f} ms ({v['latency_status']})", flush=True)
            _smi(f"{tag}_end")
            res["finished"] = dt.datetime.now().isoformat(timespec="seconds")
            res["all_final"] = all(x["latency_status"] == "final" for x in res["rows"].values())
            runs.append(res)
            part.write_text(json.dumps({"runs": runs}, indent=1, default=str))
            if res["all_final"]:
                break
            print(f"[t2] {tag}: not every row final, measuring the whole table again", flush=True)
        out = {"runs": runs, "used_run": runs[-1]["run"] if runs[-1]["all_final"] else None}
        part.write_text(json.dumps(out, indent=1, default=str))
        part.rename(OUTD / f"{name}.json")
        print(f"[t2] done, used run: {out['used_run']}", flush=True)
    finally:
        EL._resume_cpu_queue()


def _acc() -> dict[str, dict]:
    """Frozen accuracy (val / test-dev AP, AP50) and GFLOPs per image (unified method) per row."""
    rc = {r["id"]: r for r in json.loads((REPO / "runs/ext/r3_compare.json").read_text())["rows"]}
    td = {r["id"]: r for r in json.loads((REPO / "runs/ext/testdev_analysis.json").read_text())["compute_control_testdev"]}
    cr = next(q for q in json.loads((REPO / "runs/ext/res_crossing.json").read_text())["rows"] if q["exp_id"] == CROSS.name)
    g = json.loads(GFLOPS.read_text())["rows"]
    out = {i: {"val_AP": rc[i]["metrics"]["AP"], "val_AP50": rc[i]["metrics"].get("AP50"), "gflops": rc[i]["gflops_per_img"],
               "testdev_AP": td.get(i, {}).get("AP"), "testdev_AP50": None}
           for i in ("R2", "E_yolov10s_full2560_o2o", "X2", "XG_yolov10s_2048_o2m_C")}
    out[CROSS.name] = {"val_AP": cr["AP"], "val_AP50": cr["AP50"], "gflops": cr["gflops_unified_per_img"],
                       "testdev_AP": None, "testdev_AP50": None}
    mv = json.loads((X1_RUN / "metrics.json").read_text())["metrics"]
    mt = json.loads((X1_TESTDEV / "metrics.json").read_text())["metrics"]
    out["X1"] = {"val_AP": mv["AP"], "val_AP50": mv["AP50"], "testdev_AP": mt["AP"], "testdev_AP50": mt["AP50"],
                 "gflops": g[str(X1_RUN.relative_to(REPO))]["gflops_per_img"]}
    return out


def _recs(res: dict, acc: dict) -> pd.DataFrame:
    recs = []
    for i, x in res["rows"].items():
        k = x["meta"]["kind"]
        single = k == "single"
        recs.append({"row_id": i, "method": METHOD[i], "path": PATHS[k],
                     "passes_ms": x["latency_ms_mean"] if single else x["passes_ms_mean"],
                     "merge_cpu_ms": x.get("fusion_cpu_ms_mean"), "merge_gpu_ms": x.get("fusion_gpu_ms_mean"),
                     "total_ms": x["latency_ms_mean"] if k != "dense" else x["total_cpu_ms_mean"],
                     "total_gpu_merge_ms": x.get("total_gpu_ms_mean") if k == "dense" else None,
                     "total_p95_ms": x.get("latency_ms_p95") if k != "dense" else x["total_cpu_ms_p95"],
                     "gflops_per_img": acc[i]["gflops"], "val_AP": acc[i]["val_AP"], "val_AP50": acc[i]["val_AP50"],
                     "testdev_AP": acc[i]["testdev_AP"], "testdev_AP50": acc[i]["testdev_AP50"],
                     "latency_status": x["latency_status"]})
    return pd.DataFrame(recs)


PATHS = {"single": "Ultralytics predictor (fused model, one-to-one head), end to end",
         "boxfusion": "frozen X1 path: Ultralytics predictor (fused model, one-to-one head) per pass, NMS (IOS) on the CPU",
         "dense_opt": "optimised: unfused model, one-to-many head only, decoding + SCAHI on the GPU",
         "dense": "reference: unfused model, both heads, pass outputs to the CPU; SCAHI on CPU or GPU"}


def _used(name: str) -> tuple[dict, dict]:
    lat = json.loads((OUTD / f"{name}.json").read_text())
    if lat["used_run"] is None:
        sys.exit(f"{name}: no run with every row final - Table 2 not written")
    return lat, next(r for r in lat["runs"] if r["run"] == lat["used_run"])


def matched_parity_text(part: str) -> str:
    """Parity of the optimised GPU path, one-to-one matched per image and class (scripts/parity_opt_matched.py)."""
    p = json.loads((REPO / f"runs/parity_opt_matched/{part}.json").read_text())
    return (f"one-to-one matching per image and class, runs/parity_opt_matched/{part}.json: " + "; ".join(
        f"{k}: {v['n_opt']} of {v['n_ref']} detections matched in {v['images']} images, count mismatches "
        f"{len(v['count_mismatch_images'])}, unmatched {v['unmatched']}, pairs beyond {p['tolerance_relative']:g} relative "
        f"{v['pairs_beyond_tol']}, max |Δbox| {v['max_box_px']:.2g} px, max |Δscore| {v['max_score']:.2g}"
        for k, v in p["rows"].items()))


def report() -> None:
    """Paper table from the optimised run (latency_opt.json); the reference-path run of 2026-11-09 08:00 (latency.json)
    only as a report section (user 2026-11-09)."""
    lat, res = _used("latency_opt")
    ref_lat, ref = _used("latency")
    acc = _acc()
    df = _recs(res, acc)
    out = REPO / "reports/tab_compute_control"
    df.to_csv(out.with_suffix(".csv"), index=False, float_format="%.2f")
    _recs(ref, acc).to_csv(OUTD / "tab_compute_control_reference_path.csv", index=False, float_format="%.2f")
    f = lambda v, d=1: "–" if v is None or pd.isna(v) else f"{v:.{d}f}"  # noqa: E731
    x1 = res["rows"]["X1"].get("equal_to_frozen_dets", {})
    L = ["# Table 2 – VisDrone compute control, YOLOv10s (latency in ms per image, mean; exclusive GPU and CPU, "
         "RTX 5090 at 575 W, batch 1)", "",
         f"One run on {res['machine'].get('name', res['host'])}, {res['started']} – {res['finished']} (table run "
         f"{res['run']} of {len(lat['runs'])}); every row final: {res['all_final']}. These latencies replace all earlier "
         "Table 2 latencies (DECISIONS 2026-11-08 21:16 and 2026-11-09). Accuracy: frozen values (val; test-dev where "
         "evaluated). GFLOPs per image: unified method (runs/gflops_unified.json). The 2560 weights at 3072 px are "
         "exploratory, post hoc (val only); the refined grid is exploratory, post hoc.", "",
         "| method | passes | merge CPU | merge GPU | total | GFLOPs | val AP | val AP50 | test-dev AP | test-dev AP50 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        L.append(f"| {r.method} | {f(r.passes_ms)} | {f(r.merge_cpu_ms)} | {f(r.merge_gpu_ms)} | {f(r.total_ms)} | "
                 f"{f(r.gflops_per_img, 0)} | {f(r.val_AP, 2)} | {f(r.val_AP50, 2)} | {f(r.testdev_AP, 2)} | "
                 f"{f(r.testdev_AP50, 2)} |")
    L += ["", "**Footnote (code path per row).** Single passes: Ultralytics predictor, fused model, one-to-one head, end "
          "to end (image decoding, pre- and post-processing). Box fusion: the frozen path of row X1, the Ultralytics "
          "predictor (fused model, one-to-one head, tiles batched) per pass, then class-aware NMS with intersection over "
          "the smaller box on the CPU. SCAHI: the optimised path of the E2 run - the model cannot be fused (fusing removes "
          "the one-to-many branch), only the one-to-many head is run, and SCAHI needs its dense per-position outputs (all "
          "positions with a class probability >= 0.001, DFL-decoded) instead of the detector's final boxes; these stay on "
          "the GPU, where decoding and SCAHI run, and only the final detections go to the host. The dense outputs and the "
          "unfused model are a real cost of the method and are included in the passes column. Passes include image "
          "decoding. Totals = passes + merge.",
          "", f"Parity of the optimised path with the reference path (cached pass outputs + SCAHI on the CPU) on every "
          f"val image, {matched_parity_text('visdrone')}. X1 detections in this run: {x1.get('n')} vs. "
          f"{x1.get('n_frozen')} in the frozen val run (GPU inference is not bit-exact; accuracy = frozen values).", "",
          "## Reference path (run of 2026-11-09, report only - not for the paper table)", "",
          f"One run on {ref['machine'].get('name', ref['host'])}, {ref['started']} – {ref['finished']}; every row final: "
          f"{ref['all_final']}. SCAHI rows in the reference path: unfused model, both heads, pass outputs (positions "
          ">= 0.001) copied to the CPU, then SCAHI on the CPU or (on the same outputs, incl. host->GPU copy) on the GPU. "
          "Other rows as above.", "",
          "| method | passes | merge CPU | merge GPU | total (CPU merge) | total (GPU merge) |", "|---|---:|---:|---:|---:|---:|"]
    for _, r in _recs(ref, acc).iterrows():
        tg = r.total_gpu_merge_ms if r.row_id in ("X2", "XG_yolov10s_2048_o2m_C") else None
        L.append(f"| {r.method} | {f(r.passes_ms)} | {f(r.merge_cpu_ms)} | {f(r.merge_gpu_ms)} | {f(r.total_ms)} | "
                 f"{f(tg)} |")
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"smoke": smoke, "smoke_opt": lambda: smoke(True), "parity_opt": parity_opt, "measure": measure,
     "measure_opt": lambda: measure(True), "report": report}[sys.argv[1]]()
