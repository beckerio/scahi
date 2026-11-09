# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-H5 latency on host-a (addendum 4, item 7; user 2026-11-03): one exclusive run after the FCOS tuning, protocol of
``scripts/latency_host_a_dota.py`` (exclusive GPU and CPU, default power limit, batch 1, every DOTA-val image; timing
only, no detection is evaluated).

    python scripts/latency_host_a_dh5.py smoke     # 3 images per row (not reported)
    python scripts/latency_host_a_dh5.py measure   # the run -> runs/latency_host_a/dh5_latency.json
    python scripts/latency_host_a_dh5.py report    # reports/dota/tab_dota_latency_host_a_dh5.{md,csv}

Per model (YOLOv10s, YOLOv10m; 1024 weights):
* SAHI (frozen row SAHI_<m>): exactly ``latency_host_a_dota.sahi`` (library slices + standard pass, then the frozen merge);
* SAHI passes + SCAHI: the passes of the D-H5 caches computed live (fused model of SAHI's predictor path, SAHI's slice
  grid and standard pass - only if the image has more than one slice -, Ultralytics letterbox and box mapping, batch 1;
  ``sahi_scahi.load_fused`` / ``_sahi_grid``), then on the same pass outputs SCAHI of the frozen rows SAHI-F2 and SAHI-RG
  on the CPU (reference) and on the GPU (blocked ``scalespace_torch``, incl. host->GPU copy; parity:
  runs/gpu_fusion/parity_dh5.json). Totals = passes + merge step. Parity of the live passes: CPU fusion of the live
  passes = the cached path (``sahi_scahi.dh5_row_dets``) on 3 calib images.
"""

from __future__ import annotations

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

from latency_host_a_dota import NAME, OUTD, _stats, sahi  # noqa: E402

ROWS = ("SAHI-F2", "SAHI-RG")


def setup() -> list[dict]:
    from scahi.dota import final as F
    from scahi.dota import sahi_scahi as S

    fz, fzd = F.frozen(), F.frozen(S.FROZEN_DH5)
    fr, frd = {r["id"]: r for r in fz["rows"]}, {r["id"]: r for r in fzd["rows"]}
    return [{"model": m, "limit": fzd["pipeline_limit"], "SAHI": fr[f"SAHI_{m}"],
             **{x: frd[f"{x}_{m}"] for x in ROWS}} for m in S.MODELS]


def sahi_scahi(d: dict, imgs: pd.DataFrame, n_warmup: int = 2) -> dict:
    import cv2
    import torch

    from scahi.dota import data as D
    from scahi.dota import pipeline as P
    from scahi.dota import sahi_scahi as S
    from scahi.fusion import run as R
    from scahi.fusion import scalespace as ss
    from scahi.fusion.scalespace_torch import fuse_image_f2
    from scahi.infer import cache_mz as cm
    from scahi.machine import check_exclusive

    m, ov, nc = d["model"], S.overlap(d["model"]), len(D.CLASSES)
    model = S.load_fused(m)
    prm = {x: ss.FusionParams(**{**d[x]["cfg"], "top_k": d["limit"]}) for x in ROWS}

    def passes(path: str) -> tuple[list, int, int]:
        img = cv2.imread(path)
        H, W = img.shape[:2]
        n_sl = len(S.sahi_passes(S.LEVELS["s640f"], W, H, ov))
        lv = []
        for k in (S.ORDER if n_sl > 1 else S.ORDER[:1]):
            z = cm.raw_predict_level(model, img, S.LEVELS[k], ov, batch=1)
            lv.append(ss.CachedLevel(k, z["o2m_xyxy"], z["o2m_probs"], z["o2m_std"], z["o2m_tile"], z["tiles"],
                                     z["gains"], bool(int(z["n_input"]) == 0), ov, ss.LevelCal()))
        return lv, W, H

    def one(path: str) -> tuple[float, ...]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lv, W, H = passes(path)
        torch.cuda.synchronize()
        ts, mem = [1000 * (time.perf_counter() - t0)], []
        for x in ROWS:
            t1 = time.perf_counter()
            ss.fuse_image(lv, W, H, "F2", prm[x], nc=nc)
            ts.append(1000 * (time.perf_counter() - t1))
        for x in ROWS:
            st: dict = {}
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fuse_image_f2(lv, W, H, prm[x], nc=nc, stats=st)
            torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - t1))
            mem.append(st.get("peak_mem_bytes", 0) / 2**20)
        return (*ts, *mem)

    with S._sahi_grid():
        par = []
        cal = P.images("calib")
        for st, pa in list(zip(cal.stem, cal.path))[:3]:
            lv, W, H = passes(pa)
            for x in ROWS:
                live = P._to_dota(R.clip_xywh(ss.fuse_image(lv, W, H, "F2", prm[x], nc=nc), W, H), st)
                ref = P.fuse(S.load(m, "calib", "o2m", [st]), "o2m", "F2", d[x]["cfg"], limit=d["limit"])
                c = ["x1", "y1", "x2", "y2", "score"]
                par.append(bool(len(live) == len(ref) and np.allclose(live[c].to_numpy(float), ref[c].to_numpy(float),
                                                                      atol=1e-3)))
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
    out = {"passes_ms": _stats(t[:, 0]), "n_images": len(t), "parity_live_vs_cache_calib_3x2": par,
           "path": "SAHI-layout passes live (fused model of SAHI's predictor path, batch 1, outputs to CPU)"}
    for i, x in enumerate(ROWS):
        out[x] = {"row": d[x]["id"], "scahi_cpu_ms": _stats(t[:, 1 + i]), "scahi_gpu_ms": _stats(t[:, 3 + i]),
                  "total_scahi_cpu_ms": _stats(t[:, 0] + t[:, 1 + i]), "total_scahi_gpu_ms": _stats(t[:, 0] + t[:, 3 + i]),
                  "scahi_gpu_peak_mem_mb_max": float(t[:, 5 + i].max()),
                  "scahi_gpu_peak_mem_mb_median": float(np.median(t[:, 5 + i]))}
    return out


def run(n: int | None, out: Path | None) -> dict:
    from scahi.dota import pipeline as P
    from scahi.machine import CPU_SHARING, GPU_SHARING, latency_validity, machine_info, require_default_power

    require_default_power("latency_host_a_dh5")
    imgs = P.images("val")
    if n:
        imgs = imgs.head(n)
    res = {"started": dt.datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
           "machine": machine_info(), "rows": {}}
    for d in setup():
        for kind, fn in (("sahi", sahi), ("sahi_scahi", sahi_scahi)):
            key = f"{d['model']}_{kind}"
            for attempt in range(1, 4):
                GPU_SHARING.update(checks=0, other_pids=[])
                CPU_SHARING.update(checks=0, cpu_queue_busy=False, other_load_cores=0.0)
                info = fn(d, imgs)
                v = latency_validity()
                if v["latency_status"] == "final" or attempt == 3 or n:
                    break
                print(f"[lat] {key}: attempt {attempt} not exclusive ({v}), repeating", flush=True)
            res["rows"][key] = {"model": d["model"], "kind": kind, "rows": {x: d[x]["id"] for x in ("SAHI", *ROWS)},
                                **info, **v, "latency_attempts": attempt}
            if out:
                out.write_text(json.dumps(res, indent=1, default=str))
            print(f"[lat] {key}: {v['latency_status']}", flush=True)
    res["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    res["all_final"] = all(x["latency_status"] == "final" for x in res["rows"].values())
    return res


def smoke() -> None:
    r = run(3, None)
    bad = [k for k, x in r["rows"].items() if not all(x.get("parity_live_vs_cache_calib_3x2", [True]))]
    if bad:
        raise SystemExit(f"[lat] live passes differ from the cached path: {bad}")
    print("[lat] smoke passed", flush=True)


def measure() -> None:
    OUTD.mkdir(parents=True, exist_ok=True)
    part = OUTD / "dh5_latency_partial.json"
    if part.exists():
        part.rename(OUTD / f"dh5_latency_aborted_{dt.datetime.now():%Y%m%dT%H%M}.json")
    smoke()
    res = run(None, part)
    part.write_text(json.dumps(res, indent=1, default=str))
    part.rename(OUTD / "dh5_latency.json")
    print(f"[lat] done, all final: {res['all_final']}", flush=True)


def report() -> None:
    res = json.loads((OUTD / "dh5_latency.json").read_text())
    par = json.loads((REPO / "runs/gpu_fusion/parity_dh5.json").read_text())
    bad = {k: x["latency_status"] for k, x in res["rows"].items() if x["latency_status"] != "final"}
    if bad:  # never a table from non-exclusive measurements (user 2026-11-01)
        raise SystemExit(f"not all rows final, no table written: {bad}")
    recs = []
    for x in res["rows"].values():
        base = {"detector": NAME[x["model"]], "latency_status": x["latency_status"]}
        if x["kind"] == "sahi":
            recs.append({**base, "method": "SAHI (frozen)", "row_id": x["rows"]["SAHI"], "passes_ms": x["passes_ms"]["mean"],
                         "merge_ms": x["merge_ms"]["mean"], "total_ms": x["total_ms"]["mean"],
                         "total_p95_ms": x["total_ms"]["p95"], "fusion_peak_gpu_mem_mb": None})
            continue
        p = x["passes_ms"]["mean"]
        for r in ROWS:
            y = x[r]
            for dev in ("cpu", "gpu"):
                if dev == "gpu" and not par["rows"][y["row"]]["passed"]:
                    continue  # GPU fusion times only with a passed parity gate
                recs.append({**base, "method": f"SAHI passes + SCAHI, {r} (fusion on {dev.upper()})", "row_id": y["row"],
                             "passes_ms": p, "merge_ms": y[f"scahi_{dev}_ms"]["mean"],
                             "total_ms": y[f"total_scahi_{dev}_ms"]["mean"], "total_p95_ms": y[f"total_scahi_{dev}_ms"]["p95"],
                             "fusion_peak_gpu_mem_mb": y["scahi_gpu_peak_mem_mb_max"] if dev == "gpu" else None})
    df = pd.DataFrame(recs)
    df.insert(0, "measured_on", "host-a")
    out = REPO / "reports/dota/tab_dota_latency_host_a_dh5"
    df.to_csv(out.with_suffix(".csv"), index=False, float_format="%.2f")
    f = lambda v: "–" if v is None or pd.isna(v) else f"{v:.1f}"  # noqa: E731
    m = res["machine"]
    L = ["# DOTA-val latency, D-H5 (SAHI passes + SCAHI) – measured on host-a", "",
         "**Superseded (2026-11-09):** all DOTA latencies of the paper come from the single host-b run (`reports/dota/tab_dota_latency_host_b.md`, `reports/dota/tab_dota_latency_host_b_dh5.md`); this host-a table is kept for comparison only.", "",
         f"ms per image (mean over {next(iter(res['rows'].values()))['n_images']} DOTA-val images; p95 in the CSV); one "
         f"run on host-a ({m.get('gpu')}, driver {m.get('driver')}, default power limit), {res['started']} – "
         f"{res['finished']}; GPU and CPU exclusive, batch 1; all rows final: {res['all_final']}. Same protocol as "
         "`reports/dota/tab_dota_latency_host_a.md`, separate run; never mixed with the host-b tables.", "",
         "SAHI: library slices (1024 weights) + standard pass, then the frozen merge (as in the host-a DOTA table). SAHI "
         "passes + SCAHI: the same passes computed once per image (fused model of SAHI's predictor path, both heads, "
         "outputs to the CPU), then SCAHI of the frozen D-H5 rows on the CPU (reference) or the GPU (blocked "
         "implementation, incl. host→GPU copy); the pass time is shared by the four SCAHI rows of a model. Box fusion "
         "inside SAHI runs on the CPU; the table is no speed comparison of SCAHI on the GPU against SAHI's merge. GPU "
         "fusion parity: " + ", ".join(f"`{k}` {'identical' if v['passed'] else 'DIFFERS'}" for k, v in par["rows"].items())
         + " on every DOTA-val image (`runs/gpu_fusion/parity_dh5.json`).", "",
         "| detector | method | passes | merge step | total | peak GPU mem. fusion (MB) |", "|---|---|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        L.append(f"| {r.detector} | {r.method} | {f(r.passes_ms)} | {f(r.merge_ms)} | {f(r.total_ms)} | "
                 f"{f(r.fusion_peak_gpu_mem_mb)} |")
    L += _profile_section(res)
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def _profile_section(res: dict) -> list[str]:
    """Why SAHI's passes take longer than the SCAHI path's passes (user, 2026-11-03): breakdown from the profiling run
    ``scripts/latency_dh5_profile.py`` (not a latency measurement) -> runs/latency_host_a/dh5_profile.json."""
    pf = json.loads((OUTD / "dh5_profile.json").read_text())
    f = lambda v: "–" if v is None else f"{v:.1f}"  # noqa: E731
    excl = {x["model"]: {} for x in res["rows"].values()}
    for x in res["rows"].values():
        excl[x["model"]][x["kind"]] = x["passes_ms"]["mean"]
    rows = []  # (component, SAHI value, SCAHI-path value) per model
    for m, v in pf["models"].items():
        a, b = v["sahi"], v["scahi_path"]
        call = a["ultralytics call (total)"]
        u_rest = call - a["ultralytics preprocess"] - a["ultralytics inference"] - a["ultralytics postprocess"]
        read = a["sahi: read image (PIL) + cut slices"] + a["sahi: standard pass image conversion"]
        loop = a["total"] - read - call - a["sahi: ObjectPrediction creation"] - a["objects -> array"]
        b_parts = ("read image (cv2)", "crop + letterbox (cv2)", "to tensor + host->GPU", "forward (fused model)",
                   "decode (both heads)", "map boxes to source")
        rows.append((m, [
            ("image read and slicing (SAHI: PIL + slice crops; SCAHI path: cv2)", read, b["read image (cv2)"]),
            ("preprocessing (letterbox, tensor, host→GPU)", a["ultralytics preprocess"],
             b["crop + letterbox (cv2)"] + b["to tensor + host->GPU"]),
            ("forward pass (same fused weights)", a["ultralytics inference"], b["forward (fused model)"]),
            ("head output → boxes (SAHI: Ultralytics postprocess + Results; SCAHI path: decode both heads + mapping)",
             a["ultralytics postprocess"] + u_rest, b["decode (both heads)"] + b["map boxes to source"]),
            ("SAHI: one ObjectPrediction per detection", a["sahi: ObjectPrediction creation"], None),
            ("SAHI: slice loop, shifting / filtering of the objects", loop, None),
            ("SAHI: objects → arrays for the merge", a["objects -> array"], None),
            ("SCAHI path: rest (loop, bookkeeping)", None, b["total"] - sum(b[k] for k in b_parts)),
            ("**total (profiling run)**", a["total"], b["total"]),
            ("total in the exclusive run (table above)", excl[m]["sahi"], excl[m]["sahi_scahi"]),
            ("detections per image (SAHI: o2o after conf; SCAHI path: o2m candidates)", a["n_dets"], b["n_dets"]),
        ]))
    L = ["", "## Why SAHI's passes take longer than the SCAHI path's passes", "",
         "Both paths run the same passes (SAHI's slice grid and standard pass, the same fused weights, batch 1). "
         f"Breakdown from a separate profiling run (`scripts/latency_dh5_profile.py`, {pf['n_images']} DOTA-val images, "
         f"{pf['started'][:16].replace('T', ' ')}; {pf['conditions']}), ms per image (mean):", "",
         "| component | " + " | ".join(f"{NAME[m]} SAHI | {NAME[m]} SCAHI path" for m, _ in rows) + " |",
         "|---|" + "---:|---:|" * len(rows)]
    for i, (name, *_) in enumerate(rows[0][1]):
        L.append(f"| {name} | " + " | ".join(f"{f(r[i][1])} | {f(r[i][2])}" for _, r in rows) + " |")
    L += ["", "The GPU-side work (preprocessing, forward pass, head output) costs about the same in both paths. SAHI's "
          "additional time is CPU-side implementation overhead of the SAHI library: the image is decoded with PIL, and "
          "every detection of every slice becomes a Python `ObjectPrediction` that is shifted and filtered one by one "
          "and converted back to arrays for the merge (several thousand detections per image at the frozen confidence "
          "threshold). It grows with the number of detections and with lower CPU clock (larger under the load of the "
          "profiling run than in the exclusive run); how it splits over the SAHI rows varies between the models "
          "(Python's garbage collection of the many objects falls into different components), their sum is the robust "
          "figure. It is not a property of SCAHI: SCAHI's fusion uses the same passes "
          "and only replaces the merge step.", "",
          "Wording for the paper: SCAHI's merge on the GPU and SAHI's merge on the CPU run on different hardware, so "
          "their times are no speed comparison. The valid statement: replacing SAHI's merge by SCAHI does not add "
          "substantial latency, and the passes dominate the cost."]
    return L


if __name__ == "__main__":
    {"smoke": smoke, "measure": measure, "report": report}[sys.argv[1]]()
