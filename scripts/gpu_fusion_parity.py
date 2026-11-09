# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parity gate of the blocked GPU fusion (host-a brief 2026-11-01, task 3; ``scahi.fusion.scalespace_torch``).

    python scripts/gpu_fusion_parity.py dota       # every SCAHI row of both DOTA freezes, every DOTA-val image
    python scripts/gpu_fusion_parity.py visdrone   # every frozen SCAHI (F2) row of the extension freeze + X2, val
    python scripts/gpu_fusion_parity.py dh5        # the 4 D-H5 rows (SAHI passes + SCAHI, addendum 4), DOTA-val
    python scripts/gpu_fusion_parity.py report     # runs/gpu_fusion/PARITY.md from the JSONs

Reference = the CPU fusion (``scalespace.fuse_image``): for DOTA and the extension rows its stored frozen detections
(``runs/dota/val/dets/<row>.parquet``, ``runs/ext/.../<val_exp_id>/dets_val.parquet``), for rows without stored val
detections (X2, D_yolov10s_single2048_F2_o2m) the CPU fusion recomputed on the same cached levels. The GPU output goes
through the same post-processing as the frozen path (clip to the image; DOTA: devkit columns, pipeline limit 200 000;
VisDrone: sort + cap 500). Per image the detections are matched one to one (same class, box and score within 1e-4 px /
1e-4, as the earlier parity tests; greedy over the 16 nearest candidates); an image counts as identical if every
detection has a partner. A first version compared the detections after sorting (score desc, x, y) element by element and
flagged 37 DOTA images only because detections with equal scores came out in a different order
(runs/gpu_fusion/parity_dota_v1_sorted_compare.json, analysis diffs_dota_v1.json). Every differing detection is listed. No metric is computed (no evaluation).
Output: runs/gpu_fusion/parity_<part>.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUTD = REPO / "runs/gpu_fusion"
TOL = 1e-4
COLS = ["x1", "y1", "x2", "y2", "score", "cls"]


def _arr(d: pd.DataFrame) -> np.ndarray:
    return d.sort_values(["score", "x1", "y1"], ascending=[False, True, True])[COLS].to_numpy(float)


def _xyxy(d: pd.DataFrame) -> pd.DataFrame:
    """VisDrone (x, y, w, h, score, category_id) -> comparison columns."""
    return pd.DataFrame({"x1": d.x, "y1": d.y, "x2": d.x + d.w, "y2": d.y + d.h, "score": d.score, "cls": d.category_id})


def compare(ref: pd.DataFrame, gpu: pd.DataFrame, image: str, res: dict) -> None:
    from gpu_fusion_diffs import one_to_one

    a, b = _arr(ref), _arr(gpu)
    res["n_images"] += 1
    res["n_dets_ref"] += len(a)
    res["n_dets_gpu"] += len(b)
    un, mb, ms = one_to_one(a, b, TOL)
    if len(a) == len(b) and un == 0:
        res["max_box"] = max(res["max_box"], mb)
        res["max_score"] = max(res["max_score"], ms)
        return
    # differing image: list every detection without a partner (same class, all of box/score within TOL)
    from scipy.spatial import cKDTree

    def unmatched(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not len(y):
            return x
        t = cKDTree(np.c_[y[:, :5], y[:, 5] * 1e6])
        dist, _ = t.query(np.c_[x[:, :5], x[:, 5] * 1e6], p=np.inf)
        return x[dist > TOL]

    ua, ub = unmatched(a, b), unmatched(b, a)
    res["n_img_diff"] += 1
    res["n_det_only_ref"] += len(ua)
    res["n_det_only_gpu"] += len(ub)
    res["diffs"].append({"image": image, "n_ref": len(a), "n_gpu": len(b), "unmatched_one_to_one": un,
                         "only_ref": ua.round(6).tolist(), "only_gpu": ub.round(6).tolist()})


def _new_res() -> dict:
    return {"n_images": 0, "n_dets_ref": 0, "n_dets_gpu": 0, "n_img_diff": 0, "n_det_only_ref": 0, "n_det_only_gpu": 0,
            "max_box": 0.0, "max_score": 0.0, "max_peak_mem_gib": 0.0, "max_halo_px": 0.0, "max_blocks": 0,
            "gpu_s": 0.0, "diffs": []}


def _finish(res: dict) -> dict:
    res["passed"] = res["n_img_diff"] == 0
    return res


def dota(dh5: bool = False) -> dict:
    import torch

    from scahi.dota import data as D
    from scahi.dota import final as F
    from scahi.dota import pipeline as P
    from scahi.dota import sahi_scahi as S
    from scahi.fusion import run as R
    from scahi.fusion import scalespace as ss
    from scahi.fusion.scalespace_torch import fuse_image_f2

    out = {}
    for fz in ((F.frozen(S.FROZEN_DH5),) if dh5 else (F.frozen(), F.frozen(F.FROZEN_NEW))):
        limit = fz["pipeline_limit"]
        for row in fz["rows"]:
            if row["kind"] not in ("cache", "dh5") or row["variant"] != "F2":
                continue
            ref = pd.read_parquet(REPO / f"runs/dota/val/dets/{row['id']}.parquet")
            refg = {k: g for k, g in ref.groupby("image")}
            data = (S.load(row["model"], "val", row["branch"]) if dh5 else
                    P.load(row["model"], row["s"], "val", row["branch"], row["stack"]))
            prm = ss.FusionParams(**{**row["cfg"], "top_k": limit})
            res = _new_res()
            for stem, (levels, W, H) in data.items():
                st: dict = {}
                t0 = time.perf_counter()
                d = fuse_image_f2(levels, W, H, prm, nc=len(D.CLASSES), stats=st)
                torch.cuda.synchronize()
                res["gpu_s"] += time.perf_counter() - t0
                res["max_peak_mem_gib"] = max(res["max_peak_mem_gib"], st.get("peak_mem_bytes", 0) / 2**30)
                res["max_halo_px"] = max(res["max_halo_px"], st.get("max_halo", 0.0))
                res["max_blocks"] = max(res["max_blocks"], st.get("blocks", 0))
                g = P._to_dota(R.clip_xywh(d, W, H), stem)
                compare(refg.get(stem, ref.iloc[:0]), g, stem, res)
            out[row["id"]] = _finish(res)
            print(f"[parity] {row['id']}: {res['n_images']} images, {res['n_dets_ref']} dets, "
                  f"{res['n_img_diff']} differing images, peak {res['max_peak_mem_gib']:.2f} GiB", flush=True)
    return out


def visdrone() -> dict:
    import glob

    import torch

    from scahi import ext as E
    from scahi import ext_testdev as T
    from scahi.eval.detections import finalize_detections
    from scahi.fusion import run as R
    from scahi.fusion import scalespace as ss
    from scahi.fusion.scalespace_torch import fuse_image_f2

    sys.path.insert(0, str(REPO / "scripts"))
    import latency_breakdown as LB

    fz = T.freeze()
    coco, imgs = T.split_data("val")
    stem_id = dict(zip(imgs.stem, imgs.image_id))
    cases = []
    for r in fz["rows"]:
        if r["kind"] != "cache" or r["fusion"]["variant"] != "F2":
            continue
        f = r["fusion"]
        dirs = T._dirs(T.check_weights(r), f["levels"], f["overlap"], "val")
        stored = [p for p in glob.glob(str(REPO / f"runs/ext/**/{r['val_exp_id']}/dets_val.parquet"), recursive=True)
                  if "testdev" not in p and "dryrun" not in p]
        cases.append((r["id"], dirs, f["branch"], f["overlap"], dict(f["params"]), stored[0] if stored else None))
    for r in LB.rows():  # X2 (main study; no stored val detections of this path -> CPU recomputed)
        if r["id"] == "X2":
            from scahi.fusion.run import CACHE_ROOT
            from scahi.infer import cache_mz as cm

            dirs = [(k, cm.cache_dir(CACHE_ROOT, Path(r["weights"]), k, r["overlap"], "val"))
                    for k in cm.STACK_LEVELS[r["stack"]]]
            cases.append(("X2", dirs, r["branch"], r["overlap"], dict(r["cfg"]), None))
    out = {}
    for rid, dirs, branch, overlap, cfg, stored in cases:
        prm = ss.FusionParams(**cfg)
        res = _new_res()
        res["reference"] = (str(Path(stored).relative_to(REPO)) if stored else "CPU fusion recomputed")
        refd = pd.read_parquet(stored) if stored else None
        refg = {k: g for k, g in refd.groupby("image_id")} if stored else {}
        for stem in imgs.stem:
            levels, W, H = ss.load_levels(dirs, stem, branch, overlap)
            st: dict = {}
            t0 = time.perf_counter()
            d = fuse_image_f2(levels, W, H, prm, stats=st)
            torch.cuda.synchronize()
            res["gpu_s"] += time.perf_counter() - t0
            res["max_peak_mem_gib"] = max(res["max_peak_mem_gib"], st.get("peak_mem_bytes", 0) / 2**30)
            res["max_halo_px"] = max(res["max_halo_px"], st.get("max_halo", 0.0))
            res["max_blocks"] = max(res["max_blocks"], st.get("blocks", 0))
            iid = stem_id[stem]
            g = finalize_detections(R.clip_xywh(d, W, H).assign(image_id=iid)[R.DET_COLUMNS]
                                    .astype({"image_id": int, "category_id": int}))
            if stored:
                a = refg.get(iid, refd.iloc[:0])
            else:
                c = ss.fuse_image(levels, W, H, "F2", prm)
                a = finalize_detections(R.clip_xywh(c, W, H).assign(image_id=iid)[R.DET_COLUMNS]
                                        .astype({"image_id": int, "category_id": int}))
            compare(_xyxy(a), _xyxy(g), stem, res)
        out[rid] = _finish(res)
        print(f"[parity] {rid}: {res['n_images']} images, {res['n_dets_ref']} dets, {res['n_img_diff']} differing "
              f"images, peak {res['max_peak_mem_gib']:.2f} GiB ({res['reference']})", flush=True)
    del E
    return out


def main() -> None:
    part = sys.argv[1]
    OUTD.mkdir(parents=True, exist_ok=True)
    rows = {"dota": dota, "visdrone": visdrone, "dh5": lambda: dota(dh5=True)}[part]()
    o = {"part": part, "tolerance": TOL, "time": time.strftime("%Y-%m-%dT%H:%M"),
         "implementation": "scahi.fusion.scalespace_torch (blocked membership, block 1024 px)",
         "rows": rows, "passed": all(v["passed"] for v in rows.values()),
         "n_rows": len(rows), "n_img_diff": sum(v["n_img_diff"] for v in rows.values())}
    (OUTD / f"parity_{part}.json").write_text(json.dumps(o, indent=1))
    print(json.dumps({k: o[k] for k in ("part", "passed", "n_rows", "n_img_diff")}))


if __name__ == "__main__":
    main()
