# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parity of the optimised GPU path with the frozen CPU reference, with a true one-to-one matching (user 2026-11-09).

    python scripts/parity_opt_matched.py dota       # RG rows of the DOTA latency table (YOLOv10s/m, YOLOv11s 2048; YOLOv12s 1024)
    python scripts/parity_opt_matched.py visdrone   # Table 2 SCAHI rows (X2, XG_yolov10s_2048_o2m_C)

Per image and class the detections of the two paths are matched one to one on the box distance (L-inf over x, y, w, h):
mutual nearest neighbours first (KD-tree), the remainder with the Hungarian method; scores are compared only after the
matching. Reference = SCAHI on the CPU from the stored dense-head caches (the frozen path); optimised = passes and SCAHI on
the GPU (``pipeline_gpu`` / ``latency_dota_host_b._opt_levels``), exactly the code timed in the latency runs. Reported per
row: images and detections compared, count mismatches, unmatched detections, pairs beyond the tolerance (box or score
difference > 1e-4 relative, i.e. > 1e-4 * max(1, |value|)), max absolute box and score differences, the affected images.
Output: runs/parity_opt_matched/<dataset>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "runs/parity_opt_matched"
TOL = 1e-4
COLS = ["x", "y", "w", "h", "score", "category_id"]


def _match(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one matching of boxes A [n, 4] and B [m, 4] on the L-inf distance: mutual nearest neighbours, then the
    Hungarian method on the rest. Returns index arrays (ia, ib)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    if not len(A) or not len(B):
        return np.zeros(0, int), np.zeros(0, int)
    _, nb = cKDTree(B).query(A, p=np.inf)
    _, na = cKDTree(A).query(B, p=np.inf)
    mutual = na[nb] == np.arange(len(A))
    ia, ib = np.flatnonzero(mutual), nb[mutual]
    ra = np.setdiff1d(np.arange(len(A)), ia)
    rb = np.setdiff1d(np.arange(len(B)), ib)
    if len(ra) and len(rb):
        D = np.abs(A[ra][:, None, :] - B[rb][None, :, :]).max(2)
        r, c = linear_sum_assignment(D)
        ia, ib = np.concatenate([ia, ra[r]]), np.concatenate([ib, rb[c]])
    return ia, ib


def compare(a: pd.DataFrame, b: pd.DataFrame, res: dict, stem: str) -> None:
    A, B = a[COLS].to_numpy(float), b[COLS].to_numpy(float)
    res["images"] += 1
    res["n_ref"] += len(A)
    res["n_opt"] += len(B)
    if len(A) != len(B):
        res["count_mismatch_images"].append(stem)
    worst = {"box": 0.0, "score": 0.0, "n_beyond": 0, "unmatched": 0}
    for c in np.union1d(A[:, 5], B[:, 5]):
        pa, pb = A[A[:, 5] == c], B[B[:, 5] == c]
        ia, ib = _match(pa[:, :4], pb[:, :4])
        worst["unmatched"] += len(pa) + len(pb) - 2 * len(ia)
        if not len(ia):
            continue
        db = np.abs(pa[ia, :4] - pb[ib, :4])
        ds = np.abs(pa[ia, 4] - pb[ib, 4])
        rel_b = db / np.maximum(1.0, np.abs(pa[ia, :4]))
        rel_s = ds / np.maximum(1.0, np.abs(pa[ia, 4]))
        worst["box"] = max(worst["box"], float(db.max()))
        worst["score"] = max(worst["score"], float(ds.max()))
        worst["n_beyond"] += int(((rel_b > TOL).any(1) | (rel_s > TOL)).sum())
    res["max_box_px"] = max(res["max_box_px"], worst["box"])
    res["max_score"] = max(res["max_score"], worst["score"])
    res["pairs_beyond_tol"] += worst["n_beyond"]
    res["unmatched"] += worst["unmatched"]
    if worst["box"] > 0 or worst["score"] > 0 or worst["unmatched"]:
        res["images_with_any_difference"].append({"stem": stem, **worst})


def _new() -> dict:
    return {"images": 0, "n_ref": 0, "n_opt": 0, "count_mismatch_images": [], "unmatched": 0, "pairs_beyond_tol": 0,
            "max_box_px": 0.0, "max_score": 0.0, "images_with_any_difference": []}


def dota() -> dict:
    import cv2
    import torch

    import latency_dota_host_b as M
    import latency_host_a_dota as KD
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
        res = _new()
        with torch.no_grad():
            for st, pa in zip(imgs.stem, imgs.path):
                lv, W, H = P.load(d["model"], d["s"], "val", rg["branch"], "C", [st])[st]
                a = clip_xywh(ss.fuse_image(lv, W, H, "F2", prm, nc=nc), W, H)
                levels, W2, H2 = M._opt_levels(model, cv2.imread(pa), P.STACK[d["s"]], rg["branch"], nc)
                b = clip_xywh(fuse_image_f2(levels, W2, H2, prm, nc=nc), W2, H2)
                compare(a, b, res, st)
        out[rg["id"]] = res
        print(rg["id"], _summary(res), flush=True)
        del model
        torch.cuda.empty_cache()
    return out


def visdrone() -> dict:
    import cv2
    import torch

    import latency_breakdown as LB
    import latency_table2 as T2
    from scahi.fusion import pipeline_gpu as PG
    from scahi.fusion import scalespace as ss
    from scahi.fusion.run import clip_xywh
    from scahi.infer.raw_head import load_unfused

    out = {}
    for r in T2.rows():
        if r["kind"] != "dense":
            continue
        p = ss.FusionParams(**r["cfg"])
        m = load_unfused(r["weights"])
        res = _new()
        with torch.no_grad():
            for st, path in zip(r["images"].stem, r["images"].path):
                lv, W, H = LB._cache_levels(r, st)
                a = clip_xywh(ss.fuse_image(lv, W, H, "F2", p, nc=r["nc"]), W, H)
                b, _, _ = PG.predict_image(m, cv2.imread(path), r["stack"], r["overlap"], r["branch"], r["cfg"])
                compare(a, b, res, st)
        out[r["id"]] = res
        print(r["id"], _summary(res), flush=True)
        del m
        torch.cuda.empty_cache()
    return out


def _summary(res: dict) -> dict:
    return {k: (len(v) if isinstance(v, list) else v) for k, v in res.items()}


if __name__ == "__main__":
    part = sys.argv[1]
    r = {"dota": dota, "visdrone": visdrone}[part]()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{part}.json").write_text(json.dumps({"tolerance_relative": TOL, "rows": r}, indent=1, default=float))
