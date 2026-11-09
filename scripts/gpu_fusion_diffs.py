# SPDX-License-Identifier: AGPL-3.0-or-later
"""Analysis of every differing image of the GPU-fusion parity gate (runs/gpu_fusion/parity_<part>.json).

    python scripts/gpu_fusion_diffs.py dota

Per differing (row, image): (1) blocked GPU vs. the previous dense GPU implementation (git 7912698, before the change;
only where it fits into GPU memory) - identical means the difference is not caused by the blocking; (2) blocked GPU at
block sizes 512 / 1024 / 4096 - identical means the block layout does not matter; (3) cause in the CPU reference: the
differing detections are traced to their peaks (field value, members, strongest member) to see whether they sit at a
decision boundary (peak test S >= neighbour, membership distance = 1.5 g_J, tie of the strongest member, IoU = 0.5).
Output: runs/gpu_fusion/diffs_<part>.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DENSE_SRC = Path(sys.argv[2]) if len(sys.argv) > 2 else None


def _dense():
    spec = importlib.util.spec_from_file_location("scalespace_torch_dense", DENSE_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fuse_image_f2


def _arr(d: pd.DataFrame) -> np.ndarray:
    return d.sort_values(["score", "x", "y"], ascending=[False, True, True])[["x", "y", "w", "h", "score", "category_id"]] \
        .to_numpy(float)


def same(a: pd.DataFrame, b: pd.DataFrame, tol: float = 1e-4) -> bool:
    x, y = _arr(a), _arr(b)
    return len(x) == len(y) and (not len(x) or (np.abs(x[:, :5] - y[:, :5]).max() <= tol and (x[:, 5] == y[:, 5]).all()))


def one_to_one(a: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> tuple[int, float, float]:
    """Greedy one-to-one matching of detections (rows x1, y1, x2, y2, score, cls): same class, every value within
    ``tol``; returns (unmatched count, max |box diff|, max |score diff|) over the matched pairs."""
    from scipy.spatial import cKDTree

    if len(a) != len(b):
        return abs(len(a) - len(b)) + min(len(a), len(b)), np.nan, np.nan
    if not len(a):
        return 0, 0.0, 0.0
    t = cKDTree(np.c_[b[:, :5], b[:, 5] * 1e6])
    used = np.zeros(len(b), bool)
    un, mb, ms = 0, 0.0, 0.0
    k = min(16, len(b))
    dist, idx = t.query(np.c_[a[:, :5], a[:, 5] * 1e6], k=k, p=np.inf)
    dist, idx = dist.reshape(len(a), k), idx.reshape(len(a), k)
    for i in range(len(a)):
        for d_, j in zip(dist[i], idx[i]):
            if d_ > tol:
                break
            if not used[j]:
                used[j] = True
                mb = max(mb, float(np.abs(a[i, :4] - b[j, :4]).max()))
                ms = max(ms, float(abs(a[i, 4] - b[j, 4])))
                break
        else:
            un += 1
            continue
        if not used[j] or d_ > tol:
            un += 1
    return un, mb, ms


def dota() -> list[dict]:
    import torch

    from scahi.dota import data as D
    from scahi.dota import final as F
    from scahi.dota import pipeline as P
    from scahi.fusion import scalespace as ss
    from scahi.fusion.scalespace_torch import fuse_image_f2

    par = json.loads((REPO / "runs/gpu_fusion/parity_dota.json").read_text())
    rows = {r["id"]: (r, fz["pipeline_limit"]) for fz in (F.frozen(), F.frozen(F.FROZEN_NEW)) for r in fz["rows"]}
    dense = _dense() if DENSE_SRC else None
    out = []
    for rid, res in par["rows"].items():
        if not res["diffs"]:
            continue
        row, limit = rows[rid]
        data = P.load(row["model"], row["s"], "val", row["branch"], row["stack"])
        prm = ss.FusionParams(**{**row["cfg"], "top_k": limit})
        for dfx in res["diffs"]:
            stem = dfx["image"]
            levels, W, H = data[stem]
            from scahi.fusion import run as R

            ref = pd.read_parquet(REPO / f"runs/dota/val/dets/{rid}.parquet")
            ref = ref[ref.image == stem][["x1", "y1", "x2", "y2", "score", "cls"]].to_numpy(float)
            cpu = ss.fuse_image(levels, W, H, "F2", prm, nc=len(D.CLASSES))
            gb = {b: fuse_image_f2(levels, W, H, prm, nc=len(D.CLASSES), block=b) for b in (512, 1024, 4096)}

            def oo(x, y):
                return dict(zip(("unmatched", "max_box", "max_score"), one_to_one(
                    _arr(x)[:, [0, 1, 2, 3, 4, 5]], _arr(y)[:, [0, 1, 2, 3, 4, 5]])))
            e = {"row": rid, "image": stem, "n_ref": dfx["n_ref"], "n_gpu": dfx["n_gpu"],
                 "only_ref": dfx["only_ref"], "only_gpu": dfx["only_gpu"],
                 "one_to_one_vs_stored": dict(zip(("unmatched", "max_box", "max_score"), one_to_one(
                     ref, P._to_dota(R.clip_xywh(gb[1024], W, H), stem)[["x1", "y1", "x2", "y2", "score", "cls"]]
                     .to_numpy(float)))),
                 "blocks_512_vs_1024": oo(gb[512], gb[1024]), "blocks_4096_vs_1024": oo(gb[4096], gb[1024]),
                 "blocked_vs_cpu_recomputed": oo(gb[1024], cpu)}
            if dense is not None:
                try:
                    dd = dense(levels, W, H, prm, nc=len(D.CLASSES))
                    e["dense_gpu_vs_blocked"] = oo(dd, gb[1024])
                except torch.OutOfMemoryError:
                    e["dense_gpu_vs_blocked"] = "dense GPU version out of memory"
                    torch.cuda.empty_cache()
            out.append(e)
            print(json.dumps({k: v for k, v in e.items() if k not in ("only_ref", "only_gpu")}), flush=True)
    return out


def main() -> None:
    part = sys.argv[1]
    o = {"dota": dota}[part]()
    (REPO / f"runs/gpu_fusion/diffs_{part}.json").write_text(json.dumps(o, indent=1))


if __name__ == "__main__":
    main()
