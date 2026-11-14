# SPDX-License-Identifier: AGPL-3.0-or-later
"""Why B-RG_yolov10m_2048 moves by -0.42 mAP50 on calib with the Ultralytics-aligned caches (authors, 2026-11-14).

    python scripts/mapcheck_brg.py [ROW_ID]     # default B-RG_yolov10m_2048 -> runs/mapcheck/all/investigate_<id>.json

Same row, same frozen configuration, detections from the frozen and the aligned caches (scahi.dota.final.row_dets,
cache root switched only). Reports: per-class AP50 (VOC07 11-point, the primary metric) and its delta; per-image swap
(frozen detections with one image replaced by its aligned detections: delta of the dataset mAP50) and the images that
carry the total; the pipeline limit (images at the limit, the same row with the limit 10 000 000 as in the FCOS limit
check); score ties at the class cut-offs; and the 11-point steps of the class carrying the change (recall / precision at
the interpolation points). Calib only; descriptive.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scahi import mapcheck as MC  # noqa: E402

OUT = REPO / "runs/mapcheck/all"


def _dets(row: dict, root: Path, limit: int | None = None):
    from scahi.dota import final as F
    from scahi.fusion import run as R

    saved = R.CACHE_ROOT
    R.CACHE_ROOT = root
    try:
        return F.row_dets(row, "calib", limit=limit)
    finally:
        R.CACHE_ROOT = saved


def main() -> None:
    import pandas as pd

    from scahi.dota import data as DD
    from scahi.dota import evaluate as EV
    from scahi.dota import pipeline as P
    from scahi.dota import train as T
    from scahi.fusion.run import CACHE_ROOT

    rid = sys.argv[1] if len(sys.argv) > 1 else "B-RG_yolov10m_2048"
    row = next(r for p in ("configs/dota/frozen_rows.json", "configs/dota/frozen_rows_new.json")
               for r in json.loads((REPO / p).read_text())["rows"] if r["id"] == rid)
    stems = T.split_stems("calib")
    gt = EV.gt_arrays("train", stems)
    res = {"row": rid, "cfg": row["cfg"], "stored_calib_mAP50": row["calib_mAP50"], "pipeline_limit": P.PIPE}
    t0 = time.time()
    dz, iz = _dets(row, CACHE_ROOT)
    da, ia = _dets(row, MC.ROOT)
    cz, ca = EV.build(gt, dz, None, sizes=False), EV.build(gt, da, None, sizes=False)
    mz, ma = EV.map50(cz), EV.map50(ca)
    res.update(mAP50_frozen=mz, mAP50_aligned=ma, delta=ma - mz, n_dets={"frozen": len(dz), "aligned": len(da)})
    names = {i: n for n, i in DD.CLASS_ID.items()}
    pc = {}
    for c in cz.classes:
        a, b = EV.class_ap(cz, c, 0, -1), EV.class_ap(ca, c, 0, -1)  # fractions -> percent
        a, b = (None if a is None else 100 * a), (None if b is None else 100 * b)
        pc[names.get(c, c)] = {"frozen": a, "aligned": b, "delta": None if a is None else b - a}
    res["per_class_AP50"] = pc
    res["per_class_delta_sum_over_n"] = sum(v["delta"] for v in pc.values() if v["delta"] is not None) / len(
        [v for v in pc.values() if v["delta"] is not None])
    print(f"[brg] {rid}: frozen {mz:.4f} aligned {ma:.4f} delta {ma - mz:+.4f} ({time.time() - t0:.0f} s)", flush=True)
    top = sorted(pc.items(), key=lambda kv: -abs(kv[1]["delta"] or 0))[:4]
    print("[brg] largest class deltas:", [(k, round(v["delta"], 3)) for k, v in top], flush=True)

    # per-image swap
    sw = {}
    gz, ga = dict(tuple(dz.groupby("image"))), dict(tuple(da.groupby("image")))
    changed = [s for s in stems if not (s in gz and s in ga and len(gz[s]) == len(ga[s])
                                        and np.allclose(gz[s][["x1", "y1", "x2", "y2", "score"]].to_numpy(),
                                                        ga[s][["x1", "y1", "x2", "y2", "score"]].to_numpy(), atol=0))]
    for s in changed:
        d = pd.concat([v for k, v in gz.items() if k != s] + ([ga[s]] if s in ga else []), ignore_index=True)
        sw[s] = EV.map50(EV.build(gt, d, None, sizes=False)) - mz
    order = sorted(sw.items(), key=lambda kv: -abs(kv[1]))
    res["per_image_swap"] = {"n_images": len(stems), "n_changed": len(changed), "sum_of_swaps": sum(sw.values()),
                             "top": [{"image": k, "delta": v, "n_dets_frozen": len(gz.get(k, [])),
                                      "n_dets_aligned": len(ga.get(k, []))} for k, v in order[:15]],
                             "n_abs_ge_0.05": sum(abs(v) >= 0.05 for v in sw.values())}
    print(f"[brg] swaps: {len(changed)} changed images, sum {sum(sw.values()):+.4f}, top {order[:5]}", flush=True)

    # pipeline limit
    lim = {"images_at_limit_frozen": iz.get("images_at_limit", P.at_limit(dz)),
           "images_at_limit_aligned": ia.get("images_at_limit", P.at_limit(da)),
           "at_limit_images": sorted(set(dz.groupby("image").size().loc[lambda s: s >= P.PIPE].index)
                                     | set(da.groupby("image").size().loc[lambda s: s >= P.PIPE].index))}
    dz10, _ = _dets(row, CACHE_ROOT, limit=10_000_000)
    da10, _ = _dets(row, MC.ROOT, limit=10_000_000)
    lim["mAP50_frozen_limit1e7"] = EV.map50(EV.build(gt, dz10, None, sizes=False))
    lim["mAP50_aligned_limit1e7"] = EV.map50(EV.build(gt, da10, None, sizes=False))
    lim["delta_limit1e7"] = lim["mAP50_aligned_limit1e7"] - lim["mAP50_frozen_limit1e7"]
    keep = [s for s in stems if s not in lim["at_limit_images"]]
    gk = EV.gt_arrays("train", keep)
    lim["delta_without_at_limit_images"] = (EV.map50(EV.build(gk, da[da.image.isin(keep)], None, sizes=False))
                                            - EV.map50(EV.build(gk, dz[dz.image.isin(keep)], None, sizes=False)))
    res["limit"] = lim
    print(f"[brg] limit: {lim}", flush=True)

    # ties and 11-point steps of the class with the largest change
    cls_name = top[0][0]
    c = DD.CLASS_ID[cls_name] if cls_name in DD.CLASS_ID else cls_name
    st = {}
    for tag, cache, d in (("frozen", cz, dz), ("aligned", ca, da)):
        tp, fp, npos = EV._curve(cache, (c, 0, -1), np.ones(len(cache.stems)))
        rec, prec = tp / npos, tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = 100 * EV.voc07(rec, prec)
        pts = {f"{t:.1f}": float(prec[rec >= t].max()) if (rec >= t).any() else 0.0 for t in np.arange(0, 1.1, 0.1)}
        sc = d[d.cls == c].score.to_numpy()
        st[tag] = {"AP50": ap, "max_recall": float(rec.max()) if len(rec) else 0.0, "precision_at_11_points": pts,
                   "n_dets_class": int(len(sc)), "n_tied_scores": int(len(sc) - len(np.unique(sc)))}
    res["largest_class"] = {"class": cls_name, **st,
                            "point_deltas": {k: st["aligned"]["precision_at_11_points"][k]
                                             - st["frozen"]["precision_at_11_points"][k]
                                             for k in st["frozen"]["precision_at_11_points"]}}
    print(f"[brg] class {cls_name}: {res['largest_class']['point_deltas']} max recall "
          f"{st['frozen']['max_recall']:.4f} -> {st['aligned']['max_recall']:.4f}", flush=True)
    (OUT / f"investigate_{rid}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
