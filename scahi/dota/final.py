# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase D5: frozen rows (FREEZE_PROPOSAL_DOTA), replay on calib through the DOTA-val code path, DOTA-val run.

Rows per model m and training size s (docs/PREREGISTRATION_DOTA.md Section 3, addendum 1), every choice made on calib:

* ``A`` - single pass (o2o) at s, conf from the calib grid (runs/dota/fusion/<m>_<s>/single_calib.json)
* ``NMS-o2o`` / ``NMS-o2m`` - box fusion (IoS) of the stack dota<s>
* ``F2`` / ``RG`` - SCAHI (o2m) on the stack, pre-specified F2 grid / refined grid (time <= 2x the reference)
* ``B-F2`` / ``B-RG`` - SCAHI on the whole-image pass only (decomposition)
* ``X`` (s = 2048 only) - inference crossing of the 2048 weights, best of 2048 / 2560 / 3072 on calib (secondary)
* ``SAHI`` (1024 weights) - SAHI grid v2, selected merge / overlap / threshold / conf

``row_dets(row, split)`` is the one code path for calib (replay) and DOTA-val. The replay recomputes every row on calib
and requires the stored calib mAP50 exactly. Nothing here reads DOTA-val before the freeze (``run_val`` refuses
without an approved freeze file).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from scahi.dota import data as D
from scahi.dota import evaluate as EV
from scahi.dota import pipeline as P
from scahi.dota import train as T
from scahi.machine import REPO_ROOT

FROZEN = REPO_ROOT / "configs/dota/frozen_rows.json"
FREEZE_DIR = REPO_ROOT / "runs/dota/freeze"
F2_KEYS = ("iou_cluster", "tau_splat", "tau_peak", "cell_factor", "bins_per_oct")
CACHE_ROWS = {  # row -> (tuning stage, cache row, variant, branch)
    "NMS-o2o": ("NMS_o2o", "C", "F0-NMS", "o2o"), "NMS-o2m": ("NMS_o2m", "C", "F0-NMS", "o2m"),
    "F2": ("F2_o2m", "C", "F2", "o2m"), "RG": ("RGC_o2m", "C", "F2", "o2m"),
    "B-F2": ("B-F2_o2m", "B", "F2", "o2m"), "B-RG": ("RGB_o2m", "B", "F2", "o2m")}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def proposal_rows(kinds: tuple[str, ...] = ("single", "cache", "sahi"),
                  models: tuple[str, ...] = T.MODELS, sizes: tuple[int, ...] = (1024, 2048)) -> list[dict[str, Any]]:
    """All rows (of ``kinds``) with their calib choices (read from the calib artefacts; nothing re-selected).
    Single-head models (addendum 2): single pass and box fusion on their only head ("o2m"), no SAHI."""
    from scahi.common import rel

    rows = []
    for m in models:
        hb = T.heads(m)[0]
        for s in (x for x in sizes if x <= T.s_max(m)):
            w = T.weights(m, s)
            base = {"model": m, "s": s, "weights": rel(w), "weights_sha256": sha256(w)}
            if "single" in kinds:
                sc = json.loads((P.OUT / f"{m}_{s}/single_calib.json").read_text())
                rows.append({**base, "id": f"A_{m}_{s}", "row": "A", "kind": "single", "branch": hb, "imgsz": s,
                             "conf": sc[str(s)]["conf"], "calib_mAP50": sc[str(s)]["mAP50"],
                             "calib_images_at_limit": sc[str(s)]["images_at_limit"],
                             "source": f"runs/dota/fusion/{m}_{s}/single_calib.json (conf grid {list(P.CONF_GRID)})"})
                if s == 2048:
                    best = max(sc, key=lambda k: sc[k]["mAP50"])
                    rows.append({**base, "id": f"X_{m}_{s}", "row": "X", "kind": "single", "branch": hb,
                                 "imgsz": int(best), "conf": sc[best]["conf"], "calib_mAP50": sc[best]["mAP50"],
                                 "calib_images_at_limit": sc[best]["images_at_limit"],
                                 "secondary": True, "crossing_calib_mAP50": {k: v["mAP50"] for k, v in sc.items()},
                                 "source": f"runs/dota/fusion/{m}_{s}/single_calib.json (best of {sorted(sc)})"})
            if "cache" in kinds:
                sel = json.loads((P.OUT / f"{m}_{s}/tuning.json").read_text())["selected"]
                for r, (stage, crow, variant, br) in CACHE_ROWS.items():
                    if br not in T.heads(m):
                        continue
                    v = sel[stage]
                    cfg = {k: v[k] for k in (("metric", "thr", "conf") if variant == "F0-NMS" else F2_KEYS)}
                    rows.append({**base, "id": f"{r}_{m}_{s}", "row": r, "kind": "cache", "branch": br,
                                 "stack": crow, "variant": variant, "cfg": cfg, "calib_mAP50": v["mAP50"],
                                 "calib_mAP50_cap4000": v["mAP50_cap4000"],
                                 "calib_images_at_limit": v["images_at_limit"],
                                 **({"time_ratio_vs_ref": v["time_ratio_vs_ref"]} if "time_ratio_vs_ref" in v else {}),
                                 "source": f"runs/dota/fusion/{m}_{s}/tuning.json [{stage}]"})
        if "sahi" in kinds and m in T.MODELS:
            w = T.weights(m, 1024)
            st = json.loads((P.OUT / f"{m}_1024/sahi_tuning.json").read_text())["selected"]
            rows.append({"model": m, "s": 1024, "weights": rel(w), "weights_sha256": sha256(w), "id": f"SAHI_{m}",
                         "row": "SAHI", "kind": "sahi", "branch": "o2o",
                         "cfg": {k: st[k] for k in ("overlap", "merge", "metric", "match_thr", "conf")},
                         "calib_mAP50": st["mAP50"], "calib_mAP50_cap4000": st["mAP50_cap4000"],
                         "source": f"runs/dota/fusion/{m}_1024/sahi_tuning.json"})
    return rows


def row_dets(row: dict, split: str, limit: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Detections of one frozen row on ``split`` ('calib' or 'val'); the same path for replay and DOTA-val."""
    t0 = time.time()
    if row["kind"] == "single":
        src = "train" if split == "calib" else "val"
        dets, info = T.predict(T.weights(row["model"], row["s"]), src, P.split_stems(split), row["imgsz"], row["branch"],
                               max_det=limit or P.PIPE)
        dets = dets[dets.score >= row["conf"]]
    elif row["kind"] == "cache":
        data = P.load(row["model"], row["s"], split, row["branch"], row["stack"])
        dets, info = P.fuse(data, row["branch"], row["variant"], row["cfg"], limit=limit), {}
    else:
        import pickle

        c = row["cfg"]
        f = P.OUT / f"{row['model']}_1024/sahi_raw_{split}_o{int(c['overlap'] * 100):02d}.pkl"
        if not f.exists():
            f = P.sahi_raw(row["model"], split, c["overlap"])
        dets, info = P.sahi_merge(pickle.loads(f.read_bytes()), c["merge"], c["metric"], c["match_thr"], c["conf"]), {}
    info = {**info, "images_at_limit": P.at_limit(dets, limit), "wall_s": time.time() - t0}
    return dets, info


def replay(kinds: tuple[str, ...], out: Path, limit: int | None = None, models: tuple[str, ...] = T.MODELS) -> dict:
    """Recompute rows of the given kinds on calib; exact equality of mAP50 with the stored calib value required
    (``limit``: pipeline limit of the replay, default PIPE)."""
    import torch

    torch.set_num_threads(1)
    gt = EV.gt_arrays("train", T.split_stems("calib"))
    res = []
    for r in proposal_rows(kinds, models):
        dets, info = row_dets(r, "calib", limit)
        v = EV.map50(EV.build(gt, dets, None, sizes=False))
        res.append({"id": r["id"], "calib_mAP50": r["calib_mAP50"], "replay_mAP50": v,
                    "abs_diff": abs(v - r["calib_mAP50"]), "images_at_limit": info["images_at_limit"]})
        print(f"[dota] replay {r['id']}: {v:.4f} vs {r['calib_mAP50']:.4f}", flush=True)
    o = {"kinds": list(kinds), "limit": limit or P.PIPE, "rows": res, "exact": all(x["abs_diff"] == 0 for x in res),
         "max_abs_diff": max((x["abs_diff"] for x in res), default=0.0)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(o, indent=1))
    return o


def replay_cpu(limit: int | None = None, models: tuple[str, ...] = T.MODELS, tag: str = "") -> dict:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        os.sched_setaffinity(0, set(range(16, 24)))
        os.nice(19)
    except OSError:
        pass
    return replay(("cache", "sahi"), FREEZE_DIR / f"replay_cpu{_sfx(limit)}{tag}.json", limit, models)


def replay_gpu(limit: int | None = None, models: tuple[str, ...] = T.MODELS, tag: str = "") -> dict:
    return replay(("single",), FREEZE_DIR / f"replay_gpu{_sfx(limit)}{tag}.json", limit, models)


def _sfx(limit: int | None) -> str:
    return f"_limit{limit}" if limit else ""


# ------------------------------------------------------------------------------------------------ DOTA-val (once)
#
# Freeze approved 2026-10-30 (stop c, docs/DECISIONS.md): pipeline limit 200 000 for all rows, selection unchanged.
# Jobs: d_val_gpu (single-pass rows, SAHI raw), d_val_cpu (cache rows, SAHI merge), d_val_eval (metrics, paired
# bootstrap, Holm), d_val_latency (D-H3 rows, exclusive GPU / CPU, default power limit). Detections are written once per
# row (runs/dota/val/dets/<id>.parquet) and never recomputed; the evaluation reads only these files.

VAL_DIR = REPO_ROOT / "runs/dota/val"
DETS_DIR = VAL_DIR / "dets"
EPS = 0.30
HYP = {"D-H1": ("A", 2048, "NMS-o2o", 1024), "D-H2a": ("RG", 1024, "NMS-o2m", 1024),
       "D-H2b": ("RG", 2048, "NMS-o2m", 2048), "D-H3": ("RG", 2048, "A", 2048)}
SECONDARY = {  # preregistration section 4 (secondary, CI, no verdict) and addendum 1 item 4
    "F2 statt RG: D-H2a": ("F2", 1024, "NMS-o2m", 1024), "F2 statt RG: D-H2b": ("F2", 2048, "NMS-o2m", 2048),
    "F2 statt RG: D-H3": ("F2", 2048, "A", 2048),
    "D-H1, Box-Fusion o2m": ("A", 2048, "NMS-o2m", 1024), "D-H2a, Box-Fusion o2o": ("RG", 1024, "NMS-o2o", 1024),
    "D-H2b, Box-Fusion o2o": ("RG", 2048, "NMS-o2o", 2048),
    "Inferenz-Kreuzung RG_2048 − X_2048": ("RG", 2048, "X", 2048),
    **{f"Zerlegung {a}_{s} − {b}_{s}": (a, s, b, s) for s in (1024, 2048)
       for a, b in (("B-F2", "A"), ("F2", "B-F2"), ("B-RG", "A"), ("RG", "B-RG"))},
    "SAHI − A_1024": ("SAHI", 1024, "A", 1024), "SAHI − NMS-o2o_1024": ("SAHI", 1024, "NMS-o2o", 1024)}
VEC_METRICS = ("mAP50", "mAP50_cap4000", "mAP50_all_objects", "mAP50_vehicles", "AP", "R50_crowd", "R50_iso")


FROZEN_NEW = REPO_ROOT / "configs/dota/frozen_rows_new.json"  # addendum 2 (YOLO11s / YOLO12s), own approval


def frozen(path: Path = FROZEN) -> dict:
    fz = json.loads(path.read_text()) if path.exists() else {}
    if not fz.get("approved"):
        raise SystemExit(f"[dota] no approved freeze ({path.relative_to(REPO_ROOT)}) - DOTA-val is not touched")
    return fz


def approve(limit: int = 200_000) -> None:
    """Record the user's approval of stop (c) (2026-10-30): rows unchanged, pipeline limit ``limit`` for DOTA-val."""
    fz = json.loads(FROZEN.read_text())
    fz.update(approved=True, approved_at=time.strftime("%Y-%m-%dT%H:%M"), calib_selection_limit=P.PIPE,
              pipeline_limit=limit, approval="stop c approved by the user 2026-10-30 (option 1: pipeline limit "
                                             f"{limit} for all rows; selection frozen, no re-tuning)")
    FROZEN.write_text(json.dumps(fz, indent=1))


def approve_new(note: str) -> None:
    """Record the user's approval of the addendum-2 freeze (configs/dota/frozen_rows_new.json); rows unchanged."""
    fz = json.loads(FROZEN_NEW.read_text())
    fz.update(approved=True, approved_at=time.strftime("%Y-%m-%dT%H:%M"), calib_selection_limit=P.PIPE, approval=note)
    FROZEN_NEW.write_text(json.dumps(fz, indent=1))


def replay_baseline(kinds: tuple[str, ...]) -> dict:
    """Calib replay at the approved limit = the new replay baseline (docs/DECISIONS.md 2026-10-30, stop c)."""
    return replay(kinds, FREEZE_DIR / f"replay_baseline_{'gpu' if kinds == ('single',) else 'cpu'}.json",
                  frozen()["pipeline_limit"])


def _row_id(row: str, m: str, s: int) -> str:
    return f"SAHI_{m}" if row == "SAHI" else f"{row}_{m}_{s}"


def _cpu_slot() -> None:
    import os
    import subprocess

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        os.sched_setaffinity(0, set(range(16, 24)))
        os.nice(19)
        subprocess.run(["ionice", "-c3", "-p", str(os.getpid())], check=False)
    except OSError:
        pass


def _write_dets(rid: str, dets: pd.DataFrame, info: dict) -> None:
    DETS_DIR.mkdir(parents=True, exist_ok=True)
    dets.reset_index(drop=True).to_parquet(DETS_DIR / f"{rid}.parquet")
    (DETS_DIR / f"{rid}.json").write_text(json.dumps({**info, "n_dets": len(dets),
                                                      "max_dets_per_image": int(dets.groupby("image").size().max())
                                                      if len(dets) else 0}, indent=1))
    extra = (f" ({info['images_ge_20000_dets']} images with >= 20,000 detections, informational; no truncation)"
             if "images_ge_20000_dets" in info else "")
    print(f"[dota] val dets {rid}: {len(dets)} detections, {info['images_at_limit']} images at the limit{extra}", flush=True)


def val_gpu(path: Path = FROZEN) -> None:
    """GPU: single-pass rows (A; X = A when the crossing chose the training size) and SAHI raw predictions on DOTA-val."""
    import shutil

    fz = frozen(path)
    lim = fz["pipeline_limit"]
    rows = {r["id"]: r for r in fz["rows"]}
    for r in fz["rows"]:
        if r["kind"] == "single" and not (DETS_DIR / f"{r['id']}.parquet").exists():
            a = rows.get(f"A_{r['model']}_{r['s']}")
            if r["row"] == "X" and (r["imgsz"], r["conf"]) == (a["imgsz"], a["conf"]):
                (DETS_DIR / f"{a['id']}.parquet").exists() or _write_dets(a["id"], *row_dets(a, "val", lim))
                shutil.copy(DETS_DIR / f"{a['id']}.parquet", DETS_DIR / f"{r['id']}.parquet")
                info = json.loads((DETS_DIR / f"{a['id']}.json").read_text())
                (DETS_DIR / f"{r['id']}.json").write_text(json.dumps({**info, "same_as": a["id"]}, indent=1))
                continue
            _write_dets(r["id"], *row_dets(r, "val", lim))
        elif r["kind"] == "sahi":
            P.sahi_raw(r["model"], "val", r["cfg"]["overlap"], lim)


def _val_cpu_row(a: tuple[dict, int]) -> str:
    import torch

    r, lim = a
    torch.set_num_threads(1)
    if not (DETS_DIR / f"{r['id']}.parquet").exists():
        _write_dets(r["id"], *row_dets(r, "val", lim))
    return r["id"]


def val_cpu(workers: int = 7, path: Path = FROZEN) -> None:
    """CPU: every cache row (box fusion, SCAHI) and the SAHI merge on DOTA-val, from the val caches / SAHI raw pickles."""
    from concurrent.futures import ProcessPoolExecutor

    _cpu_slot()
    fz = frozen(path)
    rows = [r for r in fz["rows"] if r["kind"] in ("cache", "sahi")]
    missing = [r["id"] for r in rows if r["kind"] == "sahi" and not (
        P.OUT / f"{r['model']}_1024/sahi_raw_val_o{int(r['cfg']['overlap'] * 100):02d}.pkl").exists()]
    rows = [r for r in rows if r["id"] not in missing]
    rows.sort(key=lambda r: (r.get("stack") != "C", r["row"] not in ("RG", "F2")))  # expensive first
    with ProcessPoolExecutor(workers) as ex:
        for rid in ex.map(_val_cpu_row, [(r, fz["pipeline_limit"]) for r in rows]):
            print(f"[dota] val row {rid} done", flush=True)
    if missing:  # the runner's retry pass picks these up once d_val_gpu has written the SAHI raw pickles
        raise SystemExit(f"[dota] SAHI raw val predictions missing for {missing} (d_val_gpu first)")


def crowd_gt(gt: dict) -> dict[str, NDArray]:
    """Crowded GT (as VisDrone, scripts/ext_posthoc.py): same-class neighbour with IoU > 0.3 (all GT incl. difficult
    as neighbours); difficult GT are neither crowded nor isolated."""
    out = {}
    for s, (b, k, dif) in gt.items():
        cr = np.zeros(len(k), bool)
        for c in np.unique(k):
            ix = np.flatnonzero(k == c)
            if len(ix) < 2:
                continue
            g = b[ix].astype(np.float64)
            for j0 in range(0, len(ix), 2048):  # chunks: the densest image has 8609 GT
                q = g[j0:j0 + 2048]
                iw = np.clip(np.minimum(q[:, None, 2], g[None, :, 2]) - np.maximum(q[:, None, 0], g[None, :, 0]), 0, None)
                ih = np.clip(np.minimum(q[:, None, 3], g[None, :, 3]) - np.maximum(q[:, None, 1], g[None, :, 1]), 0, None)
                inter = iw * ih
                area = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
                iou = inter / np.maximum(area[j0:j0 + 2048, None] + area[None, :] - inter, 1e-12)
                iou[np.arange(len(q)), j0 + np.arange(len(q))] = 0
                cr[ix[j0:j0 + 2048]] = (iou > 0.3).any(1)
        out[s] = cr & ~dif
    return out


def _recall_counts(gt: dict, crowd: dict, dets: pd.DataFrame, group: str) -> tuple[NDArray, NDArray]:
    """Per image: matched GT of the group and GT of the group (devkit matching at IoU 0.5, all detections; GT outside
    the group are ignored like difficult GT)."""
    g2 = {s: (b, k, ~(crowd[s] if group == "crowd" else (~crowd[s] & ~dif))) for s, (b, k, dif) in gt.items()}
    c = EV.build(g2, dets, None, sizes=False, thrs=EV.IOU_THRS[:1])
    n = len(c.stems)
    tp, npos = np.zeros(n), np.zeros(n)
    for cl in c.classes:
        st, ix = c.dets[(cl, 0, -1)]
        tp += np.bincount(ix[st == 1], minlength=n)
        npos += c.npos[(cl, -1)]
    return tp, npos


# Vectorised paired bootstrap (user 2026-11-08; scahi.dota.boot_vec, bit-identical to one evaluation per resample);
# env SCAHI_BOOT_VEC=0 selects the per-resample path.
BOOT_VEC = os.environ.get("SCAHI_BOOT_VEC", "1") != "0"


def evaluate_row(rid: str, split: str = "val", dets: pd.DataFrame | None = None) -> tuple[dict, dict[str, NDArray]]:
    """Point metrics (no cap, cap 4000, all objects; size bins) and the bootstrap vectors of VEC_METRICS of one row."""
    from scahi.eval.bootstrap import resample_weights

    dets = pd.read_parquet(DETS_DIR / f"{rid}.parquet") if dets is None else dets
    src = "train" if split == "calib" else "val"
    stems = P.split_stems(split)
    W = resample_weights(len(stems))
    gt = EV.gt_arrays(src, stems)
    veh = tuple(D.CLASS_ID[c] for c in D.VEHICLES)
    out: dict[str, Any] = {"id": rid, "n_dets": len(dets)}
    vec: dict[str, NDArray] = {}
    if BOOT_VEC:  # same vectors (equality proof: scripts/dota/boot_vec_equality.py)
        from scahi.dota import boot_vec as BV

        m50 = lambda c, cl=None: BV.map50_b(c, W, cl)  # noqa: E731
        apc = lambda c: BV.ap_coco_b(c, W)  # noqa: E731
    else:
        m50 = lambda c, cl=None: np.array([EV.map50(c, w, cl) for w in W])  # noqa: E731
        apc = lambda c: np.array([EV.ap_coco(c, w) for w in W])  # noqa: E731
    c = EV.build(gt, dets, None, sizes=True)
    out["primary"] = EV.metrics(c)
    vec["mAP50"] = m50(c)
    vec["mAP50_vehicles"] = m50(c, veh)
    vec["AP"] = apc(c)
    del c
    c = EV.build(gt, dets, T.CAP, sizes=False, thrs=EV.IOU_THRS[:1])
    out["mAP50_cap4000"] = EV.map50(c)
    out["images_over_cap"] = int((dets.groupby("image").size() > T.CAP).sum()) if len(dets) else 0
    vec["mAP50_cap4000"] = m50(c)
    del c
    c = EV.build(EV.gt_arrays(src, stems, all_objects=True), dets, None, sizes=True)
    out["all_objects"] = EV.metrics(c)
    vec["mAP50_all_objects"] = m50(c)
    del c
    crowd = crowd_gt(gt)
    for g in ("crowd", "iso"):
        tp, npos = _recall_counts(gt, crowd, dets, g)
        out[f"R50_{g}"] = 100 * tp.sum() / max(npos.sum(), 1)
        out[f"n_{g}_gt"] = int(npos.sum())
        vec[f"R50_{g}"] = np.array([100 * (tp @ w) / max(npos @ w, 1) for w in W])
    for k in ("mAP50", "mAP50_vehicles", "AP"):
        out[k] = out["primary"][k]
    out["mAP50_all_objects"] = out["all_objects"]["mAP50"]
    return out, vec


def _eval_job(rid: str) -> str:
    import torch

    torch.set_num_threads(1)
    f = VAL_DIR / "eval" / f"{rid}.json"
    if not f.exists():
        o, v = evaluate_row(rid)
        np.savez(VAL_DIR / "eval" / f"{rid}_boot.npz", **v)
        f.write_text(json.dumps(o, indent=1))
    return rid


def _contrast(vec: dict, a: str, b: str, k: str, pt: dict) -> dict:
    from scahi.ext_verdicts import p_one_sided

    d = vec[a][k] - vec[b][k]
    delta = pt[a][k] - pt[b][k]
    return {"a": pt[a][k], "b": pt[b][k], "delta": delta, "ci95": [float(np.percentile(d, 2.5)),
                                                                   float(np.percentile(d, 97.5))],
            "p_one_sided": p_one_sided(d, EPS)}


def val_eval(workers: int = 7) -> dict:
    """CPU: evaluation of every frozen row on DOTA-val, contrasts with paired bootstrap (1000, seed 0), Holm."""
    from concurrent.futures import ProcessPoolExecutor

    from scahi.ext_verdicts import holm

    _cpu_slot()
    fz = frozen()
    ids = [r["id"] for r in fz["rows"]]
    missing = [i for i in ids if not (DETS_DIR / f"{i}.parquet").exists()]
    if missing:
        raise SystemExit(f"[dota] val detections missing: {missing}")
    (VAL_DIR / "eval").mkdir(parents=True, exist_ok=True)
    big = sorted(ids, key=lambda i: -json.loads((DETS_DIR / f"{i}.json").read_text())["n_dets"])
    with ProcessPoolExecutor(workers) as ex:
        for rid in ex.map(_eval_job, big):
            print(f"[dota] val eval {rid} done", flush=True)
    pt = {i: json.loads((VAL_DIR / "eval" / f"{i}.json").read_text()) for i in ids}
    vec = {i: dict(np.load(VAL_DIR / "eval" / f"{i}_boot.npz")) for i in ids}
    hyp, sec = {}, {}
    for h, (ra, sa, rb, sb) in HYP.items():
        per = {m: {k: _contrast(vec, _row_id(ra, m, sa), _row_id(rb, m, sb), k, pt) for k in VEC_METRICS}
               for m in T.MODELS}
        hl = holm({m: per[m]["mAP50"]["p_one_sided"] for m in T.MODELS}, m=len(T.MODELS))
        for m in T.MODELS:
            x = per[m]["mAP50"]
            crit = x["delta"] > EPS and (h != "D-H3" or x["ci95"][0] > 0)
            per[m]["holm"] = hl[m]
            per[m]["criterion"] = crit
            per[m]["met"] = bool(crit and hl[m]["p_holm"] <= 0.05)
        hyp[h] = {"contrast": f"{ra}_{sa} − {rb}_{sb}", "models": per, "holds": all(per[m]["met"] for m in T.MODELS)}
    for name, (ra, sa, rb, sb) in SECONDARY.items():
        sec[name] = {m: {k: _contrast(vec, _row_id(ra, m, sa), _row_id(rb, m, sb), k, pt) for k in VEC_METRICS}
                     for m in T.MODELS}
    res = {"split": "val", "n_images": len(P.split_stems("val")), "pipeline_limit": fz["pipeline_limit"],
           "eps": EPS, "bootstrap": {"n": 1000, "seed": 0, "ci": "percentile 2.5/97.5"},
           "rows": {i: {**pt[i], **json.loads((DETS_DIR / f"{i}.json").read_text())} for i in ids},
           "hypotheses": hyp, "secondary": sec, "frozen_code_commit": fz["code_commit"]}
    (VAL_DIR / "results.json").write_text(json.dumps(res, indent=1))
    return res


def hyp_rows(m: str) -> dict[str, tuple[str, int, str, int]]:
    """Hypotheses of one model (addendum 2): box fusion on the model's selecting head for D-H1, on o2m (= SCAHI's head)
    for D-H2; D-H3 at S_max."""
    nms1 = f"NMS-{T.heads(m)[0]}"
    sm = T.S_MAX[m]
    return {"D-H1": ("A", 2048, nms1, 1024), "D-H2a": ("RG", 1024, "NMS-o2m", 1024),
            "D-H2b": ("RG", 2048, "NMS-o2m", 2048), "D-H3": ("RG", sm, "A", sm)}


def val_eval_new(workers: int = 7) -> dict:
    """CPU (addendum 2): evaluation of the new frozen rows on DOTA-val; per hypothesis Holm over the two new models
    (not testable = not met, p = 1); verdict per model plus a descriptive count over all four models (the original
    verdicts are read from results.json, unchanged). -> runs/dota/val/results_new.json"""
    from concurrent.futures import ProcessPoolExecutor

    from scahi.ext_verdicts import holm

    _cpu_slot()
    fz = frozen(FROZEN_NEW)
    ids = [r["id"] for r in fz["rows"]]
    missing = [i for i in ids if not (DETS_DIR / f"{i}.parquet").exists()]
    if missing:
        raise SystemExit(f"[dota] val detections missing: {missing}")
    (VAL_DIR / "eval").mkdir(parents=True, exist_ok=True)
    big = sorted(ids, key=lambda i: -json.loads((DETS_DIR / f"{i}.json").read_text())["n_dets"])
    with ProcessPoolExecutor(workers) as ex:
        for rid in ex.map(_eval_job, big):
            print(f"[dota] val eval {rid} done", flush=True)
    pt = {i: json.loads((VAL_DIR / "eval" / f"{i}.json").read_text()) for i in ids}
    vec = {i: dict(np.load(VAL_DIR / "eval" / f"{i}_boot.npz")) for i in ids}
    old = json.loads((VAL_DIR / "results.json").read_text())["hypotheses"]
    hyp, sec = {}, {}
    for h in HYP:
        per = {}
        for m in T.NEW_MODELS:
            ra, sa, rb, sb = hyp_rows(m)[h]
            a, b = _row_id(ra, m, sa), _row_id(rb, m, sb)
            if a not in pt or b not in pt:
                per[m] = {"testable": False, "contrast": f"{a} − {b}", "mAP50": {"p_one_sided": None}}
                continue
            per[m] = {"testable": True, "contrast": f"{a} − {b}", **{k: _contrast(vec, a, b, k, pt) for k in VEC_METRICS}}
        hl = holm({m: per[m]["mAP50"]["p_one_sided"] for m in T.NEW_MODELS}, m=len(T.NEW_MODELS))
        for m in T.NEW_MODELS:
            x = per[m]["mAP50"]
            crit = per[m]["testable"] and x["delta"] > EPS and (h != "D-H3" or x["ci95"][0] > 0)
            per[m].update(holm=hl[m], criterion=bool(crit), met=bool(crit and hl[m]["p_holm"] <= 0.05))
        n_old = sum(bool(v["met"]) for v in old[h]["models"].values())
        hyp[h] = {"models": per, "met_new": sum(per[m]["met"] for m in T.NEW_MODELS),
                  "met_of_four_descriptive": n_old + sum(per[m]["met"] for m in T.NEW_MODELS),
                  "original_two_model_holds": old[h]["holds"]}  # unchanged, copied from results.json
    for name, (ra, sa, rb, sb) in SECONDARY.items():
        if "o2o" in name or "SAHI" in name:  # other-head contrasts and SAHI: not for single-head models (addendum 2)
            continue
        sec[name] = {m: {k: _contrast(vec, _row_id(ra, m, sa), _row_id(rb, m, sb), k, pt) for k in VEC_METRICS}
                     for m in T.NEW_MODELS if _row_id(ra, m, sa) in pt and _row_id(rb, m, sb) in pt}
    res = {"addendum": 2, "split": "val", "n_images": len(P.split_stems("val")), "pipeline_limit": fz["pipeline_limit"],
           "eps": EPS, "bootstrap": {"n": 1000, "seed": 0, "ci": "percentile 2.5/97.5"}, "s_max": dict(T.S_MAX),
           "rows": {i: {**pt[i], **json.loads((DETS_DIR / f"{i}.json").read_text())} for i in ids},
           "hypotheses": hyp, "secondary": sec, "frozen_code_commit": fz["code_commit"]}
    (VAL_DIR / "results_new.json").write_text(json.dumps(res, indent=1))
    return res


FROZEN_FCOS = REPO_ROOT / "configs/dota/frozen_rows_fcos.json"  # addendum 3 (FCOS), own approval


def val_eval_fcos(workers: int = 7) -> dict:
    """CPU (addendum 3 item 9): evaluation of the frozen FCOS rows on DOTA-val. One model -> no Holm: verdict with the
    unadjusted one-sided p <= 0.05 and the criterion (eps; D-H3 also lower CI bound > 0); descriptively the number of
    the five DOTA detectors meeting each hypothesis (the earlier verdicts read from results.json / results_new.json,
    unchanged). -> runs/dota/val/results_fcos.json"""
    from concurrent.futures import ProcessPoolExecutor

    _cpu_slot()
    m = T.FCOS
    fz = frozen(FROZEN_FCOS)
    ids = [r["id"] for r in fz["rows"]]
    missing = [i for i in ids if not (DETS_DIR / f"{i}.parquet").exists()]
    if missing:
        raise SystemExit(f"[dota] val detections missing: {missing}")
    (VAL_DIR / "eval").mkdir(parents=True, exist_ok=True)
    big = sorted(ids, key=lambda i: -json.loads((DETS_DIR / f"{i}.json").read_text())["n_dets"])
    with ProcessPoolExecutor(workers) as ex:
        for rid in ex.map(_eval_job, big):
            print(f"[dota] val eval {rid} done", flush=True)
    pt = {i: json.loads((VAL_DIR / "eval" / f"{i}.json").read_text()) for i in ids}
    vec = {i: dict(np.load(VAL_DIR / "eval" / f"{i}_boot.npz")) for i in ids}
    old = json.loads((VAL_DIR / "results.json").read_text())["hypotheses"]
    new = json.loads((VAL_DIR / "results_new.json").read_text())["hypotheses"]
    sm = T.s_max(m)
    rows_h = {"D-H1": ("A", 2048, f"NMS-{T.heads(m)[0]}", 1024), "D-H2a": ("RG", 1024, "NMS-o2m", 1024),
              "D-H2b": ("RG", 2048, "NMS-o2m", 2048), "D-H3": ("RG", sm, "A", sm)}
    hyp, sec = {}, {}
    for h, (ra, sa, rb, sb) in rows_h.items():
        a, b = _row_id(ra, m, sa), _row_id(rb, m, sb)
        x = {k: _contrast(vec, a, b, k, pt) for k in VEC_METRICS}
        crit = x["mAP50"]["delta"] > EPS and (h != "D-H3" or x["mAP50"]["ci95"][0] > 0)
        met = bool(crit and x["mAP50"]["p_one_sided"] <= 0.05)
        n_five = (sum(bool(v["met"]) for v in old[h]["models"].values())
                  + sum(bool(v["met"]) for v in new[h]["models"].values()) + met)
        hyp[h] = {"contrast": f"{a} − {b}", **x, "criterion": bool(crit), "met": met,
                  "met_of_five_descriptive": n_five}
    for name, (ra, sa, rb, sb) in SECONDARY.items():
        a, b = _row_id(ra, m, sa), _row_id(rb, m, sb)
        if "o2o" in name or "SAHI" in name or a not in pt or b not in pt:  # no o2o head, no SAHI row (addendum 3)
            continue
        sec[name] = {k: _contrast(vec, a, b, k, pt) for k in VEC_METRICS}
    res = {"addendum": 3, "model": m, "split": "val", "n_images": len(P.split_stems("val")),
           "pipeline_limit": fz["pipeline_limit"], "eps": EPS, "bootstrap": {"n": 1000, "seed": 0,
                                                                             "ci": "percentile 2.5/97.5"},
           "s_max": sm, "approved_at": fz.get("approved_at"), "approval": fz.get("approval"),
           "rows": {i: {**pt[i], **json.loads((DETS_DIR / f"{i}.json").read_text())} for i in ids},
           "hypotheses": hyp, "secondary": sec, "frozen_code_commit": fz["code_commit"]}
    (VAL_DIR / "results_fcos.json").write_text(json.dumps(res, indent=1))
    return res


def val_latency(n_warmup: int = 3) -> dict:
    """GPU (exclusive, default power limit): latency, peak memory and GFLOPs of the D-H3 rows on DOTA-val.

    A_2048 (= X_2048): Ultralytics predictor, fused weights, o2o, frozen conf. RG_2048: reference pipeline as VisDrone
    (unfused model, raw passes of dota2048 on the GPU, SCAHI on the CPU, scahi.fusion.pipeline.measure_split), timed
    as passes + fusion per image. Parity: the live pipeline reproduces the cached-path detections on 3 calib images."""
    import cv2
    import torch

    from scahi import flops as FL
    from scahi import machine as M
    from scahi.fusion import scalespace as ss
    from scahi.infer import cache_mz as cm
    from scahi.infer.raw_head import load_unfused

    M.require_default_power("d_val_latency")
    fz = frozen()
    lim = fz["pipeline_limit"]
    rows = {r["id"]: r for r in fz["rows"]}
    imgs = P.images("val")
    sizes = P._sizes()
    wh = [sizes[s] for s in imgs.stem]
    out: dict[str, Any] = {}

    def live(model, path, cfg):
        img = cv2.imread(path)
        H, W = img.shape[:2]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lv = []
        for k in cm.STACK_LEVELS[P.STACK[2048]]:
            z = cm.raw_predict_level(model, img, cm.LEVELS[k], P.OVERLAP)
            lv.append(ss.CachedLevel(k, z["o2m_xyxy"], z["o2m_probs"], z["o2m_std"], z["o2m_tile"], z["tiles"],
                                     z["gains"], cm.LEVELS[k].n_input is None, P.OVERLAP, ss.LevelCal()))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        from scahi.fusion import run as R

        d = R.clip_xywh(ss.fuse_image(lv, W, H, "F2", ss.FusionParams(**{**cfg, "top_k": lim}), nc=len(D.CLASSES)), W, H)
        return d, 1000 * (t1 - t0), 1000 * (time.perf_counter() - t1)

    for m in T.MODELS:
        a, rg = rows[f"A_{m}_2048"], rows[f"RG_{m}_2048"]
        w = T.weights(m, 2048)
        M.check_exclusive()
        _, info = T.predict(w, "val", list(imgs.stem), 2048, "o2o", conf=a["conf"], max_det=lim, measure_latency=True)
        M.check_exclusive()
        fc = FL.FlopCounter(w)
        out[a["id"]] = {**info, "gflops_per_image": FL.single_pass(fc, wh, 2048), "path": "Ultralytics predictor"}
        model = load_unfused(w)
        cal = P.load(m, 2048, "calib", "o2m", "C")
        par = []
        for st, p in list(zip(P.images("calib").stem, P.images("calib").path))[:3]:
            d_live = live(model, p, rg["cfg"])[0]
            d_cache = P.fuse({st: cal[st]}, "o2m", "F2", rg["cfg"], lim)
            par.append(bool(len(d_live) == len(d_cache) and np.allclose(
                d_live[["x", "y", "w", "h", "score"]].to_numpy(float),
                d_cache.assign(w=d_cache.x2 - d_cache.x1, h=d_cache.y2 - d_cache.y1)[["x1", "y1", "w", "h", "score"]]
                .to_numpy(float), atol=1e-3)))
        for p in imgs.path[:n_warmup]:
            live(model, p, rg["cfg"])
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for n, p in enumerate(imgs.path):
            if n % 100 == 0:
                M.check_exclusive()
            ts.append(live(model, p, rg["cfg"])[1:])
        M.check_exclusive()
        t = np.array(ts)
        out[rg["id"]] = {"latency_ms_mean": float(t.sum(1).mean()), "latency_ms_p95": float(np.percentile(t.sum(1), 95)),
                         "passes_ms_mean": float(t[:, 0].mean()), "fusion_ms_mean": float(t[:, 1].mean()),
                         "fusion_ms_p95": float(np.percentile(t[:, 1], 95)),
                         "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20, "n_images": len(ts),
                         "gflops_per_image": FL.stack(fc, wh, P.STACK[2048], P.OVERLAP),
                         "path": "reference pipeline (unfused model, GPU passes, CPU SCAHI)", "parity_calib_3": par}
        del model
        torch.cuda.empty_cache()
    out["validity"] = M.latency_validity()
    out["machine"] = M.machine_info()
    out["gflops_method"] = FL.describe()
    (VAL_DIR / "latency.json").write_text(json.dumps(out, indent=1))
    return out
