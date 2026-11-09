# SPDX-License-Identifier: AGPL-3.0-or-later
"""Port of the official DOTA_devkit HBB evaluation (task 2) with paired-bootstrap support (docs/DOTA_BRIEF.md §2.3).

Reference: ``third_party/DOTA_devkit/dota_evaluation_task2.py`` (CAPTAIN-WHU/DOTA_devkit @ 9938855, unchanged copy).
Exactly as there, per class:
  - GT box = corner 1 and corner 3 of the label polygon, each truncated to int (``data.devkit_box``);
  - detections of the class are ordered by ``np.argsort(-confidence)`` over the "file order" (images in the order of
    the image list, within an image the order of the input), and matched in that order;
  - IoU with the +1 pixel convention; a detection is matched to the GT of maximal IoU (first on ties) if that IoU is
    **strictly** greater than the threshold: difficult GT -> neither TP nor FP; already matched -> FP; else TP;
    IoU <= threshold or no GT of the class -> FP;
  - recall base npos = non-difficult GT; precision tp / max(tp + fp, eps); AP = VOC07 11-point (``use_07_metric=True``
    as in the devkit's main).
Primary metric: mean of the per-class AP at IoU 0.5 over the 16 v1.5 classes (the devkit lists the 15 v1.0 classes;
v1.5 adds container-crane), in percent. A class without non-difficult GT on the evaluated image set has no AP (the
devkit would divide by zero) and is left out of the mean; this never happens on DOTA-val.

Further metrics, all on the same detections and the same matching rule (pre-specified, docs/PREREGISTRATION_DOTA.md):
  - AP@[.5:.95]: thresholds 0.50:0.05:0.95, COCO 101-point interpolation (pycocotools ``accumulate``), mean over
    thresholds and classes ("COCO style" for comparability with VisDrone);
  - vehicles: the same two metrics over small vehicle + large vehicle;
  - class-agnostic AP50 (all classes pooled, VOC07);
  - size bins on sigma = sqrt(w h) (VisDrone bins): GT outside the bin are ignored like difficult GT, unmatched
    detections outside the bin are ignored (COCO area-range rule); AP (101-point, 0.50:0.95) and AP50 (VOC07).
Detection cap: the top ``cap`` detections per image by score (over all classes) before everything else; ``None`` =
no cap. Addendum 2026-10-30: the primary metric is the official one **without** cap; the cap of 4000 is secondary.
"All objects" analysis: ``gt_arrays(..., all_objects=True)`` (difficult GT counted as normal GT). Bootstrap: matching is per image, so a resample is a vector of image multiplicities ``w`` applied to
the cumulative TP/FP sums and to npos (as scahi.eval.bootstrap).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numba
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from scahi.dota import data as D

IOU_THRS = np.linspace(0.5, 0.95, 10)
SIZE_EDGES = (0, 8, 16, 32, 64, 128, np.inf)
SIZE_LABELS = ("<8", "8-16", "16-32", "32-64", "64-128", ">128")
REC_THRS = np.linspace(0.0, 1.00, 101)
AGNOSTIC = -1  # pseudo class index of the class-agnostic evaluation
DET_COLS = ("image", "cls", "score", "x1", "y1", "x2", "y2")


@numba.njit(cache=True)
def _match(order, d_img, d_box, g_start, g_end, g_box, g_ign, thr):
    """Devkit matching in the given global order. Returns status per detection (in input index): 1 TP, 0 FP, -1 none."""
    st = np.zeros(len(d_img), np.int8)
    used = np.zeros(len(g_box), np.bool_)
    for k in range(len(order)):
        d = order[k]
        i = d_img[d]
        a, b = g_start[i], g_end[i]
        if b == a:
            continue  # FP (status 0)
        bx0, by0, bx1, by1 = d_box[d, 0], d_box[d, 1], d_box[d, 2], d_box[d, 3]
        ovmax, jmax = -np.inf, -1
        for j in range(a, b):
            iw = min(g_box[j, 2], bx1) - max(g_box[j, 0], bx0) + 1.0
            ih = min(g_box[j, 3], by1) - max(g_box[j, 1], by0) + 1.0
            iw = max(iw, 0.0)
            ih = max(ih, 0.0)
            inters = iw * ih
            uni = ((bx1 - bx0 + 1.0) * (by1 - by0 + 1.0)
                   + (g_box[j, 2] - g_box[j, 0] + 1.0) * (g_box[j, 3] - g_box[j, 1] + 1.0) - inters)
            o = inters / uni
            if o > ovmax:  # first maximum, as np.argmax
                ovmax, jmax = o, j
        if ovmax > thr:
            if g_ign[jmax]:
                st[d] = -1
            elif not used[jmax]:
                st[d] = 1
                used[jmax] = True
    return st


def voc07(rec: NDArray, prec: NDArray) -> float:
    """dota_evaluation_task2.voc_ap(rec, prec, use_07_metric=True), verbatim."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = 0 if np.sum(rec >= t) == 0 else np.max(prec[rec >= t])
        ap = ap + p / 11.0
    return float(ap)


def coco101(rec: NDArray, prec: NDArray) -> float:
    """pycocotools COCOeval.accumulate for one class / threshold: precision envelope, 101 recall points."""
    pr = prec.tolist()
    for i in range(len(pr) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]
    inds = np.searchsorted(rec, REC_THRS, side="left")
    q = [pr[pi] if pi < len(pr) else 0.0 for pi in inds]
    return float(np.mean(q))


@dataclass
class DotaCache:
    stems: list[str]
    cap: int | None
    # (class index or AGNOSTIC, thr index, bin index or -1 = all) -> (status in global order, image index)
    dets: dict = field(default_factory=dict)
    npos: dict = field(default_factory=dict)  # (class, bin) -> [I] non-ignored GT per image
    classes: tuple = ()


def gt_arrays(split: str, stems: list[str], all_objects: bool = False) -> dict[str, tuple[NDArray, NDArray, NDArray]]:
    """GT per image in devkit form. ``all_objects``: pre-specified secondary analysis (addendum 2026-10-30) in which
    "difficult" objects count as normal ground truth (difficult flag cleared; everything else unchanged)."""
    anns = D.load(split)
    return {s: (D.devkit_box(anns[s].poly), anns[s].cls,
                np.zeros(len(anns[s].cls), bool) if all_objects else anns[s].difficult) for s in stems}


def apply_cap(dets: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    """Top ``cap`` detections per image by score (stable; ties keep input order). Input order is preserved."""
    if cap is None:
        return dets
    rank = dets.groupby("image", sort=False)["score"].rank(method="first", ascending=False)
    return dets[rank.to_numpy() <= cap]


def _sigma(b: NDArray) -> NDArray:
    return np.sqrt(np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0))


# Memory-lean build (user 2026-11-07, FCOS box-fusion rows with up to 184.6 M detections): the image index of the
# detections in score order is computed once per class and shared by every (threshold, size bin) key instead of one
# int64 copy per key; values and order unchanged (equality proof: scripts/dota/eval_lean_equality.py). False = the
# earlier path (one copy per key), kept for the proof; env SCAHI_EVAL_LEAN=0 selects it.
LEAN = os.environ.get("SCAHI_EVAL_LEAN", "1") != "0"


def build(gt: dict[str, tuple[NDArray, NDArray, NDArray]], dets: pd.DataFrame, cap: int | None = None,
          sizes: bool = True, thrs: NDArray = IOU_THRS, lean: bool | None = None) -> DotaCache:
    """Match once (all classes, thresholds and, with ``sizes``, size bins). ``gt``: stem -> (devkit boxes, class,
    difficult), in image-list order; ``dets``: columns DET_COLS, ``image`` = stem, ``cls`` = class index.
    ``lean`` (default ``LEAN``): share the image index array between the keys of a class (read-only)."""
    lean = LEAN if lean is None else lean
    stems = list(gt)
    idx = {s: i for i, s in enumerate(stems)}
    d = apply_cap(dets, cap)
    d = d[d["image"].isin(idx)]
    # file order: images in list order, within an image the input order
    d = d.assign(_i=d["image"].map(idx).to_numpy(), _o=np.arange(len(d))).sort_values(["_i", "_o"], kind="stable")
    cache = DotaCache(stems, cap, classes=tuple(range(len(D.CLASSES))))
    bins = [(-1, None)] + ([(k, (lo, hi)) for k, (lo, hi) in enumerate(zip(SIZE_EDGES[:-1], SIZE_EDGES[1:]))]
                           if sizes else [])
    for c in (*cache.classes, AGNOSTIC):
        dc = d if c == AGNOSTIC else d[d["cls"].to_numpy() == c]
        d_img = dc["_i"].to_numpy(np.int64)
        d_box = dc[["x1", "y1", "x2", "y2"]].to_numpy(np.float64)
        conf = dc["score"].to_numpy(np.float64)
        order = np.argsort(-conf)  # as the devkit (default kind), over the file order
        ix_o = d_img[order] if lean else None
        gb, gi, gs, ge, gsig = [], [], np.zeros(len(stems), np.int64), np.zeros(len(stems), np.int64), []
        n = 0
        for i, s in enumerate(stems):
            b, k, dif = gt[s]
            m = np.ones(len(k), bool) if c == AGNOSTIC else (k == c)
            gb.append(b[m])
            gi.append(dif[m])
            gs[i], n = n, n + int(m.sum())
            ge[i] = n
        g_box = np.concatenate(gb).astype(np.float64) if gb else np.zeros((0, 4))
        g_dif = np.concatenate(gi).astype(bool) if gi else np.zeros(0, bool)
        g_sig, d_sig = _sigma(g_box), _sigma(d_box)
        img_of_gt = np.repeat(np.arange(len(stems)), ge - gs)
        for bk, rng in bins:
            if c == AGNOSTIC and bk >= 0:
                continue
            ign = g_dif if rng is None else (g_dif | (g_sig < rng[0]) | (g_sig >= rng[1]))
            cache.npos[(c, bk)] = np.bincount(img_of_gt[~ign], minlength=len(stems)).astype(np.float64)
            tlist = range(len(thrs)) if c != AGNOSTIC else (0,)
            for ti in tlist:
                st = _match(order, d_img, d_box, gs, ge, g_box, ign, float(thrs[ti]))
                if rng is not None:
                    st = np.where((st == 0) & ((d_sig < rng[0]) | (d_sig >= rng[1])), -1, st).astype(np.int8)
                cache.dets[(c, ti, bk)] = (st[order], ix_o if lean else d_img[order])
    return cache


def _curve(cache: DotaCache, key: tuple, w: NDArray) -> tuple[NDArray, NDArray, float]:
    st, ix = cache.dets[key]
    wd = w[ix]
    tp = np.cumsum(wd * (st == 1))
    fp = np.cumsum(wd * (st == 0))
    npos = float(cache.npos[(key[0], key[2])] @ w)
    return tp, fp, npos


def class_ap(cache: DotaCache, c: int, ti: int = 0, bk: int = -1, w: NDArray | None = None,
             interp: str = "voc07") -> float | None:
    """AP (fraction, not %) of one class; None if the class has no non-ignored GT (left out of means)."""
    w = np.ones(len(cache.stems)) if w is None else np.asarray(w, float)
    tp, fp, npos = _curve(cache, (c, ti, bk), w)
    if npos <= 0:
        return None
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    if interp == "voc07":
        return voc07(rec, prec)
    return coco101(rec, prec) if len(rec) else 0.0


def _mean(vals: list[float | None]) -> float:
    v = [x for x in vals if x is not None]
    return float(np.mean(v) * 100) if v else float("nan")


def map50(cache: DotaCache, w: NDArray | None = None, classes: tuple[int, ...] | None = None) -> float:
    """Primary metric: official all-class mAP at IoU 0.5 (VOC07), percent."""
    return _mean([class_ap(cache, c, 0, -1, w) for c in (classes or cache.classes)])


def ap_coco(cache: DotaCache, w: NDArray | None = None, classes: tuple[int, ...] | None = None, bk: int = -1) -> float:
    """AP@[.5:.95], COCO 101-point interpolation, mean over thresholds and classes (classes without GT left out)."""
    vals = []
    for c in (classes or cache.classes):
        a = [class_ap(cache, c, ti, bk, w, "coco101") for ti in range(len(IOU_THRS))]
        vals.append(None if a[0] is None else float(np.mean(a)))
    return _mean(vals)


def metrics(cache: DotaCache, w: NDArray | None = None) -> dict:
    veh = tuple(D.CLASS_ID[c] for c in D.VEHICLES)
    out = {"mAP50": map50(cache, w), "AP": ap_coco(cache, w),
           "mAP50_vehicles": map50(cache, w, veh), "AP_vehicles": ap_coco(cache, w, veh),
           "AP50_agnostic": _mean([class_ap(cache, AGNOSTIC, 0, -1, w)]),
           "AP50_per_class": {D.CLASSES[c]: (None if (a := class_ap(cache, c, 0, -1, w)) is None else 100 * a)
                              for c in cache.classes},
           "cap": cache.cap}
    if (0, 0, 0) in cache.dets:
        out["AP_by_size"] = {lbl: ap_coco(cache, w, bk=k) for k, lbl in enumerate(SIZE_LABELS)}
        out["AP50_by_size"] = {lbl: _mean([class_ap(cache, c, 0, k, w) for c in cache.classes])
                               for k, lbl in enumerate(SIZE_LABELS)}
    return out


def evaluate(split: str, stems: list[str], dets: pd.DataFrame, cap: int | None = None, sizes: bool = True,
             all_objects: bool = False) -> dict:
    return metrics(build(gt_arrays(split, stems, all_objects), dets, cap, sizes))
