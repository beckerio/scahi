# SPDX-License-Identifier: AGPL-3.0-or-later
"""Paired bootstrap over images (AGENT_BRIEF Section 9, PREREGISTRATION): weighted re-evaluation without re-matching.

Both AP computations are separable per image: matching happens per image (toolkit: per class, IoU threshold and
maxDets; pycocotools: per category and area range), only the accumulation over images couples them. A bootstrap
resample is therefore a vector of image multiplicities ``w`` (counts from sampling images with replacement); the AP
of the resample is the AP of the per-image matches weighted by ``w``. With ``w = 1`` both functions reproduce
``visdrone_protocol.calc_accuracy`` (toolkit port) and pycocotools ``COCOeval.accumulate`` exactly
(``tests/test_bootstrap.py``). Duplicated images contribute their detections as one weighted PR step (standard
weighted-bootstrap approximation).
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from scahi.config import ClassSetting
from scahi.eval.coco_eval import EXTRA_RANGES, SIZE_EDGES, SIZE_LABELS
from scahi.eval.detections import finalize_detections
from scahi.eval.evaluate import MAX_DETS_PER_IMAGE, _to_coco_lists, prepare_images
from scahi.eval.visdrone_protocol import IOU_THRS, MAX_DETS, comp_oas, eval_res_sorted, voc_ap

N_BOOT, SEED = 1000, 0


# ------------------------------------------------------------------------------------------- toolkit port

@dataclass
class PortCache:
    class_ids: tuple[int, ...]
    n_img: int
    n_gt: NDArray  # [C, I] GT per class and image (incl. ignore GT, as calcAccuracy)
    has: NDArray  # [C, I] class present in image (images_per_class weights)
    dets: dict[tuple[int, int], tuple[NDArray, NDArray, NDArray]]  # (c, t) -> (score, match, image index), md 500


def port_cache(images: list[dict[str, Any]], class_ids: tuple[int, ...]) -> PortCache:
    """Per-image matches of the toolkit port (maxDets 500, as for AP in ``calc_accuracy``)."""
    C, I = len(class_ids), len(images)
    n_gt, has = np.zeros((C, I)), np.zeros((C, I))
    dets = {}
    md = MAX_DETS[-1]
    for ci, c in enumerate(class_ids):
        per = []
        for ii, im in enumerate(images):
            gsel = im["gt_cls"] == c
            g_ord = np.argsort(im["gt_ignore"][gsel].astype(np.int8), kind="stable")
            g_xywh, g_ign = im["gt_xywh"][gsel][g_ord], im["gt_ignore"][gsel][g_ord].astype(bool)
            n_gt[ci, ii], has[ci, ii] = len(g_xywh), float(gsel.any())
            k = min(len(im["dt_cls"]), md)
            idx = np.nonzero(im["dt_cls"][:k] == c)[0]
            s = im["dt_score"][idx]
            o = np.argsort(-s, kind="stable")
            per.append((ii, s[o], comp_oas(im["dt_xywh"][idx][o], g_xywh, g_ign), g_ign))
        for ti, thr in enumerate(IOU_THRS):
            sc, mt, ix = [], [], []
            for ii, s, oa, g_ign in per:
                _, dm = eval_res_sorted(oa, g_ign, thr)
                sc.append(s)
                mt.append(dm)
                ix.append(np.full(len(s), ii))
            s_all = np.concatenate(sc) if sc else np.zeros(0)
            rank = np.argsort(-s_all, kind="stable")
            dets[(ci, ti)] = (s_all[rank], np.concatenate(mt)[rank] if mt else np.zeros(0, np.int8),
                              np.concatenate(ix)[rank] if ix else np.zeros(0, int))
    return PortCache(tuple(class_ids), I, n_gt, has, dets)


def port_ap(pc: PortCache, w: NDArray | None = None, iou_index: int | None = None) -> float:
    """Toolkit-port AP@[.5:.95] (%) for image weights ``w`` (default 1); ``iou_index`` 0 = AP50, 5 = AP75."""
    w = np.ones(pc.n_img) if w is None else np.asarray(w, float)
    C = len(pc.class_ids)
    tis = range(len(IOU_THRS)) if iou_index is None else (iou_index,)
    T = len(tis)
    AP = np.zeros((C, T))
    for ci in range(C):
        ngt = float(pc.n_gt[ci] @ w)
        for k, ti in enumerate(tis):
            _, m, ix = pc.dets[(ci, ti)]
            wd = w[ix]
            tp = np.cumsum(wd * (m == 1))
            fp = np.cumsum(wd * (m == 0))
            rec = tp / max(1.0, ngt)
            prec = tp / np.maximum(1.0, fp + tp)
            AP[ci, k] = voc_ap(rec, prec) * 100 if len(rec) else 0.0
    weights = pc.has @ w
    return float((weights[:, None] * AP).sum() / (weights.sum() * T))


# ----------------------------------------------------------------------------------------- pycocotools

@dataclass
class CocoCache:
    n_img: int
    img_index: dict[int, int]
    eval_imgs: list  # COCOeval.evalImgs
    cat_ids: list[int]
    area_lbls: list[str]
    max_det: int
    rec_thrs: NDArray


def coco_cache(images: list[dict[str, Any]], coco: dict[str, Any], max_det: int = 500) -> CocoCache:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    imgs, anns, dts = _to_coco_lists(images, coco)
    gt = COCO()
    gt.dataset = {"images": imgs, "annotations": anns, "categories": coco["categories"]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(dts)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [max_det]
        ev.params.areaRng = ([[0, 1e10]] + [[lo**2, min(hi**2, 1e10)] for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:])]
                             + [[lo**2, hi**2] for lo, hi in EXTRA_RANGES.values()])
        ev.params.areaRngLbl = ["all", *SIZE_LABELS, *EXTRA_RANGES]
        ev.evaluate()
    p = ev.params
    return CocoCache(len(p.imgIds), {i: k for k, i in enumerate(p.imgIds)}, ev.evalImgs, list(p.catIds),
                     list(p.areaRngLbl), max_det, np.asarray(p.recThrs))


def coco_ap(cc: CocoCache, area: str = "all", w: NDArray | None = None, iou_index: int | None = None) -> float:
    """pycocotools AP (%) for one area range (``iou_index`` 0 = AP50, None = AP@[.5:.95]), image weights ``w``."""
    w = np.ones(cc.n_img) if w is None else np.asarray(w, float)
    A, I = len(cc.area_lbls), cc.n_img
    a = cc.area_lbls.index(area)
    R = len(cc.rec_thrs)
    out = []
    for k in range(len(cc.cat_ids)):
        es = [cc.eval_imgs[k * A * I + a * I + i] for i in range(I)]
        es = [(e, w[cc.img_index[e["image_id"]]]) for e in es if e is not None]
        if not es:
            continue
        sc = np.concatenate([e["dtScores"][:cc.max_det] for e, _ in es])
        wd = np.concatenate([np.full(len(e["dtScores"][:cc.max_det]), ww) for e, ww in es])
        order = np.argsort(-sc, kind="mergesort")
        dtm = np.concatenate([e["dtMatches"][:, :cc.max_det] for e, _ in es], axis=1)[:, order]
        dtig = np.concatenate([e["dtIgnore"][:, :cc.max_det] for e, _ in es], axis=1)[:, order]
        npig = sum(ww * np.count_nonzero(e["gtIgnore"] == 0) for e, ww in es)
        if npig == 0:
            continue
        wo = wd[order]
        tps = np.logical_and(dtm, np.logical_not(dtig)) * wo
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtig)) * wo
        prec_t = []
        for t, (tp, fp) in enumerate(zip(np.cumsum(tps, 1), np.cumsum(fps, 1))):
            if iou_index is not None and t != iou_index:
                continue
            rc = tp / npig
            pr = (tp / (fp + tp + np.spacing(1))).tolist()
            for i in range(len(pr) - 1, 0, -1):
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]
            q = np.zeros(R)
            inds = np.searchsorted(rc, cc.rec_thrs, side="left")
            for ri, pi in enumerate(inds):
                if pi < len(pr):
                    q[ri] = pr[pi]
            prec_t.append(q)
        out.append(np.mean(prec_t))
    return float(np.mean(out) * 100) if out else float("nan")


# --------------------------------------------------------------------------------------------- bootstrap

def prepared(coco: dict[str, Any], dets: pd.DataFrame, cs: ClassSetting, protocol: str = "visdrone_ignore") -> list[dict]:
    return prepare_images(coco, finalize_detections(dets, MAX_DETS_PER_IMAGE), cs, protocol)


def resample_weights(n_img: int, n: int = N_BOOT, seed: int = SEED) -> NDArray:
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_img, n_img), minlength=n_img) for _ in range(n)]).astype(float)


def paired(metric_a: Callable[[NDArray | None], float], metric_b: Callable[[NDArray | None], float], n_img: int,
           n: int = N_BOOT, seed: int = SEED) -> dict[str, Any]:
    """Delta = metric_a - metric_b with a 95 % percentile CI over ``n`` paired image resamples."""
    W = resample_weights(n_img, n, seed)
    d = np.array([metric_a(w) - metric_b(w) for w in W])
    a0, b0 = metric_a(None), metric_b(None)
    return {"a": a0, "b": b0, "delta": a0 - b0, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "n_boot": n, "seed": seed, "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}
