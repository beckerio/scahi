# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact Python port of the official VisDrone2018-DET toolkit (MATLAB).

Reference: github.com/VisDrone/VisDrone2018-DET-toolkit @ 005445782213e20cb91bc50a597db3dd949e749a
(``evalDET.m``, ``utils/saveAnnoRes.m``, ``utils/dropObjectsInIgr.m``, ``utils/evalRes.m``,
``utils/compOas.m``, ``utils/calcAccuracy.m``, ``utils/VOCap.m``).

Toolkit semantics reproduced here (deliberately, including quirks):

* Ignore regions (category 0) are rasterised 1-based and *inclusive* (``y:y+h``), integral image
  lookups are off by one pixel exactly as in MATLAB; GT objects and detections whose rounded
  box is covered ``>= 0.5`` by ignore regions are removed. MATLAB ``round`` rounds half away
  from zero. Only category 0 drops boxes; 'others' (11) is simply not an evaluated class.
* GT score 0 becomes an *ignore GT* (IoF matching, may be matched repeatedly).
* ``maxDets`` keeps the first ``maxDets`` rows of the per-image detection list in *file order*
  across all classes (not per class as in pycocotools).
* AP: all-point interpolated VOC AP over IoU 0.50:0.05:0.95, computed with ``maxDets = 500``.
* Class averaging: ``mean2(AP(evalClass, :))`` where ``evalClass`` receives one entry per image
  containing the class, i.e. the class mean is weighted by the number of images per class.

The only generalisation: ``drop_regions`` may name class-specific ignore regions (used by the
``person_vehicle`` setting). With ``{None: (0,)}`` the behaviour is the toolkit's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

IOU_THRS = np.round(np.arange(0.5, 0.951, 0.05), 2)
MAX_DETS = (1, 10, 100, 500)
DROP_COVERAGE = 0.5


def matlab_round(x: NDArray) -> NDArray:
    """MATLAB ``round``: half away from zero."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# --------------------------------------------------------------------------------------------
# dropObjectsInIgr
# --------------------------------------------------------------------------------------------


def toolkit_ignore_map(regions_xywh: NDArray, height: int, width: int) -> NDArray[np.float64]:
    """``igrMap`` of dropObjectsInIgr.m (0-based array, same pixels as the 1-based original)."""
    m = np.zeros((height, width), np.float64)
    for x, y, w, h in np.maximum(1, np.asarray(regions_xywh, dtype=np.float64).reshape(-1, 4)):
        # MATLAB: igrMap(y : min(H, y+h), x : min(W, x+w)) = 1   (1-based, inclusive)
        y0, x0 = int(y) - 1, int(x) - 1
        y1, x1 = int(min(height, y + h)), int(min(width, x + w))
        if y1 > y0 and x1 > x0:
            m[y0:y1, x0:x1] = 1
    return m


def integral_image(m: NDArray) -> NDArray[np.float64]:
    """``createIntImg.m`` (inclusive 2-D cumulative sum)."""
    return np.cumsum(np.cumsum(m, axis=0), axis=1)


def toolkit_coverage(boxes_xywh: NDArray, int_img: NDArray, height: int, width: int) -> NDArray[np.float64]:
    """``igrVal / (h*w)`` of dropObjectsInIgr.m for each box (vectorised, 1-based lookups)."""
    b = np.asarray(boxes_xywh, dtype=np.float64).reshape(-1, 4)
    if len(b) == 0:
        return np.zeros(0)
    pos = np.maximum(1, matlab_round(b))
    x = np.clip(pos[:, 0], 1, width).astype(np.int64)
    y = np.clip(pos[:, 1], 1, height).astype(np.int64)
    w, h = pos[:, 2], pos[:, 3]
    xr = np.minimum(width, x + w).astype(np.int64)
    yb = np.maximum(1, np.minimum(height, y + h)).astype(np.int64)
    I = int_img  # noqa: E741
    tl, tr = I[y - 1, x - 1], I[y - 1, xr - 1]
    bl, br = I[yb - 1, x - 1], I[yb - 1, xr - 1]
    return (tl + br - tr - bl) / (h * w)


def drop_masks(
    gt_xywh: NDArray, gt_cls: NDArray, det_xywh: NDArray, det_cls: NDArray,
    regions: dict[int | None, NDArray], height: int, width: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Keep-masks for GT and detections.

    ``regions[None]`` are ignore regions for all classes, ``regions[c]`` additional regions for
    class ``c`` only (union with the all-class regions, same area rule).
    """
    keep_gt = np.ones(len(gt_cls), bool)
    keep_dt = np.ones(len(det_cls), bool)
    common = np.asarray(regions.get(None, np.zeros((0, 4)))).reshape(-1, 4)
    classes = set(np.unique(gt_cls).tolist()) | set(np.unique(det_cls).tolist())
    cache: dict[int | None, NDArray | None] = {}
    for c in classes:
        extra = np.asarray(regions.get(c, np.zeros((0, 4)))).reshape(-1, 4)
        key = c if len(extra) else None
        if key not in cache:
            reg = np.concatenate([common, extra]) if len(extra) else common
            cache[key] = integral_image(toolkit_ignore_map(reg, height, width)) if len(reg) else None
        I = cache[key]  # noqa: E741
        if I is None:
            continue
        g, d = gt_cls == c, det_cls == c
        keep_gt[g] = toolkit_coverage(gt_xywh[g], I, height, width) < DROP_COVERAGE
        keep_dt[d] = toolkit_coverage(det_xywh[d], I, height, width) < DROP_COVERAGE
    return keep_gt, keep_dt


# --------------------------------------------------------------------------------------------
# compOas / evalRes
# --------------------------------------------------------------------------------------------


def comp_oas(dt: NDArray, gt: NDArray, ig: NDArray) -> NDArray[np.float64]:
    """``compOas.m``: IoU, or intersection over detection area for ignore GTs."""
    dt = np.asarray(dt, np.float64).reshape(-1, 4)
    gt = np.asarray(gt, np.float64).reshape(-1, 4)
    if len(dt) == 0 or len(gt) == 0:
        return np.zeros((len(dt), len(gt)))
    w = np.minimum(dt[:, None, 0] + dt[:, None, 2], gt[None, :, 0] + gt[None, :, 2]) - np.maximum(dt[:, None, 0], gt[None, :, 0])
    h = np.minimum(dt[:, None, 1] + dt[:, None, 3], gt[None, :, 1] + gt[None, :, 3]) - np.maximum(dt[:, None, 1], gt[None, :, 1])
    valid = (w > 0) & (h > 0)
    t = np.where(valid, w * h, 0.0)
    da = (dt[:, 2] * dt[:, 3])[:, None]
    ga = (gt[:, 2] * gt[:, 3])[None, :]
    u = np.where(np.asarray(ig, bool)[None, :], da, da + ga - t)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(valid, t / u, 0.0)


def eval_res_sorted(oa: NDArray, gt_ignore: NDArray[np.bool_], thr: float) -> tuple[NDArray, NDArray]:
    """``evalRes.m`` on pre-sorted inputs (dt by score desc, gt non-ignore first).

    Returns ``(gt_match, dt_match)`` with values -1 (ignore), 0 (FN/FP), 1 (TP).
    """
    nd, ng = oa.shape
    gt_m = np.where(gt_ignore, -1, 0).astype(np.int8)
    dt_m = np.zeros(nd, np.int8)
    n_reg = int((~gt_ignore).sum())  # regular GTs come first after sorting
    free = np.ones(n_reg, bool)
    for d in range(nd):
        if n_reg:
            row = np.where(free, oa[d, :n_reg], -1.0)
            mx = row.max()
            if mx >= thr:
                g = n_reg - 1 - int(np.argmax(row[::-1] == mx))  # ties: the last one wins
                free[g] = False
                gt_m[g] = 1
                dt_m[d] = 1
                continue
        if ng > n_reg and (oa[d, n_reg:] >= thr).any():
            dt_m[d] = -1
    return gt_m, dt_m


def voc_ap(rec: NDArray, prec: NDArray) -> float:
    """``VOCap.m``."""
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.nonzero(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[i] - mrec[i - 1]) * mpre[i]))


# --------------------------------------------------------------------------------------------
# calcAccuracy
# --------------------------------------------------------------------------------------------


@dataclass
class ImageEval:
    """One image after dropping: GT ``[x,y,w,h]``, class, ignore flag; detections in file order."""

    gt_xywh: NDArray
    gt_cls: NDArray
    gt_ignore: NDArray
    dt_xywh: NDArray
    dt_score: NDArray
    dt_cls: NDArray


@dataclass
class ToolkitResult:
    AP: float
    AP50: float
    AP75: float
    AR1: float
    AR10: float
    AR100: float
    AR500: float
    ap_class_thr: NDArray  # [n_classes, 10]
    ar_class_thr_md: NDArray  # [n_classes, 10, 4]
    class_ids: tuple[int, ...]
    images_per_class: dict[int, int] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")}

    def ap_per_class(self) -> dict[int, float]:
        """Unweighted per-class AP@[.5:.95] (%)."""
        return {c: float(self.ap_class_thr[i].mean()) for i, c in enumerate(self.class_ids)}


def calc_accuracy(images: list[ImageEval], class_ids: tuple[int, ...]) -> ToolkitResult:
    """``calcAccuracy.m``. ``images`` must be in the toolkit's image order (sorted names)."""
    nC, nT, nM = len(class_ids), len(IOU_THRS), len(MAX_DETS)
    AP = np.zeros((nC, nT))
    AR = np.zeros((nC, nT, nM))
    images_per_class = {c: sum(int((im.gt_cls == c).any()) for im in images) for c in class_ids}

    for ci, c in enumerate(class_ids):
        # per image: sorted oa matrix and prefix lengths per maxDets
        per_img = []
        n_gt_total = 0
        for im in images:
            gsel = im.gt_cls == c
            g_ord = np.argsort(im.gt_ignore[gsel].astype(np.int8), kind="stable")
            g_xywh = im.gt_xywh[gsel][g_ord]
            g_ign = im.gt_ignore[gsel][g_ord].astype(bool)
            n_gt_total += len(g_xywh)
            prefixes = []
            for md in MAX_DETS:
                k = min(len(im.dt_cls), md)
                prefixes.append(np.nonzero(im.dt_cls[:k] == c)[0])
            per_img.append((g_xywh, g_ign, prefixes))

        for ti, thr in enumerate(IOU_THRS):
            for mi in range(nM):
                scores, matches = [], []
                for im, (g_xywh, g_ign, prefixes) in zip(images, per_img):
                    idx = prefixes[mi]
                    s = im.dt_score[idx]
                    o = np.argsort(-s, kind="stable")
                    oa = comp_oas(im.dt_xywh[idx][o], g_xywh, g_ign)
                    _, dm = eval_res_sorted(oa, g_ign, thr)
                    scores.append(s[o])
                    matches.append(dm)
                s_all = np.concatenate(scores) if scores else np.zeros(0)
                m_all = np.concatenate(matches) if matches else np.zeros(0, np.int8)
                rank = np.argsort(-s_all, kind="stable")
                tp = np.cumsum(m_all[rank] == 1)
                rec = tp / max(1, n_gt_total)
                if len(rec):
                    AR[ci, ti, mi] = rec.max() * 100
            fp = np.cumsum(m_all[rank] == 0)
            prec = tp / np.maximum(1, fp + tp)
            AP[ci, ti] = voc_ap(rec, prec) * 100

    weights = np.array([images_per_class[c] for c in class_ids], dtype=np.float64)
    if weights.sum() == 0:
        raise ValueError("No evaluable ground truth.")

    def wmean(x: NDArray) -> float:
        # mean2(X(evalClass, ...)) == image-count-weighted mean over classes
        x = x.reshape(nC, -1)
        return float((weights[:, None] * x).sum() / (weights.sum() * x.shape[1]))

    return ToolkitResult(
        AP=wmean(AP), AP50=wmean(AP[:, 0]), AP75=wmean(AP[:, 5]),
        AR1=wmean(AR[:, :, 0]), AR10=wmean(AR[:, :, 1]), AR100=wmean(AR[:, :, 2]), AR500=wmean(AR[:, :, 3]),
        ap_class_thr=AP, ar_class_thr_md=AR, class_ids=tuple(class_ids), images_per_class=images_per_class,
    )
