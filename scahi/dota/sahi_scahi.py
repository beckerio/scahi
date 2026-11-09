# SPDX-License-Identifier: AGPL-3.0-or-later
"""Addendum 4 (D-H5): SCAHI as a drop-in replacement for SAHI's merge on DOTA, on exactly the passes of the frozen SAHI
rows (YOLOv10s / YOLOv10m, 1024 weights): SAHI's 640 px slices (``sahi.slicing.get_slice_bboxes``, the frozen overlap)
plus SAHI's standard pass (whole image, long side 640).

Cache levels ``w640f`` (standard pass) and ``s640f`` (slices) under ``cache/<weights_id>/<key>_o<overlap%>/<split>/``,
written with ``scahi.infer.cache_mz.raw_predict_level`` (both heads, as every cache) on the **fused** model of SAHI's own
predictor path (``load_fused``: conv + BN fused exactly as the Ultralytics predictor behind the frozen SAHI rows, only the
removal of the o2m branch skipped; TF32 as there), boxes mapped back with Ultralytics' per-axis gains (``scale_boxes``).
SAHI's slice grid and that mapping are put in for this process only (the shared modules stay untouched while the FCOS
tuning runs). Each cache directory records its model state in ``cache_state.json``. The first build (``s640`` / ``w640``,
unfused model, single gain) failed the gate for YOLOv10m and is kept only as a record.

Parity gate (addendum 4, item 4 as corrected before any gate result: the rule of ``pipeline.gate``): (1) at both new
input geometries (640 px slice, whole image at long side 640; 3 calib images each, TF32 off) own decoding vs. Ultralytics'
decoding of the same forward, per coordinate <= max(1e-4 px, 4 float32 ulp), probabilities < 1e-4, and the o2o pass
detections as the cache path forms them (``o2o_pass_dets`` on the cached anchors) vs. Ultralytics' end-to-end output;
(2) calib mAP50 (primary, no cap) of the frozen SAHI configuration merged from the cache passes vs. the frozen
``calib_mAP50``, |Δ| <= 0.01. Descriptive, no verdict: per image on calib and DOTA-val, the cache pass detections vs. the
stored raw SAHI predictions (counts, matched deviations; the stored ones come from Ultralytics' fused model with TF32,
the caches from the same fused model; no metric on DOTA-val)
-> runs/dota/fusion/<model>_1024/sahi_scahi_gate.json.
"""

from __future__ import annotations

import contextlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scahi.dota import evaluate as EV
from scahi.dota import pipeline as P
from scahi.dota import train as T
from scahi.infer.multizoom import Level

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "configs/dota/frozen_rows.json"
MODELS = ("yolov10s", "yolov10m")
# Addendum 5 (D-H6, user 2026-11-03): single-head detectors (one dense head, branch "o2m" internally); SAHI tuned on
# calib with the procedure of the frozen YOLOv10 SAHI rows, its pass postprocess is Ultralytics' NMS (not o2o)
A5_MODELS = ("yolo11s", "yolo12s")
A5_IOU = 0.7  # Ultralytics predictor default (SAHI does not pass iou)
SLICE = 640
LEVELS = {"s640f": Level(1.0, SLICE), "w640f": Level(SLICE / 1920, None, SLICE)}
ORDER = ("s640f", "w640f")  # SAHI's order: slices, then the standard pass
CACHE_STATE = {"model_state": "fused (conv+BN as the Ultralytics predictor of the frozen SAHI rows, via SAHI's "
                              "load path; Detect.fuse skipped so both heads stay)", "tf32": "PyTorch default (on)",
               "box_mapping": "Ultralytics scale_boxes (per-axis gains, float32)",
               "whole_pass_shape": "Ultralytics LetterBox (round, then pad to 32; resize by r = min(640/h, 640/w))",
               "batch": "1 pass per forward, as SAHI (batch_size 1; cuDNN picks other TF32 kernels for batches)"}
GATE_MAP_TOL = 0.01
CACHE_STATE_SINGLE = "fused (conv+BN as the Ultralytics predictor of SAHI's load path; single dense head)"


def frozen_sahi(model: str) -> dict:
    """The SAHI configuration whose passes the SCAHI rows use: frozen (YOLOv10) or the calib selection of addendum 5."""
    if model in A5_MODELS:
        s = json.loads((P.OUT / f"{model}_1024" / "sahi_tuning.json").read_text())["selected"]
        return {"id": f"SAHI_{model}", "calib_mAP50": s["mAP50"],
                "cfg": {k: s[k] for k in ("merge", "metric", "match_thr", "conf", "overlap")}}
    return next(r for r in json.loads(FROZEN.read_text())["rows"] if r["id"] == f"SAHI_{model}")


def overlap(model: str) -> float:
    return float(frozen_sahi(model)["cfg"]["overlap"])


def sahi_raw_single(model: str, split: str, ov: float) -> Path:
    """GPU, addendum 5: ``pipeline.sahi_raw`` for a single-head detector (the predictor keeps Ultralytics' NMS; YOLOv10's
    o2o switch is not applied) -> the same pickle {stem: [n, 6]} under runs/dota/fusion/<model>_1024/."""
    from sahi.predict import get_sliced_prediction

    from scahi.infer import sahi_baseline as sb

    d = P.OUT / f"{model}_1024"
    f = d / f"sahi_raw_{split}_o{int(ov * 100):02d}.pkl"
    if f.exists():
        return f
    sb._register_identity()
    dm = sb.load_model(T.weights(model, 1024), min(sb.GRID2_WBF_SKIP), "o2m")
    dm.model.overrides["max_det"] = P.PIPE
    raw = {}
    t0 = time.time()
    for stem, path in zip(P.images(split).stem, P.images(split).path):
        r = get_sliced_prediction(path, dm, slice_height=sb.SLICE, slice_width=sb.SLICE, overlap_height_ratio=ov,
                                  overlap_width_ratio=ov, perform_standard_pred=True, postprocess_type="IDENTITY",
                                  postprocess_match_threshold=0.5, postprocess_match_metric="IOS",
                                  force_postprocess_type=True, verbose=0)
        raw[stem] = np.array([[*p.bbox.to_xyxy(), p.score.value, p.category.id] for p in r.object_prediction_list],
                             np.float64).reshape(-1, 6)
    f.write_bytes(pickle.dumps(raw))
    print(f"[a5] SAHI raw {model} {split} overlap {ov}: {time.time() - t0:.0f} s, "
          f"max {max(len(v) for v in raw.values())} raw predictions per image", flush=True)
    return f


def sahi_passes(level: Level, W: int, H: int, ov: float) -> list:
    """SAHI's slices (as ``get_sliced_prediction``), input shapes as Ultralytics' LetterBox (``whole_pass``)."""
    from sahi.slicing import get_slice_bboxes

    out = []
    for x1, y1, x2, y2 in get_slice_bboxes(H, W, SLICE, SLICE, False, ov, ov):
        out.append(((x1, y1, x2, y2), whole_pass(x2 - x1, y2 - y1)[0][1]))
    return out


@contextlib.contextmanager
def _sahi_grid():
    from scahi.infer import cache_mz as cm

    orig, orig_map, orig_lb = cm.level_passes, cm.input_to_source, cm.letterbox
    cm.level_passes = lambda lv, W, H, ov: (sahi_passes(lv, W, H, ov) if lv is LEVELS["s640f"] else
                                            whole_pass(W, H) if lv is LEVELS["w640f"] else orig(lv, W, H, ov))
    cm.input_to_source, cm.letterbox = to_source_ultralytics, letterbox_ultralytics
    try:
        yield
    finally:
        cm.level_passes, cm.input_to_source, cm.letterbox = orig, orig_map, orig_lb


def letterbox_ultralytics(img: np.ndarray, out_hw: tuple[int, int]):
    """Ultralytics' LetterBox at imgsz 640 (auto): resize by r = min(640 / h, 640 / w) - not by the gain to the padded
    ``out_hw`` as ``preprocess.letterbox`` (1 px row less for e.g. 846 x 635) - then centred padding to ``out_hw``."""
    import cv2

    from scahi.infer.preprocess import PAD_VALUE, Letterbox

    h, w = img.shape[:2]
    r = min(SLICE / h, SLICE / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else img
    top, left = round((out_hw[0] - nh) / 2 - 0.1), round((out_hw[1] - nw) / 2 - 0.1)
    canvas = np.full((out_hw[0], out_hw[1], 3), PAD_VALUE, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, Letterbox(r, out_hw, (h, w), (left, top))


def whole_pass(W: int, H: int) -> list:
    """SAHI's standard pass with Ultralytics' LetterBox shape (auto): sides rounded first, then padded to a multiple of 32
    (``cache_mz.level_passes`` pads the unrounded side up: 32 px more for 5 calib / 8 DOTA-val images)."""
    r = min(SLICE / H, SLICE / W)
    nh, nw = round(H * r), round(W * r)
    return [((0, 0, W, H), (nh + (SLICE - nh) % 32, nw + (SLICE - nw) % 32))]


def to_source_ultralytics(xyxy, lb, offset_xy=(0.0, 0.0)) -> np.ndarray:
    """Network input -> source px exactly as Ultralytics' ``scale_boxes`` (gain from the padded input shape, per axis =
    rounded side / side, padding recomputed; float32), plus the slice offset (as SAHI's shift)."""
    import torch

    h, w = lb.src_hw
    g = min(lb.in_hw[0] / h, lb.in_hw[1] / w)
    nh, nw = max(round(h * g), 1), max(round(w * g), 1)
    b = torch.as_tensor(np.asarray(xyxy, np.float32)).clone()
    px, py = round((lb.in_hw[1] - nw) / 2 - 0.1), round((lb.in_hw[0] - nh) / 2 - 0.1)
    b[..., [0, 2]] -= px
    b[..., [1, 3]] -= py
    b[..., [0, 2]] /= nw / w
    b[..., [1, 3]] /= nh / h
    return b.numpy().astype(np.float64) + np.array([offset_xy[0], offset_xy[1], offset_xy[0], offset_xy[1]])


def _a5_sahi_init(paths: list[str]) -> None:
    """``pipeline._sahi_init`` without its pinning to cores 16-23 (used by the FCOS tuning): the worker stays on the
    slot cores it inherits."""
    import os

    pin = os.sched_setaffinity
    os.sched_setaffinity = lambda *a: None
    try:
        P._sahi_init(paths)
    finally:
        os.sched_setaffinity = pin


def a5_sahi_tune(model: str, workers: int = 2) -> dict:
    """CPU (10G slot, cores 0-1), addendum 5: ``pipeline.sahi_tune`` unchanged (grid v2 over the three overlaps, same
    scoring and selection) with the worker initializer kept on the slot cores -> runs/dota/fusion/<m>_1024/sahi_tuning.json."""
    from concurrent.futures import ProcessPoolExecutor

    from scahi.common import spawn_context
    from scahi.infer import sahi_baseline as sb

    d = P.OUT / f"{model}_1024"
    if (d / "sahi_tuning.json").exists():
        return json.loads((d / "sahi_tuning.json").read_text())
    paths = [str(d / f"sahi_raw_calib_o{int(ov * 100):02d}.pkl") for ov in sb.OVERLAPS]
    cfgs = [(p, m, met, t, c) for p in paths for m in ("GREEDYNMM", "NMM", "NMS") for met in sb.GRID2_METRICS
            for t in sb.GRID2_THRS for c in sb.GRID2_CONFS]
    cfgs += [(p, "WBF", "IOU", t, c) for p in paths for t in sb.GRID2_THRS for c in sb.GRID2_WBF_SKIP]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=spawn_context(), initializer=_a5_sahi_init, initargs=(paths,)) as ex:
        rows = list(ex.map(P._sahi_eval, cfgs))
    for r in rows:
        r["overlap"] = sb.OVERLAPS[[Path(p).name for p in paths].index(r["raw"])]
    best = {m: max((r for r in rows if r["merge"] == m), key=lambda r: r["mAP50"]) for m in sb.GRID2_MERGES}
    res = {"rows": rows, "best_per_merge": best, "selected": max(best.values(), key=lambda r: r["mAP50"]),
           "wall_s": time.time() - t0, "procedure": "pipeline.sahi_tune (addendum 5), workers on the slot cores"}
    (d / "sahi_tuning.json").write_text(json.dumps(res, indent=1))
    print(f"[a5] SAHI tuning {model}: {len(rows)} configs, {time.time() - t0:.0f} s, selected {res['selected']}",
          flush=True)
    return res


def load_fused(model: str):
    """The fused ``DetectionModel`` of SAHI's predictor path (``sahi_baseline.load_model`` -> Ultralytics AutoBackend,
    fuse on load), with ``Detect.fuse`` (removal of the non-selected branch) skipped so that o2m stays; o2o and the
    shared layers are the predictor's numerics (checked against an unmodified SAHI predictor)."""
    from ultralytics.nn.modules.head import Detect

    from scahi.infer import sahi_baseline as sb

    dm = sb.load_model(T.weights(model, 1024), 0.001, "o2m" if model in A5_MODELS else "o2o")
    orig = Detect.fuse
    Detect.fuse = lambda self: None
    try:
        dm.model([np.zeros((SLICE, SLICE, 3), np.uint8)], imgsz=SLICE, conf=0.001, verbose=False)
    finally:
        Detect.fuse = orig
    m = dm.model.predictor.model.model
    assert not any(hasattr(x, "bn") for x in m.modules()) and m.model[-1].cv2 is not None
    return m


def cache_dir(model: str, key: str, split: str) -> Path:
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm

    return cm.cache_dir(CACHE_ROOT, T.weights(model, 1024), key, overlap(model), split)


def caches(model: str, splits: tuple[str, ...] = ("calib", "val")) -> None:
    """GPU: both SAHI-layout levels for ``splits`` (resumable)."""
    import cv2

    from scahi.infer import cache_mz as cm

    m = load_fused(model)
    ov = overlap(model)
    with _sahi_grid():
        for split in splits:
            imgs = P.images(split)
            for key in ORDER:
                d = cache_dir(model, key, split)
                d.mkdir(parents=True, exist_ok=True)
                (d / "cache_state.json").write_text(json.dumps(
                    {**CACHE_STATE, "model_state": CACHE_STATE_SINGLE} if model in A5_MODELS else CACHE_STATE, indent=1))
                t0 = time.time()
                for stem, p in zip(imgs.stem, imgs.path):
                    f = d / f"{stem}.npz"
                    if not f.exists():
                        np.savez_compressed(f, **cm.raw_predict_level(m, cv2.imread(p), LEVELS[key], ov, batch=1))
                print(f"[sahi-scahi] cache {model} {key} o{ov} {split}: {time.time() - t0:.0f} s", flush=True)


def _passes(key: str, W: int, H: int, ov: float) -> list:
    from scahi.infer import cache_mz as cm

    return sahi_passes(LEVELS[key], W, H, ov) if key == "s640f" else whole_pass(W, H)


def o2o_pass_dets(z: dict, passes: list, conf: float = 0.001, max_det: int = P.PIPE) -> np.ndarray:
    """Ultralytics' YOLOv10 o2o postprocess per pass from the cache: both top-k stages use k = min(max_det, anchors of
    the input), the anchors of the *whole* input (``topk_like_ultralytics`` sees only the cached anchors, max class
    probability >= 0.001, and would cap the second stage there); uncached anchors score < 0.001 and fall to ``conf``."""
    import torch

    xyxy, probs, tile = z["o2o_xyxy"], z["o2o_probs"], z["o2o_tile"]
    rows = []
    for t in np.unique(tile):
        h, w = passes[t][1]
        k = min(max_det, sum((h // s) * (w // s) for s in (8, 16, 32)))
        m = tile == t
        pr = torch.from_numpy(probs[m]).float()
        nc = pr.shape[1]
        ia = pr.max(-1)[0].topk(min(k, len(pr)))[1]
        sc, ix = pr[ia].flatten().topk(min(k, len(ia) * nc))
        bx = torch.from_numpy(xyxy[m]).float()[ia[ix // nc]]
        r = torch.cat([bx, sc[:, None], (ix % nc)[:, None].float()], 1).numpy().astype(np.float64)
        x1, y1, x2, y2 = passes[t][0]  # Ultralytics clips to the pass (scale_boxes), SAHI drops empty boxes
        if x2 > x1:
            r[:, [0, 2]] = r[:, [0, 2]].clip(x1, x2)
            r[:, [1, 3]] = r[:, [1, 3]].clip(y1, y2)
        rows.append(r[(r[:, 4] > conf) & (r[:, 2] > r[:, 0]) & (r[:, 3] > r[:, 1])])
    return np.concatenate(rows).astype(np.float64) if rows else np.zeros((0, 6))


def nms_pass_dets(z: dict, passes: list, conf: float = 0.001, iou: float = A5_IOU, max_det: int = P.PIPE) -> np.ndarray:
    """Ultralytics' NMS postprocess per pass from the cache (single dense head, addendum 5): best class per anchor, score
    > ``conf``, class-wise NMS (torchvision, per class - Ultralytics separates the classes by a coordinate offset in
    input space; IoU is invariant to the per-axis mapping), kept in descending score, first ``max_det``, boxes clipped
    to the pass; uncached anchors score < 0.001 and fall to ``conf``."""
    import torch
    import torchvision

    xyxy, probs, tile = z["o2m_xyxy"], z["o2m_probs"], z["o2m_tile"]
    rows = []
    for t in np.unique(tile):
        m = tile == t
        sc, cl = torch.from_numpy(probs[m]).float().max(-1)
        bx = torch.from_numpy(xyxy[m]).float()
        f = sc > conf
        sc, cl, bx = sc[f], cl[f], bx[f]
        keep = torch.cat([torch.nonzero(cl == c)[:, 0][torchvision.ops.nms(bx[cl == c], sc[cl == c], iou)]
                          for c in cl.unique()]) if len(sc) else torch.zeros(0, dtype=torch.long)
        keep = keep[sc[keep].argsort(descending=True, stable=True)][:max_det]
        r = torch.cat([bx[keep], sc[keep, None], cl[keep, None].float()], 1).numpy().astype(np.float64)
        x1, y1, x2, y2 = passes[t][0]
        if x2 > x1:
            r[:, [0, 2]] = r[:, [0, 2]].clip(x1, x2)
            r[:, [1, 3]] = r[:, [1, 3]].clip(y1, y2)
        rows.append(r[(r[:, 2] > r[:, 0]) & (r[:, 3] > r[:, 1])])
    return np.concatenate(rows).astype(np.float64) if rows else np.zeros((0, 6))


def keys(model: str, split: str, stem: str) -> tuple[str, ...]:
    """The levels of one image: SAHI runs its standard pass only if the image has more than one slice."""
    n = len(np.load(cache_dir(model, "s640f", split) / f"{stem}.npz")["tiles"])
    return ORDER if n > 1 else ORDER[:1]


def cache_raw(model: str, split: str, stem: str) -> np.ndarray:
    """o2o pass detections of the cache in SAHI's raw format [n, 6] (x1, y1, x2, y2, score, cls), SAHI's order."""
    parts = []
    for key in keys(model, split, stem):
        z = dict(np.load(cache_dir(model, key, split) / f"{stem}.npz"))
        W, H = (int(v) for v in z["image_wh"])
        fn = nms_pass_dets if model in A5_MODELS else o2o_pass_dets
        parts.append(fn(z, _passes(key, W, H, overlap(model))))
    return np.concatenate(parts)


def _match(a: np.ndarray, b: np.ndarray) -> dict:
    """Descriptive: cache vs. stored raw of one image; each cache detection to its nearest stored one of the same class
    (max-norm over the box corners; not one-to-one - a full assignment does not fit in memory on dense images)."""
    from scipy.spatial import cKDTree

    bx, sc, miss = [], [], 0
    for c in np.unique(a[:, 5]):
        A, B = a[a[:, 5] == c], b[b[:, 5] == c]
        if not len(B):
            miss += len(A)
            continue
        dist, j = cKDTree(B[:, :4]).query(A[:, :4], p=np.inf)
        bx.append(dist)
        sc.append(np.abs(A[:, 4] - B[j, 4]))
    bx, sc = (np.concatenate(v) if v else np.zeros(1) for v in (bx, sc))
    return {"n_cache": len(a), "n_frozen": len(b), "no_same_class": miss, "box_px_max": float(bx.max()),
            "box_px_p99": float(np.quantile(bx, 0.99)), "score_max": float(sc.max()),
            "score_p99": float(np.quantile(sc, 0.99))}


def gate_decode(model: str, n: int = 3) -> list[dict]:
    """GPU: rule (1) at the two new geometries."""
    import cv2
    import torch

    from scahi.common import ulp_cmp
    from scahi.infer.dfl_decode import decode
    from scahi.infer.preprocess import input_to_source, letterbox, to_tensor
    from scahi.infer.raw_head import forward_raw
    from ultralytics.utils.tal import dist2bbox, make_anchors

    ov = overlap(model)
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    m = load_fused(model)
    m.end2end = False
    head = m.model[-1]
    rows = []
    for key in ORDER:
        for p in P.images("calib").path.iloc[:n]:
            im = cv2.imread(p)
            H, W = im.shape[:2]
            (x1, y1, x2, y2), sh = _passes(key, W, H, ov)[0]
            x, lb = letterbox(im[y1:y2, x1:x2], sh)
            xt = to_tensor([x], "cuda")
            raw = forward_raw(m, xt)
            r = {"input": key, "shape": list(x.shape[:2])}
            d = decode(raw.boxes["o2m"], raw.scores["o2m"], raw.anchors, raw.strides, raw.reg_max)
            _, preds = m(xt)
            pp = preds if "boxes" in preds else preds["one2many"]
            anc, st = (a.transpose(0, 1) for a in make_anchors(pp["feats"], head.stride, 0.5))
            ref = (dist2bbox(head.dfl(pp["boxes"]), anc.unsqueeze(0), xywh=False, dim=1) * st).transpose(1, 2)
            r["o2m"] = {**ulp_cmp(d.xyxy, ref),
                        "max_abs_prob": float((d.probs - pp["scores"].sigmoid().transpose(1, 2)).abs().max())}
            if model in A5_MODELS:  # Ultralytics' NMS on the same forward vs. the cache path
                from ultralytics.utils.nms import non_max_suppression

                from scahi.infer.cache import SCORE_MIN

                y2_ = non_max_suppression(m(xt)[0], 0.001, A5_IOU, max_det=P.PIPE)[0]
                keep = d.probs[0].max(-1)[0] >= SCORE_MIN
                z = {"o2m_xyxy": d.xyxy[0][keep].cpu().numpy(), "o2m_probs": d.probs[0][keep].cpu().numpy(),
                     "o2m_tile": np.zeros(int(keep.sum()), np.int16)}
                own = torch.from_numpy(nms_pass_dets(z, [((0, 0, 0, 0), tuple(x.shape[:2]))]).astype(np.float32)).cuda()
                same_n = len(own) == len(y2_)
                r["nms"] = {"n_own": len(own), "n_ultralytics": len(y2_)}
                if same_n:
                    r["nms"].update(ulp_cmp(own[:, :4], y2_[:, :4]),
                                    max_abs_prob=float((own[:, 4] - y2_[:, 4]).abs().max()),
                                    class_agreement=float((own[:, 5] == y2_[:, 5]).float().mean()))
                r["ok"] = (r["o2m"]["max_diff_over_tol"] <= 1.0 and r["o2m"]["max_abs_prob"] < 1e-4 and same_n
                           and r["nms"]["max_diff_over_tol"] <= 1.0 and r["nms"]["max_abs_prob"] < 1e-4
                           and r["nms"]["class_agreement"] > 0.999)
                rows.append(r)
                continue
            m.end2end, head.max_det = True, P.PIPE
            y2_ = m(xt)[0][0]
            m.end2end, head.max_det = False, 300
            y2_ = y2_[y2_[:, 4] > 0.001]
            d1 = decode(raw.boxes["o2o"], raw.scores["o2o"], raw.anchors, raw.strides, raw.reg_max)
            from scahi.infer.cache import SCORE_MIN

            keep = d1.probs[0].max(-1)[0] >= SCORE_MIN  # the cache keeps these anchors
            z = {"o2o_xyxy": d1.xyxy[0][keep].cpu().numpy(), "o2o_probs": d1.probs[0][keep].cpu().numpy(),
                 "o2o_tile": np.zeros(int(keep.sum()), np.int16)}
            own = torch.from_numpy(o2o_pass_dets(z, [((0, 0, 0, 0), tuple(x.shape[:2]))]).astype(np.float32)).cuda()
            same_n = len(own) == len(y2_)
            r["o2o"] = {"n_own": len(own), "n_ultralytics": len(y2_)}
            if same_n:
                r["o2o"].update(ulp_cmp(own[:, :4], y2_[:, :4]),
                                max_abs_prob=float((own[:, 4] - y2_[:, 4]).abs().max()),
                                class_agreement=float((own[:, 5] == y2_[:, 5]).float().mean()))
            r["ok"] = (r["o2m"]["max_diff_over_tol"] <= 1.0 and r["o2m"]["max_abs_prob"] < 1e-4 and same_n
                       and r["o2o"]["max_diff_over_tol"] <= 1.0 and r["o2o"]["max_abs_prob"] < 1e-4
                       and r["o2o"]["class_agreement"] > 0.999)
            rows.append(r)
    torch.backends.cudnn.allow_tf32 = tf32
    del m
    torch.cuda.empty_cache()
    return rows


def raw_compare(model: str, split: str, ov: float, keep: dict | None = None) -> dict:
    """Descriptive: per image, the cache path's pass detections vs. SAHI's own raw predictions (no metric)."""
    ref = pickle.loads((P.OUT / f"{model}_1024" / f"sahi_raw_{split}_o{int(ov * 100):02d}.pkl").read_bytes())
    per = {}
    for st in P.split_stems(split):
        a = cache_raw(model, split, st)
        per[st] = _match(a, ref[st])
        if keep is not None:
            keep[st] = a
    n_c, n_f = sum(v["n_cache"] for v in per.values()), sum(v["n_frozen"] for v in per.values())
    print(f"[sahi-scahi] raw vs frozen {model} {split}: {n_c} vs {n_f} dets", flush=True)
    return {"images": len(per), "dets_cache": n_c, "dets_frozen": n_f,
            "no_same_class": int(sum(v["no_same_class"] for v in per.values())),
            "images_same_count": int(sum(v["n_cache"] == v["n_frozen"] for v in per.values())),
            "box_px_max": max(v["box_px_max"] for v in per.values()),
            "box_px_p99_median": float(np.median([v["box_px_p99"] for v in per.values()])),
            "score_max": max(v["score_max"] for v in per.values()),
            "score_p99_median": float(np.median([v["score_p99"] for v in per.values()])), "per_image": per}


def gate(model: str) -> dict:
    """Parity gate (1) + (2) and the descriptive comparison with the stored raw predictions."""
    fz = frozen_sahi(model)
    ov = overlap(model)
    out = {"gate": f"{'D-H6' if model in A5_MODELS else 'D-H5'} parity (rule of pipeline.gate)", "model": model, "overlap": ov, "frozen_cfg": fz["cfg"],
           "decode": gate_decode(model)}
    raw_cal = {}
    for split in ("calib", "val"):
        if model in A5_MODELS and split == "val":
            # addendum 5, item 10: SAHI's own DOTA-val raw predictions only after the freeze approval; the descriptive
            # DOTA-val comparison (no metric, not part of the verdict) follows with them (user decision 2026-11-03)
            out["raw_vs_frozen_val"] = "deferred to the DOTA-val run (addendum 5, item 10)"
            continue
        out[f"raw_vs_frozen_{split}"] = raw_compare(model, split, ov, raw_cal if split == "calib" else None)
    c = fz["cfg"]
    d = P.sahi_merge(raw_cal, c["merge"], c["metric"], c["match_thr"], c["conf"])
    m = P._score(d, EV.gt_arrays("train", T.split_stems("calib")))["mAP50"]
    out["map"] = {"mAP50_cache": m, "mAP50_frozen": fz["calib_mAP50"], "abs_delta": abs(m - fz["calib_mAP50"]),
                  **({"reference": "calib mAP50 of the selected SAHI config (sahi_tuning.json)"}
                     if model in A5_MODELS else {})}
    out["passed_decode"] = all(r["ok"] for r in out["decode"])
    out["passed_map"] = out["map"]["abs_delta"] <= GATE_MAP_TOL
    out["passed"] = out["passed_decode"] and out["passed_map"]
    f = P.OUT / f"{model}_1024" / "sahi_scahi_gate.json"
    f.write_text(json.dumps(out, indent=1))
    print(f"[sahi-scahi] gate {model}: decode {out['passed_decode']}, calib mAP50 {m:.4f} vs frozen "
          f"{fz['calib_mAP50']:.4f} -> {'PASSED' if out['passed'] else 'FAILED'}", flush=True)
    return out


def load(model: str, split: str, branch: str, stems: list[str] | None = None) -> dict:
    """{stem: (levels, W, H)} of the SAHI layout (both levels), as ``pipeline.load``."""
    from scahi.fusion import scalespace as ss

    return {st: ss.load_levels([(k, cache_dir(model, k, split)) for k in keys(model, split, st)], st, branch,
                               overlap(model), dets=False) for st in (stems or P.split_stems(split))}


# ---------------------------------------------------------------------------------------------------- calib tuning
# Addendum 4, item 5: SAHI-F2 (VisDrone F2 grid, 24 configs) and SAHI-RG (refined grid, 108 variants + the calib-selected
# SAHI-F2 config as reference, time <= 2x the reference), o2m, calib, pipeline limit 200 000 (addendum 4; the frozen YOLO
# rows were selected at 20 000 and evaluated at 200 000). Started in parallel to the FCOS tuning (user, 2026-11-02): own
# scope (10 GB cap) on cores 0-1; one task per (image, config) in a fresh process (RG timing: option 3, reference in the
# same process per task, alternating order, as FCOS); a task whose memory estimate exceeds the budget is not started but
# deferred to a later run with a larger budget (after the FCOS tuning); no task is dropped. Scheduling only.
DH5_LIMIT = 200_000
DH5_SAFETY = 1.25  # memory estimate: 1.25 x (base + (peak on the densest image - base) x cache-size ratio)
DH5_BASE_GIB = 1.0


def dh5_dir(model: str) -> Path:
    return P.OUT / f"{model}_1024" / ("a5" if model in A5_MODELS else "dh5")


def _bytes(model: str, stem: str) -> int:
    return sum((cache_dir(model, k, "calib") / f"{stem}.npz").stat().st_size for k in keys(model, "calib", stem))


def _dh5_task(args: tuple) -> dict:
    """Worker (fresh process): fuse ``cfg`` on one image (and the reference, alternating order by ``k``, if given);
    detections -> ``part`` (parquet) if given; times and peak RSS -> ``part``.json and return value."""
    import resource

    model, stem, cfg, ref, k, part = args
    P._pin()
    data = load(model, "calib", "o2m", [stem])

    def run(c: dict) -> tuple[pd.DataFrame, float]:
        t0 = time.perf_counter()
        d = P.fuse(data, "o2m", "F2", c, limit=DH5_LIMIT)
        return d, time.perf_counter() - t0

    t_ref = None
    if ref is None:
        d, t = run(cfg)
    elif k % 2 == 0:
        _, t_ref = run(ref)
        d, t = run(cfg)
    else:
        d, t = run(cfg)
        _, t_ref = run(ref)
    out = {"t": t, "t_ref": t_ref, "n": len(d), "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
    if part is not None:
        d.to_parquet(part.with_suffix(".parquet"))
        part.with_suffix(".json").write_text(json.dumps(out))
    return out


def _dh5_profile(model: str, stage: str, cfgs: list[dict], ref: dict | None) -> dict:
    """Measured task peaks on the densest calib image (F2: every config; RG: corners of the refined grid at the smallest
    tau_peak, the reference, two interior points - same points and dominance rule as pipeline.rg_mem_profile) and the
    heaviest of them on the 10th densest image (check of the linear cache-size scaling). Serial, fresh processes."""
    f = dh5_dir(model) / f"mem_{stage}.json"
    prof = json.loads(f.read_text()) if f.exists() else {"points": {}}
    stems = sorted(T.split_stems("calib"), key=lambda st: -_bytes(model, st))
    prof.update(stem=stems[0], bytes=_bytes(model, stems[0]))
    if stage == "F2":
        pts = [(P._rg_key(c), c) for c in cfgs]
    else:
        g = json.loads((P.REPO_ROOT / "configs/ext/xgrid_frozen.json").read_text())["grid"]["C"]
        c_ = lambda a, b, cf, bi: {"iou_cluster": 0.5, "tau_splat": a, "tau_peak": b, "cell_factor": cf,  # noqa: E731
                                   "bins_per_oct": bi}
        tp = min(g["tau_peak"])
        corners = [c_(a, tp, cf, bi) for a in (min(g["tau_splat"]), max(g["tau_splat"]))
                   for cf in (min(g["cell_factor"]), max(g["cell_factor"]))
                   for bi in (min(g["bins_per_oct"]), max(g["bins_per_oct"]))]
        mid = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
        interior = [c_(mid(g["tau_splat"]), tp, mid(g["cell_factor"]), mid(g["bins_per_oct"])),
                    c_(min(g["tau_splat"]), max(g["tau_peak"]), max(g["cell_factor"]), max(g["bins_per_oct"]))]
        prof.update(grid=g, corners=[P._rg_key(x) for x in corners], interior=[P._rg_key(x) for x in interior])
        pts = [("ref", ref)] + [(P._rg_key(x), x) for x in corners + interior]
    for key, cfg in pts:
        if key not in prof["points"]:
            with P._fresh_pool(1) as ex:
                r = ex.submit(_dh5_task, (model, stems[0], cfg, ref, 0, None)).result()
            prof["points"][key] = {**r, "stem": stems[0], "bytes": prof["bytes"], "cfg": cfg}
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(prof, indent=1))
            print(f"[dh5] {model} {stage} mem {key} on {stems[0]}: {r['peak_gib']:.2f} GiB", flush=True)
    heavy_key = max((k for k in prof["points"] if "@" not in k), key=lambda k: prof["points"][k]["peak_gib"])
    second = stems[min(9, len(stems) - 1)]
    sk = f"{heavy_key}@{second}"
    if sk not in prof["points"]:
        cfg = prof["points"][heavy_key]["cfg"]
        with P._fresh_pool(1) as ex:
            r = ex.submit(_dh5_task, (model, second, cfg, ref, 0, None)).result()
        prof["points"][sk] = {**r, "stem": second, "bytes": _bytes(model, second), "cfg": cfg}
    prof["scaling_check"] = sk
    f.write_text(json.dumps(prof, indent=1))
    return prof


def _dh5_peak(prof: dict, stage: str, cfg: dict, ref: dict | None) -> float:
    if stage == "F2":
        return prof["points"][P._rg_key(cfg)]["peak_gib"]
    return P._rg_peak(prof, cfg, ref)


def _dh5_est(prof: dict, stage: str, cfg: dict, ref: dict | None, nbytes: int) -> float:
    return DH5_SAFETY * (DH5_BASE_GIB + max(_dh5_peak(prof, stage, cfg, ref) - DH5_BASE_GIB, 0.0) * nbytes / prof["bytes"])


def _dh5_score(args: tuple) -> dict:
    d, stems = args
    det = pd.concat([pd.read_parquet(d / f"{st}.parquet") for st in stems], ignore_index=True)
    return P._score(det, EV.gt_arrays("train", stems), DH5_LIMIT)


def dh5_stage(model: str, stage: str, workers: int) -> list[dict] | None:
    """One stage ('F2' or 'RG') of one model; resumable per task. Returns the rows, or None if tasks are deferred
    (estimate > env SCAHI_FUSE_BUDGET_GB) - they run in a later call with a larger budget."""
    import os
    from concurrent.futures import FIRST_COMPLETED, wait

    tf = dh5_dir(model) / "tuning.json"
    res = json.loads(tf.read_text()) if tf.exists() else {}
    if stage in res:
        return res[stage]["rows"]
    if stage == "F2":
        cfgs, ref = P.f2_grid(), None
    else:
        ref = {k: v for k, v in P.select(res["F2"]["rows"]).items()
               if k in ("iou_cluster", "tau_splat", "tau_peak", "cell_factor", "bins_per_oct")}
        cfgs = [ref, *P.refined_grid()]
    prof = _dh5_profile(model, stage, cfgs, ref)
    sc = prof["points"][prof["scaling_check"]]
    assert sc["peak_gib"] <= _dh5_est(prof, stage, sc["cfg"], ref, sc["bytes"]), "cache-size scaling underestimates"
    stems = T.split_stems("calib")
    nb = {st: _bytes(model, st) for st in stems}
    td = dh5_dir(model) / "tasks" / stage
    tasks, est = [], []
    for st in sorted(stems, key=lambda st: -nb[st]):
        for ci, cfg in enumerate(cfgs):
            (td / f"{ci:03d}").mkdir(parents=True, exist_ok=True)
            tasks.append((model, st, cfg, ref, len(tasks), td / f"{ci:03d}" / st))
            est.append(_dh5_est(prof, stage, cfg, ref, nb[st]))
    budget = float(os.environ.get("SCAHI_FUSE_BUDGET_GB", "inf"))
    todo = [i for i, t in enumerate(tasks) if not t[-1].with_suffix(".json").exists()]
    deferred = [i for i in todo if est[i] > budget]
    todo = [i for i in todo if est[i] <= budget]
    print(f"[dh5] {model} {stage}: {len(tasks)} tasks, {len(tasks) - len(todo) - len(deferred)} done, {len(todo)} to "
          f"run, {len(deferred)} deferred (estimate > budget {budget} GiB; max est {max(est):.2f} GiB), {workers} "
          "workers", flush=True)
    # the linear cache-size scaling can run low on single images: k = max(1, 1.25 x max(measured peak / raw estimate))
    # over the stage's finished tasks scales the estimates of the open tasks (restores the 1.25 margin); scheduling only
    raw = list(est)
    ratios = [json.loads(t[-1].with_suffix(".json").read_text())["peak_gib"] / raw[j] for j, t in enumerate(tasks)
              if t[-1].with_suffix(".json").exists()]
    k = max([1.0] + [DH5_SAFETY * x for x in ratios])
    est = [e * k for e in raw]
    deferred = [i for i in deferred + todo if est[i] > budget]
    todo = [i for i in todo if est[i] <= budget]
    print(f"[dh5] {model} {stage}: estimate factor k {k:.3f} (max measured peak / raw estimate "
          f"{max(ratios, default=0):.3f}); {len(todo)} to run, {len(deferred)} deferred", flush=True)
    running, used, t0, n0 = {}, 0.0, time.time(), len(todo)
    with P._fresh_pool(workers) as ex:
        while todo or running:
            for i in list(todo):
                if len(running) >= workers:
                    break
                if used + est[i] <= budget:
                    running[ex.submit(_dh5_task, tasks[i])] = (i, est[i])
                    used += est[i]
                    todo.remove(i)
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                i, e = running.pop(f)
                used -= e
                r = f.result()
                if r["peak_gib"] > e:
                    print(f"[dh5] WARNING task {tasks[i][1]} {P._rg_key(tasks[i][2])}: peak {r['peak_gib']:.2f} > est "
                          f"{e:.2f} GiB", flush=True)
                if DH5_SAFETY * r["peak_gib"] / raw[i] > k:
                    k = DH5_SAFETY * r["peak_gib"] / raw[i]
                    est = [x * k for x in raw]
                    print(f"[dh5] estimate factor k -> {k:.3f}", flush=True)
                    for j in [j for j in todo if est[j] > budget]:
                        todo.remove(j)
                        deferred.append(j)
            if done and (n0 - len(todo)) % 500 < len(done):
                print(f"[dh5] {model} {stage}: {n0 - len(todo) - len(running)}/{n0} tasks, {time.time() - t0:.0f} s",
                      flush=True)
    if deferred:
        print(f"[dh5] {model} {stage}: {len(deferred)} tasks deferred to a run with a larger budget", flush=True)
        return None
    t = {ci: [json.loads((td / f"{ci:03d}" / f"{st}.json").read_text()) for st in stems] for ci in range(len(cfgs))}
    with P._fresh_pool(workers) as ex:
        scores = list(ex.map(_dh5_score, [(td / f"{ci:03d}", stems) for ci in range(len(cfgs))]))
    rows = []
    for ci, cfg in enumerate(cfgs):
        tv = sum(x["t"] for x in t[ci])
        r = {**cfg, **scores[ci], "fuse_s_per_image": tv / len(stems), "peak_gib_max": max(x["peak_gib"] for x in t[ci])}
        if ref is not None:
            tr = sum(x["t_ref"] for x in t[ci])
            r.update(time_ratio_vs_ref=tv / tr, ref_s_per_image=tr / len(stems),
                     timing="option 3: task split, reference in the same process per image")
        rows.append(r)
    res = json.loads(tf.read_text()) if tf.exists() else {}
    res[stage] = {"rows": rows, "wall_s": time.time() - t0, "limit": DH5_LIMIT}
    if stage == "RG":
        res["selected"] = {"F2": P.select(res["F2"]["rows"]), "RG": P.select(rows, capped=True), "reference_F2": ref}
    tf.write_text(json.dumps(res, indent=1))
    print(f"[dh5] {model} {stage}: {len(rows)} configs done", flush=True)
    return rows


def dh5_tune(workers: int, models: tuple[str, ...] = MODELS) -> None:
    """Both models, F2 then RG (RG needs the F2 selection); a stage with deferred tasks blocks only its own RG."""
    for m in models:
        if dh5_stage(m, "F2", workers) is not None:
            dh5_stage(m, "RG", workers)
    print("[dh5] tuning run done", flush=True)


# ------------------------------------------------------------------------------------------- freeze proposal, replay
# Rows SAHI-F2 / SAHI-RG per model from dh5/tuning.json (nothing re-selected); ``dh5_row_dets`` is the one path for the
# calib replay and the single DOTA-val evaluation: all images of the split in one process, P.fuse, limit 200 000.
FROZEN_DH5 = REPO / "configs/dota/frozen_rows_dh5.json"
DH5_ROWS = {"SAHI-F2": "F2", "SAHI-RG": "RG"}
F2_KEYS = ("iou_cluster", "tau_splat", "tau_peak", "cell_factor", "bins_per_oct")


def dh5_rows(models: tuple[str, ...] = MODELS) -> list[dict]:
    from scahi.common import rel
    from scahi.dota import final as F

    rows = []
    for m in models:
        w = T.weights(m, 1024)
        sel = json.loads((dh5_dir(m) / "tuning.json").read_text())["selected"]
        for rid, st in DH5_ROWS.items():
            v = sel[st]
            rows.append({"model": m, "s": 1024, "weights": rel(w), "weights_sha256": F.sha256(w), "id": f"{rid}_{m}",
                         "row": rid, "kind": "dh5", "branch": "o2m", "variant": "F2", "overlap": overlap(m),
                         "levels": list(ORDER), "cfg": {k: v[k] for k in F2_KEYS}, "calib_mAP50": v["mAP50"],
                         "calib_mAP50_cap4000": v["mAP50_cap4000"], "calib_images_at_limit": v["images_at_limit"],
                         **({"time_ratio_vs_ref": v["time_ratio_vs_ref"], "reference_cfg": sel["reference_F2"]}
                            if st == "RG" else {}),
                         "source": f"{dh5_dir(m).relative_to(REPO)}/tuning.json [{st}]"})
    return rows


def dh5_row_dets(row: dict, split: str) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    dets = P.fuse(load(row["model"], split, row["branch"]), row["branch"], row["variant"], row["cfg"], limit=DH5_LIMIT)
    return dets, {"images_at_limit": P.at_limit(dets, DH5_LIMIT), "wall_s": time.time() - t0}


def dh5_replay(rows: list[dict] | None = None, row_dets=None, name: str = "replay_dh5.json") -> dict:
    """Every proposed row on calib through ``dh5_row_dets``; mAP50 and mAP50_cap4000 must equal the tuning values
    exactly -> runs/dota/freeze/replay_dh5.json (addendum 5: ``a5_replay``)."""
    from scahi.dota import final as F

    P._pin()
    gt = EV.gt_arrays("train", T.split_stems("calib"))
    res = []
    for r in rows or dh5_rows():
        dets, info = (row_dets or dh5_row_dets)(r, "calib")
        s = P._score(dets, gt, DH5_LIMIT if r["kind"] == "dh5" else None)
        res.append({"id": r["id"], "calib_mAP50": r["calib_mAP50"], "replay_mAP50": s["mAP50"],
                    "abs_diff": abs(s["mAP50"] - r["calib_mAP50"]),
                    "abs_diff_cap4000": abs(s["mAP50_cap4000"] - r["calib_mAP50_cap4000"]), **info})
        print(f"[dh5] replay {r['id']}: {s['mAP50']:.4f} vs {r['calib_mAP50']:.4f} ({info['wall_s']:.0f} s)", flush=True)
    o = {"limit": DH5_LIMIT, "rows": res, "exact": all(x["abs_diff"] == 0 and x["abs_diff_cap4000"] == 0 for x in res)}
    (F.FREEZE_DIR / name).write_text(json.dumps(o, indent=1))
    return o


# ------------------------------------------------------------------------- addendum 5 (D-H6): freeze proposal, replay
# Rows per model: SAHI (calib selection of a5_sahi_tune: raw pickle of the selected overlap, P.sahi_merge, no limit as
# the SAHI tuning), SAHI-F2 and SAHI-RG (a5/tuning.json, as dh5_rows). ``a5_row_dets`` is the one path for the calib
# replay and the single DOTA-val evaluation.
FROZEN_A5 = REPO / "configs/dota/frozen_rows_a5.json"
SAHI_KEYS = ("merge", "metric", "match_thr", "conf", "overlap")


def a5_rows() -> list[dict]:
    from scahi.common import rel
    from scahi.dota import final as F

    rows = []
    for m in A5_MODELS:
        w = T.weights(m, 1024)
        s = json.loads((P.OUT / f"{m}_1024" / "sahi_tuning.json").read_text())["selected"]
        rows.append({"model": m, "s": 1024, "weights": rel(w), "weights_sha256": F.sha256(w), "id": f"SAHI_{m}",
                     "row": "SAHI", "kind": "a5_sahi", "branch": "o2m", "cfg": {k: s[k] for k in SAHI_KEYS},
                     "calib_mAP50": s["mAP50"], "calib_mAP50_cap4000": s["mAP50_cap4000"],
                     "calib_images_at_limit": s["images_at_limit"],
                     "source": f"runs/dota/fusion/{m}_1024/sahi_tuning.json [selected]"})
        rows += dh5_rows((m,))
    return rows


def a5_row_dets(row: dict, split: str) -> tuple[pd.DataFrame, dict]:
    import pickle

    if row["kind"] != "a5_sahi":
        return dh5_row_dets(row, split)
    t0 = time.time()
    c = row["cfg"]
    f = P.OUT / f"{row['model']}_1024" / f"sahi_raw_{split}_o{int(c['overlap'] * 100):02d}.pkl"
    if not f.exists():  # DOTA-val: SAHI's raw predictions are computed once, after the freeze approval (item 10)
        f = sahi_raw_single(row["model"], split, c["overlap"])
    dets = P.sahi_merge(pickle.loads(f.read_bytes()), c["merge"], c["metric"], c["match_thr"], c["conf"])
    # SAHI rows have no per-image limit; counted against the pipeline limit of the other rows, the count at PIPE is
    # informational only (it was logged as "at the limit" on 2026-11-04 and relabelled, DECISIONS)
    return dets, {"images_at_limit": P.at_limit(dets, DH5_LIMIT), "images_ge_20000_dets": P.at_limit(dets),
                  "wall_s": time.time() - t0}


def a5_replay() -> dict:
    return dh5_replay(a5_rows(), a5_row_dets, "replay_a5.json")


# ------------------------------------------------------------------------------------------------ DOTA-val (once)
# Freeze approved 2026-11-03 (bf64c9a). Reporting as the original DOTA freeze; the frozen SAHI rows are not re-evaluated
# (their stored DOTA-val evaluation, runs/dota/val/eval/SAHI_<m>*, is the reference; same bootstrap weights, seed 0).
DH5_HYP = ("SAHI-RG", "SAHI")
DH5_SECONDARY = {"SAHI-F2 − SAHI": ("SAHI-F2", "SAHI")}


def dh5_approve(note: str) -> None:
    fz = json.loads(FROZEN_DH5.read_text())
    fz.update(approved=True, approved_at=time.strftime("%Y-%m-%dT%H:%M"), approval=note)
    FROZEN_DH5.write_text(json.dumps(fz, indent=1))


def dh5_val() -> dict:
    """Detections of the 4 frozen rows on DOTA-val (written once, never recomputed), their evaluation, D-H5 (Holm over
    the two models) and the secondary contrast -> runs/dota/val/results_dh5.json."""
    from scahi.dota import final as F
    from scahi.ext_verdicts import holm

    P._pin()
    fz = F.frozen(FROZEN_DH5)
    for r in fz["rows"]:
        if not (F.DETS_DIR / f"{r['id']}.parquet").exists():
            F._write_dets(r["id"], *dh5_row_dets(r, "val"))
    (F.VAL_DIR / "eval").mkdir(parents=True, exist_ok=True)
    for r in fz["rows"]:
        F._eval_job(r["id"])
        print(f"[dh5] val eval {r['id']} done", flush=True)
    rid = lambda row, m: f"{row}_{m}"  # noqa: E731
    ids = [rid(x, m) for m in MODELS for x in ("SAHI", *DH5_ROWS)]
    pt = {i: json.loads((F.VAL_DIR / "eval" / f"{i}.json").read_text()) for i in ids}
    vec = {i: dict(np.load(F.VAL_DIR / "eval" / f"{i}_boot.npz")) for i in ids}
    a, b = DH5_HYP
    per = {m: {k: F._contrast(vec, rid(a, m), rid(b, m), k, pt) for k in F.VEC_METRICS} for m in MODELS}
    hl = holm({m: per[m]["mAP50"]["p_one_sided"] for m in MODELS}, m=len(MODELS))
    for m in MODELS:
        crit = per[m]["mAP50"]["delta"] > F.EPS
        per[m].update(holm=hl[m], criterion=bool(crit), met=bool(crit and hl[m]["p_holm"] <= 0.05))
    sec = {n: {m: {k: F._contrast(vec, rid(x, m), rid(y, m), k, pt) for k in F.VEC_METRICS} for m in MODELS}
           for n, (x, y) in DH5_SECONDARY.items()}
    res = {"addendum": 4, "split": "val", "n_images": len(P.split_stems("val")), "pipeline_limit": fz["pipeline_limit"],
           "eps": F.EPS, "bootstrap": {"n": 1000, "seed": 0, "ci": "percentile 2.5/97.5"},
           "rows": {i: {**pt[i], **json.loads((F.DETS_DIR / f"{i}.json").read_text())} for i in ids},
           "hypotheses": {"D-H5": {"contrast": f"{a} − {b}", "models": per, "holds": all(per[m]["met"] for m in MODELS)}},
           "secondary": sec, "frozen_code_commit": fz["code_commit"], "sahi_rows": "stored, not re-evaluated"}
    (F.VAL_DIR / "results_dh5.json").write_text(json.dumps(res, indent=1))
    for m in MODELS:
        x = per[m]["mAP50"]
        print(f"[dh5] D-H5 {m}: Δ {x['delta']:+.3f} CI {x['ci95']}, p_holm {per[m]['holm']['p_holm']:.4f}, "
              f"met {per[m]['met']}", flush=True)
    return res


# ------------------------------------------------------------------------------- addendum 5: DOTA-val (once)
# Freeze approved 2026-11-04 (571a87f). All six rows are evaluated here (the SAHI rows are new rows of this addendum);
# reporting as D-H5.
A5_SECONDARY = {"SAHI-F2 − SAHI": ("SAHI-F2", "SAHI")}


def a5_val_raw() -> None:
    """GPU: SAHI's own raw predictions on DOTA-val at the selected overlap (computed once, after the approval)."""
    for m in A5_MODELS:
        sahi_raw_single(m, "val", overlap(m))


def a5_val() -> dict:
    """CPU: the deferred descriptive raw comparison on DOTA-val (gate file), detections of the six frozen rows (written
    once, never recomputed), their evaluation, D-H6 (Holm over the two models), the secondary contrast
    -> runs/dota/val/results_a5.json."""
    from scahi.dota import final as F
    from scahi.ext_verdicts import holm

    P._pin()
    for m in A5_MODELS:
        g = P.OUT / f"{m}_1024" / "sahi_scahi_gate.json"
        gj = json.loads(g.read_text())
        if not isinstance(gj.get("raw_vs_frozen_val"), dict):
            gj["raw_vs_frozen_val"] = {**raw_compare(m, "val", overlap(m)),
                                       "note": "computed after the freeze approval (deferred from the gate run, "
                                               "descriptive, not part of the verdict)"}
            g.write_text(json.dumps(gj, indent=1))
    fz = F.frozen(FROZEN_A5)
    assert fz.get("approved") is True
    for r in fz["rows"]:
        if not (F.DETS_DIR / f"{r['id']}.parquet").exists():
            F._write_dets(r["id"], *a5_row_dets(r, "val"))
            print(f"[a5] val dets {r['id']} written", flush=True)
    (F.VAL_DIR / "eval").mkdir(parents=True, exist_ok=True)
    for r in fz["rows"]:
        F._eval_job(r["id"])
        print(f"[a5] val eval {r['id']} done", flush=True)
    rid = lambda row, m: f"{row}_{m}"  # noqa: E731
    ids = [rid(x, m) for m in A5_MODELS for x in ("SAHI", *DH5_ROWS)]
    pt = {i: json.loads((F.VAL_DIR / "eval" / f"{i}.json").read_text()) for i in ids}
    vec = {i: dict(np.load(F.VAL_DIR / "eval" / f"{i}_boot.npz")) for i in ids}
    a, b = DH5_HYP
    per = {m: {k: F._contrast(vec, rid(a, m), rid(b, m), k, pt) for k in F.VEC_METRICS} for m in A5_MODELS}
    hl = holm({m: per[m]["mAP50"]["p_one_sided"] for m in A5_MODELS}, m=len(A5_MODELS))
    for m in A5_MODELS:
        crit = per[m]["mAP50"]["delta"] > F.EPS
        per[m].update(holm=hl[m], criterion=bool(crit), met=bool(crit and hl[m]["p_holm"] <= 0.05))
    sec = {n: {m: {k: F._contrast(vec, rid(x, m), rid(y, m), k, pt) for k in F.VEC_METRICS} for m in A5_MODELS}
           for n, (x, y) in A5_SECONDARY.items()}
    res = {"addendum": 5, "split": "val", "n_images": len(P.split_stems("val")), "pipeline_limit": fz["pipeline_limit"],
           "eps": F.EPS, "bootstrap": {"n": 1000, "seed": 0, "ci": "percentile 2.5/97.5"},
           "rows": {i: {**pt[i], **json.loads((F.DETS_DIR / f"{i}.json").read_text())} for i in ids},
           "hypotheses": {"D-H6": {"contrast": f"{a} − {b}", "models": per, "holds": all(per[m]["met"] for m in A5_MODELS)}},
           "secondary": sec, "frozen_code_commit": fz["code_commit"], "approved_at": fz["approved_at"]}
    (F.VAL_DIR / "results_a5.json").write_text(json.dumps(res, indent=1))
    for m in A5_MODELS:
        x = per[m]["mAP50"]
        print(f"[a5] D-H6 {m}: Δ {x['delta']:+.3f} CI {x['ci95']}, p_holm {per[m]['holm']['p_holm']:.4f}, "
              f"met {per[m]['met']}", flush=True)
    return res


if __name__ == "__main__":
    import sys

    {"caches": lambda: [caches(m) for m in MODELS], "gate": lambda: [gate(m) for m in MODELS],
     "tune": lambda: dh5_tune(int(sys.argv[2]) if len(sys.argv) > 2 else 2), "replay": dh5_replay,
     "val": dh5_val,
     # addendum 5 (D-H6): SAHI raw calib (GPU) -> SAHI tuning (CPU, slot) -> caches + gate (GPU) -> tuning (CPU, slot)
     "a5_raw": lambda: [sahi_raw_single(m, "calib", ov) for m in A5_MODELS for ov in __import__(
         "scahi.infer.sahi_baseline", fromlist=["OVERLAPS"]).OVERLAPS],
     "a5_sahi_tune": lambda: [a5_sahi_tune(m) for m in A5_MODELS],
     "a5_caches": lambda: [caches(m) for m in A5_MODELS], "a5_gate": lambda: [gate(m) for m in A5_MODELS],
     "a5_tune": lambda: dh5_tune(int(sys.argv[2]) if len(sys.argv) > 2 else 2, A5_MODELS),
     "a5_replay": a5_replay, "a5_val_raw": a5_val_raw, "a5_val": a5_val}[sys.argv[1]]()
