# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase-4 raw-output cache per zoom *level* (stacks share levels), and per-pass detections for box fusion (F0).

Cache layout: ``cache/<weights_id>/<level_key>_o<overlap%>/<split>/<stem>.npz`` with the arrays of
``scahi.infer.cache`` (per branch: global ``xyxy``, ``probs`` [n, nc], DFL ``std`` per side, ``tile``; anchors with max
class probability >= 0.001) plus per-tile ``gains``. Input geometry equals the Ultralytics predictor used by B-MZ:
whole-image pass with long side = fixed imgsz or ceil(z * long side / 32) * 32, tiles of T = N / z original px
(``tile_grid``) scaled to N on the long side; letterbox shapes padded to multiples of 32 (rect, centred, 114).
Per-pass detections reproduce the predictor: o2o = ``topk_like_ultralytics`` (parity-tested), o2m = Ultralytics'
own ``non_max_suppression`` (iou 0.7) on the dense o2m output of the pass.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from scahi.data.zoom import tile_grid
from scahi.infer.cache import BRANCHES, SCORE_MIN, weights_id
from scahi.infer.dfl_decode import decode
from scahi.infer.multizoom import Level
from scahi.infer.preprocess import STRIDE_MULTIPLE, full_image_shape, input_to_source, letterbox, to_tensor
from scahi.infer.raw_head import forward_raw

LEVELS: dict[str, Level] = {
    "w0.333": Level(1 / 3, None),
    "w0.64": Level(0.64, None),
    "w1.0": Level(1.0, None),
    "w1280": Level(1280 / 1920, None, 1280),
    "w1536": Level(1536 / 1920, None, 1536),
    "w2048": Level(2048 / 1920, None, 2048),  # exploratory (2026-10-24, defined after the val results)
    "t1.0n640": Level(1.0, 640),
    "t2.0n640": Level(2.0, 640),
    "t2.0n1024": Level(2.0, 1024),
    # Phase D (DOTA, 2026-10-30): whole-image pass at 1024 and native-resolution tiles (z = 1, 1024 px)
    "w1024": Level(1024 / 1920, None, 1024),
    "t1.0n1024": Level(1.0, 1024),
}
STACK_LEVELS: dict[str, tuple[str, ...]] = {
    "S4": ("w0.64", "t2.0n1024"),
    "S5": ("w1.0", "t2.0n1024"),
    "S1": ("w0.333", "t1.0n640", "t2.0n640"),
    "S2": ("w0.333", "t1.0n640", "t2.0n1024"),
    "native": ("w1280", "t2.0n1024"),
    "native1536": ("w1536", "t2.0n1024"),
    "native2048": ("w2048", "t2.0n1024"),  # exploratory: B-FULL 2048 pass as trained + z = 2 tiles
    "dota1024": ("w1024", "t1.0n1024"),  # Phase D: single pass as trained + tiles at native resolution
    "dota2048": ("w2048", "t1.0n1024"),
}
PASS_MAX_DET = 500  # as the B-MZ predictor passes (max_det 500)


def _pad32(x: float) -> int:
    return int(math.ceil(x / STRIDE_MULTIPLE) * STRIDE_MULTIPLE)


def level_passes(level: Level, W: int, H: int, overlap: float) -> list[tuple[tuple[int, int, int, int], tuple[int, int]]]:
    """[(tile xyxy, letterbox input shape (h, w))] of one level."""
    if level.n_input is None:
        L = level.imgsz or _pad32(level.zoom * max(W, H))
        return [((0, 0, W, H), full_image_shape(H, W, L))]
    out = []
    for t in tile_grid(W, H, level.zoom, level.n_input, overlap):
        th, tw = t.y2 - t.y1, t.x2 - t.x1
        g = level.n_input / max(th, tw)
        out.append(((t.x1, t.y1, t.x2, t.y2), (_pad32(th * g), _pad32(tw * g))))
    return out


@torch.no_grad()
def raw_predict_level(model: torch.nn.Module, img: np.ndarray, level: Level, overlap: float, batch: int = 16,
                      device: str = "cuda") -> dict[str, np.ndarray]:
    H, W = img.shape[:2]
    passes = level_passes(level, W, H, overlap)
    out: dict[str, list[np.ndarray]] = {}  # keys f"{branch}_{xyxy|probs|std|tile}" for the branches the model has
    gains = []
    for s in range(0, len(passes), batch):
        chunk = passes[s:s + batch]
        shapes = {sh for _, sh in chunk}
        for sh in sorted(shapes):  # whole-image passes are alone; tiles of a level share one shape (edge cases split)
            sub = [(k, t) for k, (t, shp) in enumerate(chunk) if shp == sh]
            ims, lbs = [], []
            for _, (x1, y1, x2, y2) in sub:
                im, lb = letterbox(img[y1:y2, x1:x2], sh)
                ims.append(im)
                lbs.append(lb)
            raw = forward_raw(model, to_tensor(ims, device))
            for b in (b for b in BRANCHES if b in raw.boxes):
                for k in ("xyxy", "probs", "std", "tile"):
                    out.setdefault(f"{b}_{k}", [])
                d = decode(raw.boxes[b], raw.scores[b], raw.anchors, raw.strides, raw.reg_max)
                keep = d.probs.max(-1)[0] >= SCORE_MIN
                for i, ((k, (x1, y1, _, _)), lb) in enumerate(zip(sub, lbs)):
                    kk = keep[i]
                    out[f"{b}_xyxy"].append(input_to_source(d.xyxy[i][kk].cpu().numpy(), lb, (x1, y1)).astype(np.float32))
                    out[f"{b}_probs"].append(d.probs[i][kk].cpu().numpy().astype(np.float32))
                    out[f"{b}_std"].append((d.ltrb_std[i][kk].cpu().numpy() / lb.gain).astype(np.float32))
                    out[f"{b}_tile"].append(np.full(int(kk.sum()), s + k, np.int16))
            for (k, _), lb in zip(sub, lbs):
                gains.append((s + k, lb.gain))
    res = {k: np.concatenate(v) if v else np.zeros(0, np.float32) for k, v in out.items()}
    res["tiles"] = np.array([t for t, _ in passes], np.int32).reshape(-1, 4)
    res["gains"] = np.array([g for _, g in sorted(gains)], np.float64)
    res["zoom"] = np.float64(level.zoom)
    res["n_input"] = np.int32(level.n_input or 0)
    res["image_wh"] = np.array([W, H], np.int32)
    return res


def cache_dir(cache_root: Path, weights: str | Path, level_key: str, overlap: float, split: str) -> Path:
    return Path(cache_root) / weights_id(weights) / f"{level_key}_o{int(round(overlap * 100)):02d}" / split


def cache_level(model: torch.nn.Module, weights: str | Path, images: pd.DataFrame, level_key: str, overlap: float,
                split: str, cache_root: Path, device: str = "cuda") -> Path:
    """Write the cache of one level for all images (resumable)."""
    d = cache_dir(cache_root, weights, level_key, overlap, split)
    d.mkdir(parents=True, exist_ok=True)
    for stem, p in zip(images.stem, images.path):
        f = d / f"{stem}.npz"
        if f.exists():
            continue
        np.savez_compressed(f, **raw_predict_level(model, cv2.imread(p), LEVELS[level_key], overlap, device=device))
    return d


def pass_detections(z: dict[str, np.ndarray], branch: str, conf: float, device: str = "cpu",
                    max_det: int = PASS_MAX_DET, nms_kw: dict | None = None) -> pd.DataFrame:
    """Per-pass detections of one cached image/level as the Ultralytics predictor would output them
    (global px): columns x1, y1, x2, y2, score, cls, tile. ``nms_kw``: extra arguments of Ultralytics'
    ``non_max_suppression`` (DOTA: ``max_wh`` above the image size - global coordinates reach 13 383 px > the default class
    offset 7680 - and no time limit; VisDrone: defaults)."""
    import torchvision  # noqa: F401  Ultralytics' NMS uses the torchvision kernel only if it is imported, else its pure-torch
    #                     TorchNMS (identical keeps: 512/512 DOTA o2m tiles, 2026-10-30; 10-15x slower on CPU)
    from ultralytics.utils.nms import non_max_suppression

    from scahi.infer.dfl_decode import topk_like_ultralytics

    xyxy, probs, tile = z[f"{branch}_xyxy"], z[f"{branch}_probs"], z[f"{branch}_tile"]
    rows = []
    for t in np.unique(tile):
        m = tile == t
        bx, pr = torch.from_numpy(xyxy[m]).float(), torch.from_numpy(probs[m]).float()
        if branch == "o2o":
            r = topk_like_ultralytics(bx[None], pr[None], max_det)[0]
            r = r[r[:, 4] > conf]
        else:
            xywh = torch.cat([(bx[:, :2] + bx[:, 2:]) / 2, bx[:, 2:] - bx[:, :2]], 1)
            pred = torch.cat([xywh, pr], 1).T[None]  # [1, 4 + nc, n]
            r = non_max_suppression(pred, conf_thres=conf, iou_thres=0.7, max_det=max_det, max_nms=max(30000, max_det),
                                    **(nms_kw or {}))[0]
        r = r.numpy()
        rows.append(np.column_stack([r[:, :6], np.full(len(r), t)]))
    a = np.concatenate(rows) if rows else np.zeros((0, 7))
    return pd.DataFrame(a, columns=["x1", "y1", "x2", "y2", "score", "cls", "tile"])
