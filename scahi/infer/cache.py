# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sparse raw-prediction cache: ``cache/<weights_id>/<zoom>/<stem>.npz`` (one file per image and zoom).

Per branch (``o2m``, ``o2o``) the arrays of all tiles are concatenated, keeping anchors whose max class
probability is ``>= SCORE_MIN``: ``{b}_xyxy`` (global px), ``{b}_probs`` [n, nc], ``{b}_std`` (DFL std per
side, global px), ``{b}_level``, ``{b}_anchor``, ``{b}_tile``. ``tiles`` holds the tile geometry
(x1, y1, x2, y2) in original pixels; ``zoom`` and ``n_input`` describe the pass. Fusion ablations run
from these files without touching the network again.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from scahi.data.zoom import Tile, tile_grid
from scahi.infer.dfl_decode import decode
from scahi.infer.preprocess import full_image_shape, input_to_source, letterbox, to_tensor
from scahi.infer.raw_head import forward_raw

SCORE_MIN = 0.001
BRANCHES = ("o2m", "o2o")


def weights_id(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def zoom_key(zoom: float) -> str:
    return f"z{zoom:.4f}"


@torch.no_grad()
def raw_predict_image(model: torch.nn.Module, img: np.ndarray, zoom: float, n_input: int = 640, overlap: float = 0.25,
                      batch: int = 16, device: str = "cuda") -> dict[str, np.ndarray]:
    H, W = img.shape[:2]
    tiles: list[Tile] = tile_grid(W, H, zoom, n_input, overlap)
    out: dict[str, list[np.ndarray]] = {f"{b}_{k}": [] for b in BRANCHES for k in ("xyxy", "probs", "std", "level", "anchor", "tile")}
    for s in range(0, len(tiles), batch):
        chunk = tiles[s:s + batch]
        ims, lbs = [], []
        for t in chunk:
            crop = img[t.y1:t.y2, t.x1:t.x2]
            shape = full_image_shape(t.y2 - t.y1, t.x2 - t.x1, int(round(max(H, W) * zoom))) if t.whole_image else (n_input, n_input)
            im, lb = letterbox(crop, shape)
            ims.append(im)
            lbs.append(lb)
        raw = forward_raw(model, to_tensor(ims, device))
        for b in BRANCHES:
            d = decode(raw.boxes[b], raw.scores[b], raw.anchors, raw.strides, raw.reg_max)
            keep = d.probs.max(-1)[0] >= SCORE_MIN
            for i, (t, lb) in enumerate(zip(chunk, lbs)):
                k = keep[i]
                out[f"{b}_xyxy"].append(input_to_source(d.xyxy[i][k].cpu().numpy(), lb, (t.x1, t.y1)).astype(np.float32))
                out[f"{b}_probs"].append(d.probs[i][k].cpu().numpy().astype(np.float32))
                out[f"{b}_std"].append((d.ltrb_std[i][k].cpu().numpy() / lb.gain).astype(np.float32))
                out[f"{b}_level"].append(raw.level[k].cpu().numpy().astype(np.int8))
                out[f"{b}_anchor"].append(torch.nonzero(k).view(-1).cpu().numpy().astype(np.int32))
                out[f"{b}_tile"].append(np.full(int(k.sum()), s + i, np.int16))
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["tiles"] = np.array([(t.x1, t.y1, t.x2, t.y2) for t in tiles], np.int32)
    res["zoom"] = np.float64(zoom)
    res["n_input"] = np.int32(n_input)
    res["image_wh"] = np.array([W, H], np.int32)
    return res


def cache_images(weights: str | Path, image_paths: list[str], zooms: list[float], cache_root: Path,
                 n_input: int = 640, overlap: float = 0.25, device: str = "cuda") -> Path:
    """Write the cache for all images and zooms; existing files are kept (resumable)."""
    from scahi.infer.raw_head import load_unfused

    root = Path(cache_root) / weights_id(weights)
    model = load_unfused(weights, device)
    for z in zooms:
        zdir = root / zoom_key(z)
        zdir.mkdir(parents=True, exist_ok=True)
        for p in image_paths:
            f = zdir / f"{Path(p).stem}.npz"
            if f.exists():
                continue
            np.savez_compressed(f, **raw_predict_image(model, cv2.imread(p), z, n_input, overlap, device=device))
    return root
