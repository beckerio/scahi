# SPDX-License-Identifier: AGPL-3.0-or-later
"""Raw access to both YOLOv10 head branches (one-to-many and one-to-one) of an *unfused* model.

Verified against ultralytics 8.4.183 (docs/DECISIONS.md, 2026-10-22): in eval mode the unfused
``DetectionModel`` returns ``(y, preds)`` with
``preds = {"one2many": {"boxes": [B, 4*reg_max, A], "scores": [B, nc, A], "feats": [P3, P4, P5]},
"one2one": {...}}``; anchors of all levels are concatenated (P3, P4, P5, row-major per level).
``y`` is the decoded output of the branch selected by ``head.end2end``: o2m -> ``[B, 4 + nc, A]``
(xywh, sigmoid scores); o2o -> ``[B, max_det, 6]`` (xyxy, score, class) after top-k.
Fusing (``model.fuse()`` / the predictor) deletes the non-selected branch, hence no fusing here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class RawHead:
    boxes: dict[str, torch.Tensor]  # branch -> [B, 4*reg_max, A] DFL logits
    scores: dict[str, torch.Tensor]  # branch -> [B, nc, A] class logits
    anchors: torch.Tensor  # [A, 2] anchor centres in grid units (cell + 0.5)
    strides: torch.Tensor  # [A] stride per anchor
    level: torch.Tensor  # [A] level index (0 = P3)
    level_hw: list[tuple[int, int]]
    y_ultralytics: torch.Tensor  # the model's own decoded output for the selected branch
    reg_max: int


def load_unfused(weights: str | Path, device: str = "cuda", end2end: bool = True) -> torch.nn.Module:
    """``DetectionModel`` in eval mode, float32, both branches kept. ``end2end`` only selects which branch
    the model's *own* output ``y`` decodes (used by the parity test)."""
    from ultralytics import YOLO

    m = YOLO(str(weights)).model.float().eval().to(device)
    if hasattr(m.model[-1], "one2one_cv2"):  # YOLOv10; single-head models (YOLO11/12) have no branch to select
        m.end2end = end2end
    return m


def anchors_for(level_hw: list[tuple[int, int]], strides: list[float], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pts, st, lv = [], [], []
    for k, ((h, w), s) in enumerate(zip(level_hw, strides)):
        sy, sx = torch.meshgrid(torch.arange(h, device=device) + 0.5, torch.arange(w, device=device) + 0.5, indexing="ij")
        pts.append(torch.stack((sx, sy), -1).view(-1, 2))
        st.append(torch.full((h * w,), float(s), device=device))
        lv.append(torch.full((h * w,), k, device=device, dtype=torch.long))
    return torch.cat(pts), torch.cat(st), torch.cat(lv)


@torch.no_grad()
def forward_raw(model: torch.nn.Module, x: torch.Tensor) -> RawHead:
    head = model.model[-1]
    y, preds = model(x)
    if "one2one" in preds:  # YOLOv10: two branches
        feats = preds["one2many"]["feats"]
        boxes = {b: preds[k]["boxes"] for b, k in (("o2m", "one2many"), ("o2o", "one2one"))}
        scores = {b: preds[k]["scores"] for b, k in (("o2m", "one2many"), ("o2o", "one2one"))}
    elif "boxes" in preds:  # single dense head (YOLO11/12): structurally the o2m head -> branch "o2m"
        feats = preds["feats"]
        boxes, scores = {"o2m": preds["boxes"]}, {"o2m": preds["scores"]}
    else:
        raise RuntimeError("unexpected head output (fused model?)")
    level_hw = [(f.shape[2], f.shape[3]) for f in feats]
    anchors, strides, level = anchors_for(level_hw, head.stride.tolist(), x.device)
    return RawHead(
        boxes=boxes,
        scores=scores,
        anchors=anchors, strides=strides, level=level, level_hw=level_hw, y_ultralytics=y, reg_max=head.reg_max,
    )


@torch.no_grad()
def forward_raw_branch(model: torch.nn.Module, x: torch.Tensor, branch: str = "o2m") -> RawHead:
    """Raw head outputs of **one** branch only (extension E2, optimised path): backbone and neck as in
    ``DetectionModel._predict_once``, then only the requested head (``Detect.forward_head``) - the other YOLOv10 branch and
    Ultralytics' own decoding are skipped. Outputs are bit-identical to ``forward_raw(...)[branch]`` (same modules, same
    inputs; parity test ``scripts/e2_head_parity.py``)."""
    y: list = []
    feats = x
    for m in model.model[:-1]:
        if m.f != -1:
            feats = y[m.f] if isinstance(m.f, int) else [feats if j == -1 else y[j] for j in m.f]
        feats = m(feats)
        y.append(feats if m.i in model.save else None)
    head = model.model[-1]
    hx = y[head.f] if isinstance(head.f, int) else [feats if j == -1 else y[j] for j in head.f]
    heads = head.one2one if branch == "o2o" else head.one2many
    p = head.forward_head(hx, **heads)
    level_hw = [(f.shape[2], f.shape[3]) for f in p["feats"]]
    anchors, strides, level = anchors_for(level_hw, head.stride.tolist(), x.device)
    return RawHead(boxes={branch: p["boxes"]}, scores={branch: p["scores"]}, anchors=anchors, strides=strides, level=level,
                   level_hw=level_hw, y_ultralytics=None, reg_max=head.reg_max)
