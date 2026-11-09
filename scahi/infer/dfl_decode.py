# SPDX-License-Identifier: AGPL-3.0-or-later
"""Own DFL decoding: softmax over bins, per-side expectation and variance, ltrb * stride.

All box outputs are in network-input pixels; map to global pixels with
:func:`scahi.infer.preprocess.input_to_source` (division by the zoom/gain plus tile offset).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Decoded:
    xyxy: torch.Tensor  # [B, A, 4] input pixels
    probs: torch.Tensor  # [B, A, nc] sigmoid class probabilities
    ltrb: torch.Tensor  # [B, A, 4] expected side distances (input pixels)
    ltrb_std: torch.Tensor  # [B, A, 4] standard deviation per side (input pixels)


def dfl_moments(box_logits: torch.Tensor, reg_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``[B, 4*reg_max, A]`` -> expectation and variance per side in bin units, each ``[B, 4, A]``.

    The channel layout matches ``ultralytics.nn.modules.block.DFL``: ``view(B, 4, reg_max, A)``."""
    b, _, a = box_logits.shape
    p = box_logits.view(b, 4, reg_max, a).softmax(2)
    bins = torch.arange(reg_max, device=p.device, dtype=p.dtype).view(1, 1, reg_max, 1)
    mean = (p * bins).sum(2)
    var = (p * bins**2).sum(2) - mean**2
    return mean, var.clamp_min(0)


def decode(box_logits: torch.Tensor, score_logits: torch.Tensor, anchors: torch.Tensor, strides: torch.Tensor,
           reg_max: int) -> Decoded:
    mean, var = dfl_moments(box_logits, reg_max)
    s = strides.view(1, 1, -1)
    ltrb = mean * s  # [B, 4, A]
    ax, ay = anchors[:, 0] * strides, anchors[:, 1] * strides
    xyxy = torch.stack([ax - ltrb[:, 0], ay - ltrb[:, 1], ax + ltrb[:, 2], ay + ltrb[:, 3]], -1)
    return Decoded(xyxy=xyxy, probs=score_logits.sigmoid().transpose(1, 2), ltrb=ltrb.transpose(1, 2),
                   ltrb_std=(var.sqrt() * s).transpose(1, 2))


def topk_like_ultralytics(xyxy: torch.Tensor, probs: torch.Tensor, max_det: int) -> torch.Tensor:
    """Re-implementation of ``Detect.postprocess`` (groups = 1, class-aware): two-stage top-k
    (anchors by max class score, then anchor x class pairs). Returns ``[B, k, 6]`` (xyxy, score, class)."""
    b, a, nc = probs.shape
    k = min(max_det, a)
    ori_index = probs.max(-1)[0].topk(k, dim=1)[1]
    sc = probs.gather(1, ori_index[..., None].expand(-1, -1, nc))
    scores, index = sc.flatten(1).topk(k, dim=1)
    anchor_idx = ori_index.gather(1, index // nc)
    boxes = xyxy.gather(1, anchor_idx[..., None].expand(-1, -1, 4))
    return torch.cat([boxes, scores[..., None], (index % nc)[..., None].float()], -1)
