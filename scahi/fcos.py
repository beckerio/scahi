# SPDX-License-Identifier: AGPL-3.0-or-later
"""FCOS feasibility spike (host-a brief 2026-11-01, task 6): torchvision ``fcos_resnet50_fpn`` (COCO_V1 weights) as a
second dense detector family for SCAHI. Smoke tests only; no full training without the user's approval.

* ``load``: the torchvision model with COCO weights; for DOTA a fresh 16-class classification layer (prior 0.01,
  the conv tower and the regression / centerness branch keep their COCO weights); fixed long side per pass
  (``min_size = max_size = L`` in torchvision's transform: scale = L / long side); raised limits for dense images.
* ``dense_pass``: dense per-position outputs of one pass (one crop) - decoded boxes (torchvision's ``BoxLinearCoder``,
  crop px), class probabilities sigmoid(cls), centerness sigmoid(ctr), and the detector's own detections of the pass
  (torchvision's ``postprocess_detections`` + ``transform.postprocess`` of the same forward).
* ``raw_predict_level``: the SCAHI cache of one level (whole image or tiles, ``scahi.infer.cache_mz`` layout and keys,
  branch name "o2m" as for the single-head YOLO11/12; ``std`` = 0: FCOS has no DFL distribution). Score = the
  detector's own score sqrt(p_c * ctr) per class (preregistration addendum 3, item 5; the spike used p_c * ctr);
  positions with max_c sqrt(p_c * ctr) >= ``SCORE_MIN`` are kept. Additionally the per-pass detections
  (``o2m_det_xyxy`` / ``_score`` / ``_cls`` / ``_tile``, image px, clipped to the pass) for box fusion.
* ``postprocess_like_torchvision``: torchvision's own post-processing re-implemented on the dense outputs (decoding
  parity check).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

SCORE_MIN = 0.001  # as scahi.infer.cache_mz.SCORE_MIN (on the score sqrt(p_c * ctr))
BRANCH = "o2m"


@dataclass
class Limits:
    score_thresh: float = 0.001
    topk_candidates: int = 200_000  # per level (torchvision default 1000)
    detections_per_img: int = 200_000  # DOTA pipeline limit (torchvision default 100)
    nms_thresh: float = 0.6


def load(num_classes: int | None = None, long_side: int = 1024, limits: Limits | None = None, device: str = "cuda",
         state_dict: dict | None = None):
    from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn
    from torchvision.models.detection.fcos import FCOSClassificationHead

    m = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1)
    if num_classes is not None and num_classes != m.head.classification_head.num_classes:
        old = m.head.classification_head
        new = FCOSClassificationHead(256, old.num_anchors, num_classes)
        new.conv.load_state_dict(old.conv.state_dict())  # COCO conv tower kept, only the output layer is new
        m.head.classification_head = new
    lim = limits or Limits()
    m.score_thresh, m.topk_candidates, m.detections_per_img, m.nms_thresh = (
        lim.score_thresh, lim.topk_candidates, lim.detections_per_img, lim.nms_thresh)
    set_long_side(m, long_side)
    if state_dict is not None:
        m.load_state_dict(state_dict)
    return m.to(device)


def set_long_side(m, long_side: int) -> None:
    m.transform.min_size, m.transform.max_size = (long_side,), long_side


def _to_tensor(img_bgr: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img_bgr[:, :, ::-1])).to(device).permute(2, 0, 1).float() / 255.0


@torch.no_grad()
def dense_pass(m, crop_bgr: np.ndarray, long_side: int, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Dense outputs of one pass; boxes in crop px. Keys: xyxy [N, 4], probs [N, C] (sigmoid), ctr [N],
    level [N], gain (input px per crop px), and the per-level split for the parity check."""
    set_long_side(m, long_side)
    m.eval()
    H, W = crop_bgr.shape[:2]
    images, _ = m.transform([_to_tensor(crop_bgr, device)])
    feats = list(m.backbone(images.tensors).values())
    head = m.head(feats)
    anchors = m.anchor_generator(images, feats)[0]
    n_per_level = [f.size(2) * f.size(3) for f in feats]
    h_in, w_in = images.image_sizes[0]
    boxes_in = m.box_coder.decode(head["bbox_regression"][0], anchors)
    gain_y, gain_x = h_in / H, w_in / W  # torchvision resizes both axes by the same scale up to rounding of the size
    scale = torch.tensor([gain_x, gain_y, gain_x, gain_y], device=boxes_in.device, dtype=boxes_in.dtype)
    split_head = {k: list(v.split(n_per_level, dim=1)) for k, v in head.items()}
    dets = m.postprocess_detections(split_head, [list(anchors.split(n_per_level))], images.image_sizes)
    dets = m.transform.postprocess(dets, images.image_sizes, [(H, W)])[0]
    return {"xyxy_in": boxes_in, "xyxy": boxes_in / scale, "probs": torch.sigmoid(head["cls_logits"][0]),
            "ctr": torch.sigmoid(head["bbox_ctrness"][0, :, 0]), "level": torch.repeat_interleave(
                torch.arange(len(n_per_level), device=device), torch.tensor(n_per_level, device=device)),
            "n_per_level": n_per_level, "image_size_in": (h_in, w_in), "gain": float(gain_x), "gain_xy": (gain_x, gain_y),
            "head": head, "anchors": anchors, "dets": dets}


def postprocess_like_torchvision(d: dict, m, original_hw: tuple[int, int]) -> dict[str, torch.Tensor]:
    """torchvision's FCOS.postprocess_detections + transform.postprocess on the dense outputs ``d`` of one image."""
    from torchvision.ops import boxes as box_ops

    h_in, w_in = d["image_size_in"]
    scores_all = torch.sqrt(d["probs"] * d["ctr"][:, None])
    boxes, scores, labels = [], [], []
    start = 0
    for n in d["n_per_level"]:
        sc = scores_all[start:start + n].flatten()
        keep = sc > m.score_thresh
        sc = sc[keep]
        idx = torch.where(keep)[0]
        k = min(m.topk_candidates, idx.numel())
        sc, o = sc.topk(k)
        idx = idx[o]
        nc = scores_all.shape[1]
        a_idx = torch.div(idx, nc, rounding_mode="floor") + start
        b = box_ops.clip_boxes_to_image(d["xyxy_in"][a_idx], (h_in, w_in))
        boxes.append(b), scores.append(sc), labels.append(idx % nc)
        start += n
    boxes, scores, labels = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
    keep = box_ops.batched_nms(boxes, scores, labels, m.nms_thresh)[: m.detections_per_img]
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    rh, rw = original_hw[0] / h_in, original_hw[1] / w_in  # transform.postprocess (resize_boxes)
    boxes = boxes * torch.tensor([rw, rh, rw, rh], device=boxes.device, dtype=boxes.dtype)
    return {"boxes": boxes, "scores": scores, "labels": labels}


def score(d: dict) -> torch.Tensor:
    """The detector's own score per position and class: sqrt(p_c * ctr) (torchvision post-processing)."""
    return torch.sqrt(d["probs"] * d["ctr"][:, None])


def raw_predict_level(m, img_bgr: np.ndarray, level, overlap: float, device: str = "cuda") -> dict[str, np.ndarray]:
    """SCAHI cache of one level (``scahi.infer.cache_mz`` layout, see the module docstring)."""
    from scahi.infer.cache_mz import level_passes

    H, W = img_bgr.shape[:2]
    passes = level_passes(level, W, H, overlap)
    keys = ("xyxy", "probs", "tile", "det_xyxy", "det_score", "det_cls", "det_tile")
    out: dict[str, list] = {k: [] for k in keys}
    gains = []
    for k, ((x1, y1, x2, y2), shape) in enumerate(passes):
        long_side = max(shape)  # the network-input long side of this pass (as the YOLO passes)
        d = dense_pass(m, img_bgr[y1:y2, x1:x2], long_side, device)
        p = score(d)
        keep = p.max(1).values >= SCORE_MIN
        off = torch.tensor([x1, y1, x1, y1], device=p.device, dtype=d["xyxy"].dtype)
        out["xyxy"].append((d["xyxy"][keep] + off).cpu().numpy().astype(np.float32))
        out["probs"].append(p[keep].cpu().numpy().astype(np.float32))
        out["tile"].append(np.full(int(keep.sum()), k, np.int16))
        dt = d["dets"]
        out["det_xyxy"].append((dt["boxes"] + off).cpu().numpy().astype(np.float32))
        out["det_score"].append(dt["scores"].cpu().numpy().astype(np.float32))
        out["det_cls"].append(dt["labels"].cpu().numpy().astype(np.int16))
        out["det_tile"].append(np.full(len(dt["scores"]), k, np.int16))
        gains.append(d["gain"])
    cat = {k: np.concatenate(v) for k, v in out.items()}
    n = len(cat["tile"])
    res = {f"{BRANCH}_{k}": v for k, v in cat.items()}
    res[f"{BRANCH}_std"] = np.zeros((n, 4), np.float32)
    res.update(tiles=np.array([t for t, _ in passes], np.int32).reshape(-1, 4), gains=np.array(gains, np.float64),
               zoom=np.float64(level.zoom), n_input=np.int32(level.n_input or 0), image_wh=np.array([W, H], np.int32))
    return res
