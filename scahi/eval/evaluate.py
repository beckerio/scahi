# SPDX-License-Identifier: AGPL-3.0-or-later
"""``scahi eval``: evaluate detections against the canonical GT under a named protocol.

Protocols:

* ``visdrone_ignore`` (and ``visdrone_strict`` for person_vehicle): toolkit drop rule with the
  setting's ``drop_regions``; headline AP/AR from the exact toolkit port; per-class, size-bin
  and a second overall number from pycocotools on the *same* dropped data (maxDets 1/10/100/500).
* ``plain``: no ignore handling at all, pycocotools with maxDets 1/10/100 (Ultralytics-style).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scahi.config import ClassSetting
from scahi.eval.coco_eval import coco_metrics
from scahi.eval.detections import MAX_DETS_PER_IMAGE, finalize_detections, sort_per_image
from scahi.eval.diagnostics import diagnostics
from scahi.eval.visdrone_protocol import ImageEval, calc_accuracy, drop_masks

DIAG_CONF = 0.25


def load_coco_gt(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _group(items: list[dict], key: str = "image_id") -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for a in items:
        out.setdefault(a[key], []).append(a)
    return out


def prepare_images(
    coco: dict[str, Any], dets: pd.DataFrame, cs: ClassSetting, protocol: str,
) -> list[dict[str, Any]]:
    """Per image (sorted by file name): GT and detections after the protocol's drop rule."""
    dets = sort_per_image(dets)
    det_groups = {k: g for k, g in dets.groupby("image_id")}
    anns, igns = _group(coco["annotations"]), _group(coco["ignore"])
    if protocol == "plain":
        drop_regions: dict[int | None, tuple[int, ...]] = {}
    else:
        drop_regions = cs.eval_protocols[protocol].drop_regions
    out = []
    for img in sorted(coco["images"], key=lambda i: i["file_name"]):
        iid, W, H = img["id"], img["width"], img["height"]
        a = anns.get(iid, [])
        gt_xywh = np.array([x["bbox"] for x in a], np.float64).reshape(-1, 4)
        gt_cls = np.array([x["category_id"] for x in a], np.int64)
        gt_ign = np.array([x["score"] == 0 for x in a], bool)
        g = det_groups.get(iid)
        if g is None:
            dt_xywh, dt_score, dt_cls = np.zeros((0, 4)), np.zeros(0), np.zeros(0, np.int64)
        else:
            dt_xywh = g[["x", "y", "w", "h"]].to_numpy(np.float64)
            dt_score = g["score"].to_numpy(np.float64)
            dt_cls = g["category_id"].to_numpy(np.int64)
        if drop_regions:
            ig = igns.get(iid, [])
            regions = {}
            for tgt, cats in drop_regions.items():
                boxes = [x["bbox"] for x in ig if x["orig_category"] in cats]
                if boxes:
                    regions[None if tgt is None else tgt + 1] = np.array(boxes, np.float64)
            if regions:
                kg, kd = drop_masks(gt_xywh, gt_cls, dt_xywh, dt_cls, regions, H, W)
                gt_xywh, gt_cls, gt_ign = gt_xywh[kg], gt_cls[kg], gt_ign[kg]
                dt_xywh, dt_score, dt_cls = dt_xywh[kd], dt_score[kd], dt_cls[kd]
        out.append({"image_id": iid, "file_name": img["file_name"], "width": W, "height": H,
                    "gt_xywh": gt_xywh, "gt_cls": gt_cls, "gt_ignore": gt_ign,
                    "dt_xywh": dt_xywh, "dt_score": dt_score, "dt_cls": dt_cls})
    return out


def _to_coco_lists(images: list[dict[str, Any]], coco: dict[str, Any]) -> tuple[list, list, list]:
    imgs = [{"id": im["image_id"], "width": im["width"], "height": im["height"]} for im in images]
    anns, dets = [], []
    for im in images:
        for (x, y, w, h), c, ig in zip(im["gt_xywh"], im["gt_cls"], im["gt_ignore"]):
            anns.append({"id": len(anns) + 1, "image_id": im["image_id"], "bbox": [x, y, w, h], "area": w * h,
                         "category_id": int(c), "iscrowd": int(ig)})
        for (x, y, w, h), s, c in zip(im["dt_xywh"], im["dt_score"], im["dt_cls"]):
            dets.append({"image_id": im["image_id"], "bbox": [x, y, w, h], "score": float(s), "category_id": int(c)})
    return imgs, anns, dets


def evaluate(
    coco: dict[str, Any], dets: pd.DataFrame, cs: ClassSetting, protocol: str,
    tiles: dict[int, np.ndarray] | None = None, diag_conf: float = DIAG_CONF, with_toolkit: bool = True,
    port_only: bool = False,
) -> dict[str, Any]:
    """``port_only=True`` (model selection / tuning): toolkit-port metrics only, no pycocotools breakdowns or
    diagnostics; the headline numbers are identical to the full evaluation."""
    n_in = len(dets)
    dets = finalize_detections(dets, MAX_DETS_PER_IMAGE)  # idempotent; enforces the common contract
    n_capped = n_in - len(dets)
    images = prepare_images(coco, dets, cs, protocol)
    cats = coco["categories"]
    names = {c["id"]: c["name"] for c in cats}
    if port_only and protocol != "plain":
        tk = calc_accuracy(
            [ImageEval(i["gt_xywh"], i["gt_cls"], i["gt_ignore"], i["dt_xywh"], i["dt_score"], i["dt_cls"]) for i in images],
            tuple(sorted(names)),
        )
        return {"eval_protocol": protocol, "metrics": {**tk.summary(), "sources": "toolkit_port"}, "diagnostics": {},
                "eval_info": {"headline": "toolkit_port", "port_only": True, "n_images": len(images), "n_det_in": n_in,
                              "n_det_capped": int(n_capped)}}
    imgs, anns, dts = _to_coco_lists(images, coco)
    if protocol == "plain":
        cm = coco_metrics(imgs, anns, cats, dts, max_dets=(1, 10, 100))
        metrics = {k: cm[k] for k in ("AP", "AP50", "AP75", "AR1", "AR10", "AR100")}
        metrics["AR500"] = None
        metrics["AP_per_class"] = {names[c]: v for c, v in cm["AP_per_class"].items()}
        source = {"headline": "pycocotools", "max_dets": cm["max_dets"]}
    else:
        cm = coco_metrics(imgs, anns, cats, dts, max_dets=(1, 10, 100, 500))
        if with_toolkit:
            tk = calc_accuracy(
                [ImageEval(i["gt_xywh"], i["gt_cls"], i["gt_ignore"], i["dt_xywh"], i["dt_score"], i["dt_cls"]) for i in images],
                tuple(sorted(names)),
            )
            metrics = tk.summary()
            metrics["AP_per_class"] = {names[c]: v for c, v in tk.ap_per_class().items()}
            source = {"headline": "toolkit_port", "toolkit_images_per_class": {names[c]: n for c, n in tk.images_per_class.items()}}
            metrics["AP_per_class_source"] = "toolkit_port (unweighted per-class AP; headline = image-count-weighted mean)"
        else:
            metrics = {k: cm[k] for k in ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")}
            metrics["AP_per_class"] = {names[c]: v for c, v in cm["AP_per_class"].items()}
            source = {"headline": "pycocotools"}
        metrics["coco"] = {k: cm[k] for k in ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")}
        metrics["coco"]["AP_per_class"] = {names[c]: v for c, v in cm["AP_per_class"].items()}
    metrics["AP_by_size"] = cm["AP_by_size"]
    metrics["AP50_by_size"] = cm["AP50_by_size"]
    metrics["AR_by_size"] = cm["AR_by_size"]
    metrics["size_metrics_source"] = "pycocotools"
    # Explicit provenance of every reported number (never mix sources unlabelled in one column).
    head = "pycocotools" if protocol == "plain" or not with_toolkit else "toolkit_port"
    metrics["sources"] = {**{k: head for k in ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500", "AP_per_class")},
                          "coco": "pycocotools", "AP_by_size": "pycocotools", "AP50_by_size": "pycocotools",
                          "AR_by_size": "pycocotools", "diagnostics": "scahi.eval.diagnostics"}
    diag = diagnostics(
        [{"image_id": i["image_id"], "width": i["width"], "height": i["height"], "gt_xywh": i["gt_xywh"],
          "gt_cls": i["gt_cls"], "dt_xywh": i["dt_xywh"], "dt_score": i["dt_score"], "dt_cls": i["dt_cls"]} for i in images],
        conf=diag_conf, tiles=tiles,
    )
    n_dropped_dt = len(dets) - sum(len(i["dt_cls"]) for i in images)
    return {"eval_protocol": protocol, "metrics": metrics, "diagnostics": diag,
            "eval_info": {**source, "n_images": len(images), "n_gt": len(anns), "n_det_in": n_in,
                          "n_det_capped": int(n_capped), "max_dets_per_image": MAX_DETS_PER_IMAGE,
                          "n_det_dropped": int(n_dropped_dt)}}
