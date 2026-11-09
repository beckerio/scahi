# SPDX-License-Identifier: AGPL-3.0-or-later
"""Convert original VisDrone annotations to the canonical COCO-JSON and YOLO formats.

Canonical COCO-JSON (one file per split and class setting) keeps every original row exactly
once: regular objects in ``annotations`` and ignore boxes in the extra field ``ignore``
(with ``ignore_type`` and the channel mask ``ignore_classes`` as COCO category ids).
Rows whose category is neither mapped nor ignored would be listed in ``unused``.

YOLO: ``yolo_ignore`` carries one extra label id per ignore type (``nc``, ``nc+1``, ...),
``yolo_plain`` drops ignore boxes (Ultralytics default behaviour, variant T0).
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from scahi.config import ClassSetting
from scahi.data.boxes import iou_matrix, xyxy_to_yolo
from scahi.data.splits import read_split, write_splits
from scahi.data.visdrone import parse_visdrone_text

SPLITS = ("train", "val", "test-dev")
YOLO_VARIANTS = ("yolo_ignore", "yolo_plain")


def convert_image(
    image_id: int, stem: str, rows: np.ndarray, width: int, height: int, cs: ClassSetting,
    ann_start: int, ign_start: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split the ``[N, 8]`` VisDrone rows of one image into annotations / ignore / unused."""
    anns, igns, unused = [], [], []
    for x, y, w, h, score, cat, trunc, occ in rows.tolist():
        base = {"image_id": image_id, "bbox": [float(x), float(y), float(w), float(h)],
                "orig_category": int(cat), "score": int(score), "truncation": int(trunc), "occlusion": int(occ)}
        itype = cs.ignore_type_for(cat, score)
        if itype is not None:
            igns.append({"id": ign_start + len(igns), **base, "ignore_type": itype.name,
                         "ignore_classes": [c + 1 for c in itype.class_ids]})
        elif cat in cs.category_to_class:
            anns.append({"id": ann_start + len(anns), **base, "category_id": cs.category_to_class[cat] + 1,
                         "area": float(w * h), "iscrowd": 0})
        else:
            unused.append(base)
    return anns, igns, unused


def coco_to_visdrone_rows(coco: dict[str, Any], image_id: int) -> np.ndarray:
    """Inverse of :func:`convert_image` for one image (row order not preserved)."""
    rows = []
    for key in ("annotations", "ignore", "unused"):
        for a in coco.get(key, []):
            if a["image_id"] == image_id:
                rows.append([*a["bbox"], a["score"], a["orig_category"], a["truncation"], a["occlusion"]])
    return np.asarray(rows, dtype=np.int64).reshape(-1, 8)


TRAIN_DUPLICATE_IOU = 0.99


def yolo_lines(
    coco_img: dict, anns: list[dict], igns: list[dict], cs: ClassSetting, with_ignore: bool
) -> tuple[list[str], dict[str, int]]:
    """YOLO label lines for *training* (boxes clipped to the image).

    Training-only GT cleaning (evaluation GT stays untouched): boxes narrower/lower than 1 px
    after clipping are dropped, and duplicate objects (same class, IoU >= 0.99) keep only the
    first occurrence. Returns ``(lines, {"degenerate": n, "duplicate": n})``.
    """
    W, H = coco_img["width"], coco_img["height"]
    ign_ids = {t.name: t.yolo_id for t in cs.ignore_types}
    items = [(a["category_id"] - 1, a["bbox"], True) for a in anns]
    if with_ignore:
        items += [(ign_ids[g["ignore_type"]], g["bbox"], False) for g in igns]
    lines: list[str] = []
    counts = {"degenerate": 0, "duplicate": 0}
    kept_obj: dict[int, list[list[float]]] = {}
    for cls, (x, y, w, h), is_object in items:
        x1, y1 = min(max(x, 0.0), W), min(max(y, 0.0), H)
        x2, y2 = min(max(x + w, 0.0), W), min(max(y + h, 0.0), H)
        if x2 - x1 < 1 or y2 - y1 < 1:
            counts["degenerate"] += 1
            continue
        if is_object:
            prev = kept_obj.setdefault(cls, [])
            if prev and iou_matrix([[x1, y1, x2, y2]], prev).max() >= TRAIN_DUPLICATE_IOU:
                counts["duplicate"] += 1
                continue
            prev.append([x1, y1, x2, y2])
        cx, cy, bw, bh = xyxy_to_yolo([[x1, y1, x2, y2]], W, H)[0]
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines, counts


def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    os.symlink(src, dst)


def convert_split(orig_split_dir: Path, split: str, cs: ClassSetting, out_root: Path) -> dict[str, Any]:
    ann_dir, img_dir = orig_split_dir / "annotations", orig_split_dir / "images"
    stems = sorted(p.stem for p in ann_dir.glob("*.txt"))
    coco: dict[str, Any] = {
        "info": {"class_setting": cs.name, "split": split, "source": str(orig_split_dir),
                 "created": dt.datetime.now().isoformat(timespec="seconds")},
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(cs.classes)],
        "ignore_types": [{"name": t.name, "yolo_id": t.yolo_id, "categories": list(t.categories),
                          "score_zero": t.score_zero, "ignore_classes": [c + 1 for c in t.class_ids]}
                         for t in cs.ignore_types],
        "images": [], "annotations": [], "ignore": [], "unused": [],
    }
    stats: Counter[str] = Counter()
    for v in YOLO_VARIANTS:
        (out_root / v / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / v / "labels" / split).mkdir(parents=True, exist_ok=True)
    for image_id, stem in enumerate(stems, start=1):
        img_path = img_dir / f"{stem}.jpg"
        with Image.open(img_path) as im:
            W, H = im.size
        img = {"id": image_id, "file_name": f"{stem}.jpg", "width": W, "height": H, "path": str(img_path)}
        coco["images"].append(img)
        rows = parse_visdrone_text((ann_dir / f"{stem}.txt").read_text())
        anns, igns, unused = convert_image(image_id, stem, rows, W, H, cs,
                                           len(coco["annotations"]) + 1, len(coco["ignore"]) + 1)
        coco["annotations"] += anns
        coco["ignore"] += igns
        coco["unused"] += unused
        stats["rows"] += len(rows)
        stats["boxes_outside_image"] += int(((rows[:, 0] < 0) | (rows[:, 1] < 0) | (rows[:, 0] + rows[:, 2] > W)
                                             | (rows[:, 1] + rows[:, 3] > H)).sum())
        stats["zero_size_boxes"] += int(((rows[:, 2] <= 0) | (rows[:, 3] <= 0)).sum())
        for v in YOLO_VARIANTS:
            lines, cnt = yolo_lines(img, anns, igns, cs, with_ignore=(v == "yolo_ignore"))
            stats[f"{v}_dropped_degenerate"] += cnt["degenerate"]
            stats[f"{v}_dropped_duplicate"] += cnt["duplicate"]
            (out_root / v / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            _link(img_path, out_root / v / "images" / split / f"{stem}.jpg")
    (out_root / "coco").mkdir(parents=True, exist_ok=True)
    (out_root / "coco" / f"{split}.json").write_text(json.dumps(coco))
    stats.update(images=len(coco["images"]), annotations=len(coco["annotations"]),
                 ignore=len(coco["ignore"]), unused=len(coco["unused"]))
    stats.update({f"ignore_{k}": v for k, v in Counter(g["ignore_type"] for g in coco["ignore"]).items()})
    stats.update({f"class_{cs.classes[k - 1]}": v for k, v in
                  sorted(Counter(a["category_id"] for a in coco["annotations"]).items())})
    return dict(stats)


def write_lists_and_yaml(out_root: Path, cs: ClassSetting, splits_dir: Path) -> None:
    lists = {
        "train": None, "val": None, "test-dev": None,
        "train_main": read_split(splits_dir / "train_main.txt"),
        "calib": read_split(splits_dir / "calib.txt"),
    }
    for v in YOLO_VARIANTS:
        root = out_root / v
        for name, stems in lists.items():
            split = name if stems is None else "train"
            stems = stems if stems is not None else sorted(p.stem for p in (root / "images" / split).glob("*.jpg"))
            (root / f"{name}.txt").write_text("".join(f"{root / 'images' / split / s}.jpg\n" for s in stems))
        names = cs.yolo_names if v == "yolo_ignore" else list(cs.classes)
        ds = {
            "path": str(root),
            "train": "train_main.txt",
            "val": "calib.txt",
            "test": "val.txt",
            "nc": len(names),
            "names": dict(enumerate(names)),
            # scahi-specific keys (ignored by Ultralytics)
            "scahi_class_setting": cs.name,
            "scahi_nc_model": cs.nc,
            "scahi_ignore_ids": ({t.yolo_id: {"name": t.name, "classes": list(t.class_ids)} for t in cs.ignore_types}
                                 if v == "yolo_ignore" else {}),
        }
        (root / "dataset.yaml").write_text(yaml.safe_dump(ds, sort_keys=False))


def run_convert(original_root: Path, split_dirs: dict[str, str], cs: ClassSetting, out_root: Path,
                splits_dir: Path) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    if not (splits_dir / "calib.txt").exists():
        train_stems = sorted(p.stem for p in (original_root / split_dirs["train"] / "annotations").glob("*.txt"))
        info = write_splits(train_stems, original_root / split_dirs["train"] / "annotations", splits_dir)
        (splits_dir / "split_info.json").write_text(json.dumps(info, indent=2))
    stats = {s: convert_split(original_root / split_dirs[s], s, cs, out_root) for s in SPLITS}
    write_lists_and_yaml(out_root, cs, splits_dir)
    (out_root / "convert_stats.json").write_text(json.dumps(stats, indent=2))
    return stats
