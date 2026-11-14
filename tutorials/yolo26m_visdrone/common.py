# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared helpers of the YOLO26m tutorial (frozen configurations, data, weights, Ultralytics-aligned caches, evaluation).

Only modules of the public repository are used. The frozen YOLO26m configurations are configs/ext/frozen_rows_ext26.json;
their stored results (calib / val / test-dev AP) are tutorials/yolo26m_visdrone/frozen_results.json.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
FREEZE = REPO / "configs/ext/frozen_rows_ext26.json"
RESULTS = HERE / "frozen_results.json"
SETTING = "visdrone10"
ALIGNED_ROOT = REPO / "cache_ualigned"  # caches with Ultralytics' input preparation (see the README, Track A)
TUTORIAL_DATA = REPO / "data/tutorial/visdrone/canonical/coco"
WEIGHTS_DIR = REPO / "weights"

LEVEL_TEXT = {"w1280": "1280 px whole-image pass", "w2048": "2048 px whole-image pass",
              "t2.0n1024": "zoom-2 tiles (1024 px input)"}


def freeze() -> dict[str, Any]:
    return json.loads(FREEZE.read_text())


def rows() -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in freeze()["rows"]}


def results() -> dict[str, Any]:
    return json.loads(RESULTS.read_text())


def describe(r: dict[str, Any]) -> str:
    """Plain description of a frozen configuration."""
    inf = r["inference"]
    head = "one-to-one head" if inf["branch"] == "o2o" else "one-to-many head"
    if r["kind"] == "single":
        return f"single pass at {inf['imgsz']} px, {head}"
    if r["kind"] == "sahi":
        return f"SAHI (640 px slices + full image, 1280 px weights, merge {inf['fusion']}), {head}"
    f = r["fusion"]
    passes = " + ".join(LEVEL_TEXT[lv] for lv in f["levels"])
    if f["variant"] == "F0-NMS":
        fus = "box fusion (NMS, intersection over smaller box)"
    elif r["id"].startswith("XG_"):
        fus = "SCAHI, refined grid (exploratory)"
    else:
        fus = "SCAHI, pre-specified setting" + (" (exploratory row)" if r["id"].startswith("D_") else "")
    return f"{passes}, {fus}, {head}"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def weights(size: int, given: str | None = None) -> Path:
    """Frozen YOLO26m weights of one training size: the frozen file or its published asset (same tensors, optimizer
    state stripped); any other file is refused (the frozen numbers would not apply)."""
    fz, res = freeze(), results()
    cands = [Path(given)] if given else [WEIGHTS_DIR / res["weights"][str(size)]["asset"],
                                         REPO / fz["weights"][str(size)]["path"]]
    ok = {fz["weights"][str(size)]["sha256"], *res["weights"][str(size)].get("asset_sha256", [])}
    for p in cands:
        if p.exists():
            if sha256(p) not in ok:
                raise SystemExit(f"{p}: SHA-256 is not that of the frozen YOLO26m {size} px weights or their asset")
            return p
    raise SystemExit(f"YOLO26m {size} px weights not found; download {res['weights'][str(size)]['asset']} from the "
                     f"repository's release assets into {WEIGHTS_DIR}/ (or pass --weights-{size})")


def split_data(split: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """COCO ground truth with the ignored regions and the image table of a VisDrone split: the canonical files of
    ``python -m scahi.reproduce convert`` if present, else the files of tutorials/yolo26m_visdrone/prepare_data.py."""
    from scahi.eval.evaluate import load_coco_gt
    from scahi.infer.full import image_table

    for f in (REPO / f"data/canonical/{SETTING}/coco/{split}.json", TUTORIAL_DATA / f"{split}.json"):
        if f.exists():
            coco = load_coco_gt(f)
            return coco, image_table(coco)
    raise SystemExit(f"no COCO file for {split}: run tutorials/yolo26m_visdrone/prepare_data.py (train / val) or "
                     "python -m scahi.reproduce convert (official data incl. test-dev)")


@contextlib.contextmanager
def ultralytics_caches(root: Path = ALIGNED_ROOT):
    """Caches built and read with Ultralytics' input preparation (size rule, letterbox factor, per-axis back-mapping),
    stored under ``root`` (default cache_ualigned/) - the convention of the frozen YOLO26m rows."""
    from scahi import mapcheck as MC
    from scahi.fusion import run as R

    saved = R.CACHE_ROOT
    R.CACHE_ROOT = Path(root)
    try:
        with MC.ultralytics_prep():
            yield Path(root)
    finally:
        R.CACHE_ROOT = saved


def evaluate(coco: dict[str, Any], dets: pd.DataFrame) -> dict[str, Any]:
    """VisDrone toolkit protocol (ignored regions, at most 500 detections per image) and COCO size bins."""
    from scahi.config import load_class_setting
    from scahi.eval.detections import finalize_detections
    from scahi.eval.evaluate import evaluate as ev

    return ev(coco, finalize_detections(dets), load_class_setting(SETTING), "visdrone_ignore")["metrics"]
