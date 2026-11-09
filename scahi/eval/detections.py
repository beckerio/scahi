# SPDX-License-Identifier: AGPL-3.0-or-later
"""Detection containers and file formats.

Canonical detections: a DataFrame with columns ``image_id, x, y, w, h, score, category_id``
(``image_id`` and ``category_id`` as in the canonical COCO-JSON; boxes in original pixels).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DET_COLUMNS = ["image_id", "x", "y", "w", "h", "score", "category_id"]


def load_detections(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".json":
        rows = json.loads(path.read_text())
        df = pd.DataFrame(
            [(r["image_id"], *r["bbox"], r["score"], r["category_id"]) for r in rows], columns=DET_COLUMNS
        )
    else:
        df = pd.read_csv(path)
    return df[DET_COLUMNS]


def save_detections(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df[DET_COLUMNS]
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def sort_per_image(df: pd.DataFrame) -> pd.DataFrame:
    """Score-descending order within each image (stable); the toolkit truncates in file order."""
    return df.sort_values(["image_id", "score"], ascending=[True, False], kind="stable").reset_index(drop=True)


def write_visdrone_results(df: pd.DataFrame, coco_gt: dict, out_dir: Path, category_to_orig: dict[int, int]) -> None:
    """One VisDrone result txt per image (``x,y,w,h,score,category,-1,-1``), score-sorted.

    Every image gets a file; the toolkit's ``load`` fails on missing files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = sort_per_image(df)
    groups = {k: g for k, g in df.groupby("image_id")}
    for img in coco_gt["images"]:
        g = groups.get(img["id"])
        lines = []
        if g is not None:
            for x, y, w, h, s, c in g[["x", "y", "w", "h", "score", "category_id"]].itertuples(index=False):
                lines.append(f"{x:.4f},{y:.4f},{w:.4f},{h:.4f},{s:.6f},{category_to_orig[int(c)]},-1,-1")
        (out_dir / f"{Path(img['file_name']).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def as_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return df[["x", "y", "w", "h"]].to_numpy(np.float64), df["score"].to_numpy(np.float64), df["category_id"].to_numpy(np.int64)


MAX_DETS_PER_IMAGE = 500


def finalize_detections(df: pd.DataFrame, max_dets: int = MAX_DETS_PER_IMAGE) -> pd.DataFrame:
    """Common output contract of *every* method (incl. the SAHI library and fusion): per image
    sorted by score (descending, stable) and capped to ``max_dets`` across all classes."""
    df = sort_per_image(df)
    return df[df.groupby("image_id").cumcount() < max_dets].reset_index(drop=True)
