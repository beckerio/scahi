# SPDX-License-Identifier: AGPL-3.0-or-later
"""Readers for the original VisDrone2019-DET annotation format.

Columns: ``bbox_left, bbox_top, bbox_width, bbox_height, score, object_category,
truncation, occlusion``. Category 0 = ignored region, 11 = others; ``score == 0`` means the
box is ignored during evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scahi.data.boxes import xywh_to_xyxy

VISDRONE_CATEGORIES: dict[int, str] = {
    0: "ignored-regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}
IGNORE_CATEGORY = 0
OTHERS_CATEGORY = 11
EVAL_CATEGORIES = tuple(range(1, 11))


@dataclass
class VisDroneAnnotation:
    """Annotations of one image; all arrays have length N."""

    xyxy: NDArray[np.float64]
    score: NDArray[np.int64]
    category: NDArray[np.int64]
    truncation: NDArray[np.int64]
    occlusion: NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.category)

    @property
    def is_ignore(self) -> NDArray[np.bool_]:
        """Boxes treated as class-agnostic ignore: ignored regions, 'others' and score 0."""
        return (
            (self.category == IGNORE_CATEGORY)
            | (self.category == OTHERS_CATEGORY)
            | (self.score == 0)
        )


def parse_visdrone_text(text: str) -> NDArray[np.int64]:
    """Parse annotation text to an ``[N, 8]`` int array (tolerates trailing commas/blank lines)."""
    rows: list[list[int]] = []
    for line in text.splitlines():
        parts = [p for p in line.strip().split(",") if p.strip() != ""]
        if not parts:
            continue
        vals = [int(float(p)) for p in parts[:8]]
        if len(vals) < 8:
            raise ValueError(f"Malformed VisDrone line: {line!r}")
        rows.append(vals)
    return np.asarray(rows, dtype=np.int64).reshape(-1, 8)


def read_visdrone_annotation(path: Path | str) -> VisDroneAnnotation:
    arr = parse_visdrone_text(Path(path).read_text())
    return VisDroneAnnotation(
        xyxy=xywh_to_xyxy(arr[:, :4]),
        score=arr[:, 4],
        category=arr[:, 5],
        truncation=arr[:, 6],
        occlusion=arr[:, 7],
    )
