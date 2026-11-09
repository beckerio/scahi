# SPDX-License-Identifier: AGPL-3.0-or-later
"""DOTA v1.5 data access (docs/DOTA_BRIEF.md §2.1).

Labels: the official DOTA v1.5 HBB files (``DOTA-v1.5_{train,val}_hbb.zip``), unpacked to ``<dota_root>/hbb/<split>``;
OBB files (``<dota_root>/obb/<split>``) only for the audit. Images: Ultralytics ``DOTAv1.5.zip`` unpacked to
``<dota_root>/DOTAv1.5/images/<split>`` (no Ultralytics conversion). ``dota_root`` comes from configs/data/dota.yaml with
the machine-local override ``data.dota_root`` in configs/local/<host>.yaml.

Line format (both files): ``x1 y1 x2 y2 x3 y3 x4 y4 class difficult`` after two header lines (imagesource, gsd).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scahi.machine import REPO_ROOT, load_data_config

# DOTA v1.5: the 15 classes of DOTA_devkit (v1.0 order) + container-crane (new in v1.5)
CLASSES = ("plane", "baseball-diamond", "bridge", "ground-track-field", "small-vehicle", "large-vehicle", "ship",
           "tennis-court", "basketball-court", "storage-tank", "soccer-ball-field", "roundabout", "harbor",
           "swimming-pool", "helicopter", "container-crane")
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
VEHICLES = ("small-vehicle", "large-vehicle")  # pre-specified subset metric
SPLITS = ("train", "val")


@functools.cache
def config() -> dict:
    return load_data_config(REPO_ROOT / "configs/data/dota.yaml")


def root() -> Path:
    return Path(config()["dota_root"])


def image_dir(split: str) -> Path:
    return root() / "DOTAv1.5/images" / split


def label_dir(split: str, kind: str = "hbb") -> Path:
    return root() / kind / split


def image_path(split: str, stem: str) -> Path:
    return image_dir(split) / f"{stem}.jpg"


@dataclass
class Ann:
    poly: np.ndarray  # (n, 8) float, the 4 corner points as written in the file
    cls: np.ndarray  # (n,) int, index into CLASSES
    difficult: np.ndarray  # (n,) bool
    header: dict


def parse(path: Path) -> Ann:
    polys, cls, diff, header = [], [], [], {}
    for line in path.read_text().splitlines():
        t = line.strip().split(" ")
        if len(t) < 9:
            if ":" in line:
                k, v = line.split(":", 1)
                header[k.strip()] = v.strip()
            continue
        polys.append([float(x) for x in t[:8]])
        cls.append(CLASS_ID[t[8]])
        diff.append(len(t) > 9 and int(t[9]) == 1)
    return Ann(np.array(polys, dtype=np.float64).reshape(-1, 8), np.array(cls, dtype=np.int64),
               np.array(diff, dtype=bool), header)


def devkit_box(poly: np.ndarray) -> np.ndarray:
    """HBB as the official task-2 evaluation reads it (dota_evaluation_task2.parse_gt): corner 1 (x1, y1) and
    corner 3 (x3, y3), each truncated to int. Returns (n, 4) float x1 y1 x2 y2."""
    return np.trunc(poly[:, [0, 1, 4, 5]]).astype(np.float64)


def minmax_box(poly: np.ndarray) -> np.ndarray:
    """Axis-aligned box of the 4 corners (min/max), (n, 4) x1 y1 x2 y2."""
    xs, ys = poly[:, 0::2], poly[:, 1::2]
    return np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], 1)


def stems(split: str) -> list[str]:
    return sorted(p.stem for p in label_dir(split).glob("*.txt"))


@functools.cache
def load(split: str) -> dict[str, Ann]:
    return {s: parse(label_dir(split) / f"{s}.txt") for s in stems(split)}
