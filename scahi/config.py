# SPDX-License-Identifier: AGPL-3.0-or-later
"""Configuration objects shared across data conversion, training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class IgnoreType:
    """A class of ignore boxes with its channel mask (0-based training class ids)."""

    name: str
    yolo_id: int
    categories: tuple[int, ...]
    score_zero: bool
    class_ids: tuple[int, ...]


@dataclass(frozen=True)
class EvalProtocol:
    name: str
    # target class id (None = all classes) -> original categories acting as ignore regions
    drop_regions: dict[int | None, tuple[int, ...]]
    toolkit_validated: bool = False


@dataclass(frozen=True)
class ClassSetting:
    name: str
    classes: tuple[str, ...]
    category_to_class: dict[int, int]
    ignore_types: tuple[IgnoreType, ...]
    eval_protocols: dict[str, EvalProtocol] = field(default_factory=dict)

    @property
    def nc(self) -> int:
        return len(self.classes)

    @property
    def yolo_names(self) -> list[str]:
        """Names for the dataset YAML: nc real classes followed by one id per ignore type."""
        return list(self.classes) + [t.name for t in self.ignore_types]

    def ignore_type_for(self, category: int, score: int) -> IgnoreType | None:
        """Ignore type of an original box, or None if it is a regular object / unused."""
        for t in self.ignore_types:
            if category in t.categories or (t.score_zero and score == 0):
                return t
        return None

    def class_id(self, name: str) -> int:
        return self.classes.index(name)


def load_class_setting(name_or_path: str | Path) -> ClassSetting:
    p = Path(name_or_path)
    if not p.suffix:
        p = REPO_ROOT / "configs" / "classes" / f"{name_or_path}.yaml"
    cfg = yaml.safe_load(p.read_text())
    classes = tuple(cfg["classes"])
    nc = len(classes)

    def _ids(spec: str | list[str]) -> tuple[int, ...]:
        return tuple(range(nc)) if spec == "all" else tuple(classes.index(c) for c in spec)

    ignore_types = tuple(
        IgnoreType(
            name=k,
            yolo_id=nc + i,
            categories=tuple(v.get("categories", [])),
            score_zero=bool(v.get("score_zero", False)),
            class_ids=_ids(v["classes"]),
        )
        for i, (k, v) in enumerate(cfg["ignore_types"].items())
    )
    protocols = {
        k: EvalProtocol(
            name=k,
            drop_regions={
                (None if tgt == "all" else classes.index(tgt)): tuple(cats)
                for tgt, cats in v["drop_regions"].items()
            },
            toolkit_validated=bool(v.get("toolkit_validated", False)),
        )
        for k, v in cfg.get("eval_protocols", {}).items()
    }
    return ClassSetting(
        name=cfg["name"],
        classes=classes,
        category_to_class={int(k): classes.index(v) for k, v in cfg["category_map"].items()},
        ignore_types=ignore_types,
        eval_protocols=protocols,
    )
