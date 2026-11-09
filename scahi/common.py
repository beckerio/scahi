# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small helpers shared by the study code and the DOTA code (moved here unchanged on 2026-11-01 from
``scripts/collect_env.py``, ``scahi.jobs`` and ``scahi.ext`` so that the released code does not depend on the
research queue drivers; the old names import from here)."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ULPS = 4  # G-E1 tolerance per coordinate max(1e-4 px, 4 float32 ulp) - addendum 0f6fecbc (replaces the 2-ulp rule)


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def spawn_context():
    """Worker processes are spawned, not forked: the parent has used torch (thread pools) before - forked children
    deadlocked on 2026-10-23 (p5_fuse_BFULL1280, all workers asleep at 0 % CPU)."""
    import multiprocessing

    return multiprocessing.get_context("spawn")


def ulp_cmp(ours: "torch.Tensor", ref: "torch.Tensor") -> dict[str, float]:  # noqa: F821
    import torch

    diff = (ours - ref).abs().double().cpu().numpy()
    # ulp of the larger magnitude of the two values: at a power-of-two boundary (e.g. 2048 px) the two float32 values can
    # sit in neighbouring binades; 1 ulp of the upper one is the float32 resolution there (decision 2026-10-26)
    coord = torch.maximum(ours.abs(), ref.abs()).float().cpu().numpy()
    ulp = np.spacing(coord).astype(np.float64)
    tol = np.maximum(1e-4, ULPS * ulp)
    return {"max_abs_box_px": float(diff.max()) if diff.size else 0.0,
            "max_diff_over_tol": float((diff / tol).max()) if diff.size else 0.0}


# ------------------------------------------------------------------------------------------------ env.json


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _version(module: str) -> str | None:
    try:
        mod = __import__(module)
        return str(getattr(mod, "__version__", None))
    except ImportError:
        return None


def collect_env() -> dict[str, Any]:
    """Return a dict with Python/Torch/CUDA/ultralytics/sahi versions, GPU and git state."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _version("torch"),
        "torchvision": _version("torchvision"),
        "ultralytics": _version("ultralytics"),
        "sahi": _version("sahi"),
        "numpy": _version("numpy"),
        "pycocotools": None,
        "cuda": None,
        "cudnn": None,
        "gpu": None,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None,
    }
    try:
        from importlib.metadata import version

        info["pycocotools"] = version("pycocotools")
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    status = _git("status", "--porcelain")
    info["git_dirty"] = None if status is None else bool(status)
    try:
        from scahi.machine import gpu_power, machine_info

        info["machine"] = {**machine_info(), **gpu_power()}
    except ImportError:
        info["machine"] = None
    return info


def write_env(path: Path) -> dict[str, Any]:
    info = collect_env()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2))
    return info
