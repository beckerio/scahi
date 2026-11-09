# SPDX-License-Identifier: AGPL-3.0-or-later
"""Versioned ``metrics.json`` schema (Section 11 of the task description); AP values in percent."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from scahi.machine import gpu_power, latency_validity, machine_info

SCHEMA_VERSION = 1

TEMPLATE: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "exp_id": "",
    "phase": None,
    "split": "",
    "eval_protocol": "",
    "class_setting": "",
    "model": "",
    "weights": "",
    "train_variant": "",
    "inference": {"zoom_stack": [], "overlap": None, "fusion": "", "branch": ""},
    # Training budget (user rule 2026-10-22): init weights, epochs and images seen.
    "training": {"init": "", "epochs": None, "images_per_epoch": None, "images_seen": None, "imgsz": None,
                 "batch": None, "seed": None, "selection": ""},
    "metrics": {"AP": None, "AP50": None, "AP75": None, "AR1": None, "AR10": None, "AR100": None, "AR500": None,
                "AP_per_class": {}, "AP_by_size": {}, "AR_by_size": {}},
    "diagnostics": {"dup_rate": None, "frag_rate": None, "fp_by_border_dist": {}},
    "efficiency": {"passes_per_img": None, "gflops_per_img": None, "latency_ms_mean": None,
                   "latency_ms_p95": None, "peak_mem_mb": None},
    # Machine the run was made on (scahi.machine); absent in runs made on "host-a" before 2026-10-23.
    "machine": {},
    "notes": "",
}


def _clean(x: Any) -> Any:
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item"):
        return _clean(x.item())
    return x


def build_metrics(eval_result: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """Merge an :func:`scahi.eval.evaluate.evaluate` result into the template."""
    m = copy.deepcopy(TEMPLATE)
    for k, v in fields.items():
        if k not in m:
            raise KeyError(f"unknown metrics.json field {k!r}")
        m[k] = v if not isinstance(m[k], dict) else {**m[k], **v}
    m["eval_protocol"] = eval_result["eval_protocol"]
    m["metrics"].update(eval_result["metrics"])
    m["diagnostics"].update(eval_result["diagnostics"])
    m["eval_info"] = eval_result["eval_info"]
    if not m["machine"]:
        m["machine"] = {**machine_info(), **gpu_power()}
    if m["efficiency"].get("latency_ms_mean") is not None and "latency_gpu_exclusive" not in m["efficiency"]:
        m["efficiency"].update(latency_validity())
    return _clean(m)


def write_metrics(m: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=2))
