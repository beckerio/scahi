# SPDX-License-Identifier: AGPL-3.0-or-later
"""Machine identity and machine-local configuration overrides (see docs/DECISIONS.md, 2026-10-23).

Machine-specific settings live in a single gitignored file ``configs/local/<host>.yaml``. Its file stem is the
machine name recorded in every ``env.json`` / ``metrics.json``; its ``data`` mapping overrides keys of
``configs/data/visdrone.yaml``. Runs without a ``machine`` block predate the move and were made on "host-a".
"""

from __future__ import annotations

import functools
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = REPO_ROOT / "configs" / "local"
LEGACY_MACHINE = "host-a"


def local_config_path() -> Path | None:
    """The single machine-local override file, or None if there is none."""
    files = sorted(LOCAL_DIR.glob("*.yaml")) if LOCAL_DIR.is_dir() else []
    if len(files) > 1:
        raise RuntimeError(f"more than one machine config in {LOCAL_DIR}: {[f.name for f in files]}")
    return files[0] if files else None


def local_config() -> dict[str, Any]:
    p = local_config_path()
    return (yaml.safe_load(p.read_text()) or {}) if p else {}


def load_data_config(path: Path) -> dict[str, Any]:
    """Data path config with the machine-local ``data`` overrides applied."""
    cfg = yaml.safe_load(path.read_text())
    cfg.update(local_config().get("data", {}))
    return cfg


def _nvidia_smi() -> dict[str, str | None]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
        name, driver = (s.strip() for s in out.split(","))
        head = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout
        cuda = head.split("CUDA Version:")[1].split()[0] if "CUDA Version:" in head else None
        return {"gpu": name, "driver": driver, "cuda_driver": cuda}
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return {"gpu": None, "driver": None, "cuda_driver": None}


# Latency is only valid if no other compute process used the GPU while it was measured (user rule 2026-10-23), and it
# is only "final" if the CPU was free as well: no job of the CPU queue running and no other process with noticeable
# CPU load (user rule 2026-10-23, second message). Otherwise the latency is informational.
GPU_SHARING: dict[str, Any] = {"checks": 0, "other_pids": []}
CPU_SHARING: dict[str, Any] = {"checks": 0, "cpu_queue_busy": False, "other_load_cores": 0.0}
OTHER_LOAD_LIMIT = 1.0  # cores used by foreign processes (not this process, its children or the GPU-queue runner)


def check_gpu_exclusive() -> bool:
    """Record whether another compute process is on the GPU right now (call around/within latency measurements)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return True
    others = sorted({int(p) for p in out if p.isdigit() and int(p) != os.getpid()})
    GPU_SHARING["checks"] += 1
    GPU_SHARING["other_pids"] = sorted(set(GPU_SHARING["other_pids"]) | set(others))
    return not others


def cpu_queue_busy() -> bool:
    """True if the CPU-queue runner (runs/queue/runner_CPU.pid) is alive and currently runs a job (has children)."""
    try:
        import psutil

        pid = int((REPO_ROOT / "runs/queue/runner_CPU.pid").read_text())
        return psutil.pid_exists(pid) and bool(psutil.Process(pid).children())
    except Exception:  # noqa: BLE001 - no pid file / process gone
        return False


def check_cpu_exclusive(sample_s: float = 0.2) -> bool:
    """Record CPU-queue activity and the CPU load of foreign processes (in cores, sampled over ``sample_s``)."""
    import psutil

    me = psutil.Process()
    own = {me.pid, *(c.pid for c in me.children(recursive=True))}
    procs = [p for p in psutil.process_iter() if p.pid not in own]
    for p in procs:
        try:
            p.cpu_percent(None)
        except Exception:  # noqa: BLE001
            pass
    import time

    time.sleep(sample_s)
    load = 0.0
    for p in procs:
        try:
            load += p.cpu_percent(None) / 100.0
        except Exception:  # noqa: BLE001
            pass
    busy = cpu_queue_busy()
    CPU_SHARING["checks"] += 1
    CPU_SHARING["cpu_queue_busy"] = CPU_SHARING["cpu_queue_busy"] or busy
    CPU_SHARING["other_load_cores"] = max(CPU_SHARING["other_load_cores"], load)
    return not busy and load <= OTHER_LOAD_LIMIT


def check_exclusive() -> bool:
    """GPU and CPU exclusivity check (call around/within latency measurements, outside the timed window)."""
    g = check_gpu_exclusive()
    c = check_cpu_exclusive()
    return g and c


def latency_validity() -> dict[str, Any]:
    """Efficiency fields: ``latency_gpu_exclusive`` / ``latency_cpu_exclusive`` are None if never checked in this
    process; ``latency_status`` = final (both exclusive) / informational."""
    out: dict[str, Any] = {"latency_gpu_exclusive": None, "latency_cpu_exclusive": None}
    if GPU_SHARING["checks"]:
        out.update(latency_gpu_exclusive=not GPU_SHARING["other_pids"], latency_other_gpu_pids=GPU_SHARING["other_pids"])
    if CPU_SHARING["checks"]:
        out.update(latency_cpu_exclusive=not CPU_SHARING["cpu_queue_busy"] and CPU_SHARING["other_load_cores"] <= OTHER_LOAD_LIMIT,
                   latency_cpu_queue_busy=CPU_SHARING["cpu_queue_busy"],
                   latency_other_cpu_load_cores=round(CPU_SHARING["other_load_cores"], 2))
    p = gpu_power()
    out.update(latency_gpu_power_limit_w=p["gpu_power_limit_w"], latency_gpu_power_default_w=p["gpu_power_default_limit_w"])
    out["latency_status"] = ("final" if out["latency_gpu_exclusive"] and out["latency_cpu_exclusive"] and p["gpu_power_capped"] is False
                             else "informational")
    return out


@functools.lru_cache(maxsize=1)
def machine_info() -> dict[str, Any]:
    """Machine block for env.json/metrics.json: name, hostname, GPU, driver, CUDA (driver and torch runtime)."""
    p = local_config_path()
    info: dict[str, Any] = {"name": p.stem if p else platform.node(), "hostname": platform.node(), **_nvidia_smi()}
    try:
        import torch

        info["cuda_runtime"] = torch.version.cuda
    except ImportError:
        info["cuda_runtime"] = None
    return info


def gpu_power() -> dict[str, float | None]:
    """Active and default GPU power limit (W), queried fresh (a cap may be set at any time; user rule 2026-10-26)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.limit,power.default_limit", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
        lim, dflt = (float(s) for s in out.split(","))
        return {"gpu_power_limit_w": lim, "gpu_power_default_limit_w": dflt, "gpu_power_capped": lim < dflt - 0.5}
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return {"gpu_power_limit_w": None, "gpu_power_default_limit_w": None, "gpu_power_capped": None}


LATENCY_DEFERRED = Path(__file__).resolve().parents[1] / "runs/queue/LATENCY_DEFERRED"


def require_default_power(job: str) -> None:
    """Latency jobs run only at the default power limit (user rule 2026-10-26): otherwise the job is deferred - a flag file
    shown in QUEUE_STATUS is written and the job exits without measuring (exit code 77 = deferred)."""
    p = gpu_power()
    if p["gpu_power_capped"]:
        LATENCY_DEFERRED.write_text(f"{job}: power limit {p['gpu_power_limit_w']} W < default {p['gpu_power_default_limit_w']} W\n")
        print(f"[latency] {job} deferred: GPU power limit capped ({p})", flush=True)
        raise SystemExit(77)
    LATENCY_DEFERRED.unlink(missing_ok=True)
