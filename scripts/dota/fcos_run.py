# SPDX-License-Identifier: AGPL-3.0-or-later
"""FCOS on DOTA (preregistration addendum 3): the GPU chain and the CPU tuning chain.

    python scripts/dota/fcos_run.py gpu   # S_max smoke -> per s: train -> gate -> single pass on calib (conf, crossing) -> caches
    python scripts/dota/fcos_run.py cpu   # per s: calib tuning (box fusion, F2, B-F2, refined grid C/B), waits for the caches
    scripts/dota/run_fcos_tuning.sh W WS MEM   # the same on all cores (cpu_par): W config-split workers, WS image-split
                                               # workers (box fusion), hard memory cap MEM, first OOM victim

Stops on a failed gate (no caches / fusion). Logs: runs/dota/fcos_logs/. Afterwards: freeze proposal + replay on calib,
then stop for the user's approval (DOTA-val only after it).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def gpu() -> None:
    os.sched_setaffinity(0, set(range(0, 16)))  # cores 16-23 belong to the CPU tuning
    from scahi.dota import fcos_train as F
    from scahi.dota import pipeline as P

    import json

    from scahi.dota import train as T

    f = T.DOTA_RUNS / "fcos_smoke.json"
    sm = json.loads(f.read_text()) if f.exists() else F.smax_smoke()
    sizes = [s for s in (1024, 2048) if s <= sm["s_max"]]
    print(f"[fcos-run] S_max {sm['s_max']} -> sizes {sizes}", flush=True)
    for s in sizes:
        F.train(s)
        F.gate(s)
        P.single_calib("fcos", s)
        F.caches(s)
    print("[fcos-run] GPU chain done", flush=True)


def cpu() -> None:
    import json

    from scahi.dota import pipeline as P
    from scahi.dota import train as T

    for s in (1024, 2048):
        smoke = T.DOTA_RUNS / "fcos_smoke.json"
        while not smoke.exists():
            __import__("time").sleep(60)
        if s > json.loads(smoke.read_text())["s_max"]:
            continue
        P.fuse_job("fcos", s)
    print("[fcos-run] CPU chain done", flush=True)


def cpu_par() -> None:
    """As cpu, with the worker counts derived from the measured memory (user, 2026-11-02; parallelisation only)."""
    from scahi.dota import pipeline as P

    w, ws = int(sys.argv[2]), int(sys.argv[3])
    print(f"[fcos-run] cpu_par: {w} config-split / {ws} image-split workers on cores {sorted(P.fuse_cores())}", flush=True)
    for s in (1024, 2048):
        P.fuse_job("fcos", s, workers=w, workers_split=ws)
    print("[fcos-run] CPU chain done", flush=True)


if __name__ == "__main__":
    {"gpu": gpu, "cpu": cpu, "cpu_par": cpu_par}[sys.argv[1]]()
