# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coverage C at every fused position (exploratory, post hoc; part of the component ablation, user request 2026-10-30).

    python scripts/ablation_coverage.py [--workers 7]

For the frozen stacks (DOTA dota2048: w2048 + native 1024 tiles; VisDrone native2048: w2048 + z = 2 tiles), branch o2m,
2048 px weights: C (scahi.fusion.scalespace.coverage) at every cell of the sparse volume, i.e. every position the
fusion splats into, for (c) F2, (e) F2 with the RG parameters and (b) F1 (no taper; C = number of covering passes).
Same cells and the same coverage function as ``fuse_image``. Positions whose cell centre lies outside the image are
counted separately (cells at the image border extend past it). Output: runs/ablation/coverage.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from ablation_components import OUT, SZ, _cpu_slot, _spawn, v_frozen  # noqa: E402


def _c_stats(args: tuple[str, str, str, str, dict]) -> tuple[str, dict]:
    from scahi.fusion import scalespace as ss

    _cpu_slot()
    ds, m, arm, variant, cfg = args
    if ds == "dota":
        from scahi.dota import pipeline as P

        data = P.load(m, SZ, "val", "o2m", "C")
    else:
        from ext_cluster_diag import _load

        data = _load(m, SZ, "o2m", "C", "val")[2]
    p = ss.FusionParams(**cfg)
    vals_in, n_out, c_out = [], 0, []
    for levels, W, H in data.values():
        v = ss.volume(levels, W, H, variant, p)
        if not len(v["x"]):
            continue
        C = ss.coverage(v["x"], v["y"], v["sigma"], levels, W, H, variant, p)
        inside = (v["x"] >= 0) & (v["x"] <= W) & (v["y"] >= 0) & (v["y"] <= H)
        vals_in.append(C[inside])
        c_out.append(C[~inside])
        n_out += int((~inside).sum())
    a, b = np.concatenate(vals_in), np.concatenate(c_out) if c_out else np.zeros(0)
    u, n = np.unique(np.round(a, 6), return_counts=True)
    top = sorted(zip(u.tolist(), n.tolist()), key=lambda t: -t[1])[:6]
    return f"{ds}/{m}/{arm}", {
        "variant": variant, "cfg": cfg, "n_positions_inside": int(len(a)), "C_min": float(a.min()),
        "C_max": float(a.max()), "C_mean": float(a.mean()), "share_C_eq_2": float(np.mean(np.isclose(a, 2.0))),
        "most_frequent_C": top, "n_positions_outside_image": n_out,
        "C_outside_min_max": [float(b.min()), float(b.max())] if len(b) else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    _cpu_slot()
    dj = json.loads((OUT / "dota.json").read_text())["arms"]
    jobs = [("dota", m, arm, dj[m][arm]["variant"], {**dj[m][arm]["cfg"]}) for m in dj for arm in ("b", "c", "e")]
    vf = v_frozen("yolov10s")
    vb = json.loads((OUT / "visdrone.json").read_text())["arms"]["yolov10s"]["b"]["cfg"]
    jobs += [("visdrone", "yolov10s", "c", "F2", vf["c"]["cfg"]), ("visdrone", "yolov10s", "e", "F2", vf["e"]["cfg"]),
             ("visdrone", "yolov10s", "b", "F1", vb)]
    with ProcessPoolExecutor(a.workers, mp_context=_spawn()) as ex:
        res = dict(ex.map(_c_stats, jobs))
    (OUT / "coverage.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, {x: v[x] for x in ("C_min", "C_max", "C_mean", "share_C_eq_2", "n_positions_outside_image")})


if __name__ == "__main__":
    main()
