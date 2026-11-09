# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-check of the refined-grid option 3 (user decision 2026-11-02) before it is used for FCOS.

    python scripts/dota/rg_option3_crosscheck.py   # YOLOv10s, 1024, row C, DOTA calib

Re-runs the refined-grid stage with scahi.dota.pipeline.tune_rg (task split, reference timed in the same process per
image) and compares it with the stored config-split stage (runs/dota/fusion/yolov10s_1024/tuning.json [RGC_o2m]).
Required: the same selected config and identical mAP50 for it. Reported: mAP50 of every variant, per variant the
time ratio under option 3 vs the stored one, and every variant whose admissibility (ratio <= 2) differs. Changes no
stored row or selection. Output: runs/dota/fusion/yolov10s_1024/rg_option3_crosscheck.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scahi.dota import pipeline as P  # noqa: E402

MODEL, S, ROW = "yolov10s", 1024, "C"
KEYS = ("iou_cluster", "tau_splat", "tau_peak", "cell_factor", "bins_per_oct")


def main() -> None:
    d = P.OUT / f"{MODEL}_{S}"
    st = json.loads((d / "tuning.json").read_text())
    old = st[f"RG{ROW}_o2m"]["rows"]
    ref = st["selected"]["reference_F2"]
    cfgs = [ref, *P.refined_grid()]
    new = P.tune_rg(MODEL, S, "o2m", ROW, cfgs, ref, int(sys.argv[1]) if len(sys.argv) > 1 else 21,
                    task_dir=d / "rg_tasks" / f"crosscheck_RG{ROW}_o2m")
    key = lambda r: tuple(r[k] for k in KEYS)  # noqa: E731
    o = {key(r): r for r in old}
    sel_old, sel_new = P.select(old, capped=True), P.select(new, capped=True)
    per = []
    for r in new:
        a = o[key(r)]
        per.append({**{k: r[k] for k in KEYS}, "mAP50_stored": a["mAP50"], "mAP50_option3": r["mAP50"],
                    "mAP50_identical": a["mAP50"] == r["mAP50"],
                    "ratio_stored": a["time_ratio_vs_ref"], "ratio_option3": r["time_ratio_vs_ref"],
                    "admissible_stored": a["time_ratio_vs_ref"] <= P.TIME_CAP,
                    "admissible_option3": r["time_ratio_vs_ref"] <= P.TIME_CAP})
    flips = [p for p in per if p["admissible_stored"] != p["admissible_option3"]]
    out = {"model": MODEL, "s": S, "row": ROW, "n_variants": len(per),
           "selected_stored": {k: sel_old[k] for k in KEYS}, "selected_option3": {k: sel_new[k] for k in KEYS},
           "selected_same": key(sel_old) == key(sel_new),
           "selected_mAP50_stored": sel_old["mAP50"], "selected_mAP50_option3": sel_new["mAP50"],
           "all_mAP50_identical": all(p["mAP50_identical"] for p in per),
           "admissibility_flips": flips, "per_variant": per,
           "passed": key(sel_old) == key(sel_new) and sel_old["mAP50"] == sel_new["mAP50"]}
    (d / "rg_option3_crosscheck.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_variant"}, indent=1), flush=True)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()
