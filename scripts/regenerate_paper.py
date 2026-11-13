# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regenerate every paper table and figure that is computed from the shipped result files (no GPU, no data, no weights)
and compare the text / data outputs byte for byte with the reference.

    python scripts/regenerate_paper.py                 # release: reference = expected/<path>
    python scripts/regenerate_paper.py --reference git # research repo: reference = the committed version (git HEAD)

Compared: CSV, .data.json, .tex, .md and .json outputs (the plotted / tabulated values). PDF and PNG files are written as
well but not compared (PDFs carry a creation date; PNGs depend on the installed fonts). Not regenerated here (they need
dataset images, raw detections or caches): the qualitative figures, the volume figure, the method figure (LaTeX) and
the per-image tables of the tiles analysis (shipped as inputs). Exit code 1 if any output differs or is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PAPER_FIGS = ["resolution", "pareto", "sizebins", "fusion_gain", "degradation", "band_edge", "border", "h1", "windows",
              "prereg_forest", "testdev"]
EXT_FIGS = ["ext_calib_curves", "ext_forest", "ext_waterfall", "ext_recall", "ext_xgrid_cost", "ext_r3_pareto",
            "ext_scaling"]
COMMANDS = [
    ["scripts/paper_figures.py", *PAPER_FIGS],
    ["scripts/paper_tables.py"],
    ["scripts/make_figures.py", *EXT_FIGS],
    ["scripts/tiles_payoff.py", "report"],
    ["scripts/ablation_report.py"],
    ["scripts/dota/d_figures.py"],
    ["scripts/dota/d_rows_table.py"],
    ["scripts/dota/d_h3_table.py"],
    ["scripts/dota/d_perclass_table.py"],
    ["scripts/dota/d_results.py"],
    ["scripts/dota/d_results_new.py"],
    ["scripts/dota/d_dh5_tables.py"],
    ["scripts/dota/d_a5_tables.py"],
    ["scripts/dota/d_results_fcos.py"],
    ["scripts/dota/fcos_diagnostics.py", "report"],
    ["scripts/weighted_box_fusion.py", "report"],
    ["scripts/latency_breakdown.py", "report"],
    ["scripts/latency_host_a_dota.py", "report"],
    ["scripts/latency_host_a_dh5.py", "report"],
    ["scripts/latency_table2.py", "report"],
    ["scripts/latency_dota_host_b.py", "report"],
    ["scripts/dota_native_pass.py", "report"],  # exploratory post-hoc analysis (supplement Tab. s-native)
    ["scripts/scahi_algorithm_doc.py"],
]
OUTPUTS = """
docs/SCAHI_ALGORITHM.md
reports/ABLATION.md
reports/ANALYSIS_WEIGHTED_BOX_FUSION.csv
reports/ANALYSIS_WEIGHTED_BOX_FUSION.md
reports/D_RESULTS_FCOS.md
reports/tab_compute_control.csv
reports/tab_compute_control.md
runs/latency_table2/tab_compute_control_reference_path.csv
reports/D_RESULTS.md
reports/D_RESULTS_NEW.md
reports/dota/FCOS_DIAGNOSTICS.md
reports/dota/tab_fcos_diag_cap.csv
reports/dota/tab_fcos_diag_limit_check.csv
reports/dota/tab_fcos_diag_per_image.csv
reports/dota/tab_fcos_diag_recall.csv
reports/dota/tab_fcos_diag_redundancy.csv
reports/dota/tab_dota_a5.md
reports/dota/NATIVE_PASS_EXPLORATORY.md
reports/dota/native_pass_exploratory.json
reports/dota/tab_dota_a5_contrasts.csv
reports/dota/tab_dota_a5_parameters.csv
reports/dota/tab_dota_a5_rows.csv
reports/dota/tab_dota_a5_sensitivity.csv
reports/dota/tab_dota_a5_sizebins.csv
reports/dota/tab_dota_dh3_metrics.csv
reports/dota/tab_dota_dh3_metrics.md
reports/dota/tab_dota_dh5.md
reports/dota/tab_dota_dh5_contrasts.csv
reports/dota/tab_dota_dh5_parameters.csv
reports/dota/tab_dota_dh5_rows.csv
reports/dota/tab_dota_dh5_sizebins.csv
reports/dota/tab_dota_latency_host_a.csv
reports/dota/tab_dota_latency_host_b.csv
reports/dota/tab_dota_latency_host_b.md
reports/dota/tab_dota_latency_host_b_dh5.csv
reports/dota/tab_dota_latency_host_b_dh5.md
reports/dota/tab_dota_latency_host_a.md
reports/dota/tab_dota_latency_host_a_dh5.csv
reports/dota/tab_dota_latency_host_a_dh5.md
reports/dota/tab_dota_perclass_map50.csv
reports/dota/tab_dota_perclass_map50.md
reports/dota/tab_dota_rows_all_models.csv
reports/dota/tab_dota_rows_all_models.md
reports/figures/ablation/ablation.csv
reports/figures/dota/fig_dota_calib_curves.csv
reports/figures/dota/fig_dota_hypotheses.csv
reports/figures/dota/fig_dota_rows.csv
reports/figures/ext/fig_ext_calib_curves.csv
reports/figures/ext/fig_ext_calib_curves.data.json
reports/figures/ext/fig_ext_forest.csv
reports/figures/ext/fig_ext_forest.data.json
reports/figures/ext/fig_ext_r3_pareto.csv
reports/figures/ext/fig_ext_r3_pareto.data.json
reports/figures/ext/fig_ext_recall.csv
reports/figures/ext/fig_ext_recall.data.json
reports/figures/ext/fig_ext_scaling.csv
reports/figures/ext/fig_ext_scaling.data.json
reports/figures/ext/fig_ext_tiles_payoff.csv
reports/figures/ext/fig_ext_waterfall.csv
reports/figures/ext/fig_ext_waterfall.data.json
reports/figures/ext/fig_ext_xgrid_cost.csv
reports/figures/ext/fig_ext_xgrid_cost.data.json
reports/figures/paper/fig_band_edge.data.json
reports/figures/paper/fig_border_fp.data.json
reports/figures/paper/fig_degradation.data.json
reports/figures/paper/fig_fusion_gain.data.json
reports/figures/paper/fig_h1.data.json
reports/figures/paper/fig_pareto.data.json
reports/figures/paper/fig_prereg_forest.data.json
reports/figures/paper/fig_resolution.data.json
reports/figures/paper/fig_sizebins.data.json
reports/figures/paper/fig_testdev.data.json
reports/figures/paper/fig_windows.data.json
reports/figures/paper/tex/tab_consistency.csv
reports/figures/paper/tex/tab_consistency.tex
reports/figures/paper/tex/tab_exploratory.tex
reports/figures/paper/tex/tab_fusion.tex
reports/figures/paper/tex/tab_main.tex
reports/figures/paper/tex/tab_prereg.tex
reports/figures/paper/tex/tab_testdev_exploratory.csv
reports/figures/paper/tex/tab_testdev_exploratory.tex
reports/figures/paper/tex/tab_testdev_main.csv
reports/figures/paper/tex/tab_testdev_main.tex
reports/figures/paper/tex/tab_testdev_prereg.csv
reports/figures/paper/tex/tab_testdev_prereg.tex
reports/figures/paper/tex/tab_testdev_sizebins.csv
reports/figures/paper/tex/tab_testdev_sizebins.tex
reports/tab_latency_breakdown.csv
reports/tab_latency_breakdown.md
reports/tab_scahi_parameters.csv
reports/tiles_payoff/TILES_PAYOFF.md
reports/tiles_payoff/tiles_payoff.json
reports/tiles_payoff/tiles_payoff_spearman.csv
""".split()


def reference(path: str, mode: str) -> bytes | None:
    if mode == "git":
        r = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{path}"], capture_output=True)
        return r.stdout if r.returncode == 0 else None
    p = REPO / "expected" / path
    return p.read_bytes() if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", choices=("expected", "git"), default="expected")
    a = ap.parse_args()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg"}
    for d in {str(Path(o).parent) for o in OUTPUTS}:
        (REPO / d).mkdir(parents=True, exist_ok=True)
    runs = []
    for cmd in COMMANDS:
        r = subprocess.run([sys.executable, *cmd], cwd=REPO, env=env, capture_output=True, text=True)
        runs.append({"cmd": " ".join(cmd), "exit": r.returncode, "stderr_tail": r.stderr.strip().splitlines()[-3:]})
        print(f"[regen] {'ok ' if r.returncode == 0 else 'ERR'} {' '.join(cmd)}", flush=True)
    res = {}
    for o in OUTPUTS:
        p = REPO / o
        ref = reference(o, a.reference)
        res[o] = ("missing output" if not p.exists() else "no reference" if ref is None
                  else "identical" if p.read_bytes() == ref else "DIFFERS")
    bad = {k: v for k, v in res.items() if v != "identical"}
    out = {"reference": a.reference, "commands": runs, "outputs": res, "n_outputs": len(res),
           "n_identical": len(res) - len(bad), "passed": not bad and all(x["exit"] == 0 for x in runs)}
    (REPO / "runs/regenerate_paper.json").write_text(json.dumps(out, indent=1))
    for k, v in bad.items():
        print(f"[regen] {v}: {k}")
    print(f"[regen] {out['n_identical']}/{len(res)} outputs identical; commands ok: "
          f"{sum(x['exit'] == 0 for x in runs)}/{len(runs)} -> {'PASSED' if out['passed'] else 'FAILED'}")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
