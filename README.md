# When Does Sliced Inference Pay Off? Resolution, Tiling and Dense Fusion for Small Object Detection

Sliced inference such as SAHI is the standard remedy for small objects in high-resolution imagery, but its gains are
usually measured against low-resolution baselines. The paper treats scale as the explicit variable and compares six
YOLO detectors on VisDrone and five detectors on DOTA (including FCOS) under pre-specified hypotheses, calibration-only
tuning and a single evaluation on held-out test data. Much of the benefit attributed to slicing turns out to be a
resolution effect, but tiles carry complementary information that box fusion wastes. **SCAHI** (Slicing with
Coverage-Aware Hyper Inference) accumulates the dense outputs of all passes in a scale space, with weights that give
every image position the same total coverage, and reads objects off a peak grid fine enough to keep crowded neighbours
apart.

## Quickstart

**(a) Install** (exact versions used for the paper, from the lock file):
```bash
conda env create -f environment.yml && conda activate scahi
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu130
pip install -e .
```

**(b) Regenerate every paper table and the data of every figure** from the stored results (`runs/`), without datasets,
weights or GPU. One command, byte-for-byte check against `expected/`:
```bash
CUDA_VISIBLE_DEVICES="" python scripts/regenerate_paper.py     # ~1 min on a CPU; prints "N/N outputs identical ... PASSED"
```

**(c) Run SCAHI on your own image** with the public COCO YOLOv10s weights (fetched by Ultralytics on first use).
The passes are one whole-image pass at long side 2048 plus tiles at zoom 2 (input 1024, overlap 0.3). The dense
one-to-many outputs are fused with SCAHI on the CPU (default) or on the GPU:
```bash
python examples/scahi_yolov10s.py path/to/image.jpg                    # SCAHI, pre-specified setting
python examples/scahi_yolov10s.py path/to/image.jpg --setting refined --fusion-device gpu --box-fusion --out dets.csv
```

**(d) SCAHI on dense predictions of any detector** (arrays in, boxes out; only numpy, pandas, scipy and torch needed):
```python
import numpy as np
from scahi.fusion.scalespace import CachedLevel, FusionParams, fuse_image

# one level per pass group: dense boxes (x1, y1, x2, y2 in image px), class probabilities, pass index, tile rectangles
whole = CachedLevel(key="whole", xyxy=boxes_w, probs=probs_w, std=np.zeros((len(boxes_w), 4), np.float32),
                    tile=np.zeros(len(boxes_w), np.int16), tiles=np.array([[0, 0, W, H]]), gains=np.ones(1),
                    whole=True, overlap=0.3)
tiles = CachedLevel(key="tiles", xyxy=boxes_t, probs=probs_t, std=np.zeros((len(boxes_t), 4), np.float32),
                    tile=tile_index, tiles=tile_rects, gains=np.ones(len(tile_rects)), whole=False, overlap=0.3)
det = fuse_image([whole, tiles], W, H, "F2", FusionParams(tau_splat=0.005, tau_peak=0.001, cell_factor=5.0,
                 bins_per_oct=5), nc=num_classes)      # DataFrame: x, y, w, h, score, category_id (1-based)
```
A complete runnable version with synthetic predictions: `python examples/scahi_arrays.py`.

**(e) Full experiment pipeline** (training, dense-head caching, calib tuning, freeze, evaluation): see `REPRODUCE.md`.
Main entry points per study:

| Study | Entry points | Frozen rows / parameters |
|---|---|---|
| VisDrone, main study (YOLOv10s) | `python -m scahi.reproduce convert / train / single / cache / tune / fuse / eval` (`scahi/reproduce.py`) | `configs/final/frozen_rows.json` |
| VisDrone, extension (five detectors, refined grid) | the same commands with the frozen parameters | `configs/ext/frozen_rows_ext.json`, `configs/ext/xgrid_frozen.json` |
| DOTA v1.5 (YOLO detectors) | `scahi/dota/train.py`, `scahi/dota/pipeline.py` (single pass, caches, box fusion, SCAHI, SAHI, calib tuning), `scahi/dota/final.py` (frozen rows, one evaluation), `scahi/dota/evaluate.py` (DOTA_devkit port); freeze proposals `scripts/dota/d_freeze_proposal*.py` | `configs/dota/frozen_rows*.json` |
| DOTA, FCOS | `scahi/fcos.py`, `scahi/dota/fcos_train.py`, `scripts/dota/fcos_run.py`, `scripts/dota/fcos_val.py`, `scripts/dota/fcos_limit_check.py` | `configs/dota/frozen_rows_fcos.json` |
| DOTA, SCAHI replacing the SAHI merge | `scahi/dota/sahi_scahi.py` | `configs/dota/frozen_rows_dh5.json`, `configs/dota/frozen_rows_a5.json` |
| Analyses (exploratory) | `scripts/weighted_box_fusion.py`, `scripts/dota/fcos_diagnostics.py`, `scripts/latency_table2.py`, `scripts/latency_breakdown.py`, `scripts/ablation_*.py` | results in `runs/` |


## Preregistrations, frozen configurations and decision log

| What | Where |
|---|---|
| Preregistration, VisDrone main study | `docs/PREREGISTRATION.md` |
| Preregistration, VisDrone extension (five detectors) | `docs/PREREGISTRATION_EXT.md` |
| Preregistration, DOTA (incl. addenda 1-5: YOLO11s / YOLO12s, FCOS, SCAHI replacing the SAHI merge) | `docs/PREREGISTRATION_DOTA.md` |
| Dated decision log (hypotheses, deviations, freezes, single evaluations) | `docs/DECISIONS.md` |
| Frozen configurations of every reported row | `configs/final/`, `configs/ext/`, `configs/dota/` |
| Formal definitions with code references | `docs/ALGORITHM.md` |

The preregistrations and the decision log are the working documents of the project (partly in German). Local paths are
removed. Trained weights are not part of this repository.

SCAHI requires dense per-position predictions as produced by anchor-free heads; query-based detectors such as RT-DETR would require a sparse variant.

## Contents

| Part | Code |
|---|---|
| Data: VisDrone → canonical COCO / YOLO (with ignore regions), calib split | `scahi/data/` |
| Evaluation: VisDrone toolkit port, COCO metrics and size bins, paired bootstrap | `scahi/eval/` |
| Training: single pass at long side s, calib model selection | `scahi/train/full.py` |
| Inference: single pass, raw dense-head cache of every pass (whole image + tiles), SAHI baseline | `scahi/infer/` |
| Fusion: SCAHI (`scalespace.py`, GPU variant `scalespace_torch.py`), box fusion (NMS / GREEDYNMM / WBF), calib tuning | `scahi/fusion/` |
| DOTA v1.5 (horizontal boxes): data, exact port of the official DOTA_devkit evaluation (task 2), training, pipeline (single pass, caches, box fusion, SCAHI, SAHI), frozen rows | `scahi/dota/`, `third_party/DOTA_devkit/`, `configs/dota/` |
| SCAHI on the GPU: blocked membership (spatial blocks with a data-driven halo), identical to the CPU reference | `scahi/fusion/scalespace_torch.py`, `tests/test_scalespace_torch.py` |
| Reproduction entry point | `scahi/reproduce.py` |
| Paper tables and figures from the shipped result files (no data, weights or GPU needed) | `scripts/regenerate_paper.py`, `runs/` (results), `expected/` (reference outputs) |
| Component ablation of SCAHI (computation needs the dense-head caches; report from the shipped results) | `scripts/ablation_components.py`, `scripts/ablation_coverage.py`, `scripts/ablation_report.py` |
| SCAHI as a drop-in replacement for the SAHI merge on DOTA (SAHI-layout caches, calib tuning, frozen rows, parity gate, latency) | `scahi/dota/sahi_scahi.py`, `configs/dota/frozen_rows_dh5.json`, `scripts/dota/d_freeze_proposal_dh5.py`, `scripts/gpu_fusion_parity.py`, `scripts/latency_host_a_dh5.py` |
| FCOS (torchvision, anchor-free) on DOTA: dense-head export, training, caches, calib tuning, pipeline-limit check, input-size check | `scahi/fcos.py`, `scahi/dota/fcos_train.py`, `scripts/dota/fcos_run.py`, `scripts/dota/fcos_limit_check.py`, `scripts/dota/fcos_size_check_runtime.py`, `scripts/fcos_spike.py` |
| Refined grid: sensitivity of the 2× time rule (reporting only), cross-check of the task split with the reference timed in the same process | `scahi/dota/time_rule.py`, `scripts/dota/rg_option3_crosscheck.py` |
| Algorithm specification (all formulas and parameters) | `docs/ALGORITHM.md` |
| Paper terms vs. names in the code | `GLOSSARY.md` |

`scripts/ablation_components.py` and the measurement mode of `scripts/latency_breakdown.py` are included as documentation of the computation: they need internal drivers and the dense-head caches, which are not part of this repository, and they are not required to reproduce the paper tables. The same holds for the VisDrone parts of `scripts/gpu_fusion_parity.py` and of the mapping checks in `scahi/mapcheck.py`.

## Install

```bash
conda env create -f environment.yml && conda activate scahi
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu130
                                    # exact versions used for the paper (incl. ultralytics 8.4.183, torch 2.14.1+cu130)
pip install -e .
pytest -q                           # unit tests (no data or GPU needed)
```

## Regenerate the paper tables and figures

```bash
python scripts/regenerate_paper.py     # ~30 s on a CPU; compares every table / plotted-data file byte for byte with expected/
```
The result files under `runs/` (JSON / CSV: metrics, bootstrap contrasts, tuning grids, latency runs) are the inputs;
the scripts write the tables (`.tex`, `.md`, `.csv`) and figures (`.pdf`, `.png` plus the plotted values as `.data.json` /
`.csv`) under `reports/` and `docs/`. Figures that need dataset images, raw detections or dense-head caches (qualitative
examples, the scale-space volume, the method figure) are not regenerated here.

## Reproduce

See `REPRODUCE.md`: data conversion, training of the single-pass ladder, caches, calib tuning of SCAHI and of box
fusion, evaluation. The frozen refined-grid parameters are in `configs/ext/xgrid_frozen.json`, all parameter values
per detector in `docs/ALGORITHM.md`.

## Citation

If you use this code in academic work, please consider citing the following paper:

```bibtex
@misc{becker2026sliced,
  title         = {When Does Sliced Inference Pay Off? Resolution, Tiling and Dense Fusion for Small Object Detection},
  author        = {Becker, Stefan and Bayer, Jens and M{\"u}nch, David and H{\"u}bner, Wolfgang},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## License

- **SCAHI core** – `scahi/fusion/scalespace.py` (CPU reference) and `scahi/fusion/scalespace_torch.py` (GPU):
  **Apache-2.0** (`LICENSES/Apache-2.0.txt`, `NOTICE`). Both files import only numpy, pandas, scipy, torch and
  torchvision and can be used without Ultralytics.
- **Everything else**: **AGPL-3.0-or-later** (`LICENSE`). The training, inference and evaluation code builds on the
  Ultralytics package (AGPL-3.0); code combined with it is distributed under the same licence. Under AGPL-3.0, using
  this code in a product or a network service requires releasing the complete source of that application under
  AGPL-3.0; Ultralytics offers a commercial licence as an alternative.
- Every source file states its licence in an SPDX header. Third-party code and dependencies: `THIRD_PARTY_NOTICES.md`.
- **Datasets and weights**: VisDrone and DOTA are not redistributed and are subject to their own terms of use (academic /
  non-commercial). Model weights start from Ultralytics' pre-trained weights (AGPL-3.0) or torchvision's COCO weights.

This note is for orientation only and is not legal advice.
