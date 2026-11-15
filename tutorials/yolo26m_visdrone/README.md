# Tutorial: YOLO26m and YOLO26s with SCAHI on VisDrone

YOLO26m and YOLO26s were added to the study after the paper's experiments, **the same way as the paper's detectors**: a
written pre-registration addendum before any training (`docs/PREREGISTRATION_EXT.md`, addenda of 2026-11-13 for YOLO26m
and 2026-11-14 for YOLO26s), the extension recipe, model selection and all tuning on the calibration split, a frozen
configuration file with weight checksums, a replay check, and then exactly one evaluation on VisDrone val and one on
test-dev. Both are reported separately from the paper's five extension detectors (they count in no family criterion)
and are part of the arXiv version and of this repository; the decision log `docs/DECISIONS.md` records every step.

This tutorial has two tracks:

- **Track A** - use the frozen YOLO26m / YOLO26s weights and parameters (on VisDrone or on your own images);
- **Track B** - add a new detector the paper way, step by step, with the runtimes of the YOLO26m run.

Shortcuts that are *not* the paper protocol are collected in one marked box at the end.

## Results (single held-out evaluation)

AP@[.5:.95], VisDrone toolkit protocol (ignored regions, at most 500 detections per image). All 25 frozen
configurations per model: `python tutorials/yolo26m_visdrone/frozen.py rows [--model yolo26s]`.

| configuration | YOLO26m val | YOLO26m test-dev | YOLO26s val | YOLO26s test-dev |
|---|---:|---:|---:|---:|
| single pass at 1280 px (one-to-one head) | 37.41 | 32.54 | 33.63 | 29.94 |
| single pass at 2048 px (one-to-one head) | 41.25 | 35.30 | 38.41 | 33.15 |
| 1280 px pass + zoom-2 tiles, box fusion (one-to-one head) | 37.89 | 32.58 | 34.97 | 30.61 |
| 2048 px pass + zoom-2 tiles, SCAHI, pre-specified setting | 41.53 | 35.82 | 39.31 | 34.18 |
| 2048 px pass + zoom-2 tiles, SCAHI, refined grid (exploratory) | **42.89** | **36.59** | **40.65** | **34.93** |
| SAHI (640 px slices, 1280 px weights) | 34.56 | 30.51 | 31.02 | 28.51 |

Pre-registered questions, verdict per detector (verdict on val; ε = 0.30 AP; paired bootstrap over the images, 1000
resamples, seed 0; test-dev reported next to it):

| question | YOLO26m: Δ on val [95 % CI] | verdict | Δ on test-dev | YOLO26s: Δ on val [95 % CI] | verdict | Δ on test-dev |
|---|---|---|---|---|---|---|
| Q1: single pass 2048 px beats 1280 px + tiles with box fusion | +3.36 [+2.91; +3.83] | met | +2.72 | +3.44 [+3.03; +3.85] | met | +2.54 |
| Q2: SCAHI beats box fusion on the same passes, 1280 px | +1.94 [+1.65; +2.21] | met | +2.24 | +1.57 [+1.34; +1.83] | met | +2.36 |
| Q2: SCAHI beats box fusion on the same passes, 2048 px | +1.77 [+1.54; +1.99] | met | +2.52 | +1.63 [+1.39; +1.85] | met | +2.27 |
| Q3: 2048 px + tiles with SCAHI beats the 2048 px single pass (also CI > 0) | +0.28 [+0.04; +0.51] | not met (Δ < ε) | +0.51 | +0.90 [+0.64; +1.16] | met | +1.03 |

For YOLO26m, as for YOLOv10m in the paper, Q3 misses the threshold on val; test-dev is reported next to it and does not
change the verdict. For YOLO26s all four questions are met.

**Equal compute (descriptive, no verdict).** Single pass of YOLO26s at 2048 px vs. YOLO26m at 1280 px, both one-to-one
head; GFLOPs per image exact per input shape (Ultralytics' counter on the fused model at each image's letterboxed shape),
mean over the images of the split; Δ with paired bootstrap 95 % CI:

| split | YOLO26s 2048 px | YOLO26m 1280 px | Δ [95 % CI] | GFLOPs / image (26s 2048 vs. 26m 1280) |
|---|---:|---:|---|---|
| val | 38.41 | 37.41 | +1.00 [+0.60; +1.37] | 126 vs. 157 (0.80x) |
| test-dev | 33.15 | 32.54 | +0.61 [+0.34; +0.89] | 137 vs. 166 (0.83x) |

The smaller model at the higher resolution is slightly more accurate at about 20 % less compute. Single-pass GFLOPs per
image on val: YOLO26s 48.5 (1280) / 126.4 (2048), YOLO26m 157.4 (1280) / 399.1 (2048).

**Documented deviation (YOLO26s, single-pass gate).** The addendum's calib gate for the single-pass rows (bit-identical
in two replays AND |Δ| to the tuning value ≤ 0.05 AP) is **not met** for YOLO26s: the replays are bit-identical, but
Δ = 0.079 to 0.094 AP. The tuning ran Ultralytics' predictor at batch 16, the replay (the val / test-dev path) at batch
1. Cause: the predictor pads a batch minimally only if all its images have the same shape, otherwise every image is
padded to the full square input size; calib has 5 image shapes and 8 of its 38 batches of 16 are mixed. Batch 16 with
one shape per batch reproduces the batch-1 values within 0.005 AP (`runs/ext/yolo26s/batch_padding_check.json` in the
research record). Because the gate protects the calib choice, the confidence grid was re-evaluated on the batch-1
detections: same choice (0.001) and same ordering in all four cases. The choice stands, the batch-1 values are the
reference for the val replay (tuning values reported alongside, `configs/ext/frozen_rows_ext26s.json`), no rule was
changed (authors' decision, `docs/DECISIONS.md` 2026-11-15). For YOLO26m the same difference stayed below 0.05 AP
(0.013-0.027).

## Quick start

```bash
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu130 && pip install -e .
python tutorials/yolo26m_visdrone/prepare_data.py          # VisDrone train / val with the ignored regions (downloads)
# cached mode (no GPU): unpack the release asset yolo26m_visdrone_cached_val.tar in the repository root
python tutorials/yolo26m_visdrone/frozen.py run --row XG_yolo26m_2048_o2m_C --split val
# -> AP 42.89006, stored 42.89006, delta +0.00000 -> PASS (exact)
# full mode (GPU): frozen weights visdrone_ext_yolo26m_1280_s0.pt / _2048_s0.pt (release assets) -> weights/
python tutorials/yolo26m_visdrone/frozen.py run --row XG_yolo26m_2048_o2m_C --split val --mode full
python tutorials/yolo26m_visdrone/frozen.py predict --row XG_yolo26m_2048_o2m_C --images my_images/ --out dets.csv
# YOLO26s: --model yolo26s (assets yolo26s_visdrone_cached_<split>.tar, visdrone_ext_yolo26s_{1280,2048}_s0.pt)
python tutorials/yolo26m_visdrone/frozen.py run --model yolo26s --row XG_yolo26s_2048_o2m_C --split val
# -> AP 40.65429, stored 40.65429, delta +0.00000 -> PASS (exact)
```

`prepare_data.py` downloads the original VisDrone2019-DET train and val zips and converts them with the repository's
converter, keeping the ignored regions that the toolkit evaluation needs (Ultralytics' `VisDrone.yaml` drops them). For
test-dev, download the official data and run `python -m scahi.reproduce convert` (see `REPRODUCE.md`); the tutorial
uses those canonical files whenever they exist. The cached-mode assets contain model outputs only (dense pass outputs
and detections, no images or annotations); the ground truth always comes from your own VisDrone download. The released
weights and cached outputs were derived from VisDrone and are subject to its terms of use
([VisDrone dataset](https://github.com/VisDrone/VisDrone-Dataset); for the DOTA weights of the main study, the
[DOTA dataset](https://captain-whu.github.io/DOTA/)).

**Splits.** Training on `train_main` (5 877 images, 178 sequences); model selection and every tuning step on `calib`
(594 images, 30 whole video sequences of VisDrone train, `data/splits/calib.txt`: the draw of 5 000 whose object-size
and class distribution is closest to all of train; no sequence is shared with `train_main`). VisDrone val (548) and
test-dev (1 610) are used once, after the freeze. For your own data, hold out whole sequences, scenes or flights in the
same way.

## Track A: use the frozen weights and parameters

```bash
python tutorials/yolo26m_visdrone/frozen.py rows [--model yolo26s]                       # 25 configurations
python tutorials/yolo26m_visdrone/frozen.py run --row <id|all> --split val|test-dev [--mode cached|full] [--model yolo26s]
python tutorials/yolo26m_visdrone/frozen.py predict --row <id> --images DIR --out dets.csv  # your own images
```

`run` recomputes frozen configurations and compares them with the stored values of the single evaluation. Two modes:

| mode | what is recomputed | needs | comparison |
|---|---|---|---|
| `cached` (default) | from the stored outputs of the single evaluation: fusion (SCAHI / box fusion) of the stored dense head outputs of every pass and evaluation; for the single-pass and SAHI rows the evaluation of their stored detections | release asset `<model>_visdrone_cached_<split>.tar` (YOLO26m: val 0.65 GB, test-dev 1.27 GB; YOLO26s: val 0.72 GB, test-dev 1.42 GB; SHA-256 in `frozen_results.json` / `frozen_results_yolo26s.json`) unpacked in the repository root; CPU only | exact: bit-identical with the versions of `requirements.lock`, on any machine |
| `full` | everything from the images: new GPU inference with the frozen weights (passes, single passes, SAHI), then fusion and evaluation | frozen weights (SHA-256 checked), GPU | \|Δ\| ≤ 0.05 AP per configuration |

Why a tolerance in `full` mode: other GPUs, drivers and CUDA / cuDNN versions change the last digits of the network
outputs, and a detection whose score or overlap sits exactly at a threshold can flip (one such flip moved a DOTA
configuration by 0.4 mAP50 on the calibration split, `reports/BACKMAP_CHECK.md`). 0.05 AP is the threshold of the
box back-mapping check. On the reference environment the full mode reproduces the stored values exactly (checked on
val: YOLO26m 25 of 25 configurations with Δ = 0; with the published weight assets, one configuration per training size
and model, Δ = 0).

Reference environment of the stored values: NVIDIA GeForce RTX 5090 (32 GB), driver 580.178.04, CUDA 13.0, cuDNN 9.24,
PyTorch 2.14.1+cu130, Ultralytics 8.4.183, Python 3.11.17, NumPy 2.4.6 (all Python versions: `requirements.lock`).
`predict` always runs new inference (your images; no stored value to compare with).

- **What is frozen** (`configs/ext/frozen_rows_ext26.json`, YOLO26s `configs/ext/frozen_rows_ext26s.json`): the weights (SHA-256 of both training sizes; the published
  assets carry the same tensors without the optimizer state), and for every configuration its parameters with their
  origin on calib: the confidence threshold of the single passes, the box-fusion setting, the SCAHI parameters (cell
  factor, bins per octave, splat and peak thresholds), the SAHI setting. Nothing is re-selected.
- **Which configuration.** Most accurate: 2048 px pass + zoom-2 tiles with SCAHI and the refined grid
  (`XG_yolo26m_2048_o2m_C`; exploratory: the grid was searched after the paper's val results, on calib only). With the
  pre-specified SCAHI setting: `E_yolo26m_mz2048_F2_o2m`. Fastest: the 2048 px single pass `E_yolo26m_full2048_o2o`
  (one pass instead of the pass plus all zoom-2 tiles).
- **Box back-mapping convention.** The cached passes of YOLO26m are built with Ultralytics' input preparation (size rule,
  letterbox factor, per-axis back-mapping), the convention under which YOLO26m was tuned and frozen; `frozen.py` and
  `aligned.py` use it (caches under `cache_ualigned/`). The paper's other detectors used one scale factor; the
  difference is below 0.05 AP for every configuration behind a main-paper number or a verdict
  (`reports/BACKMAP_CHECK.md`).
- **Runtimes** (one RTX 5090, 22 CPU cores): cached mode 1-2 min per SCAHI configuration (CPU), seconds for a stored
  single-pass row; full mode in addition the passes of one split, both sizes, about 5 min (val) and 12 min (test-dev), a
  single pass over val about 1 min (GPU, one image at a time), SAHI over val about 3 min.
- **Own images.** `predict` runs the passes in memory and writes `stem, class, score, x, y, w, h` (pixels). The VisDrone
  classes are those of the training data; for other classes, train your own detector (Track B).

## Track B: add a new detector the paper way

Every step uses the public tools; the right column gives the runtime of the YOLO26m run (one RTX 5090, 22 CPU cores).
Each choice is made on calib; val and test-dev are touched once, after the freeze.

| # | step | command / file | YOLO26m |
|---|---|---|---|
| 0 | Pre-register: detector, recipe, rows, questions, ε, rules for untestable cases; commit before any training | append to `docs/PREREGISTRATION_EXT.md` (see the YOLO26m addendum) | – |
| 1 | COCO weights and head check (branches, `reg_max`, strides) | `ultralytics.utils.downloads.attempt_download_asset("yolo26m.pt")`, `YOLO(...).model.model[-1]` | minutes |
| 2 | Decoding parity: own decoding of the dense head = Ultralytics' outputs of the same forward, both branches | `pytest tests/test_decode_reg_max1.py` (direct path for `reg_max = 1`; DFL heads as in the paper) | seconds |
| 3 | Train at both sizes, paper recipe (COCO init, 100 epochs, batch 8 / 3, nominal batch 64, Ultralytics defaults); model selection by the toolkit AP on calib at 36 scheduled epochs (one-to-one head) | `python -m scahi.reproduce train --model yolo26m --imgsz 1280` (and `--imgsz 2048`) | 3.1 h / 7.8 h of training, plus the calib evaluations |
| 4 | Single pass: confidence threshold on calib, then the held-out split | `python -m scahi.reproduce single --weights W --imgsz S --branch o2o --split val` | about 1 min per size and head on calib |
| 5 | Dense-head caches of the stack (pass at the training size + zoom-2 tiles, overlap 0.3) on calib | `python tutorials/yolo26m_visdrone/aligned.py cache --weights W --stack native2048 --split calib` (`native` for 1280) | about 2 min per size |
| 6 | Box fusion on the same passes: 12-point NMS grid (intersection over the smaller box) | `python tutorials/yolo26m_visdrone/paper_way.py tune-nms --weights W --stack native2048 --branch o2o` | steps 6 and 7 together: 8 min (8 grids, 7 processes) |
| 7 | SCAHI, pre-specified 24-point grid | `python tutorials/yolo26m_visdrone/aligned.py tune --weights W --stack native2048 --variant F2 --branch o2m` | (see step 6) |
| 8 | Refined grid (exploratory): frozen procedure, selection under a 2x fusion-time limit | `python tutorials/yolo26m_visdrone/paper_way.py refined-grid --weights W --stack native2048 --branch o2m --row C` | 1.5 h for 8 cases (7 processes) |
| 9 | Freeze: one file with the weight checksums, every choice and its calib origin; recompute every row on calib through the held-out code path (replay); approval before any held-out number | example: `configs/ext/frozen_rows_ext26.json` | calib replay about 15 min |
| 10 | Caches of val and test-dev | `aligned.py cache ... --split val` / `--split test-dev` | about 15 min (both sizes, both splits) |
| 11 | Exactly one evaluation on val; recompute it once more in fresh processes (must be identical); then exactly one on test-dev | `aligned.py fuse --weights W --stack native2048 --variant F2 --branch o2m --params '<frozen>' --split val`, `reproduce single ...` | val 17 min, val replay 12 min, test-dev 28 min (25 rows) |
| 12 | Verdicts: paired bootstrap, Δ > ε (Q3 also lower CI bound > 0) | `python tutorials/yolo26m_visdrone/paper_way.py compare --a A.parquet --b B.parquet --split val --eps 0.30` | about 1 min per comparison |

Notes. The SAHI baseline row was tuned with the paper's SAHI grid (research pipeline, not part of these tools); it is
an optional comparison. The model selection of step 3 reads the toolkit AP of the one-to-one head for two-branch heads
(YOLOv10, YOLO26) and of the only head otherwise. Report every frozen configuration, whatever its result.

> **Shortcuts - not the paper protocol.** Each of these saves time, and none of them gives numbers comparable to the
> paper's:
>
> - selecting the checkpoint with Ultralytics' built-in validator (`best.pt`: its mAP on YOLO labels without the ignored
>   regions, at most 300 detections per image) instead of the toolkit AP on calib;
> - transferring SCAHI parameters from a similar detector (e.g. the refined-grid values of YOLOv11m at 2048 px: cell
>   factor 13, 7 bins per octave, splat threshold 0.00125, peak threshold 0.001) instead of tuning on calib;
> - skipping the refined grid and using the pre-specified SCAHI setting (cell factor 5, 5 bins per octave, 0.005 / 0.001);
> - building the caches with `python -m scahi.reproduce cache` (one scale factor) instead of `aligned.py` - differences
>   are below 0.05 AP for most configurations and up to about 0.1 AP on calib for a 1280 px whole-image pass alone;
> - tuning or choosing anything on val, or evaluating on val repeatedly; skipping the freeze and the replay.
