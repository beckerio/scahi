# Tutorial: YOLO26m with SCAHI on VisDrone

YOLO26m was added to the study after the paper's experiments, **the same way as the paper's detectors**: a written
pre-registration addendum before any training (`docs/PREREGISTRATION_EXT.md`, addendum of 2026-11-13), the extension
recipe, model selection and all tuning on the calibration split, a frozen configuration file with weight checksums, a
replay check, and then exactly one evaluation on VisDrone val and one on test-dev. The results are reported separately
from the paper's five extension detectors (they count in no family criterion) and are part of the arXiv version and of
this repository; the decision log `docs/DECISIONS.md` records every step.

This tutorial has two tracks:

- **Track A** - use the frozen YOLO26m weights and parameters (on VisDrone or on your own images);
- **Track B** - add a new detector the paper way, step by step, with the runtimes of the YOLO26m run.

Shortcuts that are *not* the paper protocol are collected in one marked box at the end.

## Results (single held-out evaluation)

AP@[.5:.95], VisDrone toolkit protocol (ignored regions, at most 500 detections per image). All 25 frozen
configurations: `python tutorials/yolo26m_visdrone/frozen.py rows`.

| configuration | val | test-dev |
|---|---:|---:|
| single pass at 1280 px (one-to-one head) | 37.41 | 32.54 |
| single pass at 2048 px (one-to-one head) | 41.25 | 35.30 |
| 1280 px pass + zoom-2 tiles, box fusion (one-to-one head) | 37.89 | 32.58 |
| 2048 px pass + zoom-2 tiles, SCAHI, pre-specified setting | 41.53 | 35.82 |
| 2048 px pass + zoom-2 tiles, SCAHI, refined grid (exploratory) | **42.89** | **36.59** |
| SAHI (640 px slices, 1280 px weights) | 34.56 | 30.51 |

Pre-registered questions (verdict on val; ε = 0.30 AP; paired bootstrap over the images, 1000 resamples, seed 0):

| question | Δ on val [95 % CI] | verdict | Δ on test-dev |
|---|---|---|---|
| Q1: single pass 2048 px beats 1280 px + tiles with box fusion | +3.36 [+2.91; +3.83] | met | +2.72 |
| Q2: SCAHI beats box fusion on the same passes, 1280 px | +1.94 [+1.65; +2.21] | met | +2.24 |
| Q2: SCAHI beats box fusion on the same passes, 2048 px | +1.77 [+1.54; +1.99] | met | +2.52 |
| Q3: 2048 px + tiles with SCAHI beats the 2048 px single pass (also CI > 0) | +0.28 [+0.04; +0.51] | not met (Δ < ε) | +0.51 |

As for YOLOv10m in the paper, Q3 misses the threshold on val; test-dev is reported next to it and does not change the
verdict.

## Quick start

```bash
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu130 && pip install -e .
python tutorials/yolo26m_visdrone/prepare_data.py          # VisDrone train / val with the ignored regions (downloads)
# frozen weights: the release assets visdrone_ext_yolo26m_1280_s0.pt and visdrone_ext_yolo26m_2048_s0.pt -> weights/
python tutorials/yolo26m_visdrone/frozen.py run --row XG_yolo26m_2048_o2m_C --split val
# -> XG_yolo26m_2048_o2m_C: AP 42.89006 (identical)
python tutorials/yolo26m_visdrone/frozen.py predict --row XG_yolo26m_2048_o2m_C --images my_images/ --out dets.csv
```

`prepare_data.py` downloads the original VisDrone2019-DET train and val zips and converts them with the repository's
converter, keeping the ignored regions that the toolkit evaluation needs (Ultralytics' `VisDrone.yaml` drops them). For
test-dev, download the official data and run `python -m scahi.reproduce convert` (see `REPRODUCE.md`); the tutorial
uses those canonical files whenever they exist.

**Splits.** Training on `train_main` (5 877 images, 178 sequences); model selection and every tuning step on `calib`
(594 images, 30 whole video sequences of VisDrone train, `data/splits/calib.txt`: the draw of 5 000 whose object-size
and class distribution is closest to all of train; no sequence is shared with `train_main`). VisDrone val (548) and
test-dev (1 610) are used once, after the freeze. For your own data, hold out whole sequences, scenes or flights in the
same way.

## Track A: use the frozen weights and parameters

```bash
python tutorials/yolo26m_visdrone/frozen.py rows                              # 25 configurations, stored val / test-dev AP
python tutorials/yolo26m_visdrone/frozen.py run --row <id> --split val        # recompute one, compare with the stored AP
python tutorials/yolo26m_visdrone/frozen.py run --row all --split test-dev    # all of them
python tutorials/yolo26m_visdrone/frozen.py predict --row <id> --images DIR --out dets.csv
```

- **What is frozen** (`configs/ext/frozen_rows_ext26.json`): the weights (SHA-256 of both training sizes; the published
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
- **Runtimes** (one RTX 5090, 22 CPU cores): caches of one split, both sizes, about 5 min (val) and 12 min (test-dev);
  one SCAHI configuration from the caches 1-2 min (CPU); a single pass over val about 1 min (GPU, one image at a time);
  SAHI over val about 3 min.
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
