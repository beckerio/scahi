# Pre-registration addenda: YOLO26m and YOLO26s (extension study, VisDrone)

English rendering of the two dated addenda to the extension study's pre-registration. Both were written before any
training of the respective model; nothing of the five-detector study was changed. Frozen configurations:
`configs/ext/frozen_rows_ext26.json` (YOLO26m) and `configs/ext/frozen_rows_ext26s.json` (YOLO26s); results and
verdicts: `runs/ext/yolo26m/`, `runs/ext/yolo26s/`; paper values: `reports/ext26/YOLO26_PAPER_VALUES.md`.

## Addendum of 2026-11-13: YOLO26m as the sixth detector

Written before any YOLO26m result on VisDrone val or test-dev existed. Two aborted tutorial trainings with a different
recipe (1280 px, epochs 7 and 4) existed and were discarded; they enter nothing.

1. **Model.** YOLO26m, Ultralytics 8.4.183 (pinned), COCO weights `yolo26m.pt` (release v8.4.0, SHA-256
   401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7, recorded in `runs/ext/heads.json`). Head as
   YOLOv10: two branches, one-to-many (o2m) and one-to-one (o2o); default output o2o (NMS-free, top-k). Box encoding
   `reg_max = 1`: the box head outputs the four side distances directly (no DFL distribution). The own decoding gets a
   direct path for this case only (`scahi/infer/dfl_decode.py`; standard deviation 0, not used by SCAHI's F2 variant);
   for all other models it stays unchanged and bit-identical (checked by the regeneration and the replay gate).
2. **Classification.** Treated by the rules for two-branch models, i.e. exactly as YOLOv10m: single passes at 1280 and
   2048 px and model selection on o2o (o2m curve recorded); SAHI on the 1280 px weights (o2o); box fusion (NMS-IOS) with
   tiles in both branches; SCAHI with tiles with o2m (main row) and in addition o2o; descriptive single pass o2m + NMS
   at 1280 and 2048 px; the refined-grid rows (row B: whole-image pass, row C: with tiles; time limit at most 2x the
   pre-specified SCAHI setting, selection by maximum calib AP) and the whole-image rows as for the five detectors, with
   the same classification (exploratory).
3. **Recipe unchanged.** COCO initialisation, 100 epochs, batch 8 (1280 px) / 3 (2048 px), seed 0, AMP, Ultralytics'
   default augmentation and loss, `optimizer=auto`, training list `train_main`, same code. Model selection: toolkit-port
   AP on calib at the same 36 scheduled epochs (every 5 epochs up to 80, then each of epochs 81-100), o2o; Ultralytics'
   fitness is not used. The checkpoint of every evaluated epoch is kept (copy, no change to the training). OOM rule
   unchanged: a single reduction 8 -> 6 or 3 -> 2, batch >= 2.
4. **Questions and verdicts.** Q1, Q2 (per input size) and Q3 unchanged, eps = 0.30 AP, paired bootstrap over the val
   images (1000 resamples, seed 0), **verdict per detector**; Q4 descriptive as for YOLOv10m. The verdicts of the five
   detectors, the ">= 4 of 5" criterion and the pooled effect are unchanged; YOLO26m is reported separately and counts in
   no family criterion. Holm appendix for YOLO26m alone (a family of one model, i.e. unadjusted).
5. **Gates.** Decoding parity (own decoding vs. Ultralytics', both branches, coordinates <= 4 ulp with TF32 off, |Delta AP|
   <= 0.01 on calib) and the per-size cache gate; the s-model gate does not apply. If the decoding gate fails, no dense
   rows are produced.
6. **Untestable cases.** If batch 2 does not fit at 2048 px, a run fails for good or a gate fails, the affected rows are
   missing; every question that needs a missing row is "not testable" for YOLO26m and counts as **"not met"**. The
   remaining questions are decided normally.
7. **Procedure.** Caches and calib tuning (single-pass confidence, NMS-IOS grid, the 24-point grid of the pre-specified
   SCAHI setting, refined grid with time limit, SAHI grid) as in the extension; freeze with weight checksums and the
   origin of every choice; replay gate; then exactly **one** evaluation on val and exactly **one** on test-dev; reporting
   rules as in the extension's freeze (the val verdict decides, test-dev is reported next to it without a confirming
   role).

Decision before the freeze (2026-11-14): the box back-mapping rule of the study was applied to YOLO26m before its freeze;
its cache rows were built and tuned with Ultralytics' input preparation (size rule, letterbox gain, per-axis
back-mapping; `scahi/mapcheck.py`, `ultralytics_prep`). The single-pass rows were gated as "bit-identical in two
replays AND |Delta| to the tuning value <= 0.05 AP".

## Addendum of 2026-11-14: YOLO26s as the seventh detector

Written before any YOLO26s training or result existed (only the downloaded COCO weights existed).

1. **Model.** YOLO26s, Ultralytics 8.4.183 (pinned), COCO weights `yolo26s.pt` (release v8.4.0, SHA-256
   646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b, recorded in `runs/ext/heads.json`). Head as YOLO26m:
   two branches (o2m / o2o), `reg_max = 1`; the same direct decoding path as for YOLO26m.
2. **Classification.** Exactly as YOLO26m (two-branch rules as YOLOv10m): the same 25 rows with the same classification.
3. **Recipe unchanged** (as YOLO26m). Before the start, a diff of all training arguments against YOLO26m 1280 / 2048 px
   was recorded (only the model and the output paths differ). Model selection as YOLO26m; the checkpoint of every
   evaluated epoch is kept. OOM rule unchanged (resumption from the last checkpoint after an abort recorded as a
   deviation).
4. **Box back-mapping.** The cache rows use the Ultralytics conventions from the start (size rule, letterbox gain,
   per-axis back-mapping; `scahi/mapcheck.py`, cache root `cache_ualigned/`); calib, val and test-dev caches are built
   only this way.
5. **Questions and verdicts.** Q1, Q2 (per input size) and Q3 unchanged, eps = 0.30, paired bootstrap over the val
   images (1000, seed 0), **verdict per detector**; Q4 descriptive. Reported separately like YOLO26m, in no family
   criterion; Holm appendix for YOLO26s alone (unadjusted).
6. **Gates.** Decoding parity, the per-size cache gate and, as YOLO26s is an s model, the **s-model gate**: the 1280 px
   single pass (o2o) within +-5 AP of YOLOv10s at 1280 px (val 33.66), checked after the single val evaluation; reported,
   changes no verdict. Single-pass gate on calib as for YOLO26m (bit-identical in two replays and |Delta| to the tuning
   value <= 0.05 AP) and, before test-dev, the exact val replay of all rows.
7. **Untestable cases.** As for YOLO26m.
8. **Descriptive equal-compute comparison** (as YOLOv10s 2048 px vs. YOLOv10m 1280 px in the paper): single pass YOLO26s
   at 2048 px vs. single pass YOLO26m at 1280 px, both o2o: GFLOPs per image (exact per input shape, `scahi/flops.py`,
   over the val / test-dev images) and AP on val and test-dev, Delta with paired bootstrap CI (1000, seed 0). No verdict,
   no threshold.
9. **Procedure.** As for YOLO26m; the freeze proposal goes to the authors before any evaluation on val or test-dev; after
   their approval exactly one evaluation on val (with exact replay) and exactly one on test-dev.

Recorded deviation (2026-11-15): the single-pass gate of item 6 was **not met** as specified for YOLO26s (the two replays
were bit-identical, but |Delta| to the tuning value was 0.079-0.094 AP; tuning at batch 16, replay at batch 1). Because
the gate protects the calib choice, the confidence grid was re-evaluated on the batch-1 detections: same choice (0.001)
and same ordering in all four cases. The choice stands, the batch-1 values are the reference for the val replay, the
tuning values are reported alongside; no rule was changed. Cause: Ultralytics' predictor pads a batch minimally only if
all its images have the same shape, otherwise every image to the full square input size (5 image shapes on calib, 8 of
38 batches of 16 mixed); batch 16 with one shape per batch reproduces the batch-1 values within 0.005 AP
(`runs/ext/yolo26s/batch_padding_check.json`, `reports/ext26/single_pass_gate.csv`).
