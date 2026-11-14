# Box back-mapping check of all frozen configurations

Descriptive check after the held-out evaluations. Frozen values, selections and verdicts are kept; this report lists the recomputed values.

## What was checked

The cached whole-image passes map boxes from the network input back to the image with one scale factor. Ultralytics' own predictor differs in three details: (1) **size rule** - it rounds the resized image side before padding to a multiple of 32; (2) **letterbox factor** - it resizes with min(L/h, L/w) for the target size L; (3) **per-axis back-mapping** - it divides each axis by its own effective factor (rounded side / side). The detections differ by fractions of a pixel, and only for images whose resized side is rounded (on VisDrone at 1280 px: the 1400 x 788 images). Tiles of 512 px scaled to 1024 px are not affected on VisDrone (recomputed tiles are bit-identical); on DOTA some edge tiles differ.

## Rule (fixed before any number of the check existed)

Each configuration is recomputed on the calibration split with all three details aligned to Ultralytics at once, without re-tuning. If the change is below 0.05 (VisDrone: AP@[.5:.95]; DOTA: mAP50), nothing changes and the size of the deviation is documented. If it is 0.05 or larger, the configuration is recomputed on the held-out splits (VisDrone val and test-dev, DOTA-val) with the aligned conventions, again without re-tuning, and original and corrected values are reported side by side; pre-specified verdicts stay as originally computed. Only for configurations reaching 0.05 is the contribution of each detail reported separately.

## Scope

- 125 frozen configurations whose detections rest on a cached whole-image pass (box fusion or SCAHI on the whole-image pass with or without tiles); 5 of them were already covered by the original check.
- Not affected and therefore not recomputed: single passes, multi-zoom inference with the predictor and SAHI (the Ultralytics predictor maps boxes itself); SCAHI on SAHI's passes on DOTA (its caches were built with Ultralytics' preparation); FCOS (torchvision resizes and maps back per axis itself).
- Every configuration is fused twice through the same code path as its frozen evaluation; only the source of the cached passes differs. The value from the frozen caches reproduces the stored calibration value exactly for every configuration that has one. DOTA values on calib use the calibration selection limit of 20 000 detections per image (as stored), DOTA-val values the evaluation limit of 200 000.
- Parallel computation was cross-checked: one configuration per study recomputed in a single process gave bit-identical values (yes).

## Summary

- Configurations behind numbers of the main paper or a pre-specified verdict: largest change 0.0443 (below 0.05).
- 15 exploratory or decomposition configurations reach 0.05 on calib; on the held-out splits they change by at most 0.094. None of them enters a verdict.
- Frozen values are kept in all tables; the corrected values are listed below.

## Configurations reaching 0.05 on calib: held-out values

| dataset | detector | weights | configuration | calib change | split | frozen | recomputed | change |
|---|---|---|---|---:|---|---:|---:|---:|
| DOTA v1.5 | YOLOv10m | trained at 2048 px | 2048 px whole-image pass, SCAHI, refined grid, one-to-many head | -0.4207 | val | 68.42 | 68.44 | +0.025 |
| DOTA v1.5 | YOLOv10s | trained at 2048 px | 2048 px whole-image pass, SCAHI, refined grid, one-to-many head | -0.0653 | val | 67.27 | 67.37 | +0.094 |
| DOTA v1.5 | YOLOv12s | trained at 1024 px | 1024 px whole-image pass, SCAHI, setting from VisDrone | +0.0813 | val | 55.78 | 55.80 | +0.019 |
| VisDrone | YOLOv10m | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting, one-to-many head | +0.0897 | val | 35.53 | 35.53 | +0.000 |
| VisDrone | YOLOv10m | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting, one-to-many head | +0.0897 | test-dev | 31.23 | 31.25 | +0.027 |
| VisDrone | YOLOv10m | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting, one-to-one head | +0.0864 | val | 35.66 | 35.66 | +0.000 |
| VisDrone | YOLOv10m | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting, one-to-one head | +0.0864 | test-dev | 31.29 | 31.32 | +0.030 |
| VisDrone | YOLOv10m | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid, one-to-many head | +0.0802 | val | 36.68 | 36.68 | +0.000 |
| VisDrone | YOLOv10m | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid, one-to-many head | +0.0802 | test-dev | 31.86 | 31.89 | +0.026 |
| VisDrone | YOLOv10m | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid, one-to-one head | +0.0873 | val | 36.75 | 36.75 | +0.000 |
| VisDrone | YOLOv10m | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid, one-to-one head | +0.0873 | test-dev | 31.93 | 31.96 | +0.027 |
| VisDrone | YOLOv11m | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting | +0.0981 | val | 35.99 | 35.99 | +0.000 |
| VisDrone | YOLOv11m | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting | +0.0981 | test-dev | 31.53 | 31.57 | +0.034 |
| VisDrone | YOLOv11m | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid | +0.1034 | val | 37.25 | 37.25 | +0.000 |
| VisDrone | YOLOv11m | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid | +0.1034 | test-dev | 32.16 | 32.19 | +0.032 |
| VisDrone | YOLOv11s | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting | +0.0755 | val | 32.36 | 32.36 | +0.000 |
| VisDrone | YOLOv11s | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting | +0.0755 | test-dev | 29.10 | 29.12 | +0.021 |
| VisDrone | YOLOv11s | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid | +0.0766 | val | 33.44 | 33.44 | +0.000 |
| VisDrone | YOLOv11s | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid | +0.0766 | test-dev | 29.60 | 29.62 | +0.020 |
| VisDrone | YOLOv12m | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting | +0.0859 | val | 35.82 | 35.82 | +0.000 |
| VisDrone | YOLOv12m | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting | +0.0859 | test-dev | 31.36 | 31.39 | +0.032 |
| VisDrone | YOLOv12m | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid | +0.0790 | val | 36.99 | 36.99 | +0.000 |
| VisDrone | YOLOv12m | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid | +0.0790 | test-dev | 31.99 | 32.02 | +0.026 |
| VisDrone | YOLOv12s | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting | +0.0798 | val | 32.88 | 32.88 | +0.000 |
| VisDrone | YOLOv12s | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting | +0.0798 | test-dev | 29.95 | 29.99 | +0.038 |
| VisDrone | YOLOv12s | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid | +0.0815 | val | 34.00 | 34.00 | +0.000 |
| VisDrone | YOLOv12s | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid | +0.0815 | test-dev | 30.55 | 30.59 | +0.038 |

VisDrone val contains no 1400 x 788 image, so the 1280 px configurations are unchanged on val.

Contribution of each detail on calib (one detail aligned at a time, the others as frozen):

| dataset | detector | configuration | all three | size rule | letterbox factor | per-axis back-mapping |
|---|---|---|---:|---:|---:|---:|
| DOTA v1.5 | YOLOv10m | 2048 px whole-image pass, SCAHI, refined grid, one-to-many head | -0.4207 | -0.0059 | +0.0000 | -0.4150 |
| DOTA v1.5 | YOLOv10s | 2048 px whole-image pass, SCAHI, refined grid, one-to-many head | -0.0653 | -0.0397 | +0.0000 | -0.0274 |
| DOTA v1.5 | YOLOv12s | 1024 px whole-image pass, SCAHI, setting from VisDrone | +0.0813 | +0.0032 | +0.0000 | +0.0770 |
| VisDrone | YOLOv10m | 1280 px whole-image pass, SCAHI, pre-specified setting, one-to-many head | +0.0897 | +0.0000 | +0.0000 | +0.0897 |
| VisDrone | YOLOv10m | 1280 px whole-image pass, SCAHI, pre-specified setting, one-to-one head | +0.0864 | +0.0000 | +0.0000 | +0.0864 |
| VisDrone | YOLOv10m | 1280 px whole-image pass, SCAHI, refined grid, one-to-many head | +0.0802 | +0.0000 | +0.0000 | +0.0802 |
| VisDrone | YOLOv10m | 1280 px whole-image pass, SCAHI, refined grid, one-to-one head | +0.0873 | +0.0000 | +0.0000 | +0.0873 |
| VisDrone | YOLOv11m | 1280 px whole-image pass, SCAHI, pre-specified setting | +0.0981 | +0.0000 | +0.0000 | +0.0981 |
| VisDrone | YOLOv11m | 1280 px whole-image pass, SCAHI, refined grid | +0.1034 | +0.0000 | +0.0000 | +0.1034 |
| VisDrone | YOLOv11s | 1280 px whole-image pass, SCAHI, pre-specified setting | +0.0755 | +0.0000 | +0.0000 | +0.0755 |
| VisDrone | YOLOv11s | 1280 px whole-image pass, SCAHI, refined grid | +0.0766 | +0.0000 | +0.0000 | +0.0766 |
| VisDrone | YOLOv12m | 1280 px whole-image pass, SCAHI, pre-specified setting | +0.0859 | +0.0000 | +0.0000 | +0.0859 |
| VisDrone | YOLOv12m | 1280 px whole-image pass, SCAHI, refined grid | +0.0790 | +0.0000 | +0.0000 | +0.0790 |
| VisDrone | YOLOv12s | 1280 px whole-image pass, SCAHI, pre-specified setting | +0.0798 | +0.0000 | +0.0000 | +0.0798 |
| VisDrone | YOLOv12s | 1280 px whole-image pass, SCAHI, refined grid | +0.0815 | +0.0000 | +0.0000 | +0.0815 |

**DOTA, YOLOv10m, 2048 px pass alone + SCAHI refined grid (calib change -0.421).** Almost the whole change sits in one class and one image: helicopter AP50 60.06 -> 53.57 (the calibration split has 4 helicopter objects); replacing the detections of one image at a time, image P1131 alone accounts for -0.406. There the highest-scoring helicopter detection has an IoU of 0.500021 with a ground-truth box (true positive) on the frozen pass and 0.499968 (false positive) after a shift of 0.37 px, which moves three of the 11 interpolation points of the VOC07 precision curve. The detection limit is not involved (change -0.421 with a limit of 10 000 000 and -0.418 without the images at the limit). On DOTA-val the configuration changes by +0.025.

## All configurations, VisDrone (calib)

| detector | weights | configuration | used in | frozen | recomputed | change |
|---|---|---|---|---:|---:|---:|
| YOLOv10m | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting, one-to-many head | main paper | 41.5049 | 41.5174 | +0.0125 |
| YOLOv10m | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting, one-to-one head | pre-specified row | 41.1374 | 41.1609 | +0.0235 |
| YOLOv10m | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, refined grid, one-to-many head | exploratory (no verdict) | 42.6263 | 42.6408 | +0.0145 |
| YOLOv10m | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, refined grid, one-to-one head | exploratory (no verdict) | 42.3636 | 42.3761 | +0.0126 |
| YOLOv10m | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-many head | main paper | 39.5601 | 39.5822 | +0.0220 |
| YOLOv10m | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-one head | main paper | 40.0361 | 40.0698 | +0.0337 |
| YOLOv10m | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting, one-to-many head | exploratory (no verdict) | 37.8587 | 37.9484 | +0.0897 **>= 0.05** |
| YOLOv10m | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting, one-to-one head | exploratory (no verdict) | 37.7662 | 37.8526 | +0.0864 **>= 0.05** |
| YOLOv10m | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid, one-to-many head | exploratory (no verdict) | 38.8691 | 38.9493 | +0.0802 **>= 0.05** |
| YOLOv10m | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid, one-to-one head | exploratory (no verdict) | 38.7571 | 38.8444 | +0.0873 **>= 0.05** |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting, one-to-many head | main paper | 43.7899 | 43.7888 | -0.0011 |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting, one-to-one head | pre-specified row | 43.7081 | 43.6953 | -0.0128 |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, refined grid, one-to-many head | main paper | 45.1637 | 45.1584 | -0.0053 |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, refined grid, one-to-one head | exploratory (no verdict) | 45.0584 | 45.0473 | -0.0110 |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-many head | main paper | 42.1864 | 42.1824 | -0.0040 |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-one head | pre-specified row | 42.9202 | 42.9177 | -0.0025 |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass, SCAHI, pre-specified setting, one-to-many head | main paper | 42.1643 | 42.1559 | -0.0085 |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass, SCAHI, pre-specified setting, one-to-one head | exploratory (no verdict) | 42.1214 | 42.1138 | -0.0076 |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass, SCAHI, refined grid, one-to-many head | exploratory (no verdict) | 43.3465 | 43.3366 | -0.0100 |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass, SCAHI, refined grid, one-to-one head | exploratory (no verdict) | 43.2030 | 43.1975 | -0.0055 |
| YOLOv10s | single-pass training at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-one head | main paper | 37.7497 | 37.7760 | +0.0263 |
| YOLOv10s | single-pass training at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (WBF), one-to-one head | pre-specified row | 35.2880 | 35.2936 | +0.0056 |
| YOLOv10s | single-pass training at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (greedy NMM), one-to-one head | pre-specified row | 35.3242 | 35.3559 | +0.0317 |
| YOLOv10s | single-pass training at 1536 px | 1536 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting, one-to-many head | exploratory (no verdict) | 40.8585 | 40.8369 | -0.0216 |
| YOLOv10s | single-pass training at 1536 px | 1536 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-one head | exploratory (no verdict) | 39.1707 | 39.1629 | -0.0078 |
| YOLOv10s | single-pass training at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting, one-to-many head | main paper | 42.2538 | 42.2481 | -0.0056 |
| YOLOv10s | single-pass training at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, refined grid, one-to-many head | main paper (original check) | 43.4588 | 43.4574 | -0.0014 |
| YOLOv10s | single-pass training at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, refined grid, one-to-one head | exploratory (no verdict) | 43.1918 | 43.1909 | -0.0009 |
| YOLOv10s | single-pass training at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-one head | main paper (original check) | 41.1703 | 41.1571 | -0.0131 |
| YOLOv10s | single-pass training at 2048 px | 2048 px whole-image pass, SCAHI, pre-specified setting, one-to-many head | main paper | 40.4973 | 40.4689 | -0.0284 |
| YOLOv10s | single-pass training at 2048 px | 2048 px whole-image pass, SCAHI, refined grid, one-to-many head | exploratory (no verdict) | 41.5060 | 41.4769 | -0.0291 |
| YOLOv10s | single-pass training at 2048 px | 2048 px whole-image pass, SCAHI, refined grid, one-to-one head | exploratory (no verdict) | 41.2746 | 41.2503 | -0.0243 |
| YOLOv10s | single-pass training at 1280 px | whole-image pass at native resolution + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting, one-to-many head | pre-specified row | 39.8960 | 39.9037 | +0.0077 |
| YOLOv10s | single-pass training at 1536 px | whole-image pass at native resolution + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting, one-to-many head | pre-specified row | 41.0087 | 41.0021 | -0.0066 |
| YOLOv10s | tile training, scale-normalised | whole-image pass at native resolution + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting, one-to-many head | pre-specified row | 40.6339 | 40.6376 | +0.0037 |
| YOLOv10s | tile training, ignore regions dropped | whole-image pass at native resolution + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-one head | pre-specified row | 38.2873 | 38.2808 | -0.0065 |
| YOLOv10s | tile training, ignore regions painted grey | whole-image pass at native resolution + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-one head | pre-specified row | 38.5399 | 38.5349 | -0.0050 |
| YOLOv10s | tile training, loss-masked | whole-image pass at native resolution + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-one head | pre-specified row | 38.6426 | 38.6424 | -0.0002 |
| YOLOv10s | tile training, loss-masked + truncation ignore | whole-image pass at native resolution + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-one head | pre-specified row | 38.5999 | 38.5988 | -0.0011 |
| YOLOv10s | tile training, loss-masked, seed 1 | whole-image pass at native resolution + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-one head | seed repeat | 38.8506 | 38.8454 | -0.0052 |
| YOLOv10s | tile training, loss-masked, seed 2 | whole-image pass at native resolution + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-one head | seed repeat | 38.9474 | 38.9437 | -0.0037 |
| YOLOv10s | tile training, scale-normalised | whole-image pass at native resolution + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-one head | pre-specified row | 39.5488 | 39.5571 | +0.0083 |
| YOLOv10s | tile training, scale-normalised, seed 1 | whole-image pass at native resolution + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-one head | seed repeat | 39.0127 | 39.0158 | +0.0031 |
| YOLOv10s | tile training, scale-normalised, seed 2 | whole-image pass at native resolution + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-one head | seed repeat | 39.0284 | 39.0226 | -0.0058 |
| YOLOv10s | tile training, loss-masked | whole-image pass at zoom 0.64 + zoom-2 tiles (1024 px input), SCAHI with validity windows, one-to-many head | pre-specified row | 39.0312 | 39.0365 | +0.0053 |
| YOLOv10s | tile training, scale-normalised | whole-image pass at zoom 0.64 + zoom-2 tiles (1024 px input), SCAHI with validity windows, one-to-many head | pre-specified row | 39.7784 | 39.7952 | +0.0168 |
| YOLOv10s | tile training, scale-normalised, seed 1 | whole-image pass at zoom 0.64 + zoom-2 tiles (1024 px input), SCAHI with validity windows, one-to-many head | seed repeat | 39.4122 | 39.4254 | +0.0132 |
| YOLOv10s | tile training, scale-normalised, seed 2 | whole-image pass at zoom 0.64 + zoom-2 tiles (1024 px input), SCAHI with validity windows, one-to-many head | seed repeat | 39.5253 | 39.5413 | +0.0160 |
| YOLOv10s | tile training, scale-normalised | whole-image pass at zoom 0.64 + zoom-2 tiles (1024 px input), SCAHI with validity windows, one-to-one head | pre-specified row | 39.3055 | 39.3177 | +0.0123 |
| YOLOv10s | tile training, loss-masked | whole-image pass at zoom 0.64 + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-many head | pre-specified row | 37.0142 | 37.0229 | +0.0087 |
| YOLOv10s | tile training, scale-normalised | whole-image pass at zoom 0.64 + zoom-2 tiles (1024 px input), box fusion (NMS), one-to-many head | pre-specified row | 38.5196 | 38.5209 | +0.0014 |
| YOLOv11m | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting | main paper | 41.9551 | 41.9662 | +0.0110 |
| YOLOv11m | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, refined grid | exploratory (no verdict) | 43.1562 | 43.1702 | +0.0141 |
| YOLOv11m | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (NMS) | main paper | 39.6143 | 39.6327 | +0.0183 |
| YOLOv11m | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting | exploratory (no verdict) | 38.5671 | 38.6652 | +0.0981 **>= 0.05** |
| YOLOv11m | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid | exploratory (no verdict) | 39.5650 | 39.6685 | +0.1034 **>= 0.05** |
| YOLOv11m | trained at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting | main paper | 44.0315 | 44.0323 | +0.0009 |
| YOLOv11m | trained at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, refined grid | main paper | 45.3925 | 45.3882 | -0.0043 |
| YOLOv11m | trained at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (NMS) | main paper | 42.1404 | 42.1397 | -0.0007 |
| YOLOv11m | trained at 2048 px | 2048 px whole-image pass, SCAHI, pre-specified setting | main paper | 42.5608 | 42.5505 | -0.0103 |
| YOLOv11m | trained at 2048 px | 2048 px whole-image pass, SCAHI, refined grid | exploratory (no verdict) | 43.7199 | 43.7043 | -0.0156 |
| YOLOv11s | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting | main paper | 39.1443 | 39.1644 | +0.0202 |
| YOLOv11s | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, refined grid | exploratory (no verdict) | 40.3875 | 40.4124 | +0.0249 |
| YOLOv11s | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (NMS) | main paper | 37.1589 | 37.1799 | +0.0210 |
| YOLOv11s | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting | exploratory (no verdict) | 35.3854 | 35.4609 | +0.0755 **>= 0.05** |
| YOLOv11s | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid | exploratory (no verdict) | 36.3027 | 36.3794 | +0.0766 **>= 0.05** |
| YOLOv11s | trained at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting | main paper | 42.1208 | 42.1081 | -0.0127 |
| YOLOv11s | trained at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, refined grid | main paper | 43.4468 | 43.4384 | -0.0085 |
| YOLOv11s | trained at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (NMS) | main paper | 40.2845 | 40.2790 | -0.0055 |
| YOLOv11s | trained at 2048 px | 2048 px whole-image pass, SCAHI, pre-specified setting | main paper | 40.1586 | 40.1416 | -0.0169 |
| YOLOv11s | trained at 2048 px | 2048 px whole-image pass, SCAHI, refined grid | exploratory (no verdict) | 41.2293 | 41.2121 | -0.0171 |
| YOLOv12m | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting | main paper | 41.7482 | 41.7641 | +0.0159 |
| YOLOv12m | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, refined grid | exploratory (no verdict) | 43.0391 | 43.0498 | +0.0107 |
| YOLOv12m | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (NMS) | main paper | 39.6095 | 39.6238 | +0.0142 |
| YOLOv12m | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting | exploratory (no verdict) | 38.6208 | 38.7067 | +0.0859 **>= 0.05** |
| YOLOv12m | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid | exploratory (no verdict) | 39.7018 | 39.7808 | +0.0790 **>= 0.05** |
| YOLOv12s | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting | main paper | 40.0538 | 40.0699 | +0.0161 |
| YOLOv12s | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, refined grid | exploratory (no verdict) | 41.2615 | 41.2744 | +0.0130 |
| YOLOv12s | trained at 1280 px | 1280 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (NMS) | main paper | 37.8531 | 37.8693 | +0.0162 |
| YOLOv12s | trained at 1280 px | 1280 px whole-image pass, SCAHI, pre-specified setting | exploratory (no verdict) | 36.2387 | 36.3186 | +0.0798 **>= 0.05** |
| YOLOv12s | trained at 1280 px | 1280 px whole-image pass, SCAHI, refined grid | exploratory (no verdict) | 37.1163 | 37.1978 | +0.0815 **>= 0.05** |
| YOLOv12s | trained at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, pre-specified setting | main paper | 42.7610 | 42.7564 | -0.0046 |
| YOLOv12s | trained at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), SCAHI, refined grid | main paper | 44.1622 | 44.1611 | -0.0011 |
| YOLOv12s | trained at 2048 px | 2048 px whole-image pass + zoom-2 tiles (1024 px input), box fusion (NMS) | main paper | 41.2406 | 41.2366 | -0.0040 |
| YOLOv12s | trained at 2048 px | 2048 px whole-image pass, SCAHI, pre-specified setting | main paper | 40.8717 | 40.8587 | -0.0129 |
| YOLOv12s | trained at 2048 px | 2048 px whole-image pass, SCAHI, refined grid | exploratory (no verdict) | 41.9239 | 41.9100 | -0.0139 |

## All configurations, DOTA v1.5 (calib)

| detector | weights | configuration | used in | frozen | recomputed | change |
|---|---|---|---|---:|---:|---:|
| YOLOv10m | trained at 1024 px | 1024 px whole-image pass + native-resolution tiles, SCAHI, refined grid, one-to-many head | main paper | 72.1764 | 72.1485 | -0.0279 |
| YOLOv10m | trained at 1024 px | 1024 px whole-image pass + native-resolution tiles, SCAHI, setting from VisDrone, one-to-many head | pre-specified row | 71.5417 | 71.4974 | -0.0443 |
| YOLOv10m | trained at 1024 px | 1024 px whole-image pass + native-resolution tiles, box fusion (NMS), one-to-many head | main paper | 70.2199 | 70.2344 | +0.0145 |
| YOLOv10m | trained at 1024 px | 1024 px whole-image pass + native-resolution tiles, box fusion (NMS), one-to-one head | pre-specified row | 71.3492 | 71.3531 | +0.0039 |
| YOLOv10m | trained at 1024 px | 1024 px whole-image pass, SCAHI, refined grid, one-to-many head | decomposition (no verdict) | 68.8235 | 68.8570 | +0.0336 |
| YOLOv10m | trained at 1024 px | 1024 px whole-image pass, SCAHI, setting from VisDrone, one-to-many head | decomposition (no verdict) | 68.2313 | 68.2672 | +0.0359 |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass + native-resolution tiles, SCAHI, refined grid, one-to-many head | main paper | 73.2491 | 73.2721 | +0.0229 |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass + native-resolution tiles, SCAHI, setting from VisDrone, one-to-many head | main paper | 72.9158 | 72.9130 | -0.0028 |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass + native-resolution tiles, box fusion (NMS), one-to-many head | main paper | 70.3180 | 70.3201 | +0.0020 |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass + native-resolution tiles, box fusion (NMS), one-to-one head | pre-specified row | 70.0826 | 70.0784 | -0.0042 |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass, SCAHI, refined grid, one-to-many head | decomposition (no verdict) | 72.8651 | 72.4444 | -0.4207 **>= 0.05** |
| YOLOv10m | trained at 2048 px | 2048 px whole-image pass, SCAHI, setting from VisDrone, one-to-many head | decomposition (no verdict) | 71.6984 | 71.6649 | -0.0335 |
| YOLOv10s | trained at 1024 px | 1024 px whole-image pass + native-resolution tiles, SCAHI, refined grid, one-to-many head | main paper | 70.1081 | 70.0919 | -0.0162 |
| YOLOv10s | trained at 1024 px | 1024 px whole-image pass + native-resolution tiles, SCAHI, setting from VisDrone, one-to-many head | pre-specified row | 69.0287 | 69.0324 | +0.0036 |
| YOLOv10s | trained at 1024 px | 1024 px whole-image pass + native-resolution tiles, box fusion (NMS), one-to-many head | main paper | 68.8893 | 68.8765 | -0.0127 |
| YOLOv10s | trained at 1024 px | 1024 px whole-image pass + native-resolution tiles, box fusion (NMS), one-to-one head | pre-specified row | 68.6497 | 68.6150 | -0.0347 |
| YOLOv10s | trained at 1024 px | 1024 px whole-image pass, SCAHI, refined grid, one-to-many head | decomposition (no verdict) | 65.4820 | 65.4662 | -0.0158 |
| YOLOv10s | trained at 1024 px | 1024 px whole-image pass, SCAHI, setting from VisDrone, one-to-many head | decomposition (no verdict) | 64.4147 | 64.4578 | +0.0431 |
| YOLOv10s | trained at 2048 px | 2048 px whole-image pass + native-resolution tiles, SCAHI, refined grid, one-to-many head | main paper (original check) | 76.0173 | 76.0158 | -0.0015 |
| YOLOv10s | trained at 2048 px | 2048 px whole-image pass + native-resolution tiles, SCAHI, setting from VisDrone, one-to-many head | main paper | 75.7265 | 75.6866 | -0.0399 |
| YOLOv10s | trained at 2048 px | 2048 px whole-image pass + native-resolution tiles, box fusion (NMS), one-to-many head | main paper (original check) | 73.5709 | 73.5721 | +0.0012 |
| YOLOv10s | trained at 2048 px | 2048 px whole-image pass + native-resolution tiles, box fusion (NMS), one-to-one head | pre-specified row (original check) | 73.3884 | 73.3871 | -0.0013 |
| YOLOv10s | trained at 2048 px | 2048 px whole-image pass, SCAHI, refined grid, one-to-many head | decomposition (no verdict) | 74.9014 | 74.8361 | -0.0653 **>= 0.05** |
| YOLOv10s | trained at 2048 px | 2048 px whole-image pass, SCAHI, setting from VisDrone, one-to-many head | decomposition (no verdict) | 73.8706 | 73.8207 | -0.0499 |
| YOLOv11s | trained at 1024 px | 1024 px whole-image pass + native-resolution tiles, SCAHI, refined grid | main paper | 69.0737 | 69.0895 | +0.0157 |
| YOLOv11s | trained at 1024 px | 1024 px whole-image pass + native-resolution tiles, SCAHI, setting from VisDrone | pre-specified row | 68.4049 | 68.4351 | +0.0302 |
| YOLOv11s | trained at 1024 px | 1024 px whole-image pass + native-resolution tiles, box fusion (NMS) | main paper | 65.4878 | 65.5011 | +0.0133 |
| YOLOv11s | trained at 1024 px | 1024 px whole-image pass, SCAHI, refined grid | decomposition (no verdict) | 65.5174 | 65.5075 | -0.0099 |
| YOLOv11s | trained at 1024 px | 1024 px whole-image pass, SCAHI, setting from VisDrone | decomposition (no verdict) | 64.7913 | 64.8157 | +0.0244 |
| YOLOv11s | trained at 2048 px | 2048 px whole-image pass + native-resolution tiles, SCAHI, refined grid | main paper | 74.0154 | 74.0174 | +0.0020 |
| YOLOv11s | trained at 2048 px | 2048 px whole-image pass + native-resolution tiles, SCAHI, setting from VisDrone | main paper | 72.9003 | 72.8990 | -0.0013 |
| YOLOv11s | trained at 2048 px | 2048 px whole-image pass + native-resolution tiles, box fusion (NMS) | main paper | 69.8456 | 69.8420 | -0.0037 |
| YOLOv11s | trained at 2048 px | 2048 px whole-image pass, SCAHI, refined grid | decomposition (no verdict) | 72.6237 | 72.6245 | +0.0008 |
| YOLOv11s | trained at 2048 px | 2048 px whole-image pass, SCAHI, setting from VisDrone | decomposition (no verdict) | 71.7754 | 71.7749 | -0.0005 |
| YOLOv12s | trained at 1024 px | 1024 px whole-image pass + native-resolution tiles, SCAHI, refined grid | main paper | 68.2922 | 68.2703 | -0.0219 |
| YOLOv12s | trained at 1024 px | 1024 px whole-image pass + native-resolution tiles, SCAHI, setting from VisDrone | pre-specified row | 67.2593 | 67.2314 | -0.0279 |
| YOLOv12s | trained at 1024 px | 1024 px whole-image pass + native-resolution tiles, box fusion (NMS) | main paper | 66.2896 | 66.3152 | +0.0256 |
| YOLOv12s | trained at 1024 px | 1024 px whole-image pass, SCAHI, refined grid | decomposition (no verdict) | 65.2900 | 65.2601 | -0.0299 |
| YOLOv12s | trained at 1024 px | 1024 px whole-image pass, SCAHI, setting from VisDrone | decomposition (no verdict) | 64.0859 | 64.1672 | +0.0813 **>= 0.05** |

Metrics: VisDrone AP@[.5:.95] with the toolkit protocol (ignored regions, at most 500 detections per image); DOTA mAP50 with the devkit (VOC07, no cap). Data: reports/backmap_check.json.
