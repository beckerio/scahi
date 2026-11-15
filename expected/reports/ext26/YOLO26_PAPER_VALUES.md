# YOLO26s / YOLO26m: values of the paper (regenerated from the stored results)

AP@[.5:.95], VisDrone toolkit protocol (ignored regions, at most 500 detections per image). Single held-out evaluation of the frozen rows (val once, exact val replay, test-dev once).

## Main paper Tab. 1 (VisDrone val; Delta Q1 on val and test-dev)

| detector | single pass 1280 (o2o) | single pass 2048 (o2o) | SAHI (o2o) | 1280 px + tiles, box fusion (o2o; Q1 row) | 2048 px + tiles, SCAHI pre-specified (o2m) | 2048 px + tiles, SCAHI refined grid, row C (o2m; exploratory) | Delta Q1 val | Delta Q1 test-dev |
|---|---|---|---|---|---|---|---|---|
| YOLO26s | 33.63 | 38.41 | 31.02 | 34.97 | 39.31 | 40.65 | +3.44 | +2.54 |
| YOLO26m | 37.41 | 41.25 | 34.56 | 37.89 | 41.53 | 42.89 | +3.36 | +2.72 |

## Main paper Fig. 4 (2048 px decomposition)

| detector | configuration | AP test-dev | AP val | frozen row |
|---|---|---|---|---|
| YOLO26s | single pass 2048 (o2o) | 33.15 | 38.41 | E_yolo26s_full2048_o2o |
| YOLO26s | SCAHI on the single pass alone, row D (o2m, pre-specified) | 32.83 | 37.64 | D_yolo26s_single2048_F2_o2m |
| YOLO26s | 2048 px + tiles, SCAHI pre-specified (o2m) | 34.18 | 39.31 | E_yolo26s_mz2048_F2_o2m |
| YOLO26s | 2048 px + tiles, refined grid, row C (o2m) | 34.93 | 40.65 | XG_yolo26s_2048_o2m_C |
| YOLO26m | single pass 2048 (o2o) | 35.30 | 41.25 | E_yolo26m_full2048_o2o |
| YOLO26m | SCAHI on the single pass alone, row D (o2m, pre-specified) | 34.73 | 40.16 | D_yolo26m_single2048_F2_o2m |
| YOLO26m | 2048 px + tiles, SCAHI pre-specified (o2m) | 35.82 | 41.53 | E_yolo26m_mz2048_F2_o2m |
| YOLO26m | 2048 px + tiles, refined grid, row C (o2m) | 36.59 | 42.89 | XG_yolo26m_2048_o2m_C |

## Main paper Fig. 3: single-pass GFLOPs per image (val, exact per input shape)

| detector | input size [px] | GFLOPs per image (val) | AP val (o2o) |
|---|---|---|---|
| YOLO26s | 1280 | 48.54 | 33.63 |
| YOLO26s | 2048 | 126.45 | 38.41 |
| YOLO26m | 1280 | 157.42 | 37.41 |
| YOLO26m | 2048 | 399.14 | 41.25 |

## Equal-compute comparison (descriptive): single pass YOLO26s 2048 px vs. YOLO26m 1280 px, both o2o

| split | YOLO26s 2048 AP | YOLO26m 1280 AP | Delta | CI low | CI high | GFLOPs YOLO26s 2048 | GFLOPs YOLO26m 1280 | GFLOPs ratio |
|---|---|---|---|---|---|---|---|---|
| val | 38.41 | 37.41 | +1.00 | +0.60 | +1.37 | 126.45 | 157.42 | 0.80 |
| test-dev | 33.15 | 32.54 | +0.61 | +0.34 | +0.89 | 136.76 | 165.66 | 0.83 |

## Supplement Tab. s-y26 (Delta AP with 95 % CI on val, Delta on test-dev; verdict on val, eps 0.30; Q3 also needs CI low > 0)

| question | YOLO26s val Delta | YOLO26s CI low | YOLO26s CI high | YOLO26s test-dev Delta | YOLO26s verdict (val) | YOLO26m val Delta | YOLO26m CI low | YOLO26m CI high | YOLO26m test-dev Delta | YOLO26m verdict (val) |
|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | +3.44 | 3.03 | 3.85 | +2.54 | met | +3.36 | 2.91 | 3.83 | +2.72 | met |
| Q2, 1280 px | +1.57 | 1.34 | 1.83 | +2.36 | met | +1.94 | 1.65 | 2.21 | +2.24 | met |
| Q2, 2048 px | +1.63 | 1.39 | 1.85 | +2.27 | met | +1.77 | 1.54 | 1.99 | +2.52 | met |
| Q3 | +0.90 | 0.64 | 1.16 | +1.03 | met | +0.28 | 0.04 | 0.51 | +0.51 | not met |

## Supplement small-object table (test-dev, passes at 2048 px; AP < 8 px / 8-16 px, pycocotools size bins)

| detector | single pass | tiles, NMS | tiles, SCAHI | tiles, SCAHI refined grid |
|---|---|---|---|---|
| YOLO26s | 6.17 / 14.10 | 6.29 / 14.55 | 7.10 / 15.10 | 7.54 / 15.45 |
| YOLO26m | 7.76 / 15.93 | 8.26 / 16.18 | 7.50 / 16.78 | 7.76 / 17.21 |

## YOLO26s single-pass gate (not met as specified; choice unchanged, batch-1 values = reference)

| row | calib AP tuning (batch 16) | calib AP replay (batch 1) | difference |
|---|---|---|---|
| E_yolo26s_full1280_o2m | 35.6282 | 35.5465 | -0.0818 |
| E_yolo26s_full1280_o2o | 36.6750 | 36.5813 | -0.0937 |
| E_yolo26s_full2048_o2m | 40.2304 | 40.1502 | -0.0801 |
| E_yolo26s_full2048_o2o | 41.2355 | 41.1568 | -0.0787 |

authors 2026-11-15, option (a): the single-pass gate as specified is NOT met (Delta 0.079 to 0.094 > 0.05, tuning at batch 16 vs. replay at batch 1); the conf grid re-evaluated on the batch-1 detections gives the same choice (0.001) and the same ordering in all four cases (runs/ext/yolo26s/conf_batch1_check.json); the choice stands; the batch-1 values are the reference for the val replay, the tuning values are reported alongside; no rule was changed
