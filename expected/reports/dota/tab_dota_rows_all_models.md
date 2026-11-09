# DOTA-val, all frozen rows of the five detectors

Devkit port; primary mAP50 = difficult GT ignored, no cap; secondaries: cap 4000, all objects (difficult as positives), AP@[.5:.95]. Sizes are training = inference sizes (SAHI: 1024 weights). YOLOv12s: S_max = 1024 (no 2048 rows). YOLOv10: single pass and SAHI on the o2o head, SCAHI and box fusion on the o2m head (unless marked o2o); YOLOv11s / YOLOv12s have one (dense) head, used by every row. FCOS (addendum 3): one head, score sqrt(p · centerness), no SAHI row; values from runs/dota/val/results_fcos.json.

| detector | size | method | mAP50 | cap 4000 | all objects | AP@[.5:.95] | source |
|---|---:|---|---:|---:|---:|---:|---|
| YOLOv10s | 1024 | single pass | 55.80 | 55.76 | 50.59 | 36.14 | original freeze |
| YOLOv10s | 1024 | tiles + box fusion (o2o head) | 63.18 | 63.15 | 60.39 | 41.23 | original freeze |
| YOLOv10s | 1024 | tiles + box fusion | 64.86 | 64.63 | 61.53 | 43.50 | original freeze |
| YOLOv10s | 1024 | tiles + SCAHI | 66.17 | 66.02 | 62.65 | 44.75 | original freeze |
| YOLOv10s | 1024 | tiles + SCAHI, refined grid | 66.17 | 65.90 | 62.71 | 45.63 | original freeze |
| YOLOv10s | 1024 | SCAHI, single pass (no tiles) | 56.29 | 56.30 | 50.94 | 36.44 | original freeze |
| YOLOv10s | 1024 | SCAHI, single pass (no tiles), refined grid | 57.22 | 57.18 | 51.88 | 37.31 | original freeze |
| YOLOv10s | 1024 | SAHI | 59.20 | 59.11 | 56.31 | 37.46 | original freeze |
| YOLOv10s | 2048 | single pass | 65.50 | 65.27 | 60.71 | 44.65 | original freeze |
| YOLOv10s | 2048 | single pass of the 2048 weights, inference size selected on calib (2048 / 2560 / 3072; 2048 chosen) | 65.50 | 65.27 | 60.71 | 44.65 | original freeze |
| YOLOv10s | 2048 | tiles + box fusion (o2o head) | 68.29 | 68.33 | 64.43 | 46.85 | original freeze |
| YOLOv10s | 2048 | tiles + box fusion | 68.46 | 68.49 | 64.56 | 47.41 | original freeze |
| YOLOv10s | 2048 | tiles + SCAHI | 70.64 | 70.65 | 66.69 | 49.15 | original freeze |
| YOLOv10s | 2048 | tiles + SCAHI, refined grid | 71.00 | 71.06 | 67.03 | 50.13 | original freeze |
| YOLOv10s | 2048 | SCAHI, single pass (no tiles) | 66.58 | 66.13 | 61.18 | 44.87 | original freeze |
| YOLOv10s | 2048 | SCAHI, single pass (no tiles), refined grid | 67.27 | 66.96 | 62.04 | 45.94 | original freeze |
| YOLOv10m | 1024 | single pass | 58.44 | 58.42 | 52.82 | 38.37 | original freeze |
| YOLOv10m | 1024 | tiles + box fusion (o2o head) | 63.97 | 63.82 | 61.17 | 43.09 | original freeze |
| YOLOv10m | 1024 | tiles + box fusion | 64.57 | 64.48 | 61.37 | 43.87 | original freeze |
| YOLOv10m | 1024 | tiles + SCAHI | 66.69 | 66.54 | 63.21 | 46.33 | original freeze |
| YOLOv10m | 1024 | tiles + SCAHI, refined grid | 67.34 | 67.08 | 63.88 | 47.02 | original freeze |
| YOLOv10m | 1024 | SCAHI, single pass (no tiles) | 58.61 | 58.61 | 52.98 | 38.74 | original freeze |
| YOLOv10m | 1024 | SCAHI, single pass (no tiles), refined grid | 59.29 | 59.26 | 53.60 | 39.50 | original freeze |
| YOLOv10m | 1024 | SAHI | 60.96 | 60.87 | 58.04 | 39.78 | original freeze |
| YOLOv10m | 2048 | single pass | 67.39 | 66.99 | 63.06 | 46.13 | original freeze |
| YOLOv10m | 2048 | single pass of the 2048 weights, inference size selected on calib (2048 / 2560 / 3072; 2048 chosen) | 67.39 | 66.99 | 63.06 | 46.13 | original freeze |
| YOLOv10m | 2048 | tiles + box fusion (o2o head) | 68.71 | 68.54 | 65.63 | 47.74 | original freeze |
| YOLOv10m | 2048 | tiles + box fusion | 68.13 | 68.08 | 64.61 | 47.85 | original freeze |
| YOLOv10m | 2048 | tiles + SCAHI | 71.32 | 71.08 | 67.64 | 50.57 | original freeze |
| YOLOv10m | 2048 | tiles + SCAHI, refined grid | 71.73 | 71.44 | 68.10 | 51.30 | original freeze |
| YOLOv10m | 2048 | SCAHI, single pass (no tiles) | 67.67 | 67.20 | 62.80 | 46.48 | original freeze |
| YOLOv10m | 2048 | SCAHI, single pass (no tiles), refined grid | 68.42 | 68.02 | 63.52 | 47.30 | original freeze |
| YOLOv11s | 1024 | single pass | 56.16 | 56.16 | 50.91 | 36.30 | addendum 2 |
| YOLOv11s | 1024 | tiles + box fusion | 62.14 | 61.92 | 59.24 | 41.06 | addendum 2 |
| YOLOv11s | 1024 | tiles + SCAHI | 65.20 | 65.07 | 61.56 | 43.85 | addendum 2 |
| YOLOv11s | 1024 | tiles + SCAHI, refined grid | 65.26 | 65.09 | 61.67 | 44.61 | addendum 2 |
| YOLOv11s | 1024 | SCAHI, single pass (no tiles) | 55.68 | 55.64 | 50.54 | 35.76 | addendum 2 |
| YOLOv11s | 1024 | SCAHI, single pass (no tiles), refined grid | 56.47 | 56.46 | 51.34 | 36.45 | addendum 2 |
| YOLOv11s | 2048 | single pass | 65.19 | 64.98 | 60.28 | 44.69 | addendum 2 |
| YOLOv11s | 2048 | single pass of the 2048 weights, inference size selected on calib (2048 / 2560 / 3072; 2048 chosen) | 65.19 | 64.98 | 60.28 | 44.69 | addendum 2 |
| YOLOv11s | 2048 | tiles + box fusion | 66.68 | 66.49 | 62.91 | 45.93 | addendum 2 |
| YOLOv11s | 2048 | tiles + SCAHI | 69.14 | 68.97 | 65.38 | 48.42 | addendum 2 |
| YOLOv11s | 2048 | tiles + SCAHI, refined grid | 69.81 | 69.66 | 66.01 | 49.36 | addendum 2 |
| YOLOv11s | 2048 | SCAHI, single pass (no tiles) | 64.92 | 64.71 | 59.74 | 44.02 | addendum 2 |
| YOLOv11s | 2048 | SCAHI, single pass (no tiles), refined grid | 65.45 | 65.24 | 60.62 | 44.94 | addendum 2 |
| YOLOv12s | 1024 | single pass | 56.36 | 56.36 | 51.27 | 36.62 | addendum 2 |
| YOLOv12s | 1024 | tiles + box fusion | 62.88 | 62.79 | 59.75 | 42.17 | addendum 2 |
| YOLOv12s | 1024 | tiles + SCAHI | 65.18 | 65.07 | 61.80 | 44.16 | addendum 2 |
| YOLOv12s | 1024 | tiles + SCAHI, refined grid | 65.97 | 65.88 | 62.65 | 45.31 | addendum 2 |
| YOLOv12s | 1024 | SCAHI, single pass (no tiles) | 55.78 | 55.69 | 50.50 | 35.92 | addendum 2 |
| YOLOv12s | 1024 | SCAHI, single pass (no tiles), refined grid | 56.74 | 56.69 | 51.70 | 36.95 | addendum 2 |
| FCOS | 1024 | single pass | 42.64 | 42.63 | 38.60 | 23.97 | addendum 3 |
| FCOS | 1024 | tiles + box fusion | 52.56 | 52.56 | 48.70 | 31.45 | addendum 3 |
| FCOS | 1024 | tiles + SCAHI | 52.96 | 52.94 | 48.94 | 31.99 | addendum 3 |
| FCOS | 1024 | tiles + SCAHI, refined grid | 52.96 | 52.94 | 48.94 | 31.99 | addendum 3 |
| FCOS | 1024 | SCAHI, single pass (no tiles) | 41.89 | 41.89 | 37.81 | 22.97 | addendum 3 |
| FCOS | 1024 | SCAHI, single pass (no tiles), refined grid | 42.45 | 42.44 | 38.49 | 23.74 | addendum 3 |
| FCOS | 2048 | single pass | 55.01 | 54.97 | 49.58 | 33.91 | addendum 3 |
| FCOS | 2048 | single pass of the 2048 weights, inference size selected on calib (2048 / 2560 / 3072; 2560 chosen) | 58.12 | 58.12 | 52.67 | 36.00 | addendum 3 |
| FCOS | 2048 | tiles + box fusion | 60.49 | 60.45 | 56.21 | 37.50 | addendum 3 |
| FCOS | 2048 | tiles + SCAHI | 60.59 | 60.55 | 55.79 | 37.55 | addendum 3 |
| FCOS | 2048 | tiles + SCAHI, refined grid | 60.59 | 60.55 | 55.79 | 37.55 | addendum 3 |
| FCOS | 2048 | SCAHI, single pass (no tiles) | 54.04 | 54.04 | 48.50 | 32.21 | addendum 3 |
| FCOS | 2048 | SCAHI, single pass (no tiles), refined grid | 54.04 | 54.04 | 48.50 | 32.21 | addendum 3 |
