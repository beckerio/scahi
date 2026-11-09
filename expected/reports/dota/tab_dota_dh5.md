# DOTA-val, addendum 4 (D-H5): SCAHI on the passes of the frozen SAHI rows

Devkit port, 458 images; primary mAP50 = difficult GT ignored, no cap; secondaries: cap 4000, all objects (difficult as positives), AP@[.5:.95]. 1024 weights of the frozen SAHI rows; SAHI's passes (640 px slices, frozen overlap 0.1 for YOLOv10s / 0.2 for YOLOv10m, plus SAHI's standard pass at long side 640); SAHI on the o2o head, SCAHI on the o2m head of the same forward. SAHI rows: stored evaluation of the original freeze (not re-evaluated). Pipeline limit 200000. Values from runs/dota/val/results_dh5.json.

| detector | method | mAP50 | cap 4000 | all objects | AP@[.5:.95] | mAP50 vehicles | detections | images > cap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv10s | SAHI | 59.20 | 59.11 | 56.31 | 37.46 | 74.83 | 458,192 | 28 |
| YOLOv10s | SAHI passes + SCAHI | 61.96 | 61.71 | 58.82 | 40.62 | 75.19 | 644,295 | 38 |
| YOLOv10s | SAHI passes + SCAHI, refined grid | 62.08 | 61.91 | 58.98 | 41.60 | 78.59 | 1,891,737 | 113 |
| YOLOv10m | SAHI | 60.96 | 60.87 | 58.04 | 39.78 | 76.35 | 405,936 | 22 |
| YOLOv10m | SAHI passes + SCAHI | 62.98 | 62.77 | 59.85 | 42.56 | 75.73 | 1,274,928 | 82 |
| YOLOv10m | SAHI passes + SCAHI, refined grid | 63.22 | 63.00 | 60.02 | 43.51 | 79.54 | 1,765,089 | 118 |

## Contrasts (Δ [95 % CI], one-sided p)

Paired bootstrap over images (1000 resamples, seed 0, percentile 2.5/97.5); SAHI uses its stored bootstrap weights. D-H5 criterion: Δ mAP50 > 0.30, Holm over the two models, holds if met for both → **holds**. SAHI-F2 − SAHI: secondary, no verdict.

| contrast | detector | metric | A | B | Δ [95 % CI] | p (one-sided) | Holm p | met |
|---|---|---|---:|---:|---|---:|---:|---|
| D-H5: SAHI-RG − SAHI | YOLOv10s | mAP50 | 62.08 | 59.20 | +2.87 [+2.13, +4.33] | 0.001 | 0.002 | yes |
| D-H5: SAHI-RG − SAHI | YOLOv10s | cap 4000 | 61.91 | 59.11 | +2.80 [+2.07, +4.27] | 0.001 |  |  |
| D-H5: SAHI-RG − SAHI | YOLOv10s | all objects | 58.98 | 56.31 | +2.67 [+1.97, +3.98] | 0.001 |  |  |
| D-H5: SAHI-RG − SAHI | YOLOv10s | AP@[.5:.95] | 41.60 | 37.46 | +4.15 [+3.24, +5.18] | 0.001 |  |  |
| D-H5: SAHI-RG − SAHI | YOLOv10s | mAP50 vehicles | 78.59 | 74.83 | +3.76 [+0.97, +7.39] | 0.003 |  |  |
| D-H5: SAHI-RG − SAHI | YOLOv10s | R50 crowd | 90.89 | 90.58 | +0.32 [-1.33, +2.56] | 0.477 |  |  |
| D-H5: SAHI-RG − SAHI | YOLOv10s | R50 isolated | 96.96 | 93.67 | +3.30 [+2.80, +3.91] | 0.001 |  |  |
| D-H5: SAHI-RG − SAHI | YOLOv10m | mAP50 | 63.22 | 60.96 | +2.26 [+1.08, +3.25] | 0.001 | 0.002 | yes |
| D-H5: SAHI-RG − SAHI | YOLOv10m | cap 4000 | 63.00 | 60.87 | +2.13 [+0.86, +3.16] | 0.002 |  |  |
| D-H5: SAHI-RG − SAHI | YOLOv10m | all objects | 60.02 | 58.04 | +1.99 [+1.11, +2.83] | 0.002 |  |  |
| D-H5: SAHI-RG − SAHI | YOLOv10m | AP@[.5:.95] | 43.51 | 39.78 | +3.73 [+2.91, +4.57] | 0.001 |  |  |
| D-H5: SAHI-RG − SAHI | YOLOv10m | mAP50 vehicles | 79.54 | 76.35 | +3.19 [+0.31, +7.49] | 0.026 |  |  |
| D-H5: SAHI-RG − SAHI | YOLOv10m | R50 crowd | 97.57 | 91.29 | +6.28 [+4.01, +9.17] | 0.001 |  |  |
| D-H5: SAHI-RG − SAHI | YOLOv10m | R50 isolated | 97.52 | 94.22 | +3.30 [+2.82, +3.96] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv10s | mAP50 | 61.96 | 59.20 | +2.76 [+1.65, +4.12] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv10s | cap 4000 | 61.71 | 59.11 | +2.60 [+1.62, +4.15] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv10s | all objects | 58.82 | 56.31 | +2.51 [+1.50, +3.76] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv10s | AP@[.5:.95] | 40.62 | 37.46 | +3.16 [+2.25, +4.21] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv10s | mAP50 vehicles | 75.19 | 74.83 | +0.35 [-2.91, +4.46] | 0.565 |  |  |
| SAHI-F2 − SAHI | YOLOv10s | R50 crowd | 67.20 | 90.58 | -23.38 [-27.36, -17.99] | 1.000 |  |  |
| SAHI-F2 − SAHI | YOLOv10s | R50 isolated | 94.91 | 93.67 | +1.24 [+0.67, +1.84] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv10m | mAP50 | 62.98 | 60.96 | +2.02 [+0.83, +3.06] | 0.003 |  |  |
| SAHI-F2 − SAHI | YOLOv10m | cap 4000 | 62.77 | 60.87 | +1.90 [+0.62, +2.94] | 0.007 |  |  |
| SAHI-F2 − SAHI | YOLOv10m | all objects | 59.85 | 58.04 | +1.81 [+0.80, +2.70] | 0.002 |  |  |
| SAHI-F2 − SAHI | YOLOv10m | AP@[.5:.95] | 42.56 | 39.78 | +2.78 [+1.92, +3.67] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv10m | mAP50 vehicles | 75.73 | 76.35 | -0.62 [-4.03, +4.30] | 0.679 |  |  |
| SAHI-F2 − SAHI | YOLOv10m | R50 crowd | 69.58 | 91.29 | -21.71 [-25.93, -15.95] | 1.000 |  |  |
| SAHI-F2 − SAHI | YOLOv10m | R50 isolated | 96.13 | 94.22 | +1.91 [+1.37, +2.57] | 0.001 |  |  |

## Size bins (GT box size in px, primary protocol)

Bin < 8 px: not defined in the primary protocol (stored as NaN, shown as –; same for the frozen SAHI rows); all-objects values for every bin in tab_dota_dh5_sizebins.csv.

### AP50

| detector | method | <8 | 8-16 | 16-32 | 32-64 | 64-128 | >128 |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv10s | SAHI | – | 33.10 | 51.50 | 54.59 | 55.72 | 71.64 |
| YOLOv10s | SAHI passes + SCAHI | – | 34.66 | 53.63 | 56.30 | 58.88 | 74.46 |
| YOLOv10s | SAHI passes + SCAHI, refined grid | – | 33.70 | 53.17 | 57.08 | 59.36 | 76.06 |
| YOLOv10m | SAHI | – | 31.53 | 56.01 | 53.77 | 57.50 | 72.81 |
| YOLOv10m | SAHI passes + SCAHI | – | 40.19 | 58.51 | 56.21 | 61.16 | 73.79 |
| YOLOv10m | SAHI passes + SCAHI, refined grid | – | 39.94 | 58.32 | 56.09 | 61.81 | 77.38 |

### AP@[.5:.95]

| detector | method | <8 | 8-16 | 16-32 | 32-64 | 64-128 | >128 |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv10s | SAHI | – | 15.28 | 28.62 | 35.32 | 37.95 | 48.43 |
| YOLOv10s | SAHI passes + SCAHI | – | 18.18 | 30.76 | 37.30 | 39.98 | 54.44 |
| YOLOv10s | SAHI passes + SCAHI, refined grid | – | 16.85 | 30.83 | 38.24 | 41.28 | 56.33 |
| YOLOv10m | SAHI | – | 15.42 | 31.98 | 35.65 | 39.56 | 52.78 |
| YOLOv10m | SAHI passes + SCAHI | – | 20.39 | 34.24 | 38.06 | 42.69 | 55.63 |
| YOLOv10m | SAHI passes + SCAHI, refined grid | – | 20.33 | 34.29 | 38.66 | 44.03 | 58.96 |

Latency of these rows: not measured yet (exclusive run on host-a pending).
