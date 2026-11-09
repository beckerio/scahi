# DOTA-val, addendum 5 (D-H6): SAHI vs. SCAHI on SAHI's passes, single-head detectors

Devkit port, 458 images; primary mAP50 = difficult GT ignored, no cap; secondaries: cap 4000, all objects (difficult as positives), AP@[.5:.95]. 1024 weights (YOLOv11s, YOLOv12s; one dense head). SAHI tuned on calib with the procedure and grid of the frozen YOLOv10 SAHI rows (selected for both: overlap 0.2, NMS / IOU 0.4, conf 0.01; pass postprocess Ultralytics NMS, IoU 0.7); SCAHI rows on exactly the passes of the selected SAHI configuration (640 px slices + SAHI's standard pass at long side 640), same head. All six rows frozen and approved 2026-11-04T14:01, evaluated once. Pipeline limit of the SCAHI rows 200000. Values from runs/dota/val/results_a5.json.

| detector | method | mAP50 | cap 4000 | all objects | AP@[.5:.95] | mAP50 vehicles | detections | images > cap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv11s | SAHI | 57.90 | 57.74 | 54.75 | 37.04 | 74.66 | 432,192 | 22 |
| YOLOv11s | SAHI passes + SCAHI | 61.36 | 61.34 | 58.05 | 40.50 | 74.66 | 1,293,225 | 79 |
| YOLOv11s | SAHI passes + SCAHI, refined grid | 61.87 | 61.80 | 58.67 | 41.29 | 77.96 | 1,753,827 | 104 |
| YOLOv12s | SAHI | 60.21 | 60.20 | 57.38 | 39.02 | 74.90 | 375,164 | 21 |
| YOLOv12s | SAHI passes + SCAHI | 63.13 | 63.09 | 60.07 | 41.56 | 75.67 | 1,096,062 | 64 |
| YOLOv12s | SAHI passes + SCAHI, refined grid | 63.30 | 63.22 | 60.37 | 42.43 | 79.18 | 1,455,528 | 88 |

Images at the pipeline limit 200,000: SAHI_yolo11s 0, SAHI-F2_yolo11s 0, SAHI-RG_yolo11s 0, SAHI_yolo12s 0, SAHI-F2_yolo12s 0, SAHI-RG_yolo12s 0. SAHI rows have no per-image limit; images with >= 20,000 detections (informational; limit 200,000 not reached, no truncation): SAHI_yolo11s 2 (max 28,184 per image), SAHI_yolo12s 1 (max 21,917 per image).

## Contrasts (Δ [95 % CI], one-sided p)

Paired bootstrap over images (1000 resamples, seed 0, percentile 2.5/97.5). D-H6 criterion: Δ mAP50 > 0.30, Holm over YOLOv11s and YOLOv12s, holds if met for both → **holds**. SAHI-F2 − SAHI: secondary, no verdict. The YOLOv10 D-H5 verdict (addendum 4) is unchanged.

| contrast | detector | metric | A | B | Δ [95 % CI] | p (one-sided) | Holm p | met |
|---|---|---|---:|---:|---|---:|---:|---|
| D-H6: SAHI-RG − SAHI | YOLOv11s | mAP50 | 61.87 | 57.90 | +3.97 [+2.93, +5.06] | 0.001 | 0.002 | yes |
| D-H6: SAHI-RG − SAHI | YOLOv11s | cap 4000 | 61.80 | 57.74 | +4.05 [+2.91, +5.06] | 0.001 |  |  |
| D-H6: SAHI-RG − SAHI | YOLOv11s | all objects | 58.67 | 54.75 | +3.92 [+2.84, +4.79] | 0.001 |  |  |
| D-H6: SAHI-RG − SAHI | YOLOv11s | AP@[.5:.95] | 41.29 | 37.04 | +4.25 [+3.50, +4.99] | 0.001 |  |  |
| D-H6: SAHI-RG − SAHI | YOLOv11s | mAP50 vehicles | 77.96 | 74.66 | +3.29 [+0.76, +7.33] | 0.001 |  |  |
| D-H6: SAHI-RG − SAHI | YOLOv11s | R50 crowd | 88.79 | 84.49 | +4.29 [+0.56, +9.89] | 0.014 |  |  |
| D-H6: SAHI-RG − SAHI | YOLOv11s | R50 isolated | 96.78 | 94.45 | +2.33 [+1.98, +2.77] | 0.001 |  |  |
| D-H6: SAHI-RG − SAHI | YOLOv12s | mAP50 | 63.30 | 60.21 | +3.10 [+2.23, +4.08] | 0.001 | 0.002 | yes |
| D-H6: SAHI-RG − SAHI | YOLOv12s | cap 4000 | 63.22 | 60.20 | +3.02 [+2.17, +4.03] | 0.001 |  |  |
| D-H6: SAHI-RG − SAHI | YOLOv12s | all objects | 60.37 | 57.38 | +2.99 [+2.21, +3.95] | 0.001 |  |  |
| D-H6: SAHI-RG − SAHI | YOLOv12s | AP@[.5:.95] | 42.43 | 39.02 | +3.40 [+2.74, +4.13] | 0.001 |  |  |
| D-H6: SAHI-RG − SAHI | YOLOv12s | mAP50 vehicles | 79.18 | 74.90 | +4.28 [+1.48, +8.16] | 0.001 |  |  |
| D-H6: SAHI-RG − SAHI | YOLOv12s | R50 crowd | 89.98 | 85.09 | +4.89 [+2.82, +7.63] | 0.001 |  |  |
| D-H6: SAHI-RG − SAHI | YOLOv12s | R50 isolated | 97.18 | 95.06 | +2.12 [+1.73, +2.60] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv11s | mAP50 | 61.36 | 57.90 | +3.46 [+2.38, +4.56] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv11s | cap 4000 | 61.34 | 57.74 | +3.60 [+2.38, +4.53] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv11s | all objects | 58.05 | 54.75 | +3.30 [+2.30, +4.26] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv11s | AP@[.5:.95] | 40.50 | 37.04 | +3.46 [+2.70, +4.26] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv11s | mAP50 vehicles | 74.66 | 74.66 | -0.01 [-2.79, +4.38] | 0.572 |  |  |
| SAHI-F2 − SAHI | YOLOv11s | R50 crowd | 67.08 | 84.49 | -17.42 [-22.45, -10.58] | 1.000 |  |  |
| SAHI-F2 − SAHI | YOLOv11s | R50 isolated | 95.57 | 94.45 | +1.13 [+0.65, +1.62] | 0.002 |  |  |
| SAHI-F2 − SAHI | YOLOv12s | mAP50 | 63.13 | 60.21 | +2.92 [+1.92, +3.90] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv12s | cap 4000 | 63.09 | 60.20 | +2.89 [+1.84, +3.86] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv12s | all objects | 60.07 | 57.38 | +2.70 [+1.91, +3.69] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv12s | AP@[.5:.95] | 41.56 | 39.02 | +2.53 [+1.85, +3.26] | 0.001 |  |  |
| SAHI-F2 − SAHI | YOLOv12s | mAP50 vehicles | 75.67 | 74.90 | +0.77 [-2.33, +5.13] | 0.465 |  |  |
| SAHI-F2 − SAHI | YOLOv12s | R50 crowd | 65.81 | 85.09 | -19.28 [-23.83, -14.25] | 1.000 |  |  |
| SAHI-F2 − SAHI | YOLOv12s | R50 isolated | 95.63 | 95.06 | +0.57 [+0.10, +1.06] | 0.137 |  |  |

## Size bins (GT box size in px, primary protocol)

Bin < 8 px: not defined in the primary protocol (stored as NaN, shown as –); all-objects values for every bin in tab_dota_a5_sizebins.csv.

### AP50

| detector | method | <8 | 8-16 | 16-32 | 32-64 | 64-128 | >128 |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv11s | SAHI | – | 36.85 | 52.96 | 53.55 | 54.57 | 71.09 |
| YOLOv11s | SAHI passes + SCAHI | – | 36.71 | 58.25 | 56.75 | 59.74 | 74.14 |
| YOLOv11s | SAHI passes + SCAHI, refined grid | – | 37.29 | 57.71 | 57.09 | 60.62 | 75.62 |
| YOLOv12s | SAHI | – | 28.82 | 52.90 | 55.79 | 56.13 | 73.38 |
| YOLOv12s | SAHI passes + SCAHI | – | 33.71 | 57.04 | 57.46 | 60.95 | 76.12 |
| YOLOv12s | SAHI passes + SCAHI, refined grid | – | 34.70 | 57.08 | 58.03 | 61.57 | 76.11 |

### AP@[.5:.95]

| detector | method | <8 | 8-16 | 16-32 | 32-64 | 64-128 | >128 |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv11s | SAHI | – | 20.08 | 30.25 | 33.86 | 37.10 | 49.96 |
| YOLOv11s | SAHI passes + SCAHI | – | 20.49 | 34.46 | 37.15 | 41.18 | 55.13 |
| YOLOv11s | SAHI passes + SCAHI, refined grid | – | 20.71 | 34.20 | 37.94 | 42.17 | 56.22 |
| YOLOv12s | SAHI | – | 12.43 | 30.18 | 35.91 | 38.33 | 52.65 |
| YOLOv12s | SAHI passes + SCAHI | – | 17.24 | 33.70 | 37.92 | 41.32 | 56.02 |
| YOLOv12s | SAHI passes + SCAHI, refined grid | – | 17.57 | 33.76 | 38.62 | 42.51 | 57.41 |

## Sensitivity of the 2x time rule (calib, report only; selection rule unchanged)

Refined grid: the selected setting and every variant with higher calib mAP50 whose fusion time ratio vs. the reference lies in [1.8, 2.2] (timing noise at the cap about ±10 %).

| detector | setting | τ_splat | τ_peak | cell | bins | calib mAP50 | time ratio | admissible |
|---|---|---:|---:|---:|---:|---:|---:|---|
| YOLOv11s | selected | 0.005 | 0.0001 | 7 | 6 | 62.361 | 1.221 | yes |
| YOLOv11s | higher mAP50, ratio in [1.8, 2.2] | 0.00125 | 0.0001 | 7 | 6 | 62.396 | 2.156 | no |
| YOLOv12s | selected | 0.0003125 | 0.001 | 7 | 6 | 62.831 | 1.744 | yes |
| YOLOv12s | higher mAP50, ratio in [1.8, 2.2] | 0.00125 | 0.0001 | 7 | 6 | 62.841 | 2.172 | no |

## SAHI's own raw predictions vs. the SCAHI cache path (descriptive, no metric)

Per image: pass detections as the cache path forms them vs. SAHI's raw predictions of the selected configuration (gate rule of addendum 5; the DOTA-val part was computed after the freeze approval).

| detector | split | images | dets cache | dets SAHI | images same count | box px p99 (median) | box px max |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv11s | calib | 141 | 704,663 | 704,653 | 95 | 0.0002 | 51.91 |
| YOLOv11s | DOTA-val | 458 | 2,498,302 | 2,498,323 | 323 | 0.0002 | 50.95 |
| YOLOv12s | calib | 141 | 623,566 | 623,569 | 103 | 0.0002 | 65.49 |
| YOLOv12s | DOTA-val | 458 | 2,202,498 | 2,202,499 | 327 | 0.0002 | 53.59 |

Latency of these rows: not measured (no exclusive window before the move to host-b; addendum 5, item 8).
