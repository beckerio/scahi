# FCOS diagnostics and DOTA-val limit check (exploratory, post hoc)

exploratory, post hoc (FCOS diagnostics, user 2026-11-08); no verdict changes; reported only on the user's decision. DOTA-val, 458 images. Paired bootstrap (1000 resamples, seed 0), 95 % CIs. Source: runs/dota/val/diag_fcos/{cap,recall,redundancy}.json, runs/dota/val/limit_check_fcos.json, runs/dota/val/results_fcos.json.

## 1. Equal cap: both fusions capped at the top 200,000 detections per image (RG − box fusion, mAP50)

| contrast | pre-specified (cap 4000) | as evaluated | both capped | without limit (point value) | verdict |
|---|---|---|---|---|---|
| D-H2a: RG_fcos_1024 − NMS-o2m_fcos_1024 | +0.38 [-0.94, +1.77] | +0.40 [-0.92, +1.79] | +0.40 [-0.92, +1.79] | -1.76 | not met |
| D-H2b: RG_fcos_2048 − NMS-o2m_fcos_2048 | +0.10 [-0.96, +1.33] | +0.09 [-0.91, +1.37] | +0.09 [-0.91, +1.36] | +0.14 | not met |

Both capped, other metrics: D-H2a AP +0.54 [-0.25, +1.36], R50 crowded -30.06 [-35.00, -24.10], R50 isolated -0.63 [-1.36, +0.06]; D-H2b AP +0.05 [-0.80, +0.78], R50 crowded -30.82 [-33.48, -26.99], R50 isolated -1.27 [-2.06, -0.53].

Detections per image:

| row | max | mean | median | images over the cap |
|---|---:|---:|---:|---:|
| A_fcos_1024 | 200,000 | 135,572 | 130,271 | 0 |
| NMS-o2m_fcos_1024 | 3,495,205 | 403,143 | 107,820 | 182 |
| F2_fcos_1024 | 200,000 | 57,692 | 25,088 | 0 |
| RG_fcos_1024 | 200,000 | 57,692 | 25,088 | 0 |
| B-F2_fcos_1024 | 41,826 | 15,820 | 12,662 | 0 |
| B-RG_fcos_1024 | 46,626 | 22,845 | 20,162 | 0 |
| A_fcos_2048 | 200,000 | 160,314 | 153,806 | 0 |
| X_fcos_2048 | 200,000 | 166,920 | 161,754 | 0 |
| NMS-o2m_fcos_2048 | 3,407,292 | 240,731 | 88,504 | 159 |
| F2_fcos_2048 | 200,000 | 34,812 | 19,192 | 0 |
| RG_fcos_2048 | 200,000 | 34,812 | 19,192 | 0 |
| B-F2_fcos_2048 | 115,809 | 27,919 | 20,920 | 0 |
| B-RG_fcos_2048 | 115,809 | 27,919 | 20,920 | 0 |

## 2. Prediction redundancy (predictions with IoU >= 0.5 per non-difficult GT object, 2048; class-agnostic)

Candidates: anchors with max class probability >= tau_splat of the row; all passes ('total'), the whole-image pass, and the mean over the tile passes whose view contains the object centre.

| model | tau_splat (RG) | n GT | all passes: mean (median) all / crowded / isolated | whole image | per tile pass | zero |
|---|---|---:|---|---:|---:|---:|
| fcos | 0.02 | 34,004 | 62.7 (44) / 95.4 (85) / 60.1 (41) | 18.4 | 20.2 | 1.1 % |
| yolov10s | 0.0003125 | 34,004 | 35.5 (33) / 47.6 (42) / 34.5 (32) | 9.8 | 11.7 | 1.0 % |
| yolo11s | 0.00125 | 34,004 | 35.8 (33) / 48.1 (43) / 34.8 (32) | 9.9 | 11.8 | 1.1 % |

## 3. Crowded / isolated recall (R50), RG − box fusion (NMS-o2m)

| contrast | R50 crowded | R50 isolated |
|---|---|---|
| RG_fcos_1024 − NMS-o2m_fcos_1024 | 61.1 vs. 91.5: -30.34 [-35.18, -24.39] | 92.5 vs. 93.4: -0.87 [-1.59, -0.21] |
| RG_fcos_2048 − NMS-o2m_fcos_2048 | 62.6 vs. 93.4: -30.85 [-33.52, -27.06] | 93.9 vs. 95.3: -1.40 [-2.15, -0.70] |

Crowded / isolated GT: 2,515 / 31,489.

## 4. DOTA-val limit check (pipeline limit 200,000; mAP50 without cap)

Affected images recomputed without the limit (10,000,000); box-fusion rows: passes at the per-pass limit (their merged output is not capped). Dense outputs of the re-run passes identical to the stored caches: True (216 passes). Point values, no CIs; verdicts stay as computed under the pre-registered limit.

| row | affected images | mAP50 with limit | without limit | Δ |
|---|---:|---:|---:|---:|
| A_fcos_1024 | 24 | 42.64 | 42.64 | +0.00 |
| NMS-o2m_fcos_1024 | 30 | 52.56 | 52.56 | +0.00 |
| F2_fcos_1024 | 8 | 52.96 | 50.81 | -2.16 |
| RG_fcos_1024 | 8 | 52.96 | 50.81 | -2.16 |
| B-F2_fcos_1024 | 0 | 41.89 | 41.89 | +0.00 |
| B-RG_fcos_1024 | 0 | 42.45 | 42.45 | +0.00 |
| A_fcos_2048 | 76 | 55.01 | 55.01 | -0.00 |
| X_fcos_2048 | 82 | 58.12 | 58.12 | -0.00 |
| NMS-o2m_fcos_2048 | 78 | 60.49 | 58.96 | -1.53 |
| F2_fcos_2048 | 1 | 60.59 | 59.10 | -1.48 |
| RG_fcos_2048 | 1 | 60.59 | 59.10 | -1.48 |
| B-F2_fcos_2048 | 0 | 54.04 | 54.04 | +0.00 |
| B-RG_fcos_2048 | 0 | 54.04 | 54.04 | +0.00 |

| hypothesis | Δ mAP50 with limit [95 % CI] | verdict | Δ without limit |
|---|---|---|---:|
| D-H1: A_fcos_2048 − NMS-o2m_fcos_1024 | +2.45 [-0.49, +5.28] | not met | +2.45 |
| D-H2a: RG_fcos_1024 − NMS-o2m_fcos_1024 | +0.40 [-0.92, +1.79] | not met | -1.76 |
| D-H2b: RG_fcos_2048 − NMS-o2m_fcos_2048 | +0.09 [-0.91, +1.37] | not met | +0.14 |
| D-H3: RG_fcos_2048 − A_fcos_2048 | +5.57 [+3.77, +7.65] | met | +4.09 |
