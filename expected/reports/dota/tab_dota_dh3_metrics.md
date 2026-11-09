# DOTA-val, Q3 (D-H3): tiles + SCAHI, refined grid − single pass, both at S_max

Contrast RG_{S_max} − A_{S_max} per detector; S_max = largest training size that passes the memory smoke (2048; **YOLOv12s: 1024**, its 2048 training does not fit, so its D-H3 value is measured on the 1024 pass, where the single pass is weaker – not directly comparable with the 2048 values). YOLOv11s / YOLOv12s: addendum 2, values from runs/dota/val/results_new.json; single-head models use their only head throughout. Paired bootstrap over 458 DOTA-val images (1000 resamples, seed 0, percentile 2.5/97.5); YOLOv10s / YOLOv10m from runs/dota/val/results.json. FCOS (addendum 3, one head, S_max = 2048): runs/dota/val/results_fcos.json. Only the primary metric carries the preregistered verdict; the others are secondary, without verdict.

| detector | size (S_max) | metric | single pass | tiles + SCAHI, refined grid | Δ | 95 % CI |
|---|---:|---|---:|---:|---:|---|
| YOLOv10s | 2048 | mAP50, primary (difficult ignored, no cap) | 65.50 | 71.00 | +5.50 | [+3.55, +7.77] |
| YOLOv10s | 2048 | mAP50, all objects | 60.71 | 67.03 | +6.32 | [+4.54, +8.64] |
| YOLOv10s | 2048 | mAP50, cap 4000 | 65.27 | 71.06 | +5.79 | [+3.64, +8.18] |
| YOLOv10s | 2048 | AP@[.5:.95] | 44.65 | 50.13 | +5.48 | [+4.17, +7.03] |
| YOLOv10m | 2048 | mAP50, primary (difficult ignored, no cap) | 67.39 | 71.73 | +4.34 | [+3.02, +6.19] |
| YOLOv10m | 2048 | mAP50, all objects | 63.06 | 68.10 | +5.04 | [+3.63, +6.87] |
| YOLOv10m | 2048 | mAP50, cap 4000 | 66.99 | 71.44 | +4.45 | [+3.07, +6.24] |
| YOLOv10m | 2048 | AP@[.5:.95] | 46.13 | 51.30 | +5.17 | [+3.84, +6.50] |
| YOLOv11s | 2048 | mAP50, primary (difficult ignored, no cap) | 65.19 | 69.81 | +4.62 | [+2.76, +6.62] |
| YOLOv11s | 2048 | mAP50, all objects | 60.28 | 66.01 | +5.73 | [+3.97, +7.92] |
| YOLOv11s | 2048 | mAP50, cap 4000 | 64.98 | 69.66 | +4.68 | [+2.76, +6.66] |
| YOLOv11s | 2048 | AP@[.5:.95] | 44.69 | 49.36 | +4.67 | [+3.43, +6.20] |
| YOLOv12s | 1024 (S_max = 1024) | mAP50, primary (difficult ignored, no cap) | 56.36 | 65.97 | +9.62 | [+7.10, +12.22] |
| YOLOv12s | 1024 (S_max = 1024) | mAP50, all objects | 51.27 | 62.65 | +11.39 | [+9.09, +13.96] |
| YOLOv12s | 1024 (S_max = 1024) | mAP50, cap 4000 | 56.36 | 65.88 | +9.52 | [+7.03, +11.89] |
| YOLOv12s | 1024 (S_max = 1024) | AP@[.5:.95] | 36.62 | 45.31 | +8.69 | [+6.74, +10.60] |
| FCOS | 2048 | mAP50, primary (difficult ignored, no cap) | 55.01 | 60.59 | +5.57 | [+3.77, +7.65] |
| FCOS | 2048 | mAP50, all objects | 49.58 | 55.79 | +6.21 | [+4.25, +8.03] |
| FCOS | 2048 | mAP50, cap 4000 | 54.97 | 60.55 | +5.58 | [+3.71, +7.62] |
| FCOS | 2048 | AP@[.5:.95] | 33.91 | 37.55 | +3.64 | [+2.38, +4.94] |
