# When do tiles pay off? (exploratory, post hoc)

Per image: ΔAP = tiles + SCAHI (refined grid) 2048 − single pass 2048 (same detector and head; stored detections of the frozen rows). VisDrone test-dev: toolkit-port AP@[.5:.95] of the image; DOTA-val: devkit mAP50 of the image. Covariates over valid GT: share < 16 px (sqrt(w h)); share crowded (same-class neighbour IoU > 0.3). Spearman ρ with 95 % percentile CI over 2000 image resamples (seed 0). Rule: docs/DECISIONS.md 2026-10-31 ~14:50. Figure: reports/figures/ext/fig_ext_tiles_payoff.{pdf,png,csv}.

| dataset | detector | head | images | mean ΔAP | ρ (share < 16 px) | ρ (share crowded) |
|---|---|---|---:|---:|---|---|
| VisDrone test-dev | YOLOv10s | o2o | 1610 | +1.33 | +0.09 [+0.04, +0.14] | +0.04 [-0.00, +0.09] |
| VisDrone test-dev | YOLOv10m | o2m | 1610 | +4.74 | -0.05 [-0.10, +0.00] | -0.05 [-0.10, -0.00] |
| VisDrone test-dev | YOLOv10m | o2o | 1610 | +0.75 | +0.08 [+0.04, +0.13] | +0.08 [+0.03, +0.13] |
| VisDrone test-dev | YOLOv11s | o2m | 1610 | +4.97 | -0.05 [-0.10, -0.00] | -0.06 [-0.11, -0.01] |
| VisDrone test-dev | YOLOv11m | o2m | 1610 | +4.73 | -0.04 [-0.09, +0.01] | -0.03 [-0.08, +0.02] |
| VisDrone test-dev | YOLOv12s | o2m | 1610 | +4.98 | -0.04 [-0.09, +0.01] | -0.01 [-0.06, +0.04] |
| DOTA-val | YOLOv10s | o2o / o2m | 456 | +4.16 | +0.21 [+0.11, +0.30] | +0.03 [-0.07, +0.12] |
| DOTA-val | YOLOv10m | o2o / o2m | 456 | +3.68 | +0.15 [+0.06, +0.24] | +0.01 [-0.09, +0.11] |
