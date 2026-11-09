# Table 2 – VisDrone compute control, YOLOv10s (latency in ms per image, mean; exclusive GPU and CPU, RTX 5090 at 575 W, batch 1)

One run on host-b, 2026-11-09T09:19:48 – 2026-11-09T09:22:17 (table run 1 of 1); every row final: True. These latencies replace all earlier Table 2 latencies (DECISIONS 2026-11-08 21:16 and 2026-11-09). Accuracy: frozen values (val; test-dev where evaluated). GFLOPs per image: unified method (runs/gflops_unified.json). The 2560 weights at 3072 px are exploratory, post hoc (val only); the refined grid is exploratory, post hoc.

| method | passes | merge CPU | merge GPU | total | GFLOPs | val AP | val AP50 | test-dev AP | test-dev AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single pass 2048 | 9.6 | – | – | 9.6 | 128 | 38.47 | 62.87 | 33.48 | – |
| single pass 2560 | 13.6 | – | – | 13.6 | 203 | 39.51 | 64.42 | 33.98 | – |
| 2560 weights at 3072 px | 21.0 | – | – | 21.0 | 298 | 39.64 | 64.81 | – | – |
| 2048 + tiles, box fusion | 46.1 | 4.8 | – | 50.8 | 562 | 38.85 | 64.73 | 33.25 | 56.65 |
| 2048 + tiles, SCAHI | 67.1 | – | 10.2 | 77.3 | 562 | 39.34 | 64.19 | 34.53 | – |
| 2048 + tiles, SCAHI, refined grid | 66.9 | – | 10.3 | 77.3 | 562 | 40.72 | 66.15 | 35.28 | – |

**Footnote (code path per row).** Single passes: Ultralytics predictor, fused model, one-to-one head, end to end (image decoding, pre- and post-processing). Box fusion: the frozen path of row X1, the Ultralytics predictor (fused model, one-to-one head, tiles batched) per pass, then class-aware NMS with intersection over the smaller box on the CPU. SCAHI: the optimised path of the E2 run - the model cannot be fused (fusing removes the one-to-many branch), only the one-to-many head is run, and SCAHI needs its dense per-position outputs (all positions with a class probability >= 0.001, DFL-decoded) instead of the detector's final boxes; these stay on the GPU, where decoding and SCAHI run, and only the final detections go to the host. The dense outputs and the unfused model are a real cost of the method and are included in the passes column. Passes include image decoding. Totals = passes + merge.

Parity of the optimised path with the reference path (cached pass outputs + SCAHI on the CPU) on every val image, one-to-one matching per image and class, runs/parity_opt_matched/visdrone.json: X2: 217492 of 217492 detections matched in 548 images, count mismatches 0, unmatched 0, pairs beyond 0.0001 relative 0, max |Δbox| 1.8e-12 px, max |Δscore| 2.2e-16; XG_yolov10s_2048_o2m_C: 249999 of 249999 detections matched in 548 images, count mismatches 0, unmatched 0, pairs beyond 0.0001 relative 0, max |Δbox| 2e-12 px, max |Δscore| 2.2e-16. X1 detections in this run: 161144 vs. 161199 in the frozen val run (GPU inference is not bit-exact; accuracy = frozen values).

## Reference path (run of 2026-11-09, report only - not for the paper table)

One run on host-b, 2026-11-09T07:57:22 – 2026-11-09T08:00:51; every row final: True. SCAHI rows in the reference path: unfused model, both heads, pass outputs (positions >= 0.001) copied to the CPU, then SCAHI on the CPU or (on the same outputs, incl. host->GPU copy) on the GPU. Other rows as above.

| method | passes | merge CPU | merge GPU | total (CPU merge) | total (GPU merge) |
|---|---:|---:|---:|---:|---:|
| single pass 2048 | 11.1 | – | – | 11.1 | – |
| single pass 2560 | 13.7 | – | – | 13.7 | – |
| 2560 weights at 3072 px | 21.0 | – | – | 21.0 | – |
| 2048 + tiles, box fusion | 46.0 | 4.6 | – | 50.6 | – |
| 2048 + tiles, SCAHI | 75.6 | 30.8 | 9.2 | 106.4 | 84.8 |
| 2048 + tiles, SCAHI, refined grid | 75.9 | 59.4 | 9.7 | 135.3 | 85.5 |
