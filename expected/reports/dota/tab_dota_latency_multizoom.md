# DOTA-val latency of the multi-zoom arm (exploratory, post hoc)

One exclusive run on host-b, 2026-11-13T09:21:03 – 2026-11-13T09:24:59 (run 1 of 1); every row final: True; nvidia-smi lists: runs/latency_multizoom_host_b/nvidia_smi_run1_{start,end}.txt. ms per image, mean over the images where the native pass fits (subset F of the native-pass analysis), batch 1, RTX 5090 at 575 W, same protocol as the DOTA latency table. Peak memory: torch peak allocated (max over the images of the row; multi-zoom also the median). Parity of the timed multi-zoom path with the accuracy analysis (caches + SCAHI on the CPU), one-to-one matched: yolov10s: 842952 / 842952 detections, unmatched 0, pairs beyond tolerance 0, max |Δbox| 0.00014 px; yolo11s: 881673 / 881673 detections, unmatched 0, pairs beyond tolerance 0, max |Δbox| 0.00024 px.

| detector | method | images | native pass | 2048 pass | passes | fusion | total | peak GPU memory (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv10s | single pass 2048 | 449 | – | – | – | – | 25.4 | 0.7 |
| YOLOv10s | 2048 + tiles, SCAHI refined grid (GPU, optimised path) | 449 | – | – | 114.6 | 12.4 | 127.1 | 1.9 |
| YOLOv10s | native + 2048 pass, SCAHI refined grid (GPU, optimised path) | 449 | 64.0 | 22.6 | 86.7 | 11.4 | 98.1 | 27.6 (median 0.65) |
| YOLOv11s | single pass 2048 | 450 | – | – | – | – | 24.7 | 0.7 |
| YOLOv11s | 2048 + tiles, SCAHI refined grid (GPU, optimised path) | 450 | – | – | 115.0 | 12.7 | 127.7 | 2.5 |
| YOLOv11s | native + 2048 pass, SCAHI refined grid (GPU, optimised path) | 450 | 65.0 | 23.3 | 88.3 | 11.7 | 100.0 | 29.1 (median 0.72) |
