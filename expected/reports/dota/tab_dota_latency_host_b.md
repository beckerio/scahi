# DOTA-val latency – one run on host-b

One exclusive run on host-b, 2026-11-09T13:15:13 – 2026-11-09T13:44:44 (run 1 of 1); every row final: True; nvidia-smi lists: runs/latency_dota_host_b/nvidia_smi_run1_{start,end}.txt. ms per image, mean over the DOTA-val images, batch 1, RTX 5090 at 575 W. Replaces the host-a tables (columns "host-a" for comparison only). Same protocol and code as the host-a run; the SCAHI rows additionally in the optimised path of Table 2 (unfused model, one-to-many head only, decoding and SCAHI on the GPU); parity with the reference path (stored caches + SCAHI on the CPU) on every DOTA-val image, one-to-one matching per image and class, runs/parity_opt_matched/dota.json: RG_yolov10s_2048: 1196664 of 1196664 detections matched in 458 images, count mismatches 0, unmatched 0, pairs beyond 0.0001 relative 0, max |Δbox| 0.00049 px, max |Δscore| 3.3e-16; RG_yolov10m_2048: 1978304 of 1978304 detections matched in 458 images, count mismatches 0, unmatched 0, pairs beyond 0.0001 relative 0, max |Δbox| 0.00049 px, max |Δscore| 3.3e-16; RG_yolo11s_2048: 1240956 of 1240956 detections matched in 458 images, count mismatches 0, unmatched 0, pairs beyond 0.0001 relative 0, max |Δbox| 0.00049 px, max |Δscore| 3.3e-16; RG_yolo12s_1024: 1544934 of 1544934 detections matched in 458 images, count mismatches 0, unmatched 0, pairs beyond 0.0001 relative 0, max |Δbox| 0.00049 px, max |Δscore| 3.3e-16. Box differences of 2^-11 px are one float32 step at coordinates of 4096-8192 px.

| detector | method | passes | merge | total | host-a passes | host-a merge | host-a total |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv10s | single pass 2048 | 28.0 | – | 28.0 | 28.7 | – | 28.7 |
| YOLOv10s | 2048 + tiles, box fusion (CPU) | 137.9 | 37.7 | 175.5 | 137.6 | 45.9 | 183.5 |
| YOLOv10s | 2048 + tiles, SCAHI refined grid (CPU) | 137.9 | 163.0 | 300.9 | 137.6 | 158.4 | 296.0 |
| YOLOv10s | 2048 + tiles, SCAHI refined grid (GPU, reference passes) | 137.9 | 12.2 | 150.1 | 137.6 | 12.2 | 149.8 |
| YOLOv10s | 2048 + tiles, SCAHI refined grid (GPU, optimised path) | 120.4 | 13.2 | 133.6 | – | – | – |
| YOLOv10s | SAHI (1024 weights, frozen) | 180.3 | 14.3 | 194.6 | 176.3 | 15.3 | 191.6 |
| YOLOv10m | single pass 2048 | 34.7 | – | 34.7 | 34.9 | – | 34.9 |
| YOLOv10m | 2048 + tiles, box fusion (CPU) | 188.7 | 36.6 | 225.2 | 186.8 | 35.8 | 222.6 |
| YOLOv10m | 2048 + tiles, SCAHI refined grid (CPU) | 188.7 | 209.3 | 398.0 | 186.8 | 208.6 | 395.4 |
| YOLOv10m | 2048 + tiles, SCAHI refined grid (GPU, reference passes) | 188.7 | 12.9 | 201.5 | 186.8 | 12.7 | 199.6 |
| YOLOv10m | 2048 + tiles, SCAHI refined grid (GPU, optimised path) | 172.2 | 13.8 | 186.0 | – | – | – |
| YOLOv10m | SAHI (1024 weights, frozen) | 252.3 | 15.8 | 268.1 | 233.3 | 14.7 | 247.9 |
| YOLOv11s | single pass 2048 | 28.5 | – | 28.5 | 28.6 | – | 28.6 |
| YOLOv11s | 2048 + tiles, box fusion (CPU) | 125.9 | 39.3 | 165.3 | 124.6 | 38.1 | 162.7 |
| YOLOv11s | 2048 + tiles, SCAHI refined grid (CPU) | 125.9 | 161.0 | 286.9 | 124.6 | 161.0 | 285.6 |
| YOLOv11s | 2048 + tiles, SCAHI refined grid (GPU, reference passes) | 125.9 | 12.4 | 138.4 | 124.6 | 12.4 | 137.0 |
| YOLOv11s | 2048 + tiles, SCAHI refined grid (GPU, optimised path) | 121.0 | 13.2 | 134.2 | – | – | – |
| YOLOv12s | single pass 1024 | 21.7 | – | 21.7 | 21.0 | – | 21.0 |
| YOLOv12s | 1024 + tiles, box fusion (CPU) | 133.8 | 33.3 | 167.1 | 133.3 | 32.9 | 166.2 |
| YOLOv12s | 1024 + tiles, SCAHI refined grid (CPU) | 133.8 | 200.2 | 334.0 | 133.3 | 199.3 | 332.6 |
| YOLOv12s | 1024 + tiles, SCAHI refined grid (GPU, reference passes) | 133.8 | 12.7 | 146.5 | 133.3 | 12.7 | 146.0 |
| YOLOv12s | 1024 + tiles, SCAHI refined grid (GPU, optimised path) | 129.6 | 13.3 | 142.9 | – | – | – |
