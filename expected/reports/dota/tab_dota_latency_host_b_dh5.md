# D-H5 latency (SAHI vs. SAHI passes + SCAHI) – same run

One exclusive run on host-b, 2026-11-09T13:15:13 – 2026-11-09T13:44:44 (run 1 of 1); every row final: True; nvidia-smi lists: runs/latency_dota_host_b/nvidia_smi_run1_{start,end}.txt. ms per image, mean over the DOTA-val images, batch 1, RTX 5090 at 575 W. Replaces the host-a tables (columns "host-a" for comparison only). Same protocol and code as the host-a run; the SCAHI rows additionally in the optimised path of Table 2 (unfused model, one-to-many head only, decoding and SCAHI on the GPU); parity with the reference path (stored caches + SCAHI on the CPU) on every DOTA-val image, one-to-one matching per image and class, runs/parity_opt_matched/dota.json: RG_yolov10s_2048: 1196664 of 1196664 detections matched in 458 images, count mismatches 0, unmatched 0, pairs beyond 0.0001 relative 0, max |Δbox| 0.00049 px, max |Δscore| 3.3e-16; RG_yolov10m_2048: 1978304 of 1978304 detections matched in 458 images, count mismatches 0, unmatched 0, pairs beyond 0.0001 relative 0, max |Δbox| 0.00049 px, max |Δscore| 3.3e-16; RG_yolo11s_2048: 1240956 of 1240956 detections matched in 458 images, count mismatches 0, unmatched 0, pairs beyond 0.0001 relative 0, max |Δbox| 0.00049 px, max |Δscore| 3.3e-16; RG_yolo12s_1024: 1544934 of 1544934 detections matched in 458 images, count mismatches 0, unmatched 0, pairs beyond 0.0001 relative 0, max |Δbox| 0.00049 px, max |Δscore| 3.3e-16. Box differences of 2^-11 px are one float32 step at coordinates of 4096-8192 px.

| detector | method | passes | merge | total | host-a passes | host-a merge | host-a total |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv10s | SAHI (frozen) | 180.3 | 14.3 | 194.6 | 180.1 | 17.0 | 197.1 |
| YOLOv10s | SAHI passes + SCAHI, SAHI-F2 (fusion on CPU) | 135.8 | 100.3 | 236.1 | 145.1 | 100.5 | 245.6 |
| YOLOv10s | SAHI passes + SCAHI, SAHI-F2 (fusion on GPU) | 135.8 | 11.6 | 147.4 | 145.1 | 12.6 | 157.8 |
| YOLOv10s | SAHI passes + SCAHI, SAHI-RG (fusion on CPU) | 135.8 | 190.0 | 325.7 | 145.1 | 189.8 | 334.9 |
| YOLOv10s | SAHI passes + SCAHI, SAHI-RG (fusion on GPU) | 135.8 | 11.6 | 147.4 | 145.1 | 12.7 | 157.8 |
| YOLOv10m | SAHI (frozen) | 252.3 | 15.8 | 268.1 | 232.7 | 16.2 | 248.8 |
| YOLOv10m | SAHI passes + SCAHI, SAHI-F2 (fusion on CPU) | 206.0 | 153.7 | 359.7 | 199.5 | 155.7 | 355.2 |
| YOLOv10m | SAHI passes + SCAHI, SAHI-F2 (fusion on GPU) | 206.0 | 12.4 | 218.4 | 199.5 | 13.7 | 213.3 |
| YOLOv10m | SAHI passes + SCAHI, SAHI-RG (fusion on CPU) | 206.0 | 277.7 | 483.7 | 199.5 | 282.8 | 482.3 |
| YOLOv10m | SAHI passes + SCAHI, SAHI-RG (fusion on GPU) | 206.0 | 12.6 | 218.6 | 199.5 | 13.6 | 213.1 |
