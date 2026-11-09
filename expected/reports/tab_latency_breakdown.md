# Latency breakdown (ms per image, mean; exclusive GPU and CPU, RTX 5090 at 575 W, batch 1)

One run on host-b, 2026-10-31T15:35:11 – 2026-10-31T15:46:50; all rows final: True. Single passes: Ultralytics predictor (fused, one head). Tiles + SCAHI: passes of the stack in the reference path (unfused model, outputs to the CPU), then SCAHI on the CPU (reference) or on the GPU (scalespace_torch, incl. host→GPU copy) on the same pass outputs; totals = passes + fusion. Parity of GPU vs. CPU fusion on every val image before timing (runs/latency_breakdown/parity.json): X2: 1 of 217492 detections differ in 1 image(s) (box up to 1.90 px, same score and class); XG_yolov10s_2048_o2m_C: identical (249999 detections); RG_yolov10s_2048: GPU fusion not available (out of GPU memory: dense peak × prediction membership matrix); RG_yolov10m_2048: GPU fusion not available (out of GPU memory: dense peak × prediction membership matrix). YOLOv11s / YOLOv12s rows (DOTA addendum 2, at S_max): supplementary run on host-b, 2026-10-31T20:07:10 – 2026-10-31T20:17:15, same procedure, all rows final: True; anchors measured again in that run: A_yolov10s_2048 27.8 ms (main run 29.7); RG_yolov10s_2048 295.5 ms (main run 306.4). GPU fusion not used on DOTA (see parity).

| dataset | detector | method | passes | fusion CPU | fusion GPU | total (CPU fusion) | total (GPU fusion) |
|---|---|---|---:|---:|---:|---:|---:|
| VisDrone val | YOLOv10s | single pass 2048 | 10.3 | – | – | 10.3 | 10.3 |
| VisDrone val | YOLOv10s | single pass 2560 | 14.4 | – | – | 14.4 | 14.4 |
| VisDrone val | YOLOv10s | tiles + SCAHI (pre-specified) | 75.6 | 30.8 | 9.4 | 106.4 | 85.0 |
| VisDrone val | YOLOv10s | tiles + SCAHI, refined grid | 76.6 | 60.1 | 10.2 | 136.7 | 86.8 |
| DOTA-val | YOLOv10s | single pass 2048 | 29.7 | – | – | 29.7 | 29.7 |
| DOTA-val | YOLOv10s | tiles + SCAHI, refined grid | 140.1 | 166.3 | – | 306.4 | – |
| DOTA-val | YOLOv10m | single pass 2048 | 35.3 | – | – | 35.3 | 35.3 |
| DOTA-val | YOLOv10m | tiles + SCAHI, refined grid | 192.1 | 215.6 | – | 407.7 | – |
| DOTA-val | YOLOv11s | single pass 2048 | 26.4 | – | – | 26.4 | 26.4 |
| DOTA-val | YOLOv11s | tiles + SCAHI, refined grid (2048) | 123.5 | 162.8 | – | 286.3 | – |
| DOTA-val | YOLOv12s | single pass 1024 | 22.2 | – | – | 22.2 | 22.2 |
| DOTA-val | YOLOv12s | tiles + SCAHI, refined grid (1024) | 137.8 | 202.7 | – | 340.5 | – |
