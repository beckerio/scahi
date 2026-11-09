# DOTA-val latency, D-H5 (SAHI passes + SCAHI) – measured on host-a

**Superseded (2026-11-09):** all DOTA latencies of the paper come from the single host-b run (`reports/dota/tab_dota_latency_host_b.md`, `reports/dota/tab_dota_latency_host_b_dh5.md`); this host-a table is kept for comparison only.

ms per image (mean over 458 DOTA-val images; p95 in the CSV); one run on host-a (NVIDIA GeForce RTX 5090, driver 595.99.02, default power limit), 2026-11-03T11:47:32 – 2026-11-03T11:59:43; GPU and CPU exclusive, batch 1; all rows final: True. Same protocol as `reports/dota/tab_dota_latency_host_a.md`, separate run; never mixed with the host-b tables.

SAHI: library slices (1024 weights) + standard pass, then the frozen merge (as in the host-a DOTA table). SAHI passes + SCAHI: the same passes computed once per image (fused model of SAHI's predictor path, both heads, outputs to the CPU), then SCAHI of the frozen D-H5 rows on the CPU (reference) or the GPU (blocked implementation, incl. host→GPU copy); the pass time is shared by the four SCAHI rows of a model. Box fusion inside SAHI runs on the CPU; the table is no speed comparison of SCAHI on the GPU against SAHI's merge. GPU fusion parity: `SAHI-F2_yolov10s` identical, `SAHI-RG_yolov10s` identical, `SAHI-F2_yolov10m` identical, `SAHI-RG_yolov10m` identical on every DOTA-val image (`runs/gpu_fusion/parity_dh5.json`).

| detector | method | passes | merge step | total | peak GPU mem. fusion (MB) |
|---|---|---:|---:|---:|---:|
| YOLOv10s | SAHI (frozen) | 180.1 | 17.0 | 197.1 | – |
| YOLOv10s | SAHI passes + SCAHI, SAHI-F2 (fusion on CPU) | 145.1 | 100.5 | 245.6 | – |
| YOLOv10s | SAHI passes + SCAHI, SAHI-F2 (fusion on GPU) | 145.1 | 12.6 | 157.8 | 1477.6 |
| YOLOv10s | SAHI passes + SCAHI, SAHI-RG (fusion on CPU) | 145.1 | 189.8 | 334.9 | – |
| YOLOv10s | SAHI passes + SCAHI, SAHI-RG (fusion on GPU) | 145.1 | 12.7 | 157.8 | 1480.0 |
| YOLOv10m | SAHI (frozen) | 232.7 | 16.2 | 248.8 | – |
| YOLOv10m | SAHI passes + SCAHI, SAHI-F2 (fusion on CPU) | 199.5 | 155.7 | 355.2 | – |
| YOLOv10m | SAHI passes + SCAHI, SAHI-F2 (fusion on GPU) | 199.5 | 13.7 | 213.3 | 1492.8 |
| YOLOv10m | SAHI passes + SCAHI, SAHI-RG (fusion on CPU) | 199.5 | 282.8 | 482.3 | – |
| YOLOv10m | SAHI passes + SCAHI, SAHI-RG (fusion on GPU) | 199.5 | 13.6 | 213.1 | 1802.5 |

## Why SAHI's passes take longer than the SCAHI path's passes

Both paths run the same passes (SAHI's slice grid and standard pass, the same fused weights, batch 1). Breakdown from a separate profiling run (`scripts/latency_dh5_profile.py`, 458 DOTA-val images, 2026-11-03 14:35; profiling only, not a latency measurement: cores 0-1 next to the FCOS CPU tuning, GPU otherwise idle, wrappers synchronise CUDA; compare shares, not absolute values with the exclusive run), ms per image (mean):

| component | YOLOv10s SAHI | YOLOv10s SCAHI path | YOLOv10m SAHI | YOLOv10m SCAHI path |
|---|---:|---:|---:|---:|
| image read and slicing (SAHI: PIL + slice crops; SCAHI path: cv2) | 26.0 | 17.7 | 52.7 | 13.2 |
| preprocessing (letterbox, tensor, host→GPU) | 41.1 | 33.8 | 43.5 | 41.7 |
| forward pass (same fused weights) | 74.0 | 87.0 | 114.7 | 131.1 |
| head output → boxes (SAHI: Ultralytics postprocess + Results; SCAHI path: decode both heads + mapping) | 9.5 | 14.1 | 12.0 | 16.5 |
| SAHI: one ObjectPrediction per detection | 50.6 | – | 48.8 | – |
| SAHI: slice loop, shifting / filtering of the objects | 37.4 | – | 13.3 | – |
| SAHI: objects → arrays for the merge | 11.6 | – | 11.7 | – |
| SCAHI path: rest (loop, bookkeeping) | – | 12.3 | – | 15.4 |
| **total (profiling run)** | 250.2 | 165.0 | 296.7 | 217.9 |
| total in the exclusive run (table above) | 180.1 | 145.1 | 232.7 | 199.5 |
| detections per image (SAHI: o2o after conf; SCAHI path: o2m candidates) | 6151.5 | 9460.1 | 6109.0 | 11082.1 |

The GPU-side work (preprocessing, forward pass, head output) costs about the same in both paths. SAHI's additional time is CPU-side implementation overhead of the SAHI library: the image is decoded with PIL, and every detection of every slice becomes a Python `ObjectPrediction` that is shifted and filtered one by one and converted back to arrays for the merge (several thousand detections per image at the frozen confidence threshold). It grows with the number of detections and with lower CPU clock (larger under the load of the profiling run than in the exclusive run); how it splits over the SAHI rows varies between the models (Python's garbage collection of the many objects falls into different components), their sum is the robust figure. It is not a property of SCAHI: SCAHI's fusion uses the same passes and only replaces the merge step.

Wording for the paper: SCAHI's merge on the GPU and SAHI's merge on the CPU run on different hardware, so their times are no speed comparison. The valid statement: replacing SAHI's merge by SCAHI does not add substantial latency, and the passes dominate the cost.
