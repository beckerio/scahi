# DOTA-val latency – measured on host-a

**Superseded (2026-11-09):** all DOTA latencies of the paper come from the single host-b run (`reports/dota/tab_dota_latency_host_b.md`, `reports/dota/tab_dota_latency_host_b_dh5.md`); this host-a table is kept for comparison only.

ms per image (mean over 458 DOTA-val images; p95 in the CSV); one run on host-a (NVIDIA GeForce RTX 5090, driver 595.99.02, default power limit), 2026-11-01T19:07:03 – 2026-11-01T19:23:05; GPU and CPU exclusive, batch 1; all rows final: True. **Not comparable with the host-b tables (`reports/tab_latency_breakdown.md`); never mix the two.** For the paper, DOTA latencies come only from this table and VisDrone latencies only from the host-b table; both machines have the same GPU model (RTX 5090, 32 GB) at the same default power limit (575 W).

**How to read the merge columns.** Box fusion (SAHI's NMS, numba backend) runs on the CPU; SCAHI in the GPU rows runs on the GPU. The table is therefore **not** a speed comparison of SCAHI against box fusion (box fusion on the GPU would also be faster). The valid statement: the tile passes dominate the cost (91–94 % of the total with GPU fusion) and SCAHI on the GPU adds about 12 ms per image.

Passes of the tiled rows (whole image at S + native 1024 tiles, overlap 0.3) are measured once per image in the reference path (unfused model, outputs to the CPU); box fusion and SCAHI then run on the same pass outputs, so the pass time is shared by every tiled method. Merge step = box fusion (per-pass NMS + cross-pass NMS on the o2m head, frozen NMS-o2m row) or SCAHI refined grid (frozen RG row) on the CPU (reference) or the GPU (blocked implementation, incl. host→GPU copy). GPU fusion parity: 7/7 RG rows identical on every DOTA-val image (`runs/gpu_fusion/parity_dota.json`). Single pass: Ultralytics predictor (fused, the row's head). SAHI: library slices (1024 weights) + standard pass, then the frozen merge; rows exist for YOLOv10s / YOLOv10m only. YOLOv12s at 1024 px (largest trainable size).

**Measurement setup.** run: 2026-11-01T19:07:03 - 2026-11-01T19:23:05, started by the user from a plain terminal (scripts/run_latency_host_a.sh), VS Code closed; compute processes on gpu: none at any exclusivity check (start and end of every row and every 100 images; latency_other_gpu_pids = [] for all rows); graphics processes type G: gnome-shell (PID 3197, since 14:59:08), Xwayland (PID 5016, since 14:59:26), Firefox (PID 56595, since 19:03:56) and Nautilus (PID 57007, since 19:04:23); Firefox and Nautilus were still running after the run (process start times), i.e. present during the whole run, graphics only; vs code gpu process: PID 57941 started 19:28:31, after the end of the run; foreign cpu load max cores: 0.98; power limit w: 575.0; source: per-row exclusivity fields in runs/latency_host_a/dota_latency.json; nvidia-smi + ps start times read at 19:3x after the run (full nvidia-smi lists at start/end were not saved by this run; saved by scripts/run_latency_host_a.sh from now on).

| detector | method | passes | merge step | total | peak GPU mem. fusion (MB) |
|---|---|---:|---:|---:|---:|
| YOLOv10s | single pass 2048 | 28.7 | – | 28.7 | – |
| YOLOv10s | 2048 + tiles, box fusion (NMS) | 137.6 | 45.9 | 183.5 | – |
| YOLOv10s | 2048 + tiles, SCAHI, refined grid (fusion on CPU) | 137.6 | 158.4 | 296.0 | – |
| YOLOv10s | 2048 + tiles, SCAHI, refined grid (fusion on GPU) | 137.6 | 12.2 | 149.8 | 1923.7 |
| YOLOv10s | SAHI (1024 weights) | 176.3 | 15.3 | 191.6 | – |
| YOLOv10m | single pass 2048 | 34.9 | – | 34.9 | – |
| YOLOv10m | 2048 + tiles, box fusion (NMS) | 186.8 | 35.8 | 222.6 | – |
| YOLOv10m | 2048 + tiles, SCAHI, refined grid (fusion on CPU) | 186.8 | 208.6 | 395.4 | – |
| YOLOv10m | 2048 + tiles, SCAHI, refined grid (fusion on GPU) | 186.8 | 12.7 | 199.6 | 1823.8 |
| YOLOv10m | SAHI (1024 weights) | 233.3 | 14.7 | 247.9 | – |
| YOLOv11s | single pass 2048 | 28.6 | – | 28.6 | – |
| YOLOv11s | 2048 + tiles, box fusion (NMS) | 124.6 | 38.1 | 162.7 | – |
| YOLOv11s | 2048 + tiles, SCAHI, refined grid (fusion on CPU) | 124.6 | 161.0 | 285.6 | – |
| YOLOv11s | 2048 + tiles, SCAHI, refined grid (fusion on GPU) | 124.6 | 12.4 | 137.0 | 1898.6 |
| YOLOv12s | single pass 1024 | 21.0 | – | 21.0 | – |
| YOLOv12s | 1024 + tiles, box fusion (NMS) | 133.3 | 32.9 | 166.2 | – |
| YOLOv12s | 1024 + tiles, SCAHI, refined grid (fusion on CPU) | 133.3 | 199.3 | 332.6 | – |
| YOLOv12s | 1024 + tiles, SCAHI, refined grid (fusion on GPU) | 133.3 | 12.7 | 146.0 | 1598.1 |
