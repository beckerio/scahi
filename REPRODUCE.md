# Reproducing the experiments

All steps use `python -m scahi.reproduce <command>` (see `python -m scahi.reproduce -h`). Outputs go to `out/`.
Random seed 0 everywhere; a single seed per run.

## 0. Data and checkpoints

1. Download VisDrone2019-DET (train, val, test-dev) from the official VisDrone website and unpack it; set
   `original_root` in `configs/data/visdrone.yaml` (or a machine-local override `configs/local/<host>.yaml` with a
   `data:` block).
2. Put the COCO-pretrained Ultralytics checkpoints (`yolov10s.pt`, `yolov10m.pt`, `yolo11s.pt`, `yolo11m.pt`,
   `yolo12s.pt`, `yolo12m.pt`) into `weights/`.
3. `python -m scahi.reproduce convert` – canonical COCO / YOLO files under `data/canonical/visdrone10/`. The
   train_main / calib split used in the paper is shipped in `data/splits/` and is reused, not redrawn.

## 1. Single-pass ladder

```bash
python -m scahi.reproduce train --model yolov10s --imgsz 1280      # batch 8, 100 epochs
python -m scahi.reproduce train --model yolov10s --imgsz 2048      # batch 3
```
Model selection on calib (toolkit-port AP of the one-to-one head for YOLOv10, of the only head for YOLO11/12). The
weights are `out/train_<model>_<s>_s0/ultralytics/train/weights/best_calib.pt`.

```bash
python -m scahi.reproduce single --weights <best_calib.pt> --imgsz 2048 --split val     # conf tuned on calib
```

## 2. Passes and caches

A stack is (whole-image pass at the training size s, tiles at zoom 2 with network input 1024, overlap 0.3), e.g.
`native2048`. Cache the dense head outputs of every pass once per split:

```bash
for s in calib val test-dev; do python -m scahi.reproduce cache --weights <best_calib.pt> --stack native2048 --split $s; done
```

## 3. Fusion (tuned on calib only)

```bash
python -m scahi.reproduce tune --weights <w> --stack native2048 --variant F0-NMS --branch o2o   # box fusion
python -m scahi.reproduce tune --weights <w> --stack native2048 --variant F2 --branch o2m      # SCAHI, pre-specified grid
python -m scahi.reproduce fuse --weights <w> --stack native2048 --variant F2 --branch o2m \
       --params '<best entry printed by tune>' --split val
```
The refined grid (cell factor 7–13 × bins per octave 6–8 × reference and lowered thresholds, selection under a 2×
fusion-time limit) and the parameters selected for every detector are listed in `docs/ALGORITHM.md` and
`configs/ext/xgrid_frozen.json`; pass them to `fuse --params`.

## 4. Evaluation

`metrics.json` of every step reports the VisDrone toolkit AP (ignore regions, 500 detections per image) and the COCO
size bins. `python -m scahi.reproduce eval --dets <file> --split val` evaluates any detections file (parquet / csv /
COCO results JSON with canonical image ids).

## 5. DOTA v1.5

Unpack DOTA v1.5 (images and the official horizontal-box annotations) and set `dota_root` in `configs/data/dota.yaml`.
The split (`data/splits/dota/`), the frozen rows and their parameters (`configs/dota/frozen_rows*.json`) are shipped. The
pipeline (`scahi/dota/pipeline.py`: single pass, dense-head caches of the stacks `dota1024` / `dota2048`, box fusion,
SCAHI, SAHI) and the evaluation (`scahi/dota/evaluate.py`, exact port of DOTA_devkit task 2, mAP50 without a detection
cap) follow the same steps as above; `scahi/dota/final.py` recomputes every frozen row (`row_dets`).

## 6. Paper tables and figures

`python scripts/regenerate_paper.py` regenerates every table and figure that is computed from the shipped result files
and checks them against `expected/` (see `README.md`).
