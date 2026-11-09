# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase D training (docs/DOTA_BRIEF.md §3.1): YOLOv10s / YOLOv10m on DOTA v1.5 HBB, COCO init, 60 epochs, single-pass
ladder s in {1024, 2048, 3072}, model selection on calib by the official mAP50 (devkit port, no cap; addendum
2026-10-30) of the one-to-one head; the one-to-many head + NMS is recorded alongside. Recipe and batch rules as VisDrone (Ultralytics
8.4.183, same augmentation defaults; batch 8 at 1024, 3 at 2048 / 3072; one preregistered reduction 8 -> 6, 3 -> 2
after an OOM at the fixed batch; batch >= 2; no silent batch change - scahi.train.full.FixedBatchGuard).

Dataset (``build_dataset``): data/canonical/dota/yolo_hbb/{images,labels}/{train,val}; images are symlinks to the
Ultralytics JPEGs, labels = official HBB (min/max of the label polygon, which equals the HBB rectangle) in YOLO
format, boxes clipped to the image (audit: P0334), all GT incl. difficult (difficult objects are real objects; only
the evaluation ignores them). Image lists data/canonical/dota/yolo_hbb/{train_main,calib}.txt from data/splits/dota
(P1872 excluded there).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scahi.dota import data as D
from scahi.dota import evaluate as EV
from scahi.machine import REPO_ROOT

YOLO_ROOT = REPO_ROOT / "data/canonical/dota/yolo_hbb"
SPLITS_DIR = REPO_ROOT / "data/splits/dota"
DOTA_RUNS = REPO_ROOT / "runs/dota"
EPOCHS = 60
SIZES = (1024, 2048, 3072)
BATCH = {1024: 8, 2048: 3, 3072: 3}
FALLBACK = {8: 6, 3: 2}
MODELS = ("yolov10s", "yolov10m")  # the preregistered pair (Holm family of the original verdicts)
NEW_MODELS = ("yolo11s", "yolo12s")  # addendum 2 (2026-10-31): single-head models, own Holm family
ALL_MODELS = MODELS + NEW_MODELS
# S_max per model (memory smoke on the 48 densest train_main images, runs/dota/smoke.json; preregistered rule):
# 3072 fits for none; YOLO12s 2048 does not fit with batch 3 or 2 (2026-10-31) -> S_max = 1024
S_MAX = {"yolov10s": 2048, "yolov10m": 2048, "yolo11s": 2048, "yolo12s": 1024}
S_MAX_FCOS = 2048  # addendum 3 item 3: S_max smoke 2026-11-01 (runs/dota/fcos_smoke.json); read via s_max()
FCOS = "fcos"  # addendum 3 (2026-11-01): torchvision FCOS, scahi.dota.fcos_train; S_max by the rule (runs/dota/fcos_smoke.json)
CAP = 4000  # secondary metric only (addendum 2026-10-30): 2 x p99 of GT per training image, rounded up to 500
PIPE_MAX_DET = 20000  # pipeline limit per image / per pass, > 2 x the densest image (10 206 GT); must never bind
#                       (addendum 2026-10-30: primary metric without cap) - every row reports how often it is reached
PRED_CONF = 0.001


def s_max(model: str) -> int:
    return S_MAX_FCOS if model == FCOS else S_MAX[model]


def heads(model: str) -> tuple[str, ...]:
    """Heads of a model; the first one selects on calib. YOLOv10: o2o (+ o2m recorded); YOLO11/12: their only head ("o2m")."""
    return ("o2o", "o2m") if model.startswith("yolov10") else ("o2m",)


def split_stems(name: str) -> list[str]:
    return (SPLITS_DIR / f"{name}.txt").read_text().split()


def build_dataset() -> Path:
    """Write the YOLO-format dataset once (idempotent); returns dataset.yaml."""
    from PIL import Image

    for split in D.SPLITS:
        (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for s, a in D.load(split).items():
            img = YOLO_ROOT / "images" / split / f"{s}.jpg"
            if not img.exists():
                img.symlink_to(D.image_path(split, s))
            lab = YOLO_ROOT / "labels" / split / f"{s}.txt"
            if lab.exists():
                continue
            with Image.open(D.image_path(split, s)) as im:
                W, H = im.size
            b = D.minmax_box(a.poly)
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
            keep = (b[:, 2] - b[:, 0] >= 1) & (b[:, 3] - b[:, 1] >= 1)
            lines = [f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
                     for c, (x1, y1, x2, y2) in zip(a.cls[keep], b[keep])]
            lab.write_text("\n".join(lines) + ("\n" if lines else ""))
    for name in ("train_main", "calib"):
        (YOLO_ROOT / f"{name}.txt").write_text(
            "".join(f"{YOLO_ROOT / 'images/train' / s}.jpg\n" for s in split_stems(name)))
    ds = {"path": str(YOLO_ROOT), "train": str(YOLO_ROOT / "train_main.txt"), "val": str(YOLO_ROOT / "calib.txt"),
          "names": dict(enumerate(D.CLASSES))}
    out = YOLO_ROOT / "dataset.yaml"
    out.write_text(yaml.safe_dump(ds, sort_keys=False))
    return out


def predict(weights: str | Path, split: str, stems: list[str], imgsz: int, branch: str = "o2o",
            conf: float = PRED_CONF, max_det: int = PIPE_MAX_DET, measure_latency: bool = False) -> tuple[pd.DataFrame, dict]:
    """Single pass at long side ``imgsz``; detections as scahi.dota.evaluate.DET_COLS (cls = DOTA class index).
    FCOS weights (addendum 3) go to the torchvision predictor (``scahi.dota.fcos_train.predict``)."""
    if Path(weights).parent.parent.name.startswith(f"{FCOS}_"):
        from scahi.dota import fcos_train

        # FCOS: the YOLO calib-selection default 20 000 is replaced by the addendum-3 limit 200 000 (item 6)
        return fcos_train.predict(weights, split, stems, imgsz, conf=conf, measure_latency=measure_latency,
                                  max_det=fcos_train.LIMIT if max_det == PIPE_MAX_DET else max_det)
    import torch
    from ultralytics import YOLO

    from scahi.infer.full import BRANCH_NMS_ARG

    model = YOLO(str(weights))
    kw = {"imgsz": imgsz, "conf": conf, "max_det": max_det, "verbose": False, "nms": BRANCH_NMS_ARG[branch]}
    rows, lat, n_at_limit = [], [], 0
    torch.cuda.reset_peak_memory_stats()
    for n, s in enumerate(stems):
        p = str(D.image_path(split, s))
        if measure_latency and n == 0:
            model.predict(p, **kw)  # warm-up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model.predict(p, **kw)[0]
        torch.cuda.synchronize()
        lat.append(1000 * (time.perf_counter() - t0))
        b = r.boxes
        xyxy, sc, cl = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
        rows += [(s, int(c), float(q), *map(float, bb)) for bb, q, c in zip(xyxy, sc, cl)]
        n_at_limit += int(len(sc) >= max_det)
    info = {"peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20, "latency_ms_mean": float(np.mean(lat)),
            "latency_ms_p95": float(np.percentile(lat, 95)), "max_dets_per_image": max_det,
            "images_at_max_det": n_at_limit}
    return pd.DataFrame(rows, columns=EV.DET_COLS), info


class DotaCalibSelector:
    """``on_model_save`` callback: official mAP50 (devkit port, no cap; addendum 2026-10-30) on calib; the first of
    ``branches`` selects (YOLOv10: o2o, o2m recorded; single-head models: their only head, addendum 2)."""

    def __init__(self, imgsz: int, epochs: int, schedule=None, branches: tuple[str, ...] = ("o2o", "o2m")) -> None:
        from scahi.train.full import eval_schedule

        self.imgsz, self.epochs, self.branches = imgsz, epochs, branches
        self.schedule = schedule or eval_schedule
        self.stems = split_stems("calib")
        self.gt = EV.gt_arrays("train", self.stems)

    def __call__(self, trainer: Any) -> None:
        import shutil

        epoch1 = trainer.epoch + 1
        if not self.schedule(epoch1, self.epochs):
            return
        wdir = Path(trainer.last).parent
        cp = wdir / "calib_curve.json"
        curve = json.loads(cp.read_text()) if cp.exists() else []
        if any(c["epoch"] == epoch1 for c in curve):
            return
        e: dict[str, Any] = {"epoch": epoch1, "source": "dota_devkit_port mAP50, no cap",
                             "selection_branch": self.branches[0]}
        t0 = time.time()
        for br in self.branches:
            dets, _ = predict(trainer.last, "train", self.stems, self.imgsz, br)
            c = EV.build(self.gt, dets, None, sizes=False)
            sfx = "" if br == self.branches[0] else f"_{br}"
            e[f"mAP50{sfx}"] = EV.map50(c)
        e["eval_s"] = time.time() - t0
        best = max((c["mAP50"] for c in curve), default=-1.0)
        curve.append(e)
        if e["mAP50"] > best:
            shutil.copy(trainer.last, wdir / "best_calib.pt")
        cp.write_text(json.dumps(curve, indent=2))


def train_dir(model: str, imgsz: int, seed: int = 0) -> Path:
    return DOTA_RUNS / "train" / f"{model}_{imgsz}_s{seed}"


def weights(model: str, imgsz: int, seed: int = 0) -> Path:
    if model == FCOS:
        return train_dir(model, imgsz, seed) / "weights/best_calib.pt"
    return train_dir(model, imgsz, seed) / "ultralytics/train/weights/best_calib.pt"


def d_train(model: str, imgsz: int, seed: int = 0) -> Path:
    """One ladder training (resumable); preregistered single batch reduction after an OOM at the fixed batch."""
    import torch

    from scahi.common import rel, write_env
    from scahi.train.full import BatchReduced, train_full

    d = train_dir(model, imgsz, seed)
    d.mkdir(parents=True, exist_ok=True)
    write_env(d / "env.json")
    fb = d / "OOM_FALLBACK"
    b = int(fb.read_text()) if fb.exists() else BATCH[imgsz]
    data = build_dataset()
    try:
        best = train_full(d, "dota", imgsz, b, EPOCHS, seed, model, data_yaml=data,
                          selector=DotaCalibSelector(imgsz, EPOCHS, branches=heads(model)))
    except (torch.cuda.OutOfMemoryError, BatchReduced) as e:
        if fb.exists():
            raise
        import shutil

        print(f"[dota] OOM at the fixed batch {b} ({type(e).__name__}) -> preregistered fallback", flush=True)
        fb.write_text(str(FALLBACK[b]))
        shutil.rmtree(d / "ultralytics", ignore_errors=True)
        raise SystemExit(f"[dota] fallback batch {fb.read_text()} written; rerun in a fresh process")
    curve = json.loads((best.parent / "calib_curve.json").read_text())
    be = max(curve, key=lambda c: c["mAP50"])
    (d / "training.json").write_text(json.dumps({
        "init": f"coco:{model}.pt", "epochs": EPOCHS, "imgsz": imgsz, "batch": b, "seed": seed,
        "oom_fallback": fb.exists(), "selection": f"devkit mAP50 on calib ({heads(model)[0]}, no cap), best epoch {be['epoch']} "
                                                  f"({be['mAP50']:.2f})", "weights": rel(best)}, indent=2))
    return best
