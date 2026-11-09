# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standard Ultralytics training with model selection by the toolkit port on calib.

Ultralytics' own validation (mAP without ignore handling) is not used for selection: after
each scheduled epoch the freshly saved ``last.pt`` is evaluated on calib (protocol
``visdrone_ignore``, toolkit-port AP) and copied to ``best_calib.pt`` if it improves.
The curve lives in ``calib_curve.json`` next to the weights, so ``resume`` continues it.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from scahi.config import REPO_ROOT, load_class_setting
from scahi.eval.evaluate import evaluate, load_coco_gt
from scahi.infer.full import image_table, predict_full


def eval_schedule(epoch1: int, epochs: int, every: int = 5, last: int = 20) -> bool:
    """1-based epoch -> evaluate on calib? (every ``every`` epochs and each of the last ``last``)."""
    return epoch1 % every == 0 or epoch1 > epochs - last


def calib_subset(setting: str) -> dict[str, Any]:
    coco = load_coco_gt(REPO_ROOT / f"data/canonical/{setting}/coco/train.json")
    stems = set((REPO_ROOT / "data/splits/calib.txt").read_text().split())
    coco["images"] = [i for i in coco["images"] if Path(i["file_name"]).stem in stems]
    ids = {i["id"] for i in coco["images"]}
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in ids]
    coco["ignore"] = [g for g in coco["ignore"] if g["image_id"] in ids]
    return coco


def write_dataset_yaml(setting: str, out: Path, train_list: Path | None = None, val_list: Path | None = None) -> Path:
    root = REPO_ROOT / "data/canonical" / setting / "yolo_plain"
    ds = yaml.safe_load((root / "dataset.yaml").read_text())
    ds = {k: v for k, v in ds.items() if not k.startswith("scahi_")}
    ds["path"] = str(root)
    if train_list:
        ds["train"] = str(train_list)
    if val_list:
        ds["val"] = str(val_list)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(ds, sort_keys=False))
    return out


class CalibSelector:
    """Ultralytics callback (``on_model_save``) implementing calib-based model selection."""

    def __init__(
        self, setting: str, imgsz: int, epochs: int, image_root: Path | None = None,
        predict_fn: Callable[[Path, pd.DataFrame], pd.DataFrame] | None = None,
        schedule: Callable[[int, int], bool] = eval_schedule, selection_branch: str = "o2o",
    ) -> None:
        self.cs = load_class_setting(setting)
        self.coco = calib_subset(setting)
        self.images = image_table(self.coco, image_root)
        self.imgsz, self.epochs = imgsz, epochs
        # YOLOv10: selection on the NMS-free o2o branch, the o2m+NMS curve is recorded alongside. Single-head models
        # (YOLO11/12, extension study): ``selection_branch="o2m"`` = their only head with NMS.
        self.selection_branch = selection_branch
        if selection_branch == "o2m":
            self.predict_fns = {"o2m": predict_fn or (lambda w, imgs: predict_full(w, imgs, self.imgsz, branch="o2m")[0])}
        else:
            self.predict_fns = {"o2o": predict_fn or (lambda w, imgs: predict_full(w, imgs, self.imgsz, branch="o2o")[0])}
            if predict_fn is None:
                self.predict_fns["o2m"] = lambda w, imgs: predict_full(w, imgs, self.imgsz, branch="o2m")[0]
        self.schedule = schedule

    def __call__(self, trainer: Any) -> None:
        epoch1 = trainer.epoch + 1
        if not self.schedule(epoch1, self.epochs):
            return
        wdir = Path(trainer.last).parent
        curve_p = wdir / "calib_curve.json"
        curve = json.loads(curve_p.read_text()) if curve_p.exists() else []
        if any(c["epoch"] == epoch1 for c in curve):
            return
        entry: dict[str, Any] = {"epoch": epoch1, "source": "toolkit_port", "selection_branch": self.selection_branch}
        from scahi.machine import gpu_power

        entry["gpu_power_limit_w"] = gpu_power()["gpu_power_limit_w"]  # runtime only; logged per evaluation
        for br, fn in self.predict_fns.items():
            r = evaluate(self.coco, fn(Path(trainer.last), self.images), self.cs, "visdrone_ignore", port_only=True)
            sfx = "" if br == self.selection_branch else f"_{br}"
            entry[f"AP{sfx}"], entry[f"AP50{sfx}"] = r["metrics"]["AP"], r["metrics"]["AP50"]
        ap = entry["AP"]
        best = max((c["AP"] for c in curve), default=-1.0)
        curve.append(entry)
        if ap > best:
            shutil.copy(trainer.last, wdir / "best_calib.pt")
        curve_p.write_text(json.dumps(curve, indent=2))


QUEUE_DIR = REPO_ROOT / "runs" / "queue"
HALT_EXIT_CODE = 75  # job exit code for "halted at an epoch end, resume from last.pt"


class TrainingHalted(Exception):
    """Raised after ``last.pt`` of an epoch was written; the run resumes from it."""


class BatchReduced(Exception):
    """Ultralytics halved the batch on its own after an OOM in the first epoch (trainer auto-retry, 8.4); the recipe
    fixes the batch, so the run stops and the caller applies the preregistered fallback instead (2026-10-27)."""


class FixedBatchGuard:
    """``on_train_batch_end`` callback: raise ``BatchReduced`` if the trainer no longer runs the requested batch."""

    def __init__(self, batch: int) -> None:
        self.batch = batch

    def __call__(self, trainer: Any) -> None:
        if trainer.batch_size != self.batch:
            raise BatchReduced(f"Ultralytics reduced batch {self.batch} -> {trainer.batch_size}")


def read_deadline(path: Path = QUEUE_DIR / "DEADLINE") -> dt.datetime | None:
    """Queue deadline (ISO datetime in ``runs/queue/DEADLINE``), or None."""
    return dt.datetime.fromisoformat(path.read_text().strip()) if path.exists() else None


class EpochEndHalt:
    """Ultralytics callback (``on_model_save``, registered after the selector): clean stop at an epoch end.

    Stops when ``runs/queue/HALT`` exists or when the next epoch (longest epoch seen so far, incl. the calib
    evaluation, plus 10 % margin; before the final epoch also ``post_reserve_s`` for the evaluations after
    training) would end after the deadline in ``runs/queue/DEADLINE``. Raising here
    happens after ``last.pt`` and ``calib_curve.json`` are written and before ``final_eval`` strips the
    optimizer, so ``train(resume=True)`` continues with the next epoch.
    """

    def __init__(self, halt_file: Path = QUEUE_DIR / "HALT", deadline_file: Path = QUEUE_DIR / "DEADLINE",
                 post_reserve_s: float = 1200.0) -> None:
        self.halt_file, self.deadline_file, self.post_reserve_s = halt_file, deadline_file, post_reserve_s
        self.t_last = time.time()
        self.max_epoch_s = 0.0

    def __call__(self, trainer: Any) -> None:
        t = time.time()
        self.max_epoch_s = max(self.max_epoch_s, t - self.t_last)
        self.t_last = t
        epoch1 = trainer.epoch + 1
        if epoch1 >= trainer.epochs:
            return
        if self.halt_file.exists():
            raise TrainingHalted(f"HALT file after epoch {epoch1}")
        dl = read_deadline(self.deadline_file)
        need = 1.1 * self.max_epoch_s + (self.post_reserve_s if epoch1 + 1 >= trainer.epochs else 0.0)
        if dl and dt.datetime.now() + dt.timedelta(seconds=need) > dl:
            raise TrainingHalted(f"deadline {dl.isoformat(timespec='minutes')} after epoch {epoch1}")


def require_cuda(trainer: Any) -> None:
    """``on_train_start`` callback: never train on the CPU. After an OOM restart via ``os.execv`` (same PID, CUDA
    context of the failed run still held) CUDA initialisation failed with error 304 and Ultralytics silently trained
    YOLO12m 1280 on the CPU (2026-10-28 16:33-16:57)."""
    if getattr(trainer.device, "type", str(trainer.device)) != "cuda":
        raise RuntimeError(f"training would run on {trainer.device}, not on the GPU -> stopped")


def train_full(
    run_dir: Path, setting: str, imgsz: int, batch: int, epochs: int, seed: int, model: str = "yolov10s",
    train_list: Path | None = None, val_list: Path | None = None, image_root_calib: Path | None = None,
    data_yaml: Path | None = None, workers: int = 12, selector: CalibSelector | None = None,
    halt: EpochEndHalt | None = None,
) -> Path:
    """Train (or resume) and return the path of ``best_calib.pt``."""
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available -> no training on the CPU")
    wdir = run_dir / "ultralytics" / "train" / "weights"
    if (wdir / "best_calib.pt").exists() and (run_dir / "TRAIN_DONE").exists():
        return wdir / "best_calib.pt"
    data = data_yaml or write_dataset_yaml(setting, run_dir / "dataset.yaml", train_list, val_list)
    selector = selector or CalibSelector(setting, imgsz, epochs, image_root_calib)
    halt = halt or EpochEndHalt()
    if (wdir / "last.pt").exists():
        y = YOLO(str(wdir / "last.pt"))
        y.add_callback("on_model_save", selector)
        y.add_callback("on_model_save", halt)
        y.add_callback("on_train_batch_end", FixedBatchGuard(batch))
        y.add_callback("on_train_start", require_cuda)
        y.train(resume=True)
    else:
        y = YOLO(str(REPO_ROOT / "weights" / f"{model}.pt"))
        y.add_callback("on_model_save", selector)
        y.add_callback("on_model_save", halt)
        y.add_callback("on_train_batch_end", FixedBatchGuard(batch))
        y.add_callback("on_train_start", require_cuda)
        y.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, seed=seed, workers=workers, val=False,
                plots=False, amp=True, project=str(run_dir / "ultralytics"), name="train", exist_ok=True,
                deterministic=False, verbose=False)
    (run_dir / "TRAIN_DONE").write_text("ok\n")
    return wdir / "best_calib.pt"
