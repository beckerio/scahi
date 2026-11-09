# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reproduction entry point (``python -m scahi.reproduce <command>``), built only from the library functions that
produced the paper's numbers (scahi.data / train / infer / fusion / eval). One command per pipeline step:

  convert                         canonical COCO / YOLO data (+ ignore regions) from the original VisDrone files
  train  --imgsz S                single-pass training at long side S (COCO init, calib model selection)
  single --weights W --imgsz S    single pass on a split (conf tuned on calib) -> detections + metrics
  cache  --weights W --stack K    raw dense-head cache of every pass of a stack (calib / val / test-dev)
  tune   --weights W --stack K --variant {F2,F0-NMS}   calib grid -> best parameters (JSON)
  fuse   --weights W --stack K --variant V --params JSON --split S   fused detections + metrics
  eval   --dets FILE --split S    VisDrone evaluation (toolkit port, ignore regions) of a detections file

Outputs go to ``out/<name>/`` (detections.parquet, metrics.json). Paths of the original data: configs/data/visdrone.yaml
(or a machine-local override in configs/local/<host>.yaml). COCO-pretrained checkpoints are expected in weights/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scahi.config import REPO_ROOT, load_class_setting

SETTING = "visdrone10"
OUT = REPO_ROOT / "out"


def split_coco(split: str) -> dict[str, Any]:
    from scahi.eval.evaluate import load_coco_gt
    from scahi.train.full import calib_subset

    if split == "calib":
        return calib_subset(SETTING)
    return load_coco_gt(REPO_ROOT / f"data/canonical/{SETTING}/coco/{split}.json")


def write(name: str, dets: pd.DataFrame, split: str, fields: dict[str, Any]) -> dict[str, Any]:
    from scahi.eval.detections import finalize_detections
    from scahi.eval.evaluate import evaluate
    from scahi.report.schema import build_metrics, write_metrics

    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    dets = finalize_detections(dets)
    dets.to_parquet(d / "detections.parquet")
    cs = load_class_setting(SETTING)
    res = evaluate(split_coco(split), dets, cs, "visdrone_ignore")
    m = build_metrics(res, exp_id=name, split=split, class_setting=SETTING, eval_protocol="visdrone_ignore", **fields)
    write_metrics(m, d / "metrics.json")
    print(f"{name}: AP {m['metrics']['AP']:.2f}  AP50 {m['metrics']['AP50']:.2f}")
    return m


def cmd_convert(a: argparse.Namespace) -> None:
    from scahi.data.convert import run_convert
    from scahi.machine import load_data_config

    cfg = load_data_config(REPO_ROOT / "configs/data/visdrone.yaml")
    split_dirs = {k: v["original"] for k, v in cfg["splits"].items()}
    cs = load_class_setting(SETTING)
    stats = run_convert(Path(cfg["original_root"]), split_dirs, cs, REPO_ROOT / "data/canonical" / cs.name,
                        REPO_ROOT / "data/splits")
    print(json.dumps(stats))


def cmd_train(a: argparse.Namespace) -> None:
    from scahi.train.full import train_full

    d = OUT / f"train_{a.model}_{a.imgsz}_s{a.seed}"
    best = train_full(d, SETTING, a.imgsz, a.batch, a.epochs, a.seed, a.model)
    print(f"weights: {best}")


def cmd_single(a: argparse.Namespace) -> None:
    from scahi.eval.evaluate import evaluate
    from scahi.infer.full import image_table, predict_full

    cs = load_class_setting(SETTING)
    conf = a.conf
    if conf is None:  # tuned on calib, as in the paper
        cc = split_coco("calib")
        dc, _ = predict_full(a.weights, image_table(cc), a.imgsz, branch=a.branch)
        grid = {c: evaluate(cc, dc[dc.score >= c], cs, "visdrone_ignore", port_only=True)["metrics"]["AP"]
                for c in (0.001, 0.01, 0.05, 0.1)}
        conf = max(grid, key=grid.get)
    coco = split_coco(a.split)
    d, info = predict_full(a.weights, image_table(coco), a.imgsz, branch=a.branch)
    write(f"single_{a.imgsz}_{a.branch}_{a.split}",
          d[d.score >= conf], a.split, {"weights": str(a.weights),
                                        "inference": {"fusion": "none", "branch": a.branch, "imgsz": a.imgsz,
                                                      "conf": conf}})


def cmd_cache(a: argparse.Namespace) -> None:
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm
    from scahi.infer.full import image_table
    from scahi.infer.raw_head import load_unfused

    m = load_unfused(a.weights)
    imgs = image_table(split_coco(a.split))
    for lk in cm.STACK_LEVELS[a.stack]:
        print(cm.cache_level(m, a.weights, imgs, lk, a.overlap, a.split, CACHE_ROOT))


def _data(a: argparse.Namespace, split: str):
    from scahi.fusion import run as R

    coco = split_coco(split)
    stems = [Path(i["file_name"]).stem for i in coco["images"]]
    s2i = {Path(i["file_name"]).stem: i["id"] for i in coco["images"]}
    return coco, s2i, R.load_split(Path(a.weights), a.stack, split, a.overlap, a.branch, stems)


def cmd_tune(a: argparse.Namespace) -> None:
    from scahi.fusion import run as R

    coco, s2i, data = _data(a, "calib")
    res = R.tune(data, s2i, coco, a.branch, a.variant)
    best = max(res, key=lambda r: r["AP"])
    print(json.dumps({"best": best, "grid": res}, indent=1))


def cmd_fuse(a: argparse.Namespace) -> None:
    from scahi.fusion import run as R

    coco, s2i, data = _data(a, a.split)
    params = {k: v for k, v in json.loads(a.params).items() if k not in ("AP", "AP50", "source")}
    d = R.fuse_split(data, s2i, a.branch, a.variant, params)
    write(f"fuse_{a.stack}_{a.variant}_{a.branch}_{a.split}", d, a.split,
          {"weights": str(a.weights), "inference": {"fusion": a.variant, "branch": a.branch, "stack": a.stack,
                                                    "overlap": a.overlap, "params": params}})


def cmd_eval(a: argparse.Namespace) -> None:
    from scahi.eval.detections import load_detections

    write(a.name or Path(a.dets).stem, load_detections(a.dets), a.split, {})


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m scahi.reproduce")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("convert")
    t = sub.add_parser("train")
    t.add_argument("--model", default="yolov10s")
    t.add_argument("--imgsz", type=int, required=True)
    t.add_argument("--batch", type=int, default=None, help="default: 8 at <= 1280, else 3 (paper recipe)")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--seed", type=int, default=0)
    for name in ("single", "cache", "tune", "fuse"):
        p = sub.add_parser(name)
        p.add_argument("--weights", required=True)
        p.add_argument("--branch", default="o2m" if name in ("tune", "fuse") else "o2o")
        if name == "single":
            p.add_argument("--imgsz", type=int, required=True)
            p.add_argument("--conf", type=float, default=None)
        else:
            p.add_argument("--stack", default="native2048")
            p.add_argument("--overlap", type=float, default=0.3)
        if name in ("tune", "fuse"):
            p.add_argument("--variant", default="F2", choices=("F2", "F0-NMS"))
        if name == "fuse":
            p.add_argument("--params", required=True, help="JSON, e.g. the 'best' entry printed by tune")
        if name != "tune":
            p.add_argument("--split", default="val", choices=("calib", "val", "test-dev"))
    e = sub.add_parser("eval")
    e.add_argument("--dets", required=True)
    e.add_argument("--split", default="val", choices=("calib", "val", "test-dev"))
    e.add_argument("--name", default=None)
    a = ap.parse_args()
    if a.cmd == "train" and a.batch is None:
        a.batch = 8 if a.imgsz <= 1280 else 3
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
