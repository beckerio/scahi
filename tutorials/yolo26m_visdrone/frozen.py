# SPDX-License-Identifier: AGPL-3.0-or-later
"""Track A: use the frozen YOLO26m / YOLO26s weights and parameters (``--model``, default yolo26m).

    python tutorials/yolo26m_visdrone/frozen.py rows [--model yolo26s]                 # the 25 frozen configurations
    python tutorials/yolo26m_visdrone/frozen.py run --row ID --split val [--mode cached|full] [--model yolo26s]
    python tutorials/yolo26m_visdrone/frozen.py run --row all --split val --mode full
    python tutorials/yolo26m_visdrone/frozen.py predict --row ID --images DIR --out dets.csv   # your own images

Two modes of ``run`` (parameters always from the model's freeze, configs/ext/frozen_rows_ext26.json / _ext26s.json; stored
values from tutorials/yolo26m_visdrone/frozen_results.json / frozen_results_yolo26s.json):

* ``cached`` (default): from the stored outputs of the single evaluation - the dense head outputs of every pass
  (SCAHI / box-fusion rows: fusion and evaluation on the CPU) and the stored detections of the single-pass and SAHI rows
  (evaluation only); release asset <model>_visdrone_cached_<split>.tar, unpacked in the repository root. No GPU, no
  weights. Comparison: exact equality with the stored AP (bit-identical with the versions of requirements.lock).
* ``full``: new GPU inference from the images with the frozen weights (SHA-256 checked), caches built with Ultralytics'
  input preparation under cache_ualigned_full/. Other GPUs, drivers and CUDA / cuDNN versions change the last digits of
  the network outputs and can flip single detections at a threshold, so the comparison is |delta| <= 0.05 AP (the
  threshold of the box back-mapping check). Reference environment of the stored values: tutorial README.

Outputs under out/<model>/<mode>/<split>/<row>/.
"""

from __future__ import annotations

import sys

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tutorial helpers (also under python -I)
import common as C  # noqa: E402

FULL_ROOT = C.REPO / "cache_ualigned_full"  # caches of the full mode (never mixed with the downloaded outputs)
TOL_FULL = 0.05


def _size(r: dict) -> int:
    return int(re.search(r"yolo26[a-z]_(\d+)_", r["weights"]).group(1))


def _out() -> Path:
    return C.REPO / "out" / C.MODEL


def _stored() -> Path:
    """Stored detections of the single-pass / SAHI rows (from the cached-mode asset)."""
    return C.REPO / f"{C.MODEL}_stored"


def cached_detections(r: dict, imgs: pd.DataFrame, split: str) -> pd.DataFrame:
    """Cached mode: fusion of the stored dense outputs (no inference) or the stored detections of a predictor row."""
    if r["kind"] != "cache":
        f = _stored() / split / f"{r['id']}.parquet"
        if not f.exists():
            raise SystemExit(f"{f} missing: unpack the release asset {C.MODEL}_visdrone_cached_{split}.tar in the "
                             "repository root (or use --mode full)")
        return pd.read_parquet(f)
    from scahi.fusion import run as R
    from scahi.fusion import scalespace as ss

    f = r["fusion"]
    wid = r["weights_sha256"][:12]  # the stored outputs are filed under the frozen weights' id
    dirs = [(lk, C.ALIGNED_ROOT / wid / f"{lk}_o{int(round(f['overlap'] * 100)):02d}" / split) for lk in f["levels"]]
    miss = [str(d) for _, d in dirs if not all((d / f"{s}.npz").exists() for s in imgs.stem)]
    if miss:
        raise SystemExit(f"stored dense outputs missing in {miss}: unpack the release asset "
                         f"{C.MODEL}_visdrone_cached_{split}.tar in the repository root (or use --mode full)")
    data = {s: ss.load_levels(dirs, s, f["branch"], f["overlap"]) for s in imgs.stem}
    return R.fuse_split(data, dict(zip(imgs.stem, imgs.image_id)), f["branch"], f["variant"], f["params"])


def detections(r: dict, imgs: pd.DataFrame, split: str, w: Path, root: Path = FULL_ROOT) -> pd.DataFrame:
    """Full mode: detections of one frozen row from new inference on an image table (image_id, stem, path)."""
    inf = r["inference"]
    if r["kind"] == "single":
        from scahi.infer.full import predict_full

        d, _ = predict_full(w, imgs, inf["imgsz"], measure_latency=True, branch=inf["branch"])  # batch 1, as frozen
        return d[d.score >= inf["conf"]]
    if r["kind"] == "sahi":
        from scahi.infer import sahi_baseline as sb

        b = r["sahi"]
        m = sb.load_model(w, b["conf"], inf["branch"])
        d, _ = sb.sliced_predict(m, imgs, b["overlap"], b["merge"], b["match_thr"], force=True, measure=True,
                                 match_metric=b["metric"])
        return d
    from scahi.fusion import run as R
    from scahi.fusion import scalespace as ss
    from scahi.infer import cache_mz as cm
    from scahi.infer.raw_head import load_unfused

    f = r["fusion"]
    with C.ultralytics_caches(root):
        dirs = [(lk, cm.cache_dir(root, w, lk, f["overlap"], split)) for lk in f["levels"]]
        if any(not (d / f"{s}.npz").exists() for _, d in dirs for s in imgs.stem):
            m = load_unfused(w)
            for lk, _ in dirs:
                t0 = time.time()
                cm.cache_level(m, w, imgs, lk, f["overlap"], split, root)
                print(f"  cache {lk} ({split}): {time.time() - t0:.0f} s", flush=True)
            del m
        data = {s: ss.load_levels(dirs, s, f["branch"], f["overlap"]) for s in imgs.stem}
    return R.fuse_split(data, dict(zip(imgs.stem, imgs.image_id)), f["branch"], f["variant"], f["params"])


def cmd_rows(a) -> None:
    res = C.results()["rows"]
    print(f"{'id':32s} {'val':>6s} {'test':>6s}  configuration")
    for rid, r in C.rows().items():
        v = res[rid]
        print(f"{rid:32s} {v['val']:6.2f} {v['test-dev']:6.2f}  {C.describe(r)}")


def cmd_run(a) -> None:
    rs = C.rows()
    ids = list(rs) if a.row == "all" else [a.row]
    coco, imgs = C.split_data(a.split)
    stored = C.results()["rows"]
    n_fail = 0
    for rid in ids:
        r = rs[rid]
        t0 = time.time()
        if a.mode == "cached":
            d = cached_detections(r, imgs, a.split)
        else:
            d = detections(r, imgs, a.split, C.weights(_size(r), getattr(a, f"weights_{_size(r)}")))
        m = C.evaluate(coco, d)
        s = stored[rid][a.split]
        delta = m["AP"] - s
        ok = (delta == 0.0) if a.mode == "cached" else abs(delta) <= TOL_FULL
        n_fail += not ok
        o = _out() / a.mode / a.split / rid
        o.mkdir(parents=True, exist_ok=True)
        d.to_parquet(o / "detections.parquet")
        (o / "metrics.json").write_text(json.dumps({"row": rid, "split": a.split, "mode": a.mode, "AP": m["AP"],
                                                    "AP50": m["AP50"], "stored_AP": s, "delta": delta,
                                                    "rule": "exact" if a.mode == "cached" else f"|delta| <= {TOL_FULL}",
                                                    "passed": ok}, indent=1))
        rule = "exact" if a.mode == "cached" else f"|delta| <= {TOL_FULL}"
        print(f"{rid}: AP {m['AP']:.5f}, stored {s:.5f}, delta {delta:+.5f} -> {'PASS' if ok else 'FAIL'} ({rule}); "
              f"{time.time() - t0:.0f} s - {C.describe(r)}", flush=True)
    print(f"{len(ids) - n_fail}/{len(ids)} rows passed ({C.MODEL}, {a.mode} mode)")


def cmd_predict(a) -> None:
    r = C.rows()[a.row]
    ims = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    imgs = pd.DataFrame({"image_id": range(1, len(ims) + 1), "stem": [p.stem for p in ims], "path": [str(p) for p in ims]})
    w = C.weights(_size(r), getattr(a, f"weights_{_size(r)}"))
    d = detections(r, imgs, f"own_{Path(a.images).resolve().name}", w)  # always new inference (your images)
    from scahi.config import load_class_setting

    names = dict(enumerate(load_class_setting(C.SETTING).classes, start=1))
    d = d.merge(imgs[["image_id", "stem"]], on="image_id")
    d["class"] = d.category_id.map(names)
    d[["stem", "class", "score", "x", "y", "w", "h"]].to_csv(a.out, index=False)
    print(f"{len(d)} detections on {len(imgs)} images -> {a.out} ({C.describe(r)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rows")
    p.add_argument("--model", choices=tuple(C.MODELS), default="yolo26m")
    for name in ("run", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--model", choices=tuple(C.MODELS), default="yolo26m")
        p.add_argument("--row", required=True)
        p.add_argument("--weights-1280")
        p.add_argument("--weights-2048")
        if name == "run":
            p.add_argument("--split", choices=("val", "test-dev"), default="val")
            p.add_argument("--mode", choices=("cached", "full"), default="cached")
        else:
            p.add_argument("--images", required=True)
            p.add_argument("--out", default="detections.csv")
    a = ap.parse_args()
    C.set_model(a.model)
    {"rows": cmd_rows, "run": cmd_run, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
