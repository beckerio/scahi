# SPDX-License-Identifier: AGPL-3.0-or-later
"""Track A: use the frozen YOLO26m weights and parameters.

    python tutorials/yolo26m_visdrone/frozen.py rows                               # the 25 frozen configurations
    python tutorials/yolo26m_visdrone/frozen.py run --row ID --split val           # recompute one row, compare
    python tutorials/yolo26m_visdrone/frozen.py run --row all --split val          # every row (GPU for the passes)
    python tutorials/yolo26m_visdrone/frozen.py predict --row ID --images DIR --out dets.csv   # your own images

``run`` recomputes a frozen row exactly as in the single held-out evaluation (weights checked by SHA-256, parameters
from configs/ext/frozen_rows_ext26.json, caches with Ultralytics' input preparation) and prints its AP next to the
stored value (tutorials/yolo26m_visdrone/frozen_results.json); identical up to the last digit on the same software
stack (pinned versions, requirements.lock). Outputs under out/yolo26m/<split>/<row>/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import common as C

OUT = C.REPO / "out/yolo26m"


def _size(r: dict) -> int:
    return int(r["weights"].split("yolo26m_")[1].split("_")[0])


def detections(r: dict, imgs: pd.DataFrame, split: str, w: Path) -> pd.DataFrame:
    """Detections of one frozen row on an image table (columns image_id, stem, path)."""
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
    with C.ultralytics_caches() as root:
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
    for rid in ids:
        r = rs[rid]
        w = C.weights(_size(r), getattr(a, f"weights_{_size(r)}"))
        t0 = time.time()
        d = detections(r, imgs, a.split, w)
        m = C.evaluate(coco, d)
        o = OUT / a.split / rid
        o.mkdir(parents=True, exist_ok=True)
        d.to_parquet(o / "detections.parquet")
        (o / "metrics.json").write_text(json.dumps({"row": rid, "split": a.split, "AP": m["AP"], "AP50": m["AP50"],
                                                    "stored_AP": stored[rid].get(a.split)}, indent=1))
        s = stored[rid].get(a.split)
        same = "identical" if s is not None and m["AP"] == s else f"stored {s}"
        print(f"{rid}: AP {m['AP']:.5f} ({same}); {time.time() - t0:.0f} s - {C.describe(r)}", flush=True)


def cmd_predict(a) -> None:
    r = C.rows()[a.row]
    ims = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    imgs = pd.DataFrame({"image_id": range(1, len(ims) + 1), "stem": [p.stem for p in ims], "path": [str(p) for p in ims]})
    w = C.weights(_size(r), getattr(a, f"weights_{_size(r)}"))
    d = detections(r, imgs, f"own_{Path(a.images).resolve().name}", w)
    from scahi.config import load_class_setting

    names = dict(enumerate(load_class_setting(C.SETTING).classes, start=1))
    d = d.merge(imgs[["image_id", "stem"]], on="image_id")
    d["class"] = d.category_id.map(names)
    d[["stem", "class", "score", "x", "y", "w", "h"]].to_csv(a.out, index=False)
    print(f"{len(d)} detections on {len(imgs)} images -> {a.out} ({C.describe(r)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rows")
    for name in ("run", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--row", required=True)
        p.add_argument("--weights-1280")
        p.add_argument("--weights-2048")
        if name == "run":
            p.add_argument("--split", choices=("val", "test-dev"), default="val")
        else:
            p.add_argument("--images", required=True)
            p.add_argument("--out", default="detections.csv")
    a = ap.parse_args()
    {"rows": cmd_rows, "run": cmd_run, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
