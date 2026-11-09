# SPDX-License-Identifier: AGPL-3.0-or-later
"""Breakdown of the D-H5 pass times (user, 2026-11-03; profiling only, never reported as latency): where SAHI's passes
(``latency_host_a_dota.sahi``) spend the time they take beyond the SCAHI path's passes (``latency_host_a_dh5.sahi_scahi``)
on the same slice grid, the same fused weights and batch 1. Both paths run on every DOTA-val image with the functions of
the latency run, instrumented by wrappers (cuda-synchronised where the GPU works); per image the component times and
the uninstrumented-equivalent total. Run next to the FCOS tuning (cores 0-1, GPU otherwise idle), so absolute times are
above the exclusive run; the shares and the difference between the paths are what this run is for.

    python scripts/latency_dh5_profile.py [n_images]   -> runs/latency_host_a/dh5_profile.json
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from latency_host_a_dh5 import setup  # noqa: E402
from latency_host_a_dota import OUTD  # noqa: E402

ACC: dict[str, float] = defaultdict(float)


def _sync() -> None:
    import torch

    torch.cuda.synchronize()


def timed(key: str, fn, sync: bool = False):
    def w(*a, **k):
        if sync:
            _sync()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        if sync:
            _sync()
        ACC[key] += 1000 * (time.perf_counter() - t0)
        return r
    return w


def sahi_path(d: dict, imgs, n_warmup: int = 2) -> list[dict]:
    from sahi import predict as sp
    from sahi.predict import get_sliced_prediction

    from scahi.dota import train as DT
    from scahi.infer import sahi_baseline as sb

    c = d["SAHI"]["cfg"]
    sb._register_identity()
    dm = sb.load_model(DT.weights(d["model"], 1024), min(sb.GRID2_WBF_SKIP), "o2o")
    dm.model.overrides["max_det"] = d["limit"]
    sp.slice_image = timed("sahi: read image (PIL) + cut slices", sp.slice_image)
    sp.read_image_as_pil = timed("sahi: standard pass image conversion", sp.read_image_as_pil)
    pbi = dm.perform_batch_inference

    def batch(images):
        ACC["slices_and_passes"] += len(images)
        _sync()
        t0 = time.perf_counter()
        pbi(images)
        _sync()
        ACC["ultralytics call (total)"] += 1000 * (time.perf_counter() - t0)
        for r in dm.model.predictor.results:
            for k, v in r.speed.items():
                ACC[f"ultralytics {k}"] += v

    dm.perform_batch_inference = batch
    dm.convert_original_predictions = timed("sahi: ObjectPrediction creation", dm.convert_original_predictions)

    def one(path: str) -> dict:
        ACC.clear()
        _sync()
        t0 = time.perf_counter()
        r = get_sliced_prediction(path, dm, slice_height=sb.SLICE, slice_width=sb.SLICE, overlap_height_ratio=c["overlap"],
                                  overlap_width_ratio=c["overlap"], perform_standard_pred=True, postprocess_type="IDENTITY",
                                  postprocess_match_threshold=0.5, postprocess_match_metric="IOS",
                                  force_postprocess_type=True, verbose=0)
        t1 = time.perf_counter()
        np.array([[*p.bbox.to_xyxy(), p.score.value, p.category.id] for p in r.object_prediction_list],
                 np.float64).reshape(-1, 6)
        _sync()
        t2 = time.perf_counter()
        return {**ACC, "objects -> array": 1000 * (t2 - t1), "total": 1000 * (t2 - t0),
                "n_dets": len(r.object_prediction_list)}

    for pa in imgs.path[:n_warmup]:
        one(pa)
    out = [one(pa) for pa in imgs.path]
    del dm
    return out


def scahi_path(d: dict, imgs, n_warmup: int = 2) -> list[dict]:
    import cv2
    import torch

    from scahi.dota import sahi_scahi as S
    from scahi.fusion import scalespace as ss
    from scahi.infer import cache_mz as cm

    m, ov = d["model"], S.overlap(d["model"])
    model = S.load_fused(m)

    def one(path: str) -> dict:
        ACC.clear()
        _sync()
        t0 = time.perf_counter()
        t = time.perf_counter()
        img = cv2.imread(path)
        ACC["read image (cv2)"] += 1000 * (time.perf_counter() - t)
        H, W = img.shape[:2]
        n_sl = len(S.sahi_passes(S.LEVELS["s640f"], W, H, ov))
        lv = []
        for k in (S.ORDER if n_sl > 1 else S.ORDER[:1]):
            z = cm.raw_predict_level(model, img, S.LEVELS[k], ov, batch=1)
            ACC["slices_and_passes"] += len(z["tiles"])
            ACC["n_dets"] += len(z["o2m_xyxy"])
            lv.append(ss.CachedLevel(k, z["o2m_xyxy"], z["o2m_probs"], z["o2m_std"], z["o2m_tile"], z["tiles"],
                                     z["gains"], bool(int(z["n_input"]) == 0), ov, ss.LevelCal()))
        _sync()
        return {**ACC, "total": 1000 * (time.perf_counter() - t0)}

    with S._sahi_grid():
        orig = cm.letterbox, cm.to_tensor, cm.forward_raw, cm.decode, cm.input_to_source
        cm.letterbox = timed("crop + letterbox (cv2)", cm.letterbox)
        cm.to_tensor = timed("to tensor + host->GPU", cm.to_tensor, sync=True)
        cm.forward_raw = timed("forward (fused model)", cm.forward_raw, sync=True)
        cm.decode = timed("decode (both heads)", cm.decode, sync=True)
        cm.input_to_source = timed("map boxes to source", cm.input_to_source)
        try:
            for pa in imgs.path[:n_warmup]:
                one(pa)
            out = [one(pa) for pa in imgs.path]
        finally:
            cm.letterbox, cm.to_tensor, cm.forward_raw, cm.decode, cm.input_to_source = orig
    del model
    torch.cuda.empty_cache()
    return out


def summary(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r})
    return {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in keys}


def main() -> None:
    from scahi.dota import pipeline as P

    imgs = P.images("val")
    if len(sys.argv) > 1:
        imgs = imgs.head(int(sys.argv[1]))
    res = {"started": dt.datetime.now().isoformat(timespec="seconds"), "n_images": len(imgs),
           "conditions": "profiling only, not a latency measurement: cores 0-1 next to the FCOS CPU tuning, GPU otherwise "
                         "idle, wrappers synchronise CUDA; compare shares, not absolute values with the exclusive run",
           "models": {}}
    for d in setup():
        res["models"][d["model"]] = {"sahi": summary(sahi_path(d, imgs)), "scahi_path": summary(scahi_path(d, imgs))}
        print(json.dumps(res["models"][d["model"]], indent=1), flush=True)
    res["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    out = OUTD / ("dh5_profile.json" if len(sys.argv) == 1 else "dh5_profile_smoke.json")
    out.write_text(json.dumps(res, indent=1))
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
