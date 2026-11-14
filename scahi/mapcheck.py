# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mapping checks (user, 2026-11-02; DECISIONS 10:15 / 10:29): caches with Ultralytics-aligned input preparation.

All three details aligned to Ultralytics' predictor at once (the 0.05 rule applies to the combined effect):
(a) size rule - LetterBox rounds the resized sides first, then pads each to a multiple of 32 (auto), instead of
    ``preprocess.full_image_shape`` / ``cache_mz.level_passes`` padding the unrounded side up;
(b) letterbox gain - resize by r = min(L / h, L / w) for the pass's target size L, instead of the gain to the padded
    input shape (``preprocess.letterbox``);
(c) back-mapping - per axis as ``scale_boxes`` (``scahi.dota.sahi_scahi.to_source_ultralytics``) instead of one gain.
Everything else as the frozen caches (``load_unfused``, ``cm.cache_level`` with batch 16, TF32 default). Written to a
separate root (``cache_ualigned/``); no stored cache, row or value changes. Built now on cores 0-1 (low priority, own
memory cap); evaluated only after the FCOS tuning.

    python -m scahi.mapcheck caches dota       # YOLOv10s 2048 (DOTA), levels w2048 + t1.0n1024, calib
    python -m scahi.mapcheck caches visdrone   # YOLOv10s 2048 (VisDrone), levels w2048 + t2.0n1024, calib
    python -m scahi.mapcheck evaluate dota     # calib: RG / NMS rows from the frozen and the aligned caches
    python -m scahi.mapcheck evaluate visdrone # calib: XG_yolov10s_2048_o2m_C / X1, same

Evaluation (descriptive; decision rule DECISIONS 2026-11-02 10:15 / 10:29): each row is fused twice through the same
code path - from the frozen caches and from ``cache_ualigned/`` (only the cache root differs) - and
delta = aligned - frozen; the frozen-cache value must reproduce the stored calib value. Output runs/mapcheck/<ds>.json.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time

import numpy as np

from scahi.machine import REPO_ROOT

ROOT = REPO_ROOT / "cache_ualigned"
STATE = {"model": "unfused (load_unfused), as the frozen caches", "batch": 16, "tf32": "torch default, as the frozen caches",
         "size_rule": "Ultralytics LetterBox auto: round, then pad each side to a multiple of 32",
         "letterbox_gain": "r = min(L / h, L / w)", "back_mapping": "scale_boxes per axis (float32)"}
_R: dict = {}  # (h, w, out_hw) -> r of the pass (set by the patched level_passes, read by the patched letterbox)


def _shape(h: int, w: int, L: int) -> tuple[int, int]:
    r = L / max(h, w)
    nh, nw = round(h * r), round(w * r)
    out = (nh + (L - nh) % 32, nw + (L - nw) % 32)
    _R[(h, w, out)] = r
    return out


def level_passes_ultralytics(level, W: int, H: int, overlap: float) -> list:
    from scahi.data.zoom import tile_grid
    from scahi.infer import cache_mz as cm

    _R.clear()  # one level of one image at a time (raw_predict_level: level_passes, then letterbox per pass)
    if level.n_input is None:
        return [((0, 0, W, H), _shape(H, W, level.imgsz or cm._pad32(level.zoom * max(W, H))))]
    return [((t.x1, t.y1, t.x2, t.y2), _shape(t.y2 - t.y1, t.x2 - t.x1, level.n_input))
            for t in tile_grid(W, H, level.zoom, level.n_input, overlap)]


def letterbox_ultralytics(img: np.ndarray, out_hw: tuple[int, int]):
    import cv2

    from scahi.infer.preprocess import PAD_VALUE, Letterbox

    h, w = img.shape[:2]
    r = _R[(h, w, tuple(out_hw))]
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else img
    top, left = round((out_hw[0] - nh) / 2 - 0.1), round((out_hw[1] - nw) / 2 - 0.1)
    canvas = np.full((out_hw[0], out_hw[1], 3), PAD_VALUE, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, Letterbox(r, out_hw, (h, w), (left, top))


@contextlib.contextmanager
def ultralytics_prep():
    from scahi.dota.sahi_scahi import to_source_ultralytics
    from scahi.infer import cache_mz as cm

    orig = cm.level_passes, cm.letterbox, cm.input_to_source
    cm.level_passes, cm.letterbox, cm.input_to_source = level_passes_ultralytics, letterbox_ultralytics, to_source_ultralytics
    try:
        yield
    finally:
        cm.level_passes, cm.letterbox, cm.input_to_source = orig


_ORIG: dict = {}
DETAILS = ("size_rule", "letterbox_gain", "back_mapping")


def _lp_gain(level, W: int, H: int, overlap: float) -> list:
    """Frozen pass shapes, but register r = L / max(h, w) so that ``letterbox_ultralytics`` resizes with it."""
    from scahi.infer import cache_mz as cm

    _R.clear()
    out = _ORIG["level_passes"](level, W, H, overlap)
    for (x1, y1, x2, y2), sh in out:
        L = level.n_input if level.n_input is not None else (level.imgsz or cm._pad32(level.zoom * max(W, H)))
        _R[(y2 - y1, x2 - x1, tuple(sh))] = L / max(y2 - y1, x2 - x1)
    return out


@contextlib.contextmanager
def one_detail(detail: str):
    """One of the three details aligned to Ultralytics, the other two as the frozen caches (per-detail report of the
    10:29 rule): size_rule, letterbox_gain or back_mapping."""
    from scahi.dota.sahi_scahi import to_source_ultralytics
    from scahi.infer import cache_mz as cm

    _ORIG.update(level_passes=cm.level_passes, letterbox=cm.letterbox, input_to_source=cm.input_to_source)
    if detail == "size_rule":
        cm.level_passes = level_passes_ultralytics  # rounded-then-padded shapes; frozen letterbox and mapping
    elif detail == "letterbox_gain":
        cm.level_passes, cm.letterbox = _lp_gain, letterbox_ultralytics
    elif detail == "back_mapping":
        cm.input_to_source = to_source_ultralytics
    else:
        raise ValueError(detail)
    try:
        yield
    finally:
        cm.level_passes, cm.letterbox, cm.input_to_source = (_ORIG[k] for k in ("level_passes", "letterbox",
                                                                                 "input_to_source"))


def _jobs(ds: str):
    """(weights, images of calib, level keys, overlap) of the checked rows."""
    if ds == "dota":
        from scahi.dota import pipeline as P
        from scahi.dota import train as T

        return T.weights("yolov10s", 2048), P.images("calib"), ("w2048", P.TILES), P.OVERLAP
    from scahi import ext as E

    _, imgs = E._split_images("calib")
    return E.weights("yolov10s", 2048), imgs, ("w2048", "t2.0n1024"), E.OVERLAP


def caches(ds: str) -> None:
    from scahi.infer import cache_mz as cm
    from scahi.infer.raw_head import load_unfused

    w, imgs, lks, ov = _jobs(ds)
    m = load_unfused(w)
    with ultralytics_prep():
        for lk in lks:
            t0 = time.time()
            d = cm.cache_level(m, w, imgs, lk, ov, "calib", ROOT)
            (d / "cache_state.json").write_text(json.dumps({**STATE, "weights": str(w), "level": lk}, indent=1))
            print(f"[mapcheck] {ds} {lk}: {len(imgs)} images, {time.time() - t0:.0f} s -> {d}", flush=True)


OUT = REPO_ROOT / "runs/mapcheck"
RULE = 0.05  # |delta| below for both rows of a dataset: supplement; otherwise bug fix (DECISIONS 2026-11-02 10:15)


@contextlib.contextmanager
def cache_root(root):
    from scahi.fusion import run as R

    saved = R.CACHE_ROOT
    R.CACHE_ROOT = root
    try:
        yield
    finally:
        R.CACHE_ROOT = saved


def _dota() -> dict:
    """RG_yolov10s_2048 (tiles + SCAHI, refined grid) and its box-fusion counterparts NMS-o2m / NMS-o2o; mAP50 at the
    pipeline limit of the stored calib values (final.row_dets, as the replay)."""
    import torch

    from scahi.dota import evaluate as EV
    from scahi.dota import final as F
    from scahi.dota import train as T
    from scahi.fusion import run as R

    torch.set_num_threads(1)
    gt = EV.gt_arrays("train", T.split_stems("calib"))
    rows = {r["id"]: r for r in F.proposal_rows(("cache",), ("yolov10s",))}
    out = []
    for rid in ("RG_yolov10s_2048", "NMS-o2m_yolov10s_2048", "NMS-o2o_yolov10s_2048"):
        r, v = rows[rid], {}
        for tag, root in (("frozen", R.CACHE_ROOT), ("aligned", ROOT)):
            t0 = time.time()
            with cache_root(root):
                dets, info = F.row_dets(r, "calib")
            v[tag] = {"mAP50": EV.map50(EV.build(gt, dets, None, sizes=False)), "n_dets": len(dets),
                      "images_at_limit": info["images_at_limit"], "wall_s": time.time() - t0}
        out.append({"id": rid, "metric": "mAP50", "stored_calib": r["calib_mAP50"], **v,
                    "frozen_reproduces_stored": v["frozen"]["mAP50"] == r["calib_mAP50"],
                    "delta": v["aligned"]["mAP50"] - v["frozen"]["mAP50"]})
        print(f"[mapcheck] dota {rid}: frozen {v['frozen']['mAP50']:.5f} (stored {r['calib_mAP50']:.5f}), aligned "
              f"{v['aligned']['mAP50']:.5f}, delta {out[-1]['delta']:+.5f}", flush=True)
    return {"rows": out, "rule_rows": [x["id"] for x in out]}


def _visdrone() -> dict:
    """XG_yolov10s_2048_o2m_C (tiles + SCAHI, refined grid; configs/ext/frozen_rows_ext.json) and X1 (tiles + box
    fusion F0-NMS o2o; configs/final/frozen_rows.json); AP@[.5:.95] (headline), AP50, AP / AP50 of objects < 16 px,
    through the extension cache-row path (fuse_split + write_eval, toolkit port) into runs/mapcheck/visdrone/."""
    from scahi import ext as E
    from scahi import ext_testdev as TD
    from scahi.fusion import run as R
    from scahi.fusion import scalespace as ss

    xg = {r["id"]: r for r in TD.freeze()["rows"]}["XG_yolov10s_2048_o2m_C"]
    x1 = {r["id"]: r for r in json.loads((REPO_ROOT / "configs/final/frozen_rows.json").read_text())["rows"]
          if isinstance(r, dict) and "id" in r}["X1"]
    f1 = x1["fusion"]
    jobs = [("XG_yolov10s_2048_o2m_C", xg["fusion"], "calib AP of the X-grid selection (configs/ext/xgrid_frozen.json)",
             json.loads((REPO_ROOT / "configs/ext/xgrid_frozen.json").read_text())["weight_sets"]["yolov10s/2048/o2m/C"]["calib_AP"]),
            ("X1", {"levels": x1["inference"]["levels"], "overlap": f1["overlap"], "branch": f1["branch"],
                    "variant": f1["variant"], "params": f1["params"]}, "inference.calib_AP (configs/final)",
             x1["inference"]["calib_AP"])]
    coco, imgs = E._split_images("calib")
    w = E.weights("yolov10s", 2048)
    out = []
    for rid, f, src, stored in jobs:
        v = {}
        for tag, root in (("frozen", R.CACHE_ROOT), ("aligned", ROOT)):
            t0 = time.time()
            with cache_root(root):
                dirs = TD._dirs(w, f["levels"], f["overlap"], "calib")
            data = {s: ss.load_levels(dirs, s, f["branch"], f["overlap"]) for s in imgs.stem}
            d = R.fuse_split(data, dict(zip(imgs.stem, imgs.image_id)), f["branch"], f["variant"], f["params"])
            del data
            exp = f"{rid}_{tag}"
            E.write_eval(exp, d, E.SETTING, coco, "calib", {"model": "yolov10s", "weights": str(w), "train_variant": "BFULL2048", "inference": f,
                                                            "notes": "mapping check (descriptive, calib)"},
                         root=OUT / "visdrone", phase=9)
            m = json.loads((OUT / "visdrone" / exp / "metrics.json").read_text())["metrics"]
            v[tag] = {"AP": m["AP"], "AP50": m["AP50"], "AP_lt16": m["AP_by_size"]["<16"],
                      "AP50_lt16": m["AP50_by_size"]["<16"], "n_dets": len(d), "wall_s": time.time() - t0}
        out.append({"id": rid, "metric": "AP", "stored_calib": stored, "stored_source": src, **v,
                    "frozen_reproduces_stored": v["frozen"]["AP"] == stored,
                    "delta": v["aligned"]["AP"] - v["frozen"]["AP"],
                    "delta_secondary": {k: v["aligned"][k] - v["frozen"][k] for k in ("AP50", "AP_lt16", "AP50_lt16")}})
        print(f"[mapcheck] visdrone {rid}: frozen AP {v['frozen']['AP']:.5f} (stored {stored:.5f}), aligned "
              f"{v['aligned']['AP']:.5f}, delta {out[-1]['delta']:+.5f}, {out[-1]['delta_secondary']}", flush=True)
    return {"rows": out, "rule_rows": [x["id"] for x in out]}


def evaluate(ds: str) -> dict:
    r = {"dota": _dota, "visdrone": _visdrone}[ds]()
    r.update(dataset=ds, split="calib", rule_threshold=RULE, cache_state=STATE, time=time.strftime("%F %T"),
             max_abs_delta=max(abs(x["delta"]) for x in r["rows"]),
             below_rule=all(abs(x["delta"]) < RULE for x in r["rows"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{ds}.json").write_text(json.dumps(r, indent=1))
    print(f"[mapcheck] {ds}: max |delta| {r['max_abs_delta']:.5f} -> {'below' if r['below_rule'] else 'AT/ABOVE'} "
          f"{RULE}", flush=True)
    return r


if __name__ == "__main__":
    {"caches": caches, "evaluate": evaluate}[sys.argv[1]](sys.argv[2])
