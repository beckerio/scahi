# SPDX-License-Identifier: AGPL-3.0-or-later
"""Back-mapping check of ALL previously frozen cache rows (authors, 2026-11-14; rule DECISIONS 2026-11-02 10:15 / 10:29).

    python scripts/mapcheck_all.py list                       # inventory -> runs/mapcheck/all/inventory.json
    python scripts/mapcheck_all.py caches [--batch main|all]  # GPU: calib caches, all three details aligned
    python scripts/mapcheck_all.py check  [--batch main|all] [--workers 6]
    python scripts/mapcheck_all.py details [--batch ...]      # GPU: one detail at a time, rows with |delta| >= 0.05
    python scripts/mapcheck_all.py check --details ...        # CPU: per-detail values of those rows
    python scripts/mapcheck_all.py serial ID [ID ...]         # the same rows in this one process (no pool): must equal
                                                              # the parallel values exactly -> runs/mapcheck/all/serial.json

Rows: every frozen row whose detections rest on a cached whole-image level - VisDrone main study
(configs/final/frozen_rows.json, kind "fusion"), VisDrone extension (configs/ext/frozen_rows_ext.json, kind "cache"),
DOTA (configs/dota/frozen_rows{,_new}.json, kind "cache", YOLO detectors). Not affected: single passes, B-MZ and SAHI
rows (Ultralytics predictor, its own mapping); DOTA SAHI + SCAHI rows (D-H5 / A5: caches s640f / w640f are already
built with Ultralytics' preparation, DECISIONS 2026-11-02 10:15); FCOS cache rows (torchvision resize, per-axis
back-mapping of torchvision's own predictor, ``scahi.fcos.dense_pass``; no Ultralytics convention involved).
Covered by the original check (2026-11-03, runs/mapcheck/{visdrone,dota}.json): X1, XG_yolov10s_2048_o2m_C,
RG / NMS-o2m / NMS-o2o_yolov10s_2048 (DOTA); not recomputed here.

Each row is fused with its frozen calib configuration twice through the same code path as its replay - from the
frozen caches and from ``cache_ualigned/`` (only the cache root differs; VisDrone: ``ext_testdev._dirs`` +
``fuse_split``, toolkit-port AP; DOTA: ``scahi.dota.final.row_dets``, mAP50). delta = aligned - frozen. No re-tuning.
Batch "main": the rows behind numbers of the main paper (Tab. 1-3, Fig. decomposition, text); "all": every row.
Output runs/mapcheck/all/check.json. Calib only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scahi import mapcheck as MC  # noqa: E402

OUT = REPO / "runs/mapcheck/all"
UROOT = MC.ROOT
DROOT = REPO / "cache_ualigned_detail"
DETAILS = MC.DETAILS
RULE = MC.RULE
COVERED = {"X1": "runs/mapcheck/visdrone.json", "XG_yolov10s_2048_o2m_C": "runs/mapcheck/visdrone.json",
           "RG_yolov10s_2048": "runs/mapcheck/dota.json", "NMS-o2m_yolov10s_2048": "runs/mapcheck/dota.json",
           "NMS-o2o_yolov10s_2048": "runs/mapcheck/dota.json"}


def _main_paper(ds: str, r: dict) -> list[str]:
    """Where a row's number enters the main paper (empty: supplement / not reported in the main paper)."""
    i = r["id"]
    if ds == "dota":
        return ["Tab. 3 / text (DOTA)"]
    v10 = "_yolov10" in i
    out = []
    if i == "R-NMS":
        out.append("Tab. 1 YOLOv10s '1280 + tiles, box' and dQ1 (if R-NMS, not R, is the source: both 35.00 val)")
    if i == "X2":
        out += ["Tab. 1 YOLOv10s '2048 + tiles, SCAHI'", "Tab. 2 '2048 + tiles, SCAHI'", "Fig. decomposition '+ tiles'"]
    if i.startswith("E_") and "_mz1280_nms-ios_" in i and i.endswith("_o2o" if v10 else "_o2m"):
        out.append("Tab. 1 '1280 + tiles, box' and dQ1")
    if i.startswith("E_") and "_mz2048_F2_o2m" in i:
        out += ["Tab. 1 '2048 + tiles, SCAHI'", "Fig. decomposition '+ tiles'", "text Q2 / Q3"]
    if i.startswith("E_") and ("_mz1280_F2_o2m" in i or "_mz1280_nms-ios_o2m" in i or "_mz2048_nms-ios_o2m" in i):
        out.append("text Q2 (SCAHI vs. box fusion, range over detectors)")
    if i.startswith("XG_") and i.endswith("_2048_o2m_C"):
        out += ["Tab. 1 'refined'", "text (refined grid gain, tiles vs. single pass)"]
    if i.startswith("D_") and "_single2048_F2_o2m" in i:
        out.append("Fig. decomposition 'single pass + SCAHI' and text")
    return out


def rows_all() -> list[dict]:
    out = []
    for r in json.loads((REPO / "configs/final/frozen_rows.json").read_text())["rows"]:
        if not isinstance(r, dict) or r.get("kind") != "fusion":
            continue
        inf = r["inference"]
        out.append({"ds": "visdrone", "study": "main", "id": r["id"], "group": r["group"], "weights": r["weights"],
                    "spec": {"levels": inf["levels"], "overlap": inf["overlap"], "branch": inf["branch"],
                             "variant": inf["fusion"], "params": inf["params"]},
                    "stored_calib": inf.get("calib_AP"), "row": r})
    for r in json.loads((REPO / "configs/ext/frozen_rows_ext.json").read_text())["rows"]:
        if r["kind"] != "cache":
            continue
        f = r["fusion"]
        out.append({"ds": "visdrone", "study": "ext", "id": r["id"], "group": r["group"], "weights": r["weights"],
                    "spec": {k: f[k] for k in ("levels", "overlap", "branch", "variant", "params")},
                    "stored_calib": r["inference"].get("calib_AP"), "row": r})
    from scahi.dota import pipeline as P

    for p in ("configs/dota/frozen_rows.json", "configs/dota/frozen_rows_new.json"):
        for r in json.loads((REPO / p).read_text())["rows"]:
            if r["kind"] != "cache":
                continue
            lv = [P.WHOLE[r["s"]], P.TILES] if r["stack"] == "C" else [P.WHOLE[r["s"]]]
            out.append({"ds": "dota", "study": p.split("/")[-1], "id": r["id"], "group": r["row"],
                        "weights": r["weights"], "spec": {"levels": lv, "overlap": P.OVERLAP},
                        "stored_calib": r["calib_mAP50"], "row": r})
    for x in out:
        x["covered_by_original_check"] = COVERED.get(x["id"])
        x["main_paper"] = _main_paper(x["ds"], x["row"])
    return out


def select(batch: str) -> list[dict]:
    rs = [x for x in rows_all() if not x["covered_by_original_check"]]
    return [x for x in rs if x["main_paper"]] if batch == "main" else rs


def inventory() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rs = rows_all()
    inv = [{k: x[k] for k in ("ds", "study", "id", "group", "weights", "spec", "stored_calib",
                              "covered_by_original_check", "main_paper")} for x in rs]
    (OUT / "inventory.json").write_text(json.dumps({
        "not_affected": {
            "single passes, B-MZ (R, R5-R7), SAHI (VisDrone and DOTA)": "Ultralytics predictor (scale_boxes)",
            "DOTA SAHI + SCAHI (D-H5 / A5)": "caches s640f / w640f already built with Ultralytics' preparation",
            "DOTA FCOS cache rows": "torchvision resize + per-axis back-mapping (scahi.fcos.dense_pass)"},
        "n_rows": len(inv), "n_covered": sum(bool(x["covered_by_original_check"]) for x in inv),
        "n_main_paper": sum(bool(x["main_paper"]) for x in inv), "rows": inv}, indent=1))
    print(f"{len(inv)} rows, {sum(bool(x['covered_by_original_check']) for x in inv)} covered, "
          f"{sum(bool(x['main_paper']) for x in inv)} main paper")


def _images(ds: str):
    if ds == "dota":
        from scahi.dota import pipeline as P

        return P.images("calib")
    from scahi import ext as E

    return E._split_images("calib")[1]


def _same_npz(a: Path, b: Path) -> bool:
    za, zb = np.load(a), np.load(b)
    return sorted(za.files) == sorted(zb.files) and all(np.array_equal(za[k], zb[k]) for k in za.files)


def _jobs(rs: list[dict]) -> dict:
    """(ds, weights) -> sorted levels needed."""
    j: dict = {}
    for x in rs:
        j.setdefault((x["ds"], x["weights"], x["spec"]["overlap"]), set()).update(x["spec"]["levels"])
    return {k: sorted(v, key=lambda s: (not s.startswith("w"), s)) for k, v in sorted(j.items())}


def caches(batch: str) -> None:
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm
    from scahi.infer.raw_head import load_unfused

    OUT.mkdir(parents=True, exist_ok=True)
    rf = OUT / "caches.json"
    rep = json.loads(rf.read_text()) if rf.exists() else {}
    for (ds, wr, ov), lks in _jobs(select(batch)).items():
        w = REPO / wr
        imgs = _images(ds)
        m = None
        for lk in lks:
            key = f"{ds}/{wr}/{lk}"
            if key in rep:
                continue
            m = m or load_unfused(w)
            t0 = time.time()
            with MC.ultralytics_prep():
                d = cm.cache_level(m, w, imgs, lk, ov, "calib", UROOT)
            (d / "cache_state.json").write_text(json.dumps({**MC.STATE, "weights": str(w), "level": lk}, indent=1))
            fz = cm.cache_dir(CACHE_ROOT, w, lk, ov, "calib")
            same = sum(_same_npz(fz / f"{s}.npz", d / f"{s}.npz") for s in imgs.stem)
            rep[key] = {"images": len(imgs), "bit_identical_to_frozen": int(same), "build_s": round(time.time() - t0)}
            rf.write_text(json.dumps(rep, indent=1))
            print(f"[mapall] {key}: {rep[key]}", flush=True)
        del m


def details(batch: str) -> None:
    """Per-detail whole-image caches (and tile levels where the aligned tiles differ) for rows with |delta| >= 0.05."""
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm
    from scahi.infer.raw_head import load_unfused

    res = json.loads((OUT / "check.json").read_text())["rows"]
    cs = json.loads((OUT / "caches.json").read_text())
    big = [x for x in select(batch) if res.get(x["id"], {}).get("at_or_above_rule")]
    for (ds, wr, ov), lks in _jobs(big).items():
        w = REPO / wr
        imgs = _images(ds)
        m = load_unfused(w)
        for det in DETAILS:
            root = DROOT / det
            with MC.one_detail(det):
                for lk in lks:
                    d = cm.cache_dir(root, w, lk, ov, "calib")
                    if cs[f"{ds}/{wr}/{lk}"]["bit_identical_to_frozen"] == len(imgs):
                        if not d.exists():
                            d.parent.mkdir(parents=True, exist_ok=True)
                            d.symlink_to(cm.cache_dir(CACHE_ROOT, w, lk, ov, "calib"), target_is_directory=True)
                        continue
                    t0 = time.time()
                    cm.cache_level(m, w, imgs, lk, ov, "calib", root)
                    (d / "cache_state.json").write_text(json.dumps({"aligned_detail": det, "others": "frozen",
                                                                    "weights": str(w), "level": lk}, indent=1))
                    print(f"[mapall] detail {det} {ds} {wr} {lk}: {time.time() - t0:.0f} s", flush=True)
        del m


def _fuse(args) -> tuple[str, str, dict]:
    import torch

    from scahi.fusion import run as R

    torch.set_num_threads(1)
    x, tag, root = args
    R.CACHE_ROOT = Path(root)
    t0 = time.time()
    if x["ds"] == "dota":
        from scahi.dota import evaluate as EV
        from scahi.dota import final as F
        from scahi.dota import train as T

        gt = EV.gt_arrays("train", T.split_stems("calib"))
        dets, info = F.row_dets(x["row"], "calib")
        return x["id"], tag, {"metric": "mAP50", "value": EV.map50(EV.build(gt, dets, None, sizes=False)),
                              "n_dets": len(dets), "images_at_limit": info.get("images_at_limit"),
                              "wall_s": time.time() - t0}
    from scahi import ext as E
    from scahi import ext_testdev as TD
    from scahi.config import load_class_setting
    from scahi.eval.evaluate import evaluate
    from scahi.fusion import scalespace as ss

    f = x["spec"]
    coco, imgs = E._split_images("calib")
    if x["study"] == "main" and x["row"]["fusion"]["variant"] in ("F3e", "F4", "F5"):
        # main study's own replay path (scahi.testdev.run_fusion_row): per-level calib estimates, always taken from
        # the frozen caches (the file of Phase 5, or for seed rows re-estimated on the frozen calib caches)
        from scahi import testdev as TDM

        R.CACHE_ROOT = TDM.REPO_ROOT / "cache"
        cal = R.cal_from_json(TDM._cal(x["row"]))
        R.CACHE_ROOT = Path(root)
        fr = x["row"]["fusion"]
        data = R.load_split(REPO / x["weights"], fr["stack"], "calib", fr["overlap"], fr["branch"], list(imgs.stem), cal)
    else:
        dirs = TD._dirs(REPO / x["weights"], f["levels"], f["overlap"], "calib")
        data = {s: ss.load_levels(dirs, s, f["branch"], f["overlap"]) for s in imgs.stem}
    d = R.fuse_split(data, dict(zip(imgs.stem, imgs.image_id)), f["branch"], f["variant"], f["params"])
    del data
    m = evaluate(coco, d, load_class_setting(E.SETTING), "visdrone_ignore")["metrics"]
    return x["id"], tag, {"metric": "AP", "value": m["AP"], "AP50": m["AP50"], "AP_lt16": m["AP_by_size"]["<16"],
                          "n_dets": len(d), "wall_s": time.time() - t0}


def check(batch: str, workers: int, with_details: bool) -> None:
    from scahi.fusion.run import CACHE_ROOT

    rs = select(batch)
    of = OUT / "check.json"
    res = json.loads(of.read_text())["rows"] if of.exists() else {}
    if with_details:
        jobs = [(x, det, str(DROOT / det)) for x in rs if res.get(x["id"], {}).get("at_or_above_rule")
                for det in DETAILS if det not in res[x["id"]]]
    else:
        jobs = [(x, t, str(rt)) for x in rs for t, rt in (("frozen", CACHE_ROOT), ("aligned", UROOT))
                if t not in res.get(x["id"], {})]
    jobs.sort(key=lambda j: j[0]["ds"] != "dota")  # DOTA first (fewer, larger tasks)
    with ProcessPoolExecutor(workers, mp_context=MC_spawn()) as ex:
        for rid, tag, v in ex.map(_fuse, jobs):
            res.setdefault(rid, {})[tag] = v
            print(f"[mapall] {rid} {tag}: {v['metric']} {v['value']:.5f}", flush=True)
            _write(of, rs, res)
    _write(of, rs, res)


def serial(ids: list[str]) -> None:
    from scahi.fusion.run import CACHE_ROOT

    par = json.loads((OUT / "check.json").read_text())["rows"]
    meta = {x["id"]: x for x in rows_all()}
    sf = OUT / "serial.json"
    out = json.loads(sf.read_text()) if sf.exists() else {}
    for rid in ids:
        r = {}
        for tag, root in (("frozen", CACHE_ROOT), ("aligned", UROOT)):
            _, _, v = _fuse((meta[rid], tag, str(root)))
            r[tag] = {"serial": v["value"], "parallel": par[rid][tag]["value"],
                      "identical": v["value"] == par[rid][tag]["value"]}
        out[rid] = r
        print(f"[mapall] serial {rid}: {r}", flush=True)
        sf.write_text(json.dumps(out, indent=1))


def MC_spawn():
    import multiprocessing as mp

    return mp.get_context("spawn")


def _write(of: Path, rs: list[dict], res: dict) -> None:
    meta = {x["id"]: x for x in rows_all()}
    for rid, v in res.items():
        x = meta[rid]
        v.update(ds=x["ds"], study=x["study"], group=x["group"], main_paper=x["main_paper"],
                 stored_calib=x["stored_calib"])
        if "frozen" in v and "aligned" in v:
            fz = v["frozen"]["value"]
            v["frozen_reproduces_stored"] = None if x["stored_calib"] is None else fz == x["stored_calib"]
            v["delta"] = v["aligned"]["value"] - fz
            v["at_or_above_rule"] = abs(v["delta"]) >= RULE
            if x["ds"] == "visdrone":
                v["delta_secondary"] = {k: v["aligned"][k] - v["frozen"][k] for k in ("AP50", "AP_lt16")}
            v["delta_per_detail"] = {d: v[d]["value"] - fz for d in DETAILS if d in v}
    done = [v for v in res.values() if "delta" in v]
    of.write_text(json.dumps({
        "rule": "DECISIONS 2026-11-02 10:15 / 10:29 (|delta| on calib, all three details at once; >= 0.05: bug fix, "
                "held-out recomputation without re-tuning, side by side; verdicts unchanged)",
        "threshold": RULE, "aligned_cache_state": MC.STATE, "time": time.strftime("%F %T"),
        "n_done": len(done), "n_at_or_above_rule": sum(v["at_or_above_rule"] for v in done),
        "max_abs_delta": max((abs(v["delta"]) for v in done), default=None), "rows": res}, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("list", "caches", "check", "details", "serial"))
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--batch", choices=("main", "all"), default="main")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--details", action="store_true")
    a = ap.parse_args()
    if a.stage == "list":
        inventory()
    elif a.stage == "caches":
        caches(a.batch)
    elif a.stage == "details":
        details(a.batch)
    elif a.stage == "serial":
        serial(a.ids)
    else:
        check(a.batch, a.workers, a.details)


if __name__ == "__main__":
    main()
