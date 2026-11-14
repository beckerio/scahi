# SPDX-License-Identifier: AGPL-3.0-or-later
"""Back-mapping rule as worded (DECISIONS 2026-11-02 10:15): rows with |delta| >= 0.05 on calib are recomputed on the
held-out splits with the Ultralytics conventions, WITHOUT re-tuning, original and corrected values side by side;
verdicts stay as computed (authors, 2026-11-14).

    python scripts/mapcheck_heldout.py caches               # GPU: aligned caches of the rows' levels, held-out splits
    python scripts/mapcheck_heldout.py eval [--workers 12]  # CPU: each row from the frozen and the aligned caches

Rows: those with at_or_above_rule in runs/mapcheck/all/check.json (VisDrone extension -> val and test-dev through
scahi.ext_testdev.run_cache_row; DOTA -> DOTA-val through scahi.dota.final.row_dets + evaluate_row). Each row is
computed twice through the same path (only the cache root differs); the frozen-cache value must reproduce the stored
held-out value exactly. Outputs runs/mapcheck/all/heldout/ and runs/mapcheck/all/heldout.json. No stored row, cache or
result is changed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scahi import mapcheck as MC  # noqa: E402

OUT = REPO / "runs/mapcheck/all"
HO = OUT / "heldout"
SPLITS = {"visdrone": ("val", "test-dev"), "dota": ("val",)}


def rows() -> list[dict]:
    import mapcheck_all as A

    res = json.loads((OUT / "check.json").read_text())["rows"]
    return [x for x in A.rows_all() if res.get(x["id"], {}).get("at_or_above_rule")]


def _images(ds: str, split: str):
    if ds == "dota":
        from scahi.dota import pipeline as P

        return P.images(split)
    from scahi import ext_testdev as TD

    return TD.split_data(split)[1]


def caches() -> None:
    from scahi.infer import cache_mz as cm
    from scahi.infer.raw_head import load_unfused

    rep_f = OUT / "heldout_caches.json"
    rep = json.loads(rep_f.read_text()) if rep_f.exists() else {}
    jobs: dict = {}
    for x in rows():
        jobs.setdefault((x["ds"], x["weights"], x["spec"]["overlap"]), set()).update(x["spec"]["levels"])
    for (ds, wr, ov), lks in sorted(jobs.items()):
        w = REPO / wr
        m = load_unfused(w)
        for split in SPLITS[ds]:
            imgs = _images(ds, split)
            for lk in sorted(lks):
                t0 = time.time()
                with MC.ultralytics_prep():
                    d = cm.cache_level(m, w, imgs, lk, ov, split, MC.ROOT)
                (d / "cache_state.json").write_text(json.dumps({**MC.STATE, "weights": str(w), "level": lk}, indent=1))
                n = sum((d / f"{s}.npz").exists() for s in imgs.stem)
                rep[f"{ds}/{wr}/{lk}/{split}"] = {"images": len(imgs), "present": n, "build_s": round(time.time() - t0)}
                rep_f.write_text(json.dumps(rep, indent=1))
                print(f"[heldout] cache {ds} {wr} {lk} {split}: {n}/{len(imgs)} ({time.time() - t0:.0f} s)", flush=True)
        del m


def _eval(args) -> tuple[str, str, str, float, dict]:
    import torch

    from scahi.fusion import run as R

    torch.set_num_threads(1)
    x, split, tag, root = args
    R.CACHE_ROOT = Path(root)
    out = HO / tag / split
    out.mkdir(parents=True, exist_ok=True)
    if x["ds"] == "dota":
        from scahi.dota import final as F

        lim = json.loads((REPO / "configs/dota" / x["study"]).read_text())["pipeline_limit"]  # as DOTA-val (200 000)
        dets, info = F.row_dets(x["row"], split, lim)
        dets.to_parquet(out / f"{x['id']}.parquet")
        m, _ = F.evaluate_row(x["id"], split, dets)
        (out / f"{x['id']}.json").write_text(json.dumps(m, indent=1, default=float))
        return x["id"], split, tag, m["primary"]["mAP50"], {"n_dets": len(dets)}
    from scahi import ext_testdev as TD

    TD.TD, TD.REPLAY = out, out  # out_root() of the val / test-dev path (both splits written below this tag's folder)
    fz = TD.freeze()
    TD.run_cache_row((x["row"], split, fz))
    m = json.loads((out / x["row"]["val_exp_id"] / "metrics.json").read_text())["metrics"]
    return x["id"], split, tag, m["AP"], {"AP50": m["AP50"], "AP_lt16": m["AP_by_size"]["<16"]}


def _stored(x: dict, split: str) -> float:
    if x["ds"] == "dota":
        f = "results.json" if x["study"] == "frozen_rows.json" else "results_new.json"
        return json.loads((REPO / "runs/dota/val" / f).read_text())["rows"][x["id"]]["mAP50"]
    if split == "val":
        return x["row"]["val_AP"]
    return json.loads((REPO / "runs/ext/testdev" / x["row"]["val_exp_id"] / "metrics.json").read_text())["metrics"]["AP"]


def evaluate(workers: int) -> None:
    from scahi.fusion.run import CACHE_ROOT

    rs = rows()
    jobs = [(x, sp, tag, str(rt)) for x in rs for sp in SPLITS[x["ds"]]
            for tag, rt in (("frozen", CACHE_ROOT), ("aligned", MC.ROOT))]
    jobs.sort(key=lambda j: j[0]["ds"] != "dota")
    res: dict = {}
    with ProcessPoolExecutor(workers, mp_context=__import__("multiprocessing").get_context("spawn")) as ex:
        for rid, sp, tag, v, extra in ex.map(_eval, jobs):
            res.setdefault(rid, {}).setdefault(sp, {})[tag] = {"value": v, **extra}
            print(f"[heldout] {rid} {sp} {tag}: {v:.5f}", flush=True)
    calib = json.loads((OUT / "check.json").read_text())["rows"]
    out = {}
    for x in rs:
        r = {"ds": x["ds"], "group": x["group"], "metric": "mAP50" if x["ds"] == "dota" else "AP",
             "calib_delta": calib[x["id"]]["delta"], "splits": {}}
        for sp, v in res[x["id"]].items():
            st = _stored(x, sp)
            r["splits"][sp] = {"original_stored": st, "frozen_recomputed": v["frozen"]["value"],
                               "frozen_reproduces_stored": v["frozen"]["value"] == st,
                               "corrected": v["aligned"]["value"], "delta": v["aligned"]["value"] - st, **(
                                   {"delta_AP50": v["aligned"]["AP50"] - v["frozen"]["AP50"],
                                    "delta_AP_lt16": v["aligned"]["AP_lt16"] - v["frozen"]["AP_lt16"]}
                                   if x["ds"] == "visdrone" else {})}
        out[x["id"]] = r
    (OUT / "heldout.json").write_text(json.dumps({
        "rule": "DECISIONS 2026-11-02 10:15: rows >= 0.05 on calib recomputed on the held-out splits with the per-axis "
                "mapping (all three Ultralytics details, 10:29), without re-tuning; original and corrected side by side; "
                "verdicts as originally computed", "time": time.strftime("%F %T"), "rows": out}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("caches", "eval"))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    caches() if a.stage == "caches" else evaluate(a.workers)


if __name__ == "__main__":
    main()
