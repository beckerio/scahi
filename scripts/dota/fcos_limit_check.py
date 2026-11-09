# SPDX-License-Identifier: AGPL-3.0-or-later
"""FCOS: effect of the pipeline limit 200 000 (user decision 2026-11-01: limit kept, same pipeline as all YOLO rows).

    python scripts/dota/fcos_limit_check.py calib   # before the freeze: every FCOS row on calib (GPU, then CPU fusion)
    python scripts/dota/fcos_limit_check.py calib 1024   # only the rows of one size (user 2026-11-02: 1024 rows as
                                                         # soon as the 1024 stages are done, 2048 after the 2048 stages)
    python scripts/dota/fcos_limit_check.py calib 1024 --row ID   # one row -> ..._parts/ID.json (parallel run)
    python scripts/dota/fcos_limit_check.py calib 1024 --merge    # parts -> the file of the serial run
    python scripts/dota/fcos_limit_check.py val     # after the single DOTA-val evaluation, exploratory, post hoc;
        # one row at a time; box-fusion rows only counted (passes at the per-pass limit) unless named in
        # SCAHI_LIMIT_BF_UNLIMITED=ID[,ID] (recomputation without the per-pass limit, after the user saw the cost)

Does not change any row, selection or reported number. Two places where 200 000 can bind:
* per image: torchvision ``detections_per_img`` of the single pass (rows A, X) and the SCAHI ``top_k`` (cache rows);
  box fusion has no final ``max_det``: its merged output is not capped (frozen, executed path; user 2026-11-07);
* per pass: torchvision ``detections_per_img`` of each pass whose detections box fusion uses (rows NMS; SCAHI reads the
  dense outputs, which have no per-pass cap).
For box-fusion rows "images at the limit" counts merged outputs with >= 200 000 detections; they are not truncated.
An image is affected for a row if it reaches the limit at either place. Affected images are recomputed without the
limit (10 000 000; the per-level candidate limit of torchvision, 200 000 per FPN level, addendum 3 item 6, is kept), their
detections replace the limited ones, and mAP50 (no cap) is recomputed. Passes are re-run without the limit into
``runs/dota/fcos_nolimit_cache/`` (dense outputs compared with the stored caches: must be identical, otherwise the Δ
would mix GPU non-determinism into the limit effect). Output: runs/dota/freeze/limit_calib_fcos.json (on calib it
is also the replay of every row: the limited detections come from ``scahi.dota.final.row_dets``, the DOTA-val path) or
runs/dota/val/limit_check_fcos.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scahi.dota import evaluate as EV  # noqa: E402
from scahi.dota import fcos_train as FT  # noqa: E402
from scahi.dota import final as F  # noqa: E402
from scahi.dota import pipeline as P  # noqa: E402
from scahi.dota import train as T  # noqa: E402

LIMIT = FT.LIMIT
NO_LIMIT = 10_000_000
NOLIM_ROOT = REPO / "runs/dota/fcos_nolimit_cache"
OUT = {"calib": F.FREEZE_DIR / "limit_calib_fcos.json", "val": F.VAL_DIR / "limit_check_fcos.json"}
FROZEN_FCOS = REPO / "configs/dota/frozen_rows_fcos.json"


def _lks(r: dict) -> tuple[str, ...]:
    return (P.WHOLE[r["s"]], P.TILES) if r["stack"] == "C" else (P.WHOLE[r["s"]],)


def pass_hits(r: dict, split: str) -> set[str]:
    """Images with a pass at the per-pass limit (box-fusion rows only)."""
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm

    if r["kind"] != "cache" or r["variant"] != "F0-NMS":
        return set()
    hit = set()
    for lk in _lks(r):
        d = cm.cache_dir(CACHE_ROOT, T.weights(r["model"], r["s"]), lk, P.OVERLAP, split)
        for st in P.split_stems(split):
            t = np.load(d / f"{st}.npz")[f"{r['branch']}_det_tile"]
            if len(t) and np.bincount(t).max() >= LIMIT:
                hit.add(st)
    return hit


def pass_counts(r: dict, split: str) -> dict[str, int]:
    """Box-fusion rows: per image the number of passes (whole image or tile, every level) at the per-pass limit."""
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm

    out: dict[str, int] = {}
    for lk in _lks(r):
        d = cm.cache_dir(CACHE_ROOT, T.weights(r["model"], r["s"]), lk, P.OVERLAP, split)
        for st in P.split_stems(split):
            t = np.load(d / f"{st}.npz")[f"{r['branch']}_det_tile"]
            k = int((np.bincount(t) >= LIMIT).sum()) if len(t) else 0
            if k:
                out[st] = out.get(st, 0) + k
    return out


def nolimit_caches(s: int, split: str, stems: list[str]) -> dict:
    """GPU: both levels of dota<s> for ``stems`` without the per-pass limit; dense outputs vs. the stored caches."""
    import cv2

    from scahi import fcos as FC
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm

    w = T.weights(T.FCOS, s)
    m = FT.load(w, s, max_det=NO_LIMIT)
    imgs = P.images(split).set_index("stem")
    same = {}
    for lk in (P.WHOLE[s], P.TILES):
        d = cm.cache_dir(NOLIM_ROOT, w, lk, P.OVERLAP, split)
        d.mkdir(parents=True, exist_ok=True)
        ref = cm.cache_dir(CACHE_ROOT, w, lk, P.OVERLAP, split)
        for st in stems:
            f = d / f"{st}.npz"
            if not f.exists():
                np.savez_compressed(f, **FC.raw_predict_level(m, cv2.imread(imgs.path[st]), cm.LEVELS[lk], P.OVERLAP))
            a, b = np.load(f), np.load(ref / f"{st}.npz")
            same[f"{lk}/{st}"] = all(np.array_equal(a[k], b[k]) for k in ("o2m_xyxy", "o2m_probs", "o2m_tile"))
    return same


def _load(r: dict, split: str, stems: list[str], root: Path) -> dict:
    from scahi.fusion import scalespace as ss
    from scahi.infer import cache_mz as cm

    w = T.weights(r["model"], r["s"])
    dirs = [(k, cm.cache_dir(root, w, k, P.OVERLAP, split)) for k in _lks(r)]
    return {st: ss.load_levels(dirs, st, r["branch"], P.OVERLAP) for st in stems}


IMG_WORKERS = 1  # > 1: cache rows fused by image in parallel (set by --row; env SCAHI_LIMIT_IMG_WORKERS)
PART_DIR: Path | None = None


def unlimited(r: dict, split: str, hit: list[str]) -> pd.DataFrame:
    if r["kind"] == "cache" and IMG_WORKERS > 1:
        return by_image(r, split, hit, True, IMG_WORKERS, PART_DIR)
    if r["kind"] == "single":
        d, _ = T.predict(T.weights(r["model"], r["s"]), "train" if split == "calib" else "val", hit, r["imgsz"],
                         r["branch"], max_det=NO_LIMIT)
        return d[d.score >= r["conf"]]
    root = NOLIM_ROOT if r["variant"] == "F0-NMS" else __import__("scahi.fusion.run", fromlist=["x"]).CACHE_ROOT
    return P.fuse(_load(r, split, hit, root), r["branch"], r["variant"], r["cfg"], NO_LIMIT)


def _fuse_image(args: tuple) -> Path:
    """Worker (fresh process): one image of a cache row, as ``row_dets`` (limited) or ``unlimited`` (no limit) fuse it;
    -> parquet ``f`` (kept, so a restarted row skips finished images)."""
    r, split, st, unlim, f = args
    if unlim:
        root = NOLIM_ROOT if r["variant"] == "F0-NMS" else __import__("scahi.fusion.run", fromlist=["x"]).CACHE_ROOT
        d = P.fuse(_load(r, split, [st], root), r["branch"], r["variant"], r["cfg"], NO_LIMIT)
    else:
        d = P.fuse(P.load(r["model"], r["s"], split, r["branch"], r["stack"], [st]), r["branch"], r["variant"], r["cfg"],
                   limit=LIMIT)
    t = f.with_suffix(".tmp")
    d.to_parquet(t)
    t.replace(f)
    return f


def by_image(r: dict, split: str, stems: list[str], unlim: bool, workers: int, part_dir: Path) -> pd.DataFrame:
    """Parallelisation only (user 2026-11-04): ``fuse`` handles every image on its own, so the per-image results
    concatenated in ``stems`` order equal the serial call. Densest images are submitted first."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    d = part_dir / ("unlimited" if unlim else "limited")
    d.mkdir(parents=True, exist_ok=True)
    todo = sorted((st for st in stems if not (d / f"{st}.parquet").exists()),
                  key=lambda st: -P._cache_bytes(r["model"], r["s"], r["stack"], st, split))
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp.get_context("spawn"), max_tasks_per_child=1) as ex:
        list(ex.map(_fuse_image, [(r, split, st, unlim, d / f"{st}.parquet") for st in todo]))
        print(f"[fcos-limit] {r['id']} {'unlimited' if unlim else 'limited'}: {len(todo)} images fused "
              f"({len(stems) - len(todo)} kept), {time.time() - t0:.0f} s", flush=True)
    out = [pd.read_parquet(d / f"{st}.parquet") for st in stems]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=EV.DET_COLS)


def out_path(split: str, sizes: tuple[int, ...] | None = None):
    """Whole check -> OUT[split]; one size only -> ..._<s>.json (the freeze proposal merges the parts)."""
    f = OUT[split]
    return f if not sizes else f.with_name(f"{f.stem}_{'_'.join(map(str, sizes))}.json")


def check(rows: list[dict], split: str, limited, out_file=None) -> dict:
    gt = EV.gt_arrays("train" if split == "calib" else "val", T.split_stems(split))
    out = {"label": "before the freeze (calib)" if split == "calib" else "exploratory, post hoc (DOTA-val)",
           "split": split, "sizes": sorted({r["s"] for r in rows}), "limit": LIMIT, "no_limit": NO_LIMIT, "rows": {}}
    # GPU first: passes without the per-pass limit for every image any box-fusion row needs
    lim = {r["id"]: limited(r) for r in rows}
    hits = {}
    for r in rows:
        n = lim[r["id"]].groupby("image").size()
        hits[r["id"]] = (set(n[n >= LIMIT].index), pass_hits(r, split))
    same = {}
    for s in sorted({r["s"] for r in rows}):
        need = sorted(set().union(*(a | b for r in rows if r["s"] == s and r["kind"] == "cache"
                                    and r["variant"] == "F0-NMS" for a, b in [hits[r["id"]]])))
        if need:
            same.update(nolimit_caches(s, split, need))
    out["nolimit_dense_identical"] = {"n": len(same), "all": all(same.values()),
                                      "differs": [k for k, v in same.items() if not v]}
    for r in rows:
        img_hit, pass_hit = hits[r["id"]]
        hit = sorted(img_hit | pass_hit)
        d = lim[r["id"]]
        e = {"images_at_image_limit": len(img_hit), "images_with_pass_at_limit": len(pass_hit), "images_affected": len(hit)}
        a = EV.map50(EV.build(gt, d, None, sizes=False))
        e["mAP50_limited"] = a
        if "calib_mAP50" in r and split == "calib":  # replay of the row on calib (the DOTA-val path, F.row_dets)
            e.update(calib_mAP50=r["calib_mAP50"], replay_abs_diff=abs(a - r["calib_mAP50"]))
        if hit:
            t0 = time.time()
            new = unlimited(r, split, hit)
            n_new = new.groupby("image").size()
            d2 = pd.concat([d[~d.image.isin(hit)], new], ignore_index=True)
            b = EV.map50(EV.build(gt, d2, None, sizes=False))
            e.update(mAP50_unlimited=b, delta=b - a, wall_s=time.time() - t0,
                     images={st: {"dets_limited": int((d.image == st).sum()), "dets_unlimited": int(n_new.get(st, 0))}
                             for st in hit})
        else:
            e.update(mAP50_unlimited=a, delta=0.0)
        out["rows"][r["id"]] = e
        print(f"[fcos-limit] {split} {r['id']}: affected {len(hit)} (image {len(img_hit)}, pass {len(pass_hit)}), "
              f"Δ mAP50 {e['delta']:+.2e}", flush=True)
    out_f = out_file or OUT[split]
    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(out, indent=1))
    return out


def check_val(rows: list[dict], boxfusion_unlimited: tuple[str, ...] = ()) -> dict:
    """DOTA-val, after the evaluation (user 2026-11-07): one row at a time (the 13 rows hold 640 M detections). Rows with
    a capped output (A, X: torchvision per image; SCAHI: top_k): images at the limit, recomputed without the limit, mAP50
    (no cap) with and without. Box-fusion rows (merged output not capped, frozen path): images with a merged output
    >= 200 000 (not truncated) and the passes at the per-pass limit; only the images with a pass at the limit can
    change, and they are recomputed only for the rows in ``boxfusion_unlimited`` (after the user saw the cost)."""
    f = OUT["val"]
    out = json.loads(f.read_text()) if f.exists() else {
        "label": "exploratory, post hoc (DOTA-val)", "split": "val", "limit": LIMIT, "no_limit": NO_LIMIT, "rows": {},
        "nolimit_dense_identical": {"n": 0, "all": True, "differs": []}}
    gt = EV.gt_arrays("val", T.split_stems("val"))
    for r in rows:
        bf = r["kind"] == "cache" and r["variant"] == "F0-NMS"
        if r["id"] in out["rows"] and not (bf and r["id"] in boxfusion_unlimited
                                           and "mAP50_unlimited" not in out["rows"][r["id"]]):
            continue
        d = pd.read_parquet(F.DETS_DIR / f"{r['id']}.parquet")
        n = d.groupby("image").size()
        img_hit = set(n[n >= LIMIT].index)
        a = EV.map50(EV.build(gt, d, None, sizes=False))
        e = {"kind": "box fusion (merged output not capped)" if bf else "capped output", "n_dets": len(d),
             "max_dets_per_image": int(n.max()), "images_at_image_limit": len(img_hit), "mAP50_limited": a}
        ev = F.VAL_DIR / "eval" / f"{r['id']}.json"
        if ev.exists():
            e["eval_mAP50_equal"] = json.loads(ev.read_text())["mAP50"] == a
        if bf:
            pc = pass_counts(r, "val")
            hit = sorted(pc)
            e.update(passes_at_pass_limit=sum(pc.values()), images_with_pass_at_limit=len(hit),
                     dets_on_those_images=int(n.reindex(hit).fillna(0).sum()))
            if r["id"] not in boxfusion_unlimited:
                e["unlimited"] = "not run (cost reported to the user first)"
                out["rows"][r["id"]] = e
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(out, indent=1))
                print(f"[fcos-limit] val {r['id']}: {e['passes_at_pass_limit']} passes at the limit on {len(hit)} images "
                      f"(merged >= {LIMIT}: {len(img_hit)} images, not truncated)", flush=True)
                continue
            same = nolimit_caches(r["s"], "val", hit) if hit else {}
            nd = out["nolimit_dense_identical"]
            nd.update(n=nd["n"] + len(same), all=nd["all"] and all(same.values()),
                      differs=nd["differs"] + [k for k, v in same.items() if not v])
        else:
            hit = sorted(img_hit)
        if hit:
            t0 = time.time()
            new = unlimited(r, "val", hit)
            n_new = new.groupby("image").size()
            d2 = pd.concat([d[~d.image.isin(hit)], new], ignore_index=True)
            b = EV.map50(EV.build(gt, d2, None, sizes=False))
            e.update(mAP50_unlimited=b, delta=b - a, wall_s=time.time() - t0,
                     images={st: {"dets_limited": int(n.get(st, 0)), "dets_unlimited": int(n_new.get(st, 0))}
                             for st in hit})
        else:
            e.update(mAP50_unlimited=a, delta=0.0)
        out["rows"][r["id"]] = e
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out, indent=1))
        print(f"[fcos-limit] val {r['id']}: recomputed {len(hit)} images, Δ mAP50 {e['delta']:+.2e}", flush=True)
    return out


def part_path(split: str, sizes: tuple[int, ...] | None, row_id: str) -> Path:
    f = out_path(split, sizes)
    return f.parent / f"{f.stem}_parts" / f"{row_id}.json"


def merge(split: str, rows: list[dict], sizes: tuple[int, ...] | None = None) -> dict:
    """Per-row parts (``--row``, run in parallel; rows are independent) -> the file ``check`` writes for all rows."""
    parts = [json.loads(part_path(split, sizes, r["id"]).read_text()) for r in rows]
    same = [p["nolimit_dense_identical"] for p in parts]
    out = {**{k: parts[0][k] for k in ("label", "split", "limit", "no_limit")}, "sizes": sorted({r["s"] for r in rows}),
           "rows": {r["id"]: p["rows"][r["id"]] for r, p in zip(rows, parts)},
           "nolimit_dense_identical": {"n": sum(x["n"] for x in same), "all": all(x["all"] for x in same),
                                       "differs": [k for x in same for k in x["differs"]]}}
    out = {k: out[k] for k in ("label", "split", "sizes", "limit", "no_limit", "rows", "nolimit_dense_identical")}
    out_path(split, sizes).write_text(json.dumps(out, indent=1))
    return out


def main(split: str, sizes: tuple[int, ...] | None = None, row: str | None = None, do_merge: bool = False) -> None:
    global IMG_WORKERS, PART_DIR
    if split == "calib":
        rows = F.proposal_rows(("single", "cache"), (T.FCOS,), sizes or (1024, 2048))
        if do_merge:
            merge("calib", rows, sizes)
        elif row:  # one row (parallel run, user 2026-11-04: parallelisation only, rows and rule unchanged)
            r = [x for x in rows if x["id"] == row]
            assert len(r) == 1, row
            IMG_WORKERS = int(os.environ.get("SCAHI_LIMIT_IMG_WORKERS", "20"))
            PART_DIR = part_path("calib", sizes, row).with_suffix("")
            check(r, "calib", lambda r: by_image(r, "calib", P.split_stems("calib"), False, IMG_WORKERS, PART_DIR)
                  if r["kind"] == "cache" and IMG_WORKERS > 1 else F.row_dets(r, "calib", LIMIT)[0],
                  part_path("calib", sizes, row))
        else:
            check(rows, "calib", lambda r: F.row_dets(r, "calib", LIMIT)[0], out_path("calib", sizes))
    else:
        rows = F.frozen(FROZEN_FCOS)["rows"]
        # box-fusion rows without the per-pass limit: by image, parallel
        IMG_WORKERS = int(os.environ.get("SCAHI_LIMIT_IMG_WORKERS", "1"))
        PART_DIR = OUT["val"].with_name("limit_check_fcos_parts")
        check_val(rows, tuple(x for x in os.environ.get("SCAHI_LIMIT_BF_UNLIMITED", "").split(",") if x))


if __name__ == "__main__":
    a = sys.argv[1:]
    row = a[a.index("--row") + 1] if "--row" in a else None
    a = [x for i, x in enumerate(a) if x not in ("--row", "--merge") and (i == 0 or a[i - 1] != "--row")]
    main(a[0], tuple(int(x) for x in a[1:]) or None, row, "--merge" in sys.argv)
