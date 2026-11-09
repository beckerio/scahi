# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exploratory FCOS diagnostics on DOTA-val (post hoc, user 2026-11-08; framing in docs/DECISIONS.md before any number).
Existing detections and caches only, CPU; never toward any verdict; reported to the user only.

    python scripts/dota/fcos_diagnostics.py cap          # 1. equal cap for both fusions + detections per image
    python scripts/dota/fcos_diagnostics.py redundancy   # 2. predictions above tau_splat per GT object (3 models, 2048)
    python scripts/dota/fcos_diagnostics.py recall       # 3. crowded / isolated recall, NMS-o2m vs. RG (stored eval)
    python scripts/dota/fcos_diagnostics.py report       # tables from the stored JSON files only (no data; release
                                                         # regeneration, user 2026-11-09), incl. the DOTA-val limit check

Output: runs/dota/val/diag_fcos/<cmd>.json (+ cap: <id>_cap200k.json / _boot.npz); report: reports/dota/FCOS_DIAGNOSTICS.md
and reports/dota/tab_fcos_diag_{cap,per_image,redundancy,recall,limit_check}.csv.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scahi.dota import evaluate as EV  # noqa: E402
from scahi.dota import final as F  # noqa: E402
from scahi.dota import pipeline as P  # noqa: E402
from scahi.dota import train as T  # noqa: E402

LABEL = "exploratory, post hoc (FCOS diagnostics, user 2026-11-08); no verdict changes; reported only on the user's decision"
OUT = F.VAL_DIR / "diag_fcos"
CAP = 200_000  # the SCAHI rows' top_k
SZ = 2048
QS = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def _load(rid: str, d: Path = F.VAL_DIR / "eval") -> tuple[dict, dict]:
    return json.loads((d / f"{rid}.json").read_text()), dict(np.load(d / f"{rid}_boot.npz"))


def _fcos_ids() -> list[str]:
    return [r["id"] for r in F.frozen(F.FROZEN_FCOS)["rows"]]


# ------------------------------------------------------------------------------------------------------- 1. cap

def cap() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"label": LABEL, "cap": CAP, "per_image": {}, "contrasts": {}}
    for rid in _fcos_ids():
        n = pd.read_parquet(F.DETS_DIR / f"{rid}.parquet", columns=["image"]).groupby("image").size()
        res["per_image"][rid] = {"max": int(n.max()), "mean": float(n.mean()), "median": float(n.median()),
                                 "images_over_cap": int((n > CAP).sum()), "n_images_with_dets": int(len(n))}
        print(f"[diag] {rid}: max {n.max()}, mean {n.mean():.0f}", flush=True)
    rows = {"D-H2a": ("RG_fcos_1024", "NMS-o2m_fcos_1024"), "D-H2b": ("RG_fcos_2048", "NMS-o2m_fcos_2048")}
    pre = json.loads((F.VAL_DIR / "results_fcos.json").read_text())["hypotheses"]
    for h, (a, b) in rows.items():
        cid = f"{b}_cap200k"
        if not (OUT / f"{cid}.json").exists():
            t0 = time.time()
            dets = EV.apply_cap(pd.read_parquet(F.DETS_DIR / f"{b}.parquet"), CAP)
            o, v = F.evaluate_row(cid, dets=dets)
            o.update(label=LABEL, source=b, cap=CAP, wall_s=time.time() - t0)
            np.savez(OUT / f"{cid}_boot.npz", **v)
            (OUT / f"{cid}.json").write_text(json.dumps(o, indent=1, default=float))
            del dets
        pt, vec = {}, {}
        pt[a], vec[a] = _load(a)
        pt[cid], vec[cid] = _load(cid, OUT)
        res["contrasts"][h] = {
            "pre-specified secondary, cap 4000 (results_fcos.json)": pre[h]["mAP50_cap4000"],
            f"exploratory, box fusion capped to top {CAP} per image": {
                k: F._contrast(vec, a, cid, k, pt) for k in ("mAP50", "AP", "R50_crowd", "R50_iso")},
            "under the pre-registered evaluation (results_fcos.json)": pre[h]["mAP50"]}
        print(f"[diag] {h}: {json.dumps(res['contrasts'][h], default=float)}", flush=True)
    (OUT / "cap.json").write_text(json.dumps(res, indent=1, default=float))
    return res


# ------------------------------------------------------------------------------------------------ 2. redundancy

def _taus(model: str) -> dict[str, float]:
    fz = F.frozen(F.FROZEN_FCOS) if model == T.FCOS else (F.frozen() if model in T.MODELS else F.frozen(F.FROZEN_NEW))
    r = {x["id"]: x for x in fz["rows"]}
    return {k: r[f"{k}_{model}_{SZ}"]["cfg"]["tau_splat"] for k in ("F2", "RG")}


def _match(gb: np.ndarray, cb: np.ndarray, ix: np.ndarray, order: np.ndarray, cxs: np.ndarray) -> list[np.ndarray]:
    """Per GT box: indices (into cb) of candidates with IoU >= 0.5. IoU >= 0.5 needs the candidate's centre inside the
    GT box (intersection width >= half of each width), so only candidates whose centre x lies in [gx1, gx2] are tested."""
    out = []
    lo = np.searchsorted(cxs, gb[:, 0], "left")
    hi = np.searchsorted(cxs, gb[:, 2], "right")
    ga = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
    for g in range(len(gb)):
        j = order[lo[g]:hi[g]]
        if not len(j):
            out.append(j)
            continue
        c = cb[j]
        cy = (c[:, 1] + c[:, 3]) / 2
        m = (cy >= gb[g, 1]) & (cy <= gb[g, 3])
        j, c = j[m], c[m]
        iw = np.minimum(c[:, 2], gb[g, 2]) - np.maximum(c[:, 0], gb[g, 0])
        ih = np.minimum(c[:, 3], gb[g, 3]) - np.maximum(c[:, 1], gb[g, 1])
        inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
        ca = (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
        out.append(ix[j[inter / np.maximum(ga[g] + ca - inter, 1e-9) >= 0.5]])
    return out


def _red_image(args: tuple[str, str]) -> dict:
    """One image of one model: per non-difficult GT the counts (agnostic / same class) per tau setting and pass group."""
    model, stem = args
    taus = _taus(model)
    tmin = min(taus.values())
    levels, W, H = P.load(model, SZ, "val", "o2m", "C", [stem], keep_dets=False)[stem]
    b, k, dif = EV.gt_arrays("val", [stem])[stem]
    crowd = F.crowd_gt({stem: (b, k, dif)})[stem]
    keep = ~dif
    gb, gk = b[keep].astype(np.float64), k[keep]
    gcx, gcy = (gb[:, 0] + gb[:, 2]) / 2, (gb[:, 1] + gb[:, 3]) / 2
    res = {"sigma": EV._sigma(gb), "crowd": crowd[keep], "cls": gk}
    for lv_i, lv in enumerate(levels):
        pmax = lv.probs.max(1)
        sel = np.flatnonzero(pmax >= tmin)
        cb = lv.xyxy[sel].astype(np.float64)
        cx = (cb[:, 0] + cb[:, 2]) / 2
        order = np.argsort(cx, kind="stable")
        hits = _match(gb, cb, np.arange(len(sel)), order, cx[order])
        p_sel, a_sel = pmax[sel], lv.probs[sel].argmax(1)
        t_sel = lv.tile[sel].astype(int)
        n_t = len(lv.tiles)
        # tile passes whose view contains the GT centre (whole-image pass: its single view)
        tl = lv.tiles
        inview = ((gcx[:, None] >= tl[None, :, 0]) & (gcx[:, None] < tl[None, :, 2])
                  & (gcy[:, None] >= tl[None, :, 1]) & (gcy[:, None] < tl[None, :, 3]))
        for tn, tau in taus.items():
            for cls_mode in ("agnostic", "same_class"):
                per_tile = np.zeros((len(gb), n_t))
                for g, h in enumerate(hits):
                    ok = p_sel[h] >= tau
                    if cls_mode == "same_class":
                        ok &= a_sel[h] == gk[g]
                    per_tile[g] = np.bincount(t_sel[h[ok]], minlength=n_t)
                key = f"{tn}/{cls_mode}"
                tot = per_tile.sum(1)
                res.setdefault(f"{key}/total", np.zeros(len(gb)))
                res[f"{key}/total"] += tot
                if lv.whole:
                    res[f"{key}/whole"] = tot
                else:
                    nv = inview.sum(1)
                    res[f"{key}/tile_pass_mean"] = np.where(nv > 0, (per_tile * inview).sum(1) / np.maximum(nv, 1), np.nan)
                    res[f"{key}/tiles_sum"] = tot
                    res["n_tile_views"] = nv
    return res


def _summ(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "share_zero": float((x == 0).mean()),
            **{f"q{int(q * 100)}": float(np.quantile(x, q)) for q in QS}, "max": float(x.max())}


def redundancy(models: tuple[str, ...] = (T.FCOS, "yolov10s", "yolo11s"), workers: int = 8) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "redundancy.json"
    res = json.loads(f.read_text()) if f.exists() else {"label": LABEL, "models": {}}
    stems = P.split_stems("val")
    for m in models:
        if m in res["models"]:
            continue
        t0 = time.time()
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_red_image, [(m, s) for s in stems], chunksize=4))
        cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(OUT / f"redundancy_{m}_per_gt.npz", **cat)
        sig, crowd = cat["sigma"], cat["crowd"].astype(bool)
        groups = {"all": np.ones(len(sig), bool), "crowded": crowd, "isolated": ~crowd}
        for i, (lo, hi) in enumerate(zip(EV.SIZE_EDGES[:-1], EV.SIZE_EDGES[1:])):
            groups[f"size {lo}-{hi}"] = (sig >= lo) & (sig < hi)
        out = {"tau_splat": _taus(m), "n_gt": int(len(sig)), "wall_s": time.time() - t0, "stats": {}}
        for k in cat:
            if "/" not in k:
                continue
            out["stats"][k] = {g: _summ(cat[k][msk]) for g, msk in groups.items()}
        res["models"][m] = out
        f.write_text(json.dumps(res, indent=1, default=float))
        a = out["stats"]["F2/agnostic/total"]["all"]
        print(f"[diag] redundancy {m}: {len(sig)} GT, mean {a['mean']:.1f} (F2 tau, agnostic, all passes), "
              f"{time.time() - t0:.0f} s", flush=True)
    return res


# ------------------------------------------------------------------------------------------------------ 3. recall

def recall() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"label": LABEL, "rows": {}, "contrasts": {}}
    pt, vec = {}, {}
    for s in (1024, 2048):
        for r in ("NMS-o2m", "RG"):
            rid = f"{r}_fcos_{s}"
            pt[rid], vec[rid] = _load(rid)
            res["rows"][rid] = {k: pt[rid][k] for k in ("R50_crowd", "n_crowd_gt", "R50_iso", "n_iso_gt")}
        a, b = f"RG_fcos_{s}", f"NMS-o2m_fcos_{s}"
        res["contrasts"][f"{a} − {b}"] = {k: F._contrast(vec, a, b, k, pt) for k in ("R50_crowd", "R50_iso")}
    (OUT / "recall.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return res


# ------------------------------------------------------------------------------------------------------ report

def report() -> None:
    """Exploratory tables of the FCOS diagnostics and of the DOTA-val limit check from the stored JSON files."""
    D = OUT
    cap, rec, red = (json.loads((D / f"{k}.json").read_text()) for k in ("cap", "recall", "redundancy"))
    lim = json.loads((F.VAL_DIR / "limit_check_fcos.json").read_text())
    hyp = json.loads((F.VAL_DIR / "results_fcos.json").read_text())["hypotheses"]
    rd = REPO / "reports/dota"
    f = lambda v, d=2: "–" if v is None else f"{v:+.{d}f}"  # noqa: E731
    ci = lambda c: f"{f(c['delta'])} [{f(c['ci95'][0])}, {f(c['ci95'][1])}]"  # noqa: E731
    nl = lambda h: lim["rows"][hyp[h]["contrast"].split(" − ")[0]]["mAP50_unlimited"] - lim["rows"][  # noqa: E731
        hyp[h]["contrast"].split(" − ")[1]]["mAP50_unlimited"]
    PRE, CAPK, EVK = ("pre-specified secondary, cap 4000 (results_fcos.json)",
                      f"exploratory, box fusion capped to top {CAP} per image",
                      "under the pre-registered evaluation (results_fcos.json)")
    L = ["# FCOS diagnostics and DOTA-val limit check (exploratory, post hoc)", "",
         f"{LABEL}. DOTA-val, 458 images. Paired bootstrap (1000 resamples, seed 0), 95 % CIs. Source: "
         "runs/dota/val/diag_fcos/{cap,recall,redundancy}.json, runs/dota/val/limit_check_fcos.json, "
         "runs/dota/val/results_fcos.json.", "",
         f"## 1. Equal cap: both fusions capped at the top {CAP:,} detections per image (RG − box fusion, mAP50)", "",
         "| contrast | pre-specified (cap 4000) | as evaluated | both capped | without limit (point value) | verdict |",
         "|---|---|---|---|---|---|"]
    rows_cap = []
    for h, c in cap["contrasts"].items():
        cc = c[CAPK]
        L.append(f"| {h}: {hyp[h]['contrast']} | {ci(c[PRE])} | {ci(c[EVK])} | {ci(cc['mAP50'])} | {f(nl(h))} | "
                 f"{'met' if hyp[h]['met'] else 'not met'} |")
        for k, v in [("pre-specified cap 4000 mAP50", c[PRE]), ("as evaluated mAP50", c[EVK])] + [
                (f"both capped {m}", cc[m]) for m in cc]:
            rows_cap.append({"hypothesis": h, "contrast": hyp[h]["contrast"], "setting": k, "a": v["a"], "b": v["b"],
                             "delta": v["delta"], "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1],
                             "p_one_sided": v["p_one_sided"]})
        rows_cap.append({"hypothesis": h, "contrast": hyp[h]["contrast"], "setting": "without limit mAP50 (point)",
                         "delta": nl(h)})
    L += ["", "Both capped, other metrics: " + "; ".join(
        f"{h} AP {ci(c[CAPK]['AP'])}, R50 crowded {ci(c[CAPK]['R50_crowd'])}, R50 isolated {ci(c[CAPK]['R50_iso'])}"
        for h, c in cap["contrasts"].items()) + ".", "", "Detections per image:", "",
          "| row | max | mean | median | images over the cap |", "|---|---:|---:|---:|---:|"]
    pi = [{"row": k, **v} for k, v in cap["per_image"].items()]
    for r in pi:
        L.append(f"| {r['row']} | {r['max']:,} | {r['mean']:,.0f} | {r['median']:,.0f} | {r['images_over_cap']} |")
    L += ["", "## 2. Prediction redundancy (predictions with IoU >= 0.5 per non-difficult GT object, 2048; class-agnostic)",
          "", "Candidates: anchors with max class probability >= tau_splat of the row; all passes ('total'), the "
          "whole-image pass, and the mean over the tile passes whose view contains the object centre.", "",
          "| model | tau_splat (RG) | n GT | all passes: mean (median) all / crowded / isolated | whole image | "
          "per tile pass | zero |", "|---|---|---:|---|---:|---:|---:|"]
    rr = []
    for m, o in red["models"].items():
        st = o["stats"]
        g = lambda k, grp, q="mean": st[k][grp].get(q)  # noqa: E731
        t = "RG/agnostic/total"
        L.append(f"| {m} | {o['tau_splat']['RG']} | {o['n_gt']:,} | " + " / ".join(
            f"{g(t, x):.1f} ({g(t, x, 'q50'):.0f})" for x in ("all", "crowded", "isolated"))
                 + f" | {g('RG/agnostic/whole', 'all'):.1f} | {g('RG/agnostic/tile_pass_mean', 'all'):.1f} | "
                 f"{100 * g(t, 'all', 'share_zero'):.1f} % |")
        for k, grps in st.items():
            for grp, v in grps.items():
                rr.append({"model": m, "tau": k.split("/")[0], "tau_splat": o["tau_splat"][k.split("/")[0]],
                           "class_mode": k.split("/")[1], "count": k.split("/")[2], "group": grp, **v})
    L += ["", "## 3. Crowded / isolated recall (R50), RG − box fusion (NMS-o2m)", "",
          "| contrast | R50 crowded | R50 isolated |", "|---|---|---|"]
    rc = []
    for k, c in rec["contrasts"].items():
        L.append(f"| {k} | {c['R50_crowd']['a']:.1f} vs. {c['R50_crowd']['b']:.1f}: {ci(c['R50_crowd'])} | "
                 f"{c['R50_iso']['a']:.1f} vs. {c['R50_iso']['b']:.1f}: {ci(c['R50_iso'])} |")
        for m in ("R50_crowd", "R50_iso"):
            rc.append({"contrast": k, "metric": m, "a": c[m]["a"], "b": c[m]["b"], "delta": c[m]["delta"],
                       "ci_lo": c[m]["ci95"][0], "ci_hi": c[m]["ci95"][1], "p_one_sided": c[m]["p_one_sided"]})
    L += ["", f"Crowded / isolated GT: {next(iter(rec['rows'].values()))['n_crowd_gt']:,} / "
          f"{next(iter(rec['rows'].values()))['n_iso_gt']:,}.", "",
          f"## 4. DOTA-val limit check (pipeline limit {lim['limit']:,}; mAP50 without cap)", "",
          "Affected images recomputed without the limit (10,000,000); box-fusion rows: passes at the per-pass limit "
          "(their merged output is not capped). Dense outputs of the re-run passes identical to the stored caches: "
          f"{lim['nolimit_dense_identical']['all']} ({lim['nolimit_dense_identical']['n']} passes). Point values, no "
          "CIs; verdicts stay as computed under the pre-registered limit.", "",
          "| row | affected images | mAP50 with limit | without limit | Δ |", "|---|---:|---:|---:|---:|"]
    rl = []
    for i, r in lim["rows"].items():
        n = r.get("images_with_pass_at_limit", r["images_at_image_limit"])
        L.append(f"| {i} | {n} | {r['mAP50_limited']:.2f} | {r['mAP50_unlimited']:.2f} | {r['delta']:+.2f} |")
        rl.append({"row": i, "kind": r["kind"], "affected_images": n, "mAP50_limited": r["mAP50_limited"],
                   "mAP50_unlimited": r["mAP50_unlimited"], "delta": r["delta"]})
    L += ["", "| hypothesis | Δ mAP50 with limit [95 % CI] | verdict | Δ without limit |", "|---|---|---|---:|"]
    for h, v in hyp.items():
        L.append(f"| {h}: {v['contrast']} | {ci(v['mAP50'])} | {'met' if v['met'] else 'not met'} | {f(nl(h))} |")
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "FCOS_DIAGNOSTICS.md").write_text("\n".join(L) + "\n")
    for name, recs in (("cap", rows_cap), ("per_image", pi), ("redundancy", rr), ("recall", rc), ("limit_check", rl)):
        pd.DataFrame(recs).to_csv(rd / f"tab_fcos_diag_{name}.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
        sys.exit(0)
    import torch

    torch.set_num_threads(1)
    cmd = sys.argv[1]
    if cmd == "cap":
        cap()
    elif cmd == "redundancy":
        redundancy(workers=int(os.environ.get("SCAHI_DIAG_WORKERS", "8")))
    elif cmd == "recall":
        recall()
