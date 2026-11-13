# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two exploratory post-hoc analyses for the supplement (user 2026-11-13; framing in docs/DECISIONS.md 2026-11-13 08:40,
recorded before any number). Descriptive only: no verdict, no frozen row or setting changes.

    python scripts/dota_native_more.py caches   # GPU alone: native dense o2m caches (z = 1) of YOLOv10s / YOLO11s on F
    python scripts/dota_native_more.py fcos     # GPU alone: FCOS native whole-image pass on all DOTA-val images
    python scripts/dota_native_more.py fuse     # CPU: MZ-RG (SCAHI) and MZ-NMS (box fusion) on native + 2048 pass
    python scripts/dota_native_more.py eval     # CPU: row and size-bin evaluations (bootstrap vectors) on the subsets
    python scripts/dota_native_more.py analyse  # contrasts -> runs/dota/native_more/report.json, then report
    python scripts/dota_native_more.py report   # report.json -> reports/dota/NATIVE_MORE_EXPLORATORY.md (+ JSON)

(1) Multi-zoom without tiles (YOLOv10s, YOLO11s; subset F of the native-pass analysis): native pass (z = 1, padding
    only) + the frozen w2048 pass, o2m head; MZ-RG = SCAHI F2 with the RG_2048 parameters (whole-image passes: no taper,
    C = 2), MZ-NMS = box fusion with the NMS-o2m_2048 setting; vs. RG_2048, A_2048, N_z1.
(2) FCOS control: frozen 2048 weights as a native whole-image pass (scale 1), own score, limits / conf of A_fcos_2048;
    subset F_fcos = images without OOM; vs. A_fcos_2048, RG_fcos_2048, NMS-o2m_fcos_2048.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import dota_native_pass as NP  # noqa: E402

OUT = REPO / "runs/dota/native_more"
REP = REPO / "reports/dota"
CACHE = OUT / "cache"
MODELS = NP.MODELS
S = 2048
PAD = 114


def _fz(m: str):
    from scahi.dota import final as F

    path = {"yolov10s": F.FROZEN, "yolo11s": F.FROZEN_NEW, "fcos": F.FROZEN_FCOS}[m]
    fz = F.frozen(path)
    return {r["id"]: r for r in fz["rows"]}, fz["pipeline_limit"]


# ------------------------------------------------------------------------------------------------ GPU

def native_level(model, img: np.ndarray) -> dict[str, np.ndarray]:
    """Dense o2m outputs of the whole image at z = 1 (padded bottom/right to multiples of 32, gain 1), in the cache format
    of ``scahi.infer.cache_mz.raw_predict_level`` (o2m keys only)."""
    import cv2
    import torch

    from scahi.infer.cache import SCORE_MIN
    from scahi.infer.dfl_decode import decode
    from scahi.infer.preprocess import to_tensor
    from scahi.infer.raw_head import forward_raw_branch

    H, W = img.shape[:2]
    Hp, Wp = -(-H // 32) * 32, -(-W // 32) * 32
    x = cv2.copyMakeBorder(img, 0, Hp - H, 0, Wp - W, cv2.BORDER_CONSTANT, value=(PAD, PAD, PAD))
    with torch.no_grad():
        raw = forward_raw_branch(model, to_tensor([x], "cuda"), "o2m")
        d = decode(raw.boxes["o2m"], raw.scores["o2m"], raw.anchors, raw.strides, raw.reg_max)
        keep = d.probs.max(-1)[0] >= SCORE_MIN
        k = keep[0]
        out = {"o2m_xyxy": d.xyxy[0][k].cpu().numpy().astype(np.float32),
               "o2m_probs": d.probs[0][k].cpu().numpy().astype(np.float32),
               "o2m_std": d.ltrb_std[0][k].cpu().numpy().astype(np.float32),
               "o2m_tile": np.zeros(int(k.sum()), np.int16)}
    out.update(tiles=np.array([[0, 0, W, H]], np.int32), gains=np.array([1.0]), zoom=np.float64(1.0),
               n_input=np.int32(0), image_wh=np.array([W, H], np.int32))
    return out


def _parity(model) -> dict:
    """On an image whose sides are multiples of 32 (no padding, gain 1 in both paths): native_level == the o2m keys of
    ``cache_mz.raw_predict_level`` at level w1.0 (the pipeline's whole-image forward, both branches)."""
    import cv2

    from scahi.dota import data as D
    from scahi.infer import cache_mz as cm

    st = "P1759"  # 4000 x 4000
    img = cv2.imread(str(D.image_path("val", st)))
    assert img.shape[0] % 32 == 0 and img.shape[1] % 32 == 0
    a = native_level(model, img)
    b = cm.raw_predict_level(model, img, cm.LEVELS["w1.0"], 0.3)
    return {k: bool(np.array_equal(a[k], b[k])) for k in ("o2m_xyxy", "o2m_probs", "o2m_std", "o2m_tile", "tiles", "gains")}


def caches() -> None:
    import cv2
    import torch

    from scahi.dota import data as D
    from scahi.dota import train as T
    from scahi.infer.raw_head import load_unfused

    res = {}
    for m in MODELS:
        d = CACHE / m
        d.mkdir(parents=True, exist_ok=True)
        model = load_unfused(T.weights(m, S))
        par = _parity(model)
        assert all(par.values()), par
        per = {}
        for st in NP._subset(m):
            f = d / f"{st}.npz"
            if f.exists():
                continue
            img = cv2.imread(str(D.image_path("val", st)))
            torch.cuda.reset_peak_memory_stats()
            try:
                np.savez_compressed(f, **native_level(model, img))
                per[st] = {"fits": True, "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}
            except torch.cuda.OutOfMemoryError:
                per[st] = {"fits": False}
                print(m, st, "OOM", flush=True)
            torch.cuda.empty_cache()
        res[m] = {"parity_vs_raw_predict_level": par, "images": per,
                  "missing": [st for st in NP._subset(m) if not (d / f"{st}.npz").exists()]}
        print(m, "caches", len(NP._subset(m)) - len(res[m]["missing"]), "of", len(NP._subset(m)), flush=True)
        del model
        torch.cuda.empty_cache()
    (OUT / "caches.json").write_text(json.dumps(res, indent=1))


def fcos() -> None:
    import cv2
    import torch

    from scahi import fcos as FC
    from scahi.dota import data as D
    from scahi.dota import evaluate as EV
    from scahi.dota import fcos_train as FT
    from scahi.dota import pipeline as P
    from scahi.dota import train as T

    rows, lim = _fz("fcos")
    a = rows[f"A_fcos_{S}"]
    stems = P.split_stems("val")
    from PIL import Image

    hw = {}
    for s in stems:
        with Image.open(D.image_path("val", s)) as im:
            hw[s] = (im.size[1], im.size[0])
    order = sorted(stems, key=lambda s: hw[s][0] * hw[s][1])
    m = FT.load(T.weights("fcos", S), S, lim)
    m.eval()
    out, per = [], {}
    for n, st in enumerate(order):
        img = cv2.imread(str(D.image_path("val", st)))
        H, W = img.shape[:2]
        FC.set_long_side(m, max(H, W))  # scale = long side / long side = 1
        rec = {"W": W, "H": H}
        try:
            with torch.no_grad():
                if n == 0:
                    m([FC._to_tensor(img, "cuda")])  # warm-up
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                o = m([FC._to_tensor(img, "cuda")])[0]
                torch.cuda.synchronize()
                rec["ms"] = 1000 * (time.perf_counter() - t0)
                rec["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
            k = o["scores"] >= a["conf"]
            bx, sc, cl = o["boxes"][k].cpu().numpy(), o["scores"][k].cpu().numpy(), o["labels"][k].cpu().numpy()
            out.append(pd.DataFrame({"image": st, "cls": cl.astype(int), "score": sc.astype(float),
                                     "x1": bx[:, 0].astype(float), "y1": bx[:, 1].astype(float),
                                     "x2": bx[:, 2].astype(float), "y2": bx[:, 3].astype(float)}))
            rec.update(fits=True, n_dets=int(k.sum()), at_limit=bool(len(o["scores"]) >= lim))
            del o
        except torch.cuda.OutOfMemoryError:
            rec.update(fits=False, error="CUDA out of memory")
        torch.cuda.empty_cache()
        per[st] = rec
        if n % 50 == 0 or not rec["fits"]:
            print("fcos", n, st, rec, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=EV.DET_COLS)).to_parquet(OUT / "N_z1_fcos.parquet")
    (OUT / "N_z1_fcos.json").write_text(json.dumps({
        "model": "fcos", "A_row": a["id"], "conf": a["conf"], "pipeline_limit": lim, "dtype": "float32", "batch": 1,
        "gpu": torch.cuda.get_device_name(0), "images": per}, indent=1))
    print("fcos fit", sum(v["fits"] for v in per.values()), "of", len(per), flush=True)


# ------------------------------------------------------------------------------------------------ CPU

def subset(m: str) -> list[str]:
    from scahi.dota import pipeline as P

    if m != "fcos":
        return NP._subset(m)
    per = json.loads((OUT / "N_z1_fcos.json").read_text())["images"]
    return [s for s in P.split_stems("val") if per[s]["fits"]]


def _mz_data(m: str, stems: list[str]) -> dict:
    from scahi.dota import pipeline as P
    from scahi.dota import train as T
    from scahi.fusion import scalespace as ss
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm

    w2048 = cm.cache_dir(CACHE_ROOT, T.weights(m, S), P.WHOLE[S], P.OVERLAP, "val")
    dirs = [("w1.0", CACHE / m), (P.WHOLE[S], w2048)]
    return {st: ss.load_levels(dirs, st, "o2m", P.OVERLAP) for st in stems}


def _fuse_job(job: tuple[str, str]) -> str:
    import torch

    from scahi.dota import final as F
    from scahi.dota import pipeline as P

    torch.set_num_threads(1)
    m, name = job
    f = OUT / "dets" / f"{name}_{m}.parquet"
    if f.exists():
        return f.stem
    rows, lim = _fz(m)
    src = rows[f"{'RG' if name == 'MZ-RG' else 'NMS-o2m'}_{m}_{S}"]
    t0 = time.time()
    data = _mz_data(m, subset(m))
    d = P.fuse(data, "o2m", src["variant"], src["cfg"], limit=lim)
    d.reset_index(drop=True).to_parquet(f)
    (f.with_suffix(".json")).write_text(json.dumps({"source_row": src["id"], "variant": src["variant"], "cfg": src["cfg"],
                                                    "pipeline_limit": lim, "levels": ["w1.0", P.WHOLE[S]],
                                                    "images_at_limit": P.at_limit(d, lim), "n_dets": len(d),
                                                    "wall_s": time.time() - t0}, indent=1))
    del F
    return f.stem


def fuse(workers: int = 4) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from scahi.dota import final as F

    F._cpu_slot()
    (OUT / "dets").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        for k in ex.map(_fuse_job, [(m, n) for m in MODELS for n in ("MZ-RG", "MZ-NMS")]):
            print("done", k, flush=True)


def _dets(m: str, name: str) -> pd.DataFrame:
    from scahi.dota import final as F

    if name in ("MZ-RG", "MZ-NMS"):
        return pd.read_parquet(OUT / "dets" / f"{name}_{m}.parquet")
    if name == "N_z1":
        return pd.read_parquet((OUT if m == "fcos" else NP.OUT) / f"N_z1_{m}.parquet")
    return pd.read_parquet(F.DETS_DIR / f"{name}_{m}_{S}.parquet")


def _eval_job(job: tuple[str, str, str]) -> str:
    """job = (model, row, kind): kind 'rows' (evaluate_row metrics + bootstrap vectors) or 'sizes' (size bins)."""
    import torch

    from scahi.dota import final as F
    from scahi.dota import pipeline as P

    torch.set_num_threads(1)
    m, name, kind = job
    f = OUT / kind / f"{name}_{m}_F.json"
    if f.exists():
        return f.stem
    sub = subset(m)
    d = _dets(m, name)
    d = d[d.image.isin(set(sub))].reset_index(drop=True)
    if kind == "rows":
        with mock.patch.object(P, "split_stems", lambda split: list(sub)):
            o, v = F.evaluate_row(f"{name}_{m}_F", "val", dets=d)
        o["n_images"] = len(sub)
    else:
        pts, v = NP.size_metrics(d, sub)
        o = {"n_images": len(sub), "bins": pts}
    np.savez(OUT / kind / f"{f.stem}_boot.npz", **v)
    f.write_text(json.dumps(o, indent=1))
    return f.stem


def jobs() -> list[tuple[str, str, str]]:
    js = [(m, n, k) for m in MODELS for n in ("MZ-RG", "MZ-NMS") for k in ("rows", "sizes")]
    js += [("fcos", n, k) for n in ("N_z1", "A", "RG", "NMS-o2m") for k in ("rows", "sizes")]
    js += [("yolov10s", "A", "check")]  # size_metrics refactor: must reproduce the stored native-pass size bins
    return js


def _check_job() -> str:
    pts, v = NP.size_metrics(_dets("yolov10s", "A"), NP._subset("yolov10s"))
    ref = json.loads((NP.OUT / "sizes" / "A_yolov10s_F.json").read_text())["bins"]
    rv = dict(np.load(NP.OUT / "sizes" / "A_yolov10s_F_boot.npz"))
    eq = lambda a, b: a == b or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))  # noqa: E731
    ok = all(eq(pts[b][k], ref[b][k]) for b in ref for k in ref[b]) and all(
        np.array_equal(v[k], rv[k], equal_nan=True) for k in rv)
    (OUT / "check_size_metrics.json").write_text(json.dumps({"identical": ok, "note": "NaN-aware comparison (empty <8 bin)"}))
    return f"check identical={ok}"


def _job(j):
    return _check_job() if j[2] == "check" else _eval_job(j)


def evaluate(workers: int = 7) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from scahi.dota import final as F

    F._cpu_slot()
    for k in ("rows", "sizes"):
        (OUT / k).mkdir(parents=True, exist_ok=True)
    js = sorted(jobs(), key=lambda j: (j[1] not in ("MZ-RG", "RG"), j[2] != "rows"))
    with ProcessPoolExecutor(workers) as ex:
        for k in ex.map(_job, js):
            print("done", k, flush=True)


# ------------------------------------------------------------------------------------------------ report

KEYS = NP.KEYS


def _ld(m: str, name: str, kind: str) -> tuple[dict, dict]:
    """Evaluation (point dict, bootstrap vectors) of a row on F of the detector; reuses the native-pass artefacts."""
    if m != "fcos" and name in ("N_z1", "A", "RG", "NMS-o2m"):
        base = NP.OUT / ("eval" if kind == "rows" else "sizes") / f"{name}_{m}_F"
    else:
        base = OUT / kind / f"{name}_{m}_F"
    o = json.loads(base.with_suffix(".json").read_text())
    return (o if kind == "rows" else o["bins"]), dict(np.load(f"{base}_boot.npz"))


def _con(va, vb, pa, pb, k) -> dict:
    d = va[k] - vb[k]
    return {"a": pa[k], "b": pb[k], "delta": pa[k] - pb[k],
            "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}


def _size_con(za: tuple[dict, dict], zb: tuple[dict, dict]) -> dict:
    (pa, va), (pb, vb) = za, zb
    out = {}
    for lbl in pa:
        if pa[lbl]["n_gt"] == 0:
            continue
        out[lbl] = {k: {"a": pa[lbl][k], "b": pb[lbl][k], "delta": pa[lbl][k] - pb[lbl][k],
                        "ci95": [float(np.nanpercentile(va[f"{lbl}|{k}"] - vb[f"{lbl}|{k}"], q)) for q in (2.5, 97.5)]}
                    for k in ("AP50", "AP", "R50")} | {"n_gt": pa[lbl]["n_gt"]}
    return out


def _counts(stems: list[str]) -> dict:
    from scahi.dota import data as D

    ann = D.load("val")
    return {"images": len(stems), "of": len(ann),
            "objects": sum(len(ann[s].cls) for s in stems), "objects_total": sum(len(a.cls) for a in ann.values()),
            "objects_non_difficult": int(sum((~ann[s].difficult).sum() for s in stems)),
            "objects_non_difficult_total": int(sum((~a.difficult).sum() for a in ann.values()))}


def analyse() -> dict:
    res: dict = {"framing": "docs/DECISIONS.md 2026-11-13 08:40 (exploratory, post hoc, descriptive; no verdict)",
                 "bootstrap": {"n": 1000, "seed": 0, "ci": "percentile 2.5/97.5, paired over the images of the subset"},
                 "check_size_metrics_refactor": json.loads((OUT / "check_size_metrics.json").read_text()),
                 "native_caches": {m: v["parity_vs_raw_predict_level"]
                                   for m, v in json.loads((OUT / "caches.json").read_text()).items()},
                 "detectors": {}}
    plan = {m: (["MZ-RG", "MZ-NMS", "RG", "A", "N_z1", "NMS-o2m"],
                [("MZ-RG", "RG"), ("MZ-RG", "A"), ("MZ-RG", "N_z1"), ("MZ-RG", "MZ-NMS"), ("MZ-NMS", "A"),
                 ("MZ-NMS", "NMS-o2m")],
                [("MZ-RG", "RG"), ("MZ-RG", "A"), ("MZ-NMS", "A")]) for m in MODELS}
    plan["fcos"] = (["N_z1", "A", "RG", "NMS-o2m"], [("N_z1", "A"), ("N_z1", "RG"), ("N_z1", "NMS-o2m")],
                    [("N_z1", "A"), ("N_z1", "RG")])
    for m, (names, cons, scons) in plan.items():
        sub = subset(m)
        R = {n: _ld(m, n, "rows") for n in names}
        Z = {n: _ld(m, n, "sizes") for n in names}
        r = {"subset": _counts(sub), "points": {n: {k: R[n][0][k] for k in KEYS} for n in names},
             "contrasts": {f"{a} − {b}": {k: _con(R[a][1], R[b][1], R[a][0], R[b][0], k) for k in KEYS} for a, b in cons},
             "size_points": {n: Z[n][0] for n in names},
             "size_contrasts": {f"{a} − {b}": _size_con(Z[a], Z[b]) for a, b in scons}}
        if m == "fcos":
            meta = json.loads((OUT / "N_z1_fcos.json").read_text())["images"]
            ms, mem = [meta[s]["ms"] for s in sub], [meta[s]["peak_mem_mb"] for s in sub]
            r["failing_images"] = {s: {"W": v["W"], "H": v["H"]} for s, v in meta.items() if not v["fits"]}
            r["runtime_ms"] = {"mean": float(np.mean(ms)), "median": float(np.median(ms)), "max": float(np.max(ms))}
            r["peak_mem_mb"] = {"median": float(np.median(mem)), "max": float(np.max(mem))}
            r["images_at_limit"] = sum(bool(meta[s].get("at_limit")) for s in sub)
        else:
            r["fusion"] = {n: json.loads((OUT / "dets" / f"{n}_{m}.json").read_text()) for n in ("MZ-RG", "MZ-NMS")}
        res["detectors"][m] = r
    (OUT / "report.json").write_text(json.dumps(res, indent=1))
    return render(res)


def report() -> dict:
    return render(json.loads((OUT / "report.json").read_text()))


NAMES = {"MZ-RG": "native + 2048 pass, SCAHI (RG parameters)", "MZ-NMS": "native + 2048 pass, box fusion",
         "RG": "tiles + SCAHI (RG_2048)", "A": "single pass 2048 (A_2048)", "N_z1": "native single pass (z = 1)",
         "NMS-o2m": "tiles + box fusion (NMS-o2m_2048)"}


def render(res: dict) -> dict:
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "native_more_exploratory.json").write_text(json.dumps(res, indent=1))
    L = ["# Multi-zoom without tiles and FCOS native control on DOTA (exploratory, post hoc)", "",
         f"Framing: {res['framing']}. DOTA-val, devkit mAP50 without cap, AP@[.5:.95], recall@0.5 crowded / isolated GT, "
         "size bins on sigma = sqrt(w h) of the GT; paired image bootstrap (1000, seed 0), 95 % percentile CI, over the "
         "images of the subset. Not for paper tables without the authors' decision.", "",
         "Rows: " + "; ".join(f"{k} = {v}" for k, v in NAMES.items()) + ".", ""]
    for m, r in res["detectors"].items():
        s = r["subset"]
        L += [f"## {m} (2048 weights)", "",
              f"Subset: {s['images']} / {s['of']} images; objects {s['objects']} / {s['objects_total']} "
              f"({100 * s['objects'] / s['objects_total']:.1f} %), non-difficult {s['objects_non_difficult']} / "
              f"{s['objects_non_difficult_total']} ({100 * s['objects_non_difficult'] / s['objects_non_difficult_total']:.1f} %)."]
        if "runtime_ms" in r:
            L.append(f"Native FCOS pass: {r['runtime_ms']['mean']:.0f} ms mean, {r['runtime_ms']['median']:.0f} median, "
                     f"{r['runtime_ms']['max']:.0f} max; peak memory {r['peak_mem_mb']['median'] / 1024:.2f} GB median, "
                     f"{r['peak_mem_mb']['max'] / 1024:.1f} GB max; images at the limit {r['images_at_limit']}; failing (OOM): "
                     + (", ".join(f"{k} ({v['W']}x{v['H']})" for k, v in r["failing_images"].items()) or "none") + ".")
        else:
            L.append("Images at the pipeline limit: " + ", ".join(f"{n} {v['images_at_limit']}" for n, v in r["fusion"].items()) + ".")
        L += ["", "| row | mAP50 | AP | R50 crowded | R50 isolated |", "|---|---|---|---|---|"]
        for n, p in r["points"].items():
            L.append(f"| {n} | " + " | ".join(f"{p[k]:.2f}" for k in KEYS) + " |")
        L += ["", "| contrast | Δ mAP50 [95 % CI] | Δ AP | Δ R50 crowded | Δ R50 isolated |", "|---|---|---|---|---|"]
        for nm, c in r["contrasts"].items():
            L.append(f"| {nm} | " + " | ".join(f"{c[k]['delta']:+.2f} [{c[k]['ci95'][0]:+.2f}, {c[k]['ci95'][1]:+.2f}]"
                                                for k in KEYS) + " |")
        L.append("")
        for nm, cs in r["size_contrasts"].items():
            L += [f"Size bins, {nm}:", "",
                  "| bin | GT | AP50 a / b | Δ AP50 [95 % CI] | Δ AP [95 % CI] | R50 a / b | Δ R50 [95 % CI] |",
                  "|---|---|---|---|---|---|---|"]
            for lbl, c in cs.items():
                f_ = lambda k: f"{c[k]['delta']:+.2f} [{c[k]['ci95'][0]:+.2f}, {c[k]['ci95'][1]:+.2f}]"  # noqa: E731
                L.append(f"| {lbl} | {c['n_gt']} | {c['AP50']['a']:.1f} / {c['AP50']['b']:.1f} | {f_('AP50')} | {f_('AP')} | "
                         f"{c['R50']['a']:.1f} / {c['R50']['b']:.1f} | {f_('R50')} |")
            L.append("")
    (REP / "NATIVE_MORE_EXPLORATORY.md").write_text("\n".join(L))
    print("\n".join(L))
    return res


if __name__ == "__main__":
    {"caches": caches, "fcos": fcos, "fuse": fuse, "eval": evaluate, "analyse": analyse, "report": report}[sys.argv[1]]()
