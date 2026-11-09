# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-image analysis "when do tiles pay off" (exploratory, post hoc; rule in docs/DECISIONS.md, 2026-10-31 ~14:50).

    python scripts/tiles_payoff.py compute [--workers 8]   # per-image ΔAP + covariates -> reports/tiles_payoff/*.csv
    python scripts/tiles_payoff.py report                  # Spearman + bootstrap CI, figure, markdown

ΔAP per image = tiles + SCAHI (refined grid) 2048 - single pass 2048, same detector and head, from the stored
detections of the frozen rows only. VisDrone test-dev: toolkit-port AP@[.5:.95] of the image (as fig_qualitative_v2);
DOTA-val: devkit mAP50 of the image (as d_qualitative.py). Covariates over the valid GT of the image (not ignored / not
difficult): share with sqrt(w h) < 16 px; share crowded (same-class GT neighbour with IoU > 0.3).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/dota"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = REPO / "reports/tiles_payoff"
FIG = REPO / "reports/figures/ext/fig_ext_tiles_payoff"
LABEL = "exploratory, post hoc"
N_BOOT, SEED = 2000, 0
BINS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0 + 1e-9)
MIN_BIN = 10
TD = REPO / "runs/ext/testdev"
R2 = REPO / "runs/testdev/20261022_v10s_BFULL2048_full2048_none_s0"
NAME = {"yolov10s": "YOLOv10s", "yolov10m": "YOLOv10m", "yolo11s": "YOLOv11s", "yolo11m": "YOLOv11m", "yolo12s": "YOLOv12s"}
VD_PAIRS = (("yolov10s", "o2o"), ("yolov10m", "o2m"), ("yolov10m", "o2o"), ("yolo11s", "o2m"), ("yolo11m", "o2m"),
            ("yolo12s", "o2m"))
DOTA_MODELS = ("yolov10s", "yolov10m")


def vd_runs(m: str, h: str) -> tuple[Path, Path]:
    single = R2 if m == "yolov10s" else TD / f"E_{m}_full2048_{h}"
    xg = [p for p in TD.glob(f"XG_{m}_mz2048_*_{h}") if p.is_dir()]
    assert len(xg) == 1, (m, h, xg)
    return single, xg[0]


def _vd_ap(args: tuple[str, int, int]) -> tuple[str, np.ndarray]:
    """Per-image toolkit-port AP of one run (one-hot image weights on the full split cache, as fig_qualitative_v2)."""
    from scahi import ext_testdev as T
    from scahi.config import load_class_setting
    from scahi.eval import bootstrap as Bt

    run, lo, hi = args
    coco, _ = T.split_data(T.SPLIT)
    ims = Bt.prepared(coco, pd.read_parquet(Path(run) / f"dets_{T.SPLIT}.parquet"), load_class_setting("visdrone10"))
    pc = Bt.port_cache(ims, tuple(sorted(c["id"] for c in coco["categories"])))
    out = np.full(hi - lo, np.nan)
    for i in range(lo, hi):
        w = np.zeros(len(ims))
        w[i] = 1
        out[i - lo] = Bt.port_ap(pc, w)
    return run, out


def vd_covariates() -> pd.DataFrame:
    from ext_posthoc import iou_xywh

    from scahi import ext_testdev as T
    from scahi.config import load_class_setting
    from scahi.eval import bootstrap as Bt

    coco, _ = T.split_data(T.SPLIT)
    ims = Bt.prepared(coco, pd.read_parquet(R2 / f"dets_{T.SPLIT}.parquet"), load_class_setting("visdrone10"))
    rows = []
    for im in ims:
        gx, gc, gi = im["gt_xywh"], im["gt_cls"], im["gt_ignore"].astype(bool)
        crowd = np.zeros(len(gx), bool)
        for c in np.unique(gc):  # as scripts/ext_posthoc.per_image
            ix = np.flatnonzero(gc == c)
            if len(ix) > 1:
                gg = iou_xywh(gx[ix], gx[ix])
                np.fill_diagonal(gg, 0)
                crowd[ix] = (gg > 0.3).any(1)
        v = ~gi
        n = int(v.sum())
        small = np.sqrt(gx[v, 2] * gx[v, 3]) < 16
        rows.append({"image_id": im["image_id"], "n_gt": n, "share_lt16": small.mean() if n else np.nan,
                     "share_crowded": (crowd & v).sum() / n if n else np.nan})
    return pd.DataFrame(rows)


def compute_visdrone(workers: int) -> None:
    from scahi import ext_testdev as T

    cov = vd_covariates()
    n = len(cov)
    runs = sorted({str(r) for m, h in VD_PAIRS for r in vd_runs(m, h)})
    chunks = [(r, 0, n) for r in runs]  # one task per run (cache built once)
    ap: dict[str, np.ndarray] = {r: np.full(n, np.nan) for r in runs}
    with ProcessPoolExecutor(workers) as ex:
        for (run, lo, _), (_, v) in zip(chunks, ex.map(_vd_ap, chunks)):
            ap[run][lo:lo + len(v)] = v
    recs = []
    for m, h in VD_PAIRS:
        s, x = vd_runs(m, h)
        recs.append(cov.assign(dataset="VisDrone test-dev", detector=NAME[m], head=h, single_run=str(s.relative_to(REPO)),
                               scahi_run=str(x.relative_to(REPO)), AP_single=ap[str(s)], AP_scahi=ap[str(x)]))
    df = pd.concat(recs, ignore_index=True)
    df["dAP"] = df.AP_scahi - df.AP_single
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"per_image_visdrone_{T.SPLIT}.csv", index=False)
    print("visdrone", df.groupby("detector").dAP.describe(), flush=True)


def compute_dota() -> None:
    import d_qualitative as DQ

    from scahi.dota import evaluate as EV
    from scahi.dota import final as F
    from scahi.dota import pipeline as P

    stems = P.split_stems("val")
    gt = EV.gt_arrays("val", stems)
    crowd = F.crowd_gt(gt)
    cov = []
    for s in stems:
        b, _, dif = gt[s]
        v = ~dif
        n = int(v.sum())
        w, h = b[v, 2] - b[v, 0], b[v, 3] - b[v, 1]
        cov.append({"image": s, "n_gt": n, "share_lt16": (np.sqrt(w * h) < 16).mean() if n else np.nan,
                    "share_crowded": crowd[s].sum() / n if n else np.nan})
    cov = pd.DataFrame(cov)
    recs = []
    for m in DOTA_MODELS:
        ap = {k: DQ.per_image_ap(gt, pd.read_parquet(F.DETS_DIR / f"{k}_{m}_2048.parquet")) for k in ("A", "RG")}
        recs.append(cov.assign(dataset="DOTA-val", detector=NAME[m], head="o2o / o2m", single_run=f"A_{m}_2048",
                               scahi_run=f"RG_{m}_2048", AP_single=cov.image.map(ap["A"]),
                               AP_scahi=cov.image.map(ap["RG"])))
    df = pd.concat(recs, ignore_index=True)
    df["dAP"] = df.AP_scahi - df.AP_single
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "per_image_dota_val.csv", index=False)
    print("dota", df.groupby("detector").dAP.describe(), flush=True)


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.n_gt > 0) & df.dAP.notna()]


def spearman_ci(x: np.ndarray, y: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    rng = np.random.default_rng(SEED)
    rho = float(spearmanr(x, y).statistic)
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        bs.append(spearmanr(x[i], y[i]).statistic)
    bs = np.asarray(bs, float)
    return {"rho": rho, "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "n_img": int(len(x))}


def binned(x: np.ndarray, y: np.ndarray) -> list[dict]:
    rng = np.random.default_rng(SEED)
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = (x >= lo) & (x < hi)
        v = y[sel]
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)] if len(v) else []
        out.append({"bin": f"[{lo:g}, {min(hi, 1):g}{']' if hi > 1 else ')'}", "lo": lo, "hi": min(hi, 1.0),
                    "n_img": int(len(v)), "mean_dAP": float(v.mean()) if len(v) else None,
                    "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(v) else None,
                    "plotted": bool(len(v) >= MIN_BIN)})
    return out


def display(name: str) -> str:
    """Displayed model name (paper decision 2026-11-07): YOLOv11 / YOLOv12; stored data keep Ultralytics' names."""
    return re.sub(r"\bYOLO(1[12])", r"YOLOv\1", name)


def report() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import figstyle as FS

    vd = _valid(pd.read_csv(OUT / "per_image_visdrone_test-dev.csv"))
    dt = _valid(pd.read_csv(OUT / "per_image_dota_val.csv"))
    vd["detector"], dt["detector"] = vd.detector.map(display), dt.detector.map(display)  # stored with the old names
    res = {"label": LABEL, "rule": "docs/DECISIONS.md 2026-10-31 ~14:50", "n_boot": N_BOOT, "seed": SEED, "rows": []}
    for df in (vd, dt):
        for (ds, det, head), g in df.groupby(["dataset", "detector", "head"], sort=False):
            r = {"dataset": ds, "detector": det, "head": head, "n_img": len(g), "mean_dAP": float(g.dAP.mean()),
                 "single_run": g.single_run.iloc[0], "scahi_run": g.scahi_run.iloc[0]}
            for cv in ("share_lt16", "share_crowded"):
                r[f"spearman_{cv}"] = spearman_ci(g[cv].to_numpy(float), g.dAP.to_numpy(float))
                r[f"binned_{cv}"] = binned(g[cv].to_numpy(float), g.dAP.to_numpy(float))
            res["rows"].append(r)
    (OUT / "tiles_payoff.json").write_text(json.dumps(res, indent=1))
    flat = [{k: v for k, v in r.items() if not k.startswith("binned")} | {
        f"rho_{cv}": r[f"spearman_{cv}"]["rho"], f"rho_{cv}_lo": r[f"spearman_{cv}"]["ci95"][0],
        f"rho_{cv}_hi": r[f"spearman_{cv}"]["ci95"][1]} for r in res["rows"] for cv in ("share_lt16",)]
    for f, r in zip(flat, res["rows"]):
        f.pop("spearman_share_lt16"), f.pop("spearman_share_crowded")
        c = r["spearman_share_crowded"]
        f |= {"rho_share_crowded": c["rho"], "rho_share_crowded_lo": c["ci95"][0], "rho_share_crowded_hi": c["ci95"][1]}
    pd.DataFrame(flat).to_csv(OUT / "tiles_payoff_spearman.csv", index=False, float_format="%.4f")

    # ---- figure: rows = covariate, columns = dataset; YOLOv10s highlighted, other detectors thin grey
    FS.apply()
    fig, axs = plt.subplots(2, 2, figsize=(FS.FULL_W, 4.4), sharey="col")
    cov_lbl = {"share_lt16": "share of GT objects < 16 px", "share_crowded": "share of crowded GT objects"}
    col = {"YOLOv10s": FS.BLUE, "YOLOv10m": FS.ORANGE}
    rec = []
    for j, ds in enumerate(("VisDrone test-dev", "DOTA-val")):
        rows = [r for r in res["rows"] if r["dataset"] == ds]
        for i, cv in enumerate(("share_lt16", "share_crowded")):
            ax = axs[i, j]
            ax.axhline(0, color=FS.MUTED, lw=0.6)
            for r in rows:
                b = [x for x in r[f"binned_{cv}"] if x["plotted"]]
                xc = np.array([(x["lo"] + x["hi"]) / 2 for x in b])
                mu = np.array([x["mean_dAP"] for x in b])
                lo = np.array([x["ci95"][0] for x in b])
                hi = np.array([x["ci95"][1] for x in b])
                name = r["detector"] + (f" ({r['head']})" if ds.startswith("VisDrone") and r["detector"] == "YOLOv10m" else "")
                hl = ds.startswith("DOTA") or r["detector"] == "YOLOv10s"
                c = col.get(r["detector"], FS.BLUE) if hl else FS.MUTED
                rho = r[f"spearman_{cv}"]
                lab = f"{name}: ρ {rho['rho']:+.2f} [{rho['ci95'][0]:+.2f}, {rho['ci95'][1]:+.2f}]"
                if hl:
                    ax.fill_between(xc, lo, hi, color=c, alpha=0.18, lw=0)
                    ax.plot(xc, mu, "-o", color=c, lw=FS.LINE_W, ms=FS.MARKER_SIZE, label=lab)
                else:
                    first = not any(l.get_label().startswith("5 other") for l in ax.get_lines())
                    ax.plot(xc, mu, "-", color=c, lw=0.8, alpha=0.8,
                            label="5 other detector rows (ρ in table)"
                            if first else "_nolegend_")
                for x, m_, l_, h_, bb in zip(xc, mu, lo, hi, b):
                    rec.append({"dataset": ds, "covariate": cv, "detector": name, "bin": bb["bin"], "x": x, "mean_dAP": m_,
                                "ci_lo": l_, "ci_hi": h_, "n_img": bb["n_img"]})
            ax.set_xlim(0, 1)
            ax.set_xlabel(cov_lbl[cv])
            ax.set_ylabel(("ΔAP" if ds.startswith("VisDrone") else "ΔAP50") + " per image\n(SCAHI, refined grid − single pass)")
            ax.set_title(f"{ds}, 2048 px", fontsize=FS.TITLE_SIZE)
            if ds.startswith("VisDrone"):
                ax.set_ylim(-6.5, 6.2)
            ax.legend(fontsize=5.6, loc="lower right" if ds.startswith("VisDrone") else "upper right", frameon=False,
                      handlelength=1.4)
    fig.suptitle(f"When do tiles pay off? Mean per-image difference in bins, 95 % bootstrap CI ({LABEL})",
                 fontsize=FS.TITLE_SIZE, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(FIG.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIG.with_suffix(".png"), bbox_inches="tight", dpi=FS.DPI_PNG)
    pd.DataFrame(rec).to_csv(FIG.with_suffix(".csv"), index=False, float_format="%.4f")

    L = ["# When do tiles pay off? (exploratory, post hoc)", "",
         "Per image: ΔAP = tiles + SCAHI (refined grid) 2048 − single pass 2048 (same detector and head; stored detections "
         "of the frozen rows). VisDrone test-dev: toolkit-port AP@[.5:.95] of the image; DOTA-val: devkit mAP50 of the image. "
         "Covariates over valid GT: share < 16 px (sqrt(w h)); share crowded (same-class neighbour IoU > 0.3). Spearman ρ "
         f"with 95 % percentile CI over {N_BOOT} image resamples (seed {SEED}). Rule: docs/DECISIONS.md 2026-10-31 ~14:50. "
         "Figure: reports/figures/ext/fig_ext_tiles_payoff.{pdf,png,csv}.", "",
         "| dataset | detector | head | images | mean ΔAP | ρ (share < 16 px) | ρ (share crowded) |", "|---|---|---|---:|---:|---|---|"]
    for r in res["rows"]:
        a, c = r["spearman_share_lt16"], r["spearman_share_crowded"]
        L.append(f"| {r['dataset']} | {r['detector']} | {r['head']} | {r['n_img']} | {r['mean_dAP']:+.2f} | "
                 f"{a['rho']:+.2f} [{a['ci95'][0]:+.2f}, {a['ci95'][1]:+.2f}] | {c['rho']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] |")
    (OUT / "TILES_PAYOFF.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("compute", "report"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "compute":
        compute_dota()
        compute_visdrone(a.workers)
    else:
        report()


if __name__ == "__main__":
    main()
