# SPDX-License-Identifier: AGPL-3.0-or-later
"""Paper-quality figures (vector PDF + PNG) from files under runs/ (no new experiments).

    python scripts/paper_figures.py [figure ...]

Style (dataviz method, light paper surface): Inter, thin marks, recessive grid, categorical slots in fixed order
(blue, orange, aqua; validated all-pairs, aqua below 3:1 contrast -> always direct-labelled), sequential = blue ramp,
diverging = blue <-> red around a gray midpoint, text in ink colours only. Figures -> reports/figures/paper/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "reports/figures/paper"

sys.path.insert(0, str(REPO / "scripts"))
import figstyle as FS  # noqa: E402  (central style: colours, fonts, sizes, dpi)
from figstyle import (AQUA, BLUE, BLUE_D, COL_W, FULL_W, GRID, INK, INK2, MUTED, ORANGE, ORD4, SEQ, SURF)  # noqa: E402,F401

_SOURCES: list[str] = []  # input files of the figure being built (written next to it)
_EXTRA: dict = {}  # non-plotted context of the figure being built (e.g. image path and crop)


# Paper terms (user 2026-10-30): no internal row / variant / stack names in any figure. Main study: passes are written
# out ("1280 + tiles" = whole-image pass at 1280 px plus zoom-2 tiles; "z0.64 + z2 tiles" = whole image at zoom 0.64),
# SCAHI variants by what they add.
STACK_TERM = {"S1": "z1/3 + z1 + z2 tiles", "S2": "z1/3 + z1 + z2 tiles (1024)", "S4": "z0.64 + z2 tiles", "S5": "z1 + z2 tiles",
              "native": "1280 + tiles", "native1536": "1536 + tiles", "native2048": "2048 + tiles"}
# relative-zoom stacks by absolute zoom (pixel sizes vary per image); pattern "<pass> + tiles, <merge>" (DECISIONS 2026-10-31)
VALIDITY = "SCAHI + validity windows (estimated on calib)"  # main-study variant F3e
MAB = "selected main-study configuration\n(scale-normalised training, SCAHI + validity windows)"  # M_ab


def style() -> None:
    FS.apply()


def save(fig, name: str, out: Path | None = None) -> None:
    """Vector PDF + PNG (``FS.DPI_PNG``) + ``<name>.data.json`` (plotted values, sources) next to each other."""
    out = out or OUT
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=FS.DPI_PNG)
    data = FS.figure_data(fig, _SOURCES, dict(_EXTRA) or None)
    (out / f"{name}.data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    _SOURCES.clear()
    _EXTRA.clear()
    plt.close(fig)
    print("wrote", name)


def jload(p: str | Path) -> dict:
    _SOURCES.append(str(Path(p)))
    return json.loads((REPO / p).read_text())


def latest(glob: str) -> Path:
    return sorted(REPO.glob(glob))[-1]


# ------------------------------------------------------------------------------------------- figures

def fig_resolution() -> None:
    f = jload(latest("runs/p2/*_phase2_focus/focus.json").relative_to(REPO))
    single = sorted([r for r in f["rows"] if r["label"].startswith("B-FULL")], key=lambda r: r["median_effective_zoom"])
    multi = {r["label"].split(" {")[0]: r for r in f["rows"] if r["label"].startswith(("B-MZ-native", "B-MZ-1536"))}
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.35), constrained_layout=True)
    for ax, key, title in ((axes[0], None, "(a) Overall AP"), (axes[1], "<16", "(b) AP on objects with σ < 16 px")):
        y = [r["AP_port"] if key is None else r["AP_coco"][key] for r in single]
        z = [r["median_effective_zoom"] for r in single]
        ax.plot(z, y, "-o", color=BLUE, zorder=3)
        for zi, yi, r in zip(z, y, single):
            lab = r["label"].split(" @")[0].replace("B-FULL ", "")
            below = lab in ("1280", "1536")
            ax.annotate(lab, (zi, yi), xytext=(6 if below else 0, -10 if below else 6), textcoords="offset points",
                        ha="left" if below else "center", fontsize=6.6, color=INK2)
        for name, ls in (("B-MZ-native", "--"), ("B-MZ-1536-native", ":")):
            r = multi[name]
            v = r["AP_port"] if key is None else r["AP_coco"][key]
            ax.axhline(v, color=ORANGE, lw=1.0, ls=ls, zorder=2)
            pas = "1280" if name == "B-MZ-native" else "1536"
            ax.text(0.45, v + 0.15 if key is None else v + 0.1, f"{pas} + tiles, box fusion (≈9 passes)",
                    fontsize=6.0, color=INK2, va="bottom")
        ax.set_xlabel("median effective zoom (network px per image px), single pass")
        ax.set_ylabel("AP [%]" if key is None else "AP σ<16 [%]")
        ax.set_title(title)
    fig.suptitle("Resolution beats tiling: single pass (train = test size) vs. tiles + box fusion", fontsize=8.2,
                 x=0.01, ha="left", color=INK)
    save(fig, "fig_resolution")


def _gf_unified(exp_id: str) -> float | None:
    """GFLOPs per val image with the one method of runs/gflops_unified.json (user 2026-10-29: exact per input shape on
    the fused model, tiles included); the GFLOPs/MP values of the finished study stay archived in their files."""
    f = REPO / "runs/gflops_unified.json"
    if not f.exists():
        return None
    rows = jload("runs/gflops_unified.json")["rows"]
    for root in ("runs/p2", "runs/p5", "runs/ext/rows", "runs/ext/gridext"):
        r = rows.get(f"{root}/{exp_id}")
        if r and "gflops_per_img" in r:
            return r["gflops_per_img"]
    return None


def _pareto_rows() -> list[dict]:
    rows = jload(latest("runs/p6/*_exploratory/explore.json").relative_to(REPO))["rows"]
    return [{**r, "gflops_per_img": _gf_unified(r["exp_id"])} for r in rows]


def fig_pareto() -> None:
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    rows = _pareto_rows()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), constrained_layout=True)
    names = {"B-FULL 640 @640": ("640", (4, -9)), "B-FULL 1280 @1280": ("1280", (5, -8)), "B-FULL 1536 @1536": ("1536", (5, -8)),
             "B-FULL 2048 @2048": ("2048", (-8, 6)), "R = B-MZ-native (NMS/IOS, Predictor)": ("box fusion", (6, -3)),
             "B-SAHI-FT": ("SAHI (fine-tuned)", (-58, 5))}
    for ax, cost, xl, ticks in ((axes[0], "latency_ms_mean", "latency per image [ms]  (RTX 5090, exclusive)", [5, 10, 20, 50, 100]),
                                (axes[1], "gflops_per_img", "GFLOPs per image", [10, 30, 100, 300, 600])):
        s = sorted([r for r in rows if r["kind"] == "single" and r.get(cost)], key=lambda r: r[cost])
        ax.plot([r[cost] for r in s], [r["AP"] for r in s], "-o", color=BLUE, zorder=4, label="single pass 640–2048")
        for kind, mk, col, fc, lbl in (("reference", "s", ORANGE, ORANGE, "multi-pass baselines (SAHI, tiles + box fusion)"),
                                       ("method", "o", AQUA, AQUA, "SCAHI variants (pre-registered)"),
                                       ("exploratory", "^", INK2, "none", "exploratory, defined after val")):
            p = [r for r in rows if r["kind"] == kind and r.get(cost) and (cost != "latency_ms_mean" or r.get("latency_status") == "final")]
            ax.scatter([r[cost] for r in p], [r["AP"] for r in p], marker=mk, s=22, edgecolor=col, facecolor=fc, lw=1.0,
                       zorder=5, label=lbl)
        for r in rows:
            if r.get(cost) is None:
                continue
            tag, off = names.get(r["label"], (None, None))
            if r["label"].startswith("M_ab"):
                tag, off = "selected configuration", ((-84, -2) if cost == "latency_ms_mean" else (-130, 4))
            if r["label"] == "B-SAHI-FT" and cost == "gflops_per_img":
                off = (5, 4)
            if r["kind"] == "exploratory" and "2048" in r["label"] and "F2" in r["label"]:
                tag, off = "2048 + tiles, SCAHI", ((-80, 3) if cost == "latency_ms_mean" else (-118, 8))
            if tag:
                lead = dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=3) if tag.startswith("SCAHI") else None
                ax.annotate(tag, (r[cost], r["AP"]), xytext=off, textcoords="offset points", fontsize=6.2, color=INK2,
                            arrowprops=lead)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(v) for v in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xl)
        ax.set_ylabel("AP [%]  (VisDrone val, toolkit)")
    axes[0].text(0.98, 0.04, "every preregistered multi-pass row is\ndominated by a single pass", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=6.2, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=6.2, columnspacing=1.2, handlelength=1.8)
    save(fig, "fig_pareto")


def fig_sizebins() -> None:
    p6 = jload(latest("runs/p6/*_phase6/phase6.json").relative_to(REPO))
    rows = {r["label"]: r for r in p6["rows"] if not r.get("missing")}
    base = rows["B-FULL 1280 @1280"]["AP_size"]
    bins = ["<8", "8-16", "16-32", "32-64", "64-128", ">128"]
    series = (("B-FULL 2048 @2048", "single pass 2048", BLUE, "o"),
              ("R = B-MZ-native (NMS/IOS, Predictor)", "1280 + tiles, box fusion", ORANGE, "s"),
              (next(k for k in rows if k.startswith("M_ab")), MAB, AQUA, "D"))
    fig, ax = plt.subplots(figsize=(COL_W, 2.35), constrained_layout=True)
    x = np.arange(len(bins))
    ax.axhline(0, color=MUTED, lw=0.8)
    for key, lbl, col, mk in series:
        y = [rows[key]["AP_size"][b] - base[b] for b in bins]
        ax.plot(x, y, "-", marker=mk, color=col, label=lbl, zorder=3)
        ax.annotate(f"{y[-1]:+.1f}", (x[-1], y[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=6.4, color=INK2)
    ax.set_xticks(x, [f"{b} px" for b in bins])
    ax.set_xlabel("object size σ = √(wh) in the original image")
    ax.set_ylabel("ΔAP vs. single pass 1280 [%]")
    ax.legend(loc="upper left", fontsize=6.2)
    save(fig, "fig_sizebins")


def fig_fusion_gain() -> None:
    models = (("BFULL1280", "single-pass 1280 w."), ("BFULL1536", "single-pass 1536 w."), ("P3T2_s0", "loss-masked training"),
              ("P3T4_s0", "scale-norm. training"), ("BFULL2048", "single-pass 2048 w. *"))
    stacks = ("S1", "S2", "S4", "S5", "native", "native1536", "native2048")
    dense = ("F1", "F2", "F3", "F3e", "F4", "F5")
    G = np.full((len(models), len(stacks)), np.nan)
    for i, (m, _) in enumerate(models):
        tf = REPO / f"runs/p5/20261023_{m}_fusion/tuning.json"
        if not tf.exists():
            continue
        t = jload(tf.relative_to(REPO))
        for j, s in enumerate(stacks):
            cand_f0 = [(t[f"{s}/{b}/F0-NMS"]["best"]["AP"], b) for b in ("o2o", "o2m") if f"{s}/{b}/F0-NMS" in t]
            cand_d = [(t[f"{s}/{b}/{v}"]["best"]["AP"], b, v) for b in ("o2o", "o2m") for v in dense if f"{s}/{b}/{v}" in t]
            if not cand_f0 or not cand_d:
                continue
            _, b0 = max(cand_f0)
            _, b1, v1 = max(cand_d)
            ap = lambda e: jload(f"runs/p5/{e}/metrics.json")["metrics"]["AP"]  # noqa: E731
            G[i, j] = ap(f"20261023_v10s_{m}_{s}_{v1}_{b1}") - ap(f"20261023_v10s_{m}_{s}_F0-NMS_{b0}")
    fig, ax = plt.subplots(figsize=(COL_W + 0.4, 2.1), constrained_layout=True)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", SEQ)
    im = ax.imshow(G, cmap=cmap, vmin=0, vmax=2.0, aspect="auto")
    ax.grid(False)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            if np.isfinite(G[i, j]):
                ax.text(j, i, f"{G[i, j]:+.1f}", ha="center", va="center", fontsize=6.6,
                        color=SURF if G[i, j] > 1.2 else INK)
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=6.6, color=MUTED)
    ax.set_xticks(range(len(stacks)), ["z1/3+z1+z2", "z1/3+z1+z2 (1024)", "z0.64+z2", "z1+z2", "1280+tiles", "1536+tiles",
                                       "2048+tiles*"], rotation=35, ha="right", rotation_mode="anchor", fontsize=5.8)
    ax.set_yticks(range(len(models)), [m[1] for m in models])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("ΔAP SCAHI − box fusion [pp]", fontsize=6.6)
    cb.outline.set_visible(False)
    ax.set_title("SCAHI (best variant) vs. box fusion, identical passes (val)", fontsize=7.2)
    fig.text(0.0, -0.03, "columns: whole-image pass [px] + tiles (z = zoom); rows: weights. "
             "* exploratory, defined after the val results", fontsize=6, color=MUTED)
    save(fig, "fig_fusion_gain")


def fig_degradation() -> None:
    files = [("BFULL1280", "single pass 1280"), ("BFULL1536", "single pass 1536"), ("BFULL2048", "single pass 2048")]
    conds = [("A_native", "native", INK2), ("B_degraded_d2", "degraded ×2", BLUE), ("C_small_up2", "orig. small ×2", ORANGE),
             ("B_degraded_d4", "degraded ×4", BLUE), ("C_small_up4", "orig. small ×4", ORANGE)]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.2), sharey=True, constrained_layout=True)
    for ax, (tag, title) in zip(axes, files):
        r = jload(latest(f"runs/p2/*_degradation_{tag}/degradation.json").relative_to(REPO))
        for k, (c, lbl, col) in enumerate(conds):
            x = k + (0.3 if k > 2 else 0)
            if c not in r["conditions"]:
                ax.text(x, 0.36, "n/a\n(8192 px)", ha="center", fontsize=6, color=MUTED)
                continue
            v = r["conditions"][c]
            ax.errorbar(x, v["recall"], yerr=[[v["recall"] - v["recall_ci95"][0]], [v["recall_ci95"][1] - v["recall"]]],
                        fmt="o" if c.startswith(("A", "B")) else "s", color=col, ms=4.5, capsize=2, lw=1.0)
        ax.set_xticks([0, 1, 2, 3.3, 4.3], ["A\nnative", "B\n↓2", "C\nsmall ×2", "B\n↓4", "C\nsmall ×4"])
        ax.set_title(title)
        ax.set_ylim(0.3, 0.9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("recall @ score ≥ 0.25 (95% CI)")
    fig.suptitle("Same apparent size σ_in, same information loss: originally small objects stay harder (population effect)",
                 fontsize=8, x=0.01, ha="left")
    save(fig, "fig_degradation")


def fig_band_edge() -> None:
    r = jload("runs/p2/20261022_reliability_calib/reliability.json")
    keys = [k for k in r["rows"] if r["rows"][k]["branch"] == "o2o" and r["rows"][k]["model"] == f"BFULL{r['rows'][k]['imgsz']}"]
    keys = sorted(keys, key=lambda k: r["rows"][k]["imgsz"])
    fig, ax = plt.subplots(figsize=(COL_W, 2.3), constrained_layout=True)
    ax.axvspan(6, 14, color="#f0efec", zorder=0)
    ax.axvline(8, color=MUTED, lw=0.8, ls="--")
    ax.text(6.3, 1.02, "a = 8 px (soft 6–14)", fontsize=6.2, color=INK2)
    for col, k in zip(ORD4, keys):
        rr = r["rows"][k]
        e = rr["edge"]
        cx = [p["center"] for p in e["curve"]]
        y = [p["recall"] / e["plateau_recall"] for p in e["curve"]]
        ax.plot(cx, y, "-", color=col, lw=1.5, label=f"single pass {rr['imgsz']}")
        ax.annotate(str(rr["imgsz"]), (cx[-1], y[-1]), xytext=(3, 0), textcoords="offset points", fontsize=6.2, color=INK2, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xlim(2, 160)
    ax.set_xlabel("apparent size in the network input σ_in [px]")
    ax.set_ylabel("recall / plateau recall (calib)")
    ax.legend(loc="lower right", fontsize=6.2, title="train = test size", title_fontsize=6.2)
    save(fig, "fig_band_edge")


def fig_border() -> None:
    def fp(e: str) -> dict:
        return jload(f"runs/p5/{e}/metrics.json")["diagnostics"]["fp_by_border_dist"]

    a, b = fp("20261023_v10s_P3T4_s0_S4_F0-NMS_o2m"), fp("20261023_v10s_P3T4_s0_S4_F3e_o2m")
    ks = list(a)
    fig, ax = plt.subplots(figsize=(COL_W, 2.1), constrained_layout=True)
    x = np.arange(len(ks))
    ax.plot(x, [a[k]["fp_rate"] for k in ks], "-s", color=ORANGE, label="box fusion (NMS)")
    ax.plot(x, [b[k]["fp_rate"] for k in ks], "-o", color=BLUE, label=VALIDITY)
    ax.set_xticks(x, [k.replace("-inf", "+") for k in ks])
    ax.set_xlabel("distance of detection centre to nearest inner tile edge [px]")
    ax.set_ylabel("FP rate (score ≥ 0.25)")
    ax.set_ylim(0, 0.45)
    ax.legend(loc="lower right", fontsize=6.4)
    ax.set_title("Same weights (scale-normalised training) and passes (z0.64 + z2 tiles), val", fontsize=6.6)
    save(fig, "fig_border_fp")


def fig_h1() -> None:
    d = jload(latest("runs/p6/*_h1_diagnostic.json").relative_to(REPO))
    pos = jload("runs/p3/20261024_positives_T2_T4.json")
    bins = ["<8", "8-16", "16-32", "32-64", "64-128", ">128"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.2), constrained_layout=True)
    x = np.arange(len(bins))
    w = 0.38
    for k, (name, col) in enumerate((("T2", ORANGE), ("T4", BLUE))):
        v = [pos[name]["pos_by_sigma_in_train"][b] / 1e3 for b in bins]
        axes[0].bar(x + (k - 0.5) * w, v, width=w - 0.04, color=col,
                    label="loss-masked training" if name == "T2" else "scale-normalised training (zoom-stratified + band)")
    axes[0].set_xticks(x, [f"{b}" for b in bins])
    axes[0].set_xlabel("σ_in of positives in the training input [px]")
    axes[0].set_ylabel("positives per epoch [k]")
    axes[0].legend(fontsize=6.2, loc="upper right")
    axes[0].set_title("(a) Where the supervision goes")
    axes[0].grid(axis="x", visible=False)
    for cfg, lbl, col, mk in (("S5_F0-NMS_o2o", "box fusion", ORANGE, "s"), ("S5_F2_o2m", "SCAHI", BLUE, "o")):
        y = [d["delta_T4_minus_T2_seed0"][cfg]["AP_size"][b] for b in bins]
        axes[1].plot(x, y, "-", marker=mk, color=col, label=f"{lbl}: ΔAP {d['delta_T4_minus_T2_seed0']['AP'][cfg]:+.2f}")
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].set_xticks(x, bins)
    axes[1].set_xlabel("object size σ in the original image [px]")
    axes[1].set_ylabel("ΔAP  scale-norm. − loss-masked [%]")
    axes[1].legend(fontsize=6.2, loc="upper left")
    axes[1].set_title("(b) Effect of scale-normalised training, z1 + z2 tiles")
    save(fig, "fig_h1")


def fig_windows() -> None:
    from scahi.fusion import scalespace as ss
    from scahi.infer import cache_mz as cm

    p = ss.FusionParams()
    lks = cm.STACK_LEVELS["S1"]
    gains = [cm.LEVELS[k].n_input / (cm.LEVELS[k].n_input / cm.LEVELS[k].zoom) if cm.LEVELS[k].n_input else 640 / 1920 for k in lks]
    sig = 2.0 ** np.linspace(0.5, 10, 600)
    raw = np.stack([ss.validity_raw(sig * g, p) for g in gains], 1)
    vn = raw / np.maximum(raw.sum(1, keepdims=True), 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.0), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 1]})
    names = ["z = 1/3 (whole image)", "z = 1 (tiles 640)", "z = 2 (tiles 320)"]
    for k, (col, nm) in enumerate(zip((BLUE, ORANGE, AQUA), names)):
        axes[0].plot(sig, vn[:, k], color=col, label=nm)
    axes[0].plot(sig, vn.sum(1), color=INK, lw=1.0, ls="--", label="Σ v̂ (partition of unity)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("object size σ in the original image [px]")
    axes[0].set_ylabel("validity window v̂_k(σ)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, columnspacing=1.2, handlelength=1.8)
    axes[0].set_title("(a) Validity windows, passes z1/3 + z1 + z2 tiles (a = 8, b = 192 px in σ_in)")
    # border windows along x for the z = 2 tiles of a 1920-px row
    lv = ss.CachedLevel("t", np.zeros((0, 4)), np.zeros((0, 10)), np.zeros((0, 4)), np.zeros(0),
                        np.array([t for t, _ in cm.level_passes(cm.LEVELS["t2.0n1024"], 1920, 1080, 0.3)]), np.ones(1), False, 0.3)
    xs = np.linspace(0, 1920, 1200)
    bn, _ = ss.border_norm(xs, np.full_like(xs, 300.0), lv, 1920, 1080)
    row_tiles = [t for t in range(len(lv.tiles)) if lv.tiles[t][1] == 0]
    for i, t in enumerate(row_tiles):
        axes[1].plot(xs, bn[:, t], color=SEQ[2 + i % 4], lw=1.2)
    axes[1].plot(xs, bn[:, row_tiles].sum(1), color=INK, lw=1.0, ls="--")
    axes[1].set_xlabel("x [px] (z = 2 tiles of 512 px, overlap 0.3)")
    axes[1].set_ylabel("border window b̂_t(x)")
    axes[1].set_title("(b) Border windows sum to one")
    save(fig, "fig_windows")


def _qual_data(stem: str | None = None):
    """B-FULL-1280 weights, passes of R (native stack): box fusion vs. dense fusion on one val image (illustration)."""
    from scahi.eval.evaluate import load_coco_gt
    from scahi.fusion import run as R
    from scahi.fusion import scalespace as ss
    from scahi.infer import cache_mz as cm

    coco = load_coco_gt(REPO / "data/canonical/visdrone10/coco/val.json")
    w = REPO / "runs/p2/20261022_v10s_BFULL1280_train_s0/ultralytics/train/weights/best_calib.pt"
    t = jload("runs/p5/20261023_BFULL1280_fusion/tuning.json")
    cal = R.cal_from_json(jload("runs/p5/20261023_BFULL1280_fusion/calibration.json")["native/o2m"])
    cfg0 = {k: v for k, v in t["native/o2o/F0-NMS"]["best"].items() if k not in ("AP", "AP50", "source")}
    cfg2 = {k: v for k, v in t["native/o2m/F2"]["best"].items() if k not in ("AP", "AP50", "source")}
    imgs = {i["file_name"].rsplit(".", 1)[0]: i for i in coco["images"]}
    stem = stem or "0000295_02000_d_0000031"
    if stem not in imgs:
        stem = sorted(imgs)[200]
    im = imgs[stem]
    dirs = R.level_dirs(w, "native", "val", 0.3)
    lv0, W, H = ss.load_levels(dirs, stem, "o2o", 0.3)
    lv2, _, _ = ss.load_levels(dirs, stem, "o2m", 0.3, cal)
    d0 = R.f0_image(lv0, W, H, "o2o", "F0-NMS", cfg0)
    d2 = R.clip_xywh(ss.fuse_image(lv2, W, H, "F2", ss.FusionParams(**cfg2)), W, H)
    gt = [a for a in coco["annotations"] if a["image_id"] == im["id"]]
    vol = ss.volume(lv2, W, H, "F2", ss.FusionParams(**cfg2))
    tiles = lv2[1].tiles
    _SOURCES.extend([str(Path(d).relative_to(REPO) / f"{stem}.npz") for _, d in dirs] + [im["path"], "data/canonical/visdrone10/coco/val.json"])
    _EXTRA.update(image=im["path"], stem=stem, weights=str(w.relative_to(REPO)), cfg_F0=cfg0, cfg_F2=cfg2,
                  dets_box_fusion=d0.round(3).to_dict("list"), dets_dense_fusion=d2.round(3).to_dict("list"))
    return im, gt, d0, d2, vol, tiles, cfg2


def _crop(d0, d2, gt, W: int, H: int, cw: int = 440, ch: int = 270) -> tuple[int, int, int, int]:
    """Window with the largest excess of box-fusion over dense-fusion detections (score >= 0.3) and >= 8 GT objects."""
    def cnt(boxes, x0, y0):
        return sum(1 for bx, by, bw, bh in boxes if x0 <= bx + bw / 2 < x0 + cw and y0 <= by + bh / 2 < y0 + ch)
    b0 = [(r.x, r.y, r.w, r.h) for r in d0.itertuples() if r.score >= 0.3]
    b2 = [(r.x, r.y, r.w, r.h) for r in d2.itertuples() if r.score >= 0.3]
    g = [a["bbox"] for a in gt if a.get("score", 1) == 1]
    best, arg = -1e9, (0, 0)
    for y0 in range(0, max(1, H - ch), 30):
        for x0 in range(0, max(1, W - cw), 30):
            if cnt(g, x0, y0) < 8:
                continue
            s = cnt(b0, x0, y0) - cnt(b2, x0, y0)
            if s > best:
                best, arg = s, (x0, y0)
    return arg[0], arg[1], arg[0] + cw, arg[1] + ch


def fig_qualitative() -> None:
    im, gt, d0, d2, vol, tiles, cfg2 = _qual_data()
    img = cv2.imread(im["path"])[..., ::-1]
    H, W = img.shape[:2]
    x0, y0, x1, y1 = _crop(d0, d2, gt, W, H)
    _EXTRA["crop_xyxy"] = [x0, y0, x1, y1]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.62), constrained_layout=True)
    for ax, title, dets, col in ((axes[0], "Ground truth", None, "#ffd84d"), (axes[1], "Box fusion (NMS)", d0, ORANGE),
                                 (axes[2], "SCAHI", d2, BLUE)):
        ax.imshow(img[y0:y1, x0:x1])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for tt in tiles:
            for v in (tt[0], tt[2]):
                if x0 < v < x1:
                    ax.axvline(v - x0, color=SURF, lw=0.7, ls=(0, (3, 3)))
            for v in (tt[1], tt[3]):
                if y0 < v < y1:
                    ax.axhline(v - y0, color=SURF, lw=0.7, ls=(0, (3, 3)))
        boxes = [(a["bbox"], 1.0) for a in gt if a.get("score", 1) == 1] if dets is None else \
            [((r.x, r.y, r.w, r.h), r.score) for r in dets.itertuples() if r.score >= 0.3]
        n = 0
        for (bx, by, bw, bh), _s in boxes:
            if not (x0 <= bx + bw / 2 < x1 and y0 <= by + bh / 2 < y1):
                continue
            n += 1
            ax.add_patch(Rectangle((bx - x0, by - y0), bw, bh, fill=False, lw=0.8, edgecolor=col))
        ax.set_title(f"{title}: {n} boxes", fontsize=7)
    fig.text(0.0, -0.04, "Single-pass 1280 weights, passes 1280 + z = 2 tiles (dashed = tile edges); score ≥ 0.3; "
             "val image " + im["file_name"], fontsize=5.8, color=MUTED)
    save(fig, "fig_qualitative")


def fig_volume() -> None:
    im, gt, d0, d2, vol, tiles, cfg2 = _qual_data()
    img = cv2.imread(im["path"])[..., ::-1]
    H, W = img.shape[:2]
    x0, y0, x1, y1 = _crop(d0, d2, gt, W, H)
    _EXTRA["crop_xyxy"] = [x0, y0, x1, y1]
    bpo = cfg2["bins_per_oct"]
    # the four sigma bins with the most ground-truth objects inside the crop (sorted by size)
    jbin = lambda s: int(np.floor(np.log2(max(s, 2.0) / 2.0) * bpo))  # noqa: E731
    cnt: dict[int, int] = {}
    for a in gt:
        bx, by, bw, bh = a["bbox"]
        if a.get("score", 1) == 1 and x0 <= bx + bw / 2 < x1 and y0 <= by + bh / 2 < y1:
            cnt[jbin(np.sqrt(bw * bh))] = cnt.get(jbin(np.sqrt(bw * bh)), 0) + 1
    js = sorted(sorted(cnt, key=lambda k: -cnt[k])[:4])
    targets = [2.0 * 2 ** ((j + 0.5) / bpo) for j in js]
    fig, axes = plt.subplots(1, len(targets), figsize=(FULL_W, 1.42), constrained_layout=True)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue_alpha", [(0.16, 0.47, 0.84, 0.0), (0.06, 0.26, 0.51, 0.92)])
    for ax, s in zip(axes, targets):
        j = jbin(s)
        m = vol["j"] == j
        g = float(np.median(vol["g"][m])) if m.any() else 1.0
        grid = np.zeros((int(H / g) + 2, int(W / g) + 2))
        ix, iy = (vol["x"][m] / g).astype(int), (vol["y"][m] / g).astype(int)
        np.maximum.at(grid, (iy, ix), vol["S"][m])
        from scipy.ndimage import maximum_filter

        disp = maximum_filter(grid, size=max(3, int(round(s / (2 * g)) * 2 + 1)))  # display only: footprint ~ sigma/2
        vmax = max(float(np.percentile(disp[disp > 0], 99)) if (disp > 0).any() else 1.0, 1e-3)
        ax.imshow((img[y0:y1, x0:x1] * 0.5 + 255 * 0.5).astype(np.uint8), extent=(x0, x1, y1, y0))
        ax.imshow(disp, extent=(0, grid.shape[1] * g, grid.shape[0] * g, 0), cmap=cmap, vmin=0, vmax=vmax,
                  interpolation="nearest")
        lo, hi = 2.0 * 2 ** (j / bpo), 2.0 * 2 ** ((j + 1) / bpo)
        for a in gt:
            bw, bh = a["bbox"][2], a["bbox"][3]
            if a.get("score", 1) == 1 and lo <= np.sqrt(bw * bh) < hi:
                ax.add_patch(Rectangle(a["bbox"][:2], bw, bh, fill=False, lw=0.8, edgecolor=ORANGE))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"σ {lo:.0f}–{hi:.0f} px · cell {g:.0f} px", fontsize=6.8)
    fig.text(0.0, -0.05, "Field S(x, y, log σ) of SCAHI per σ-slice (blue, darker = higher, scaled to the slice's "
             "99th percentile; display max-filter ≈ σ/2); orange = ground truth of that size; same crop as the qualitative figure.",
             fontsize=5.8, color=MUTED)
    save(fig, "fig_volume")


# ------------------------------------------------------------------------------------------- test-dev

GROUP_STYLE = {"single": ("o", BLUE, BLUE, "single pass"), "reference": ("s", ORANGE, ORANGE, "multi-pass box fusion"),
               "dense": ("D", AQUA, AQUA, "SCAHI variants (pre-registered)"),
               "exploratory": ("^", INK2, "none", "exploratory, defined after val")}
SINGLE_IDS = ("R1", "R2a", "R2b", "R2")
DENSE_VARIANTS = ("F1", "F2", "F3", "F3e", "F4", "F5")


def _td_group(r: dict) -> str:
    if r["group"] == "exploratory":
        return "exploratory"
    if r["id"] in SINGLE_IDS:
        return "single"
    v = r["val_exp_id"].rsplit("_", 2)[-2]
    return "dense" if v in DENSE_VARIANTS else "reference"


def fig_prereg_forest() -> None:
    a = jload("runs/testdev/analysis.json")
    items = [("a", "(a) ΔAP vs. 1280 + tiles, box fusion", 1.0), ("b", "(b) ΔAP σ<16 vs. 1280 + tiles, box fusion", 2.0),
             ("a_R2", "(a) ΔAP vs. single pass 2048", 1.0), ("b_R2", "(b) ΔAP σ<16 vs. single pass 2048", 2.0),
             ("H1", "scale-norm. − loss-masked training", a["preregistration_val"]["eps"]),
             ("H2", f"{VALIDITY} − box fusion", a["preregistration_val"]["eps"]), ("H3", "head o2m − o2o", None)]
    fig, ax = plt.subplots(figsize=(FULL_W, 2.5), constrained_layout=True)
    y = np.arange(len(items))[::-1]
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    for split, off, col, mk, lbl in (("val", 0.15, BLUE, "o", "val (548 images)"), ("test-dev", -0.15, ORANGE, "s", "test-dev (1610 images)")):
        p = a[f"preregistration_{split}"]
        for yi, (k, _, thr) in zip(y, items):
            d = p[k]
            ax.plot(d["ci95"], [yi + off] * 2, color=col, lw=1.6, solid_capstyle="round", zorder=3)
            ax.scatter([d["delta"]], [yi + off], marker=mk, s=24, color=col, edgecolor=SURF, lw=0.8, zorder=4,
                       label=lbl if k == "a" else None)
            v = d.get("verdict", d.get("direction"))
            v = {"erfüllt": "met", "nicht erfüllt": "not met", "unklar": "unclear"}.get(v, v)
            ax.annotate(v, (d["ci95"][1], yi + off), xytext=(5, 0), textcoords="offset points", va="center",
                        fontsize=5.8, color=INK2)
    for yi, (_, _, thr) in zip(y, items):
        if thr is not None:
            ax.plot([thr, thr], [yi - 0.38, yi + 0.38], color=INK, lw=0.9, ls=(0, (2, 1.5)), zorder=2)
    ax.set_yticks(y, [lbl for _, lbl, _ in items])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ AP [%] with paired-bootstrap 95 % CI (1000 resamples over images); dashed = preregistered threshold")
    ax.set_xlim(-3.25, 2.6)
    ax.legend(loc="lower left", fontsize=6.4)
    ax.set_title("Preregistered criteria on val and test-dev (frozen rows, no retuning)")
    save(fig, "fig_prereg_forest")


def fig_testdev() -> None:
    a = jload("runs/testdev/analysis.json")
    rows = [{**r, "gflops_per_img_val": _gf_unified(r["val_exp_id"])} for r in a["rows"] if r["group"] != "seed"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.45), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    # (a) val vs. test-dev AP
    ax = axes[0]
    xs = [r["val"]["AP"] for r in rows]
    ys = [r["test-dev"]["AP"] for r in rows]
    lo, hi = min(xs + ys) - 1, max(xs + ys) + 1
    off = float(np.mean(np.array(ys) - np.array(xs)))
    ax.plot([lo, hi], [lo + off, hi + off], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    from scipy.stats import spearmanr

    m = [r for r in rows if r["group"] != "exploratory"]
    rho = spearmanr([r["val"]["AP"] for r in m], [r["test-dev"]["AP"] for r in m]).statistic
    ax.text(0.97, 0.05, f"dashed: test-dev = val {off:+.1f} AP\nSpearman ρ = {rho:.2f} (preregistered + reference rows)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=INK2)
    for g, (mk, col, fc, lbl) in GROUP_STYLE.items():
        p = [r for r in rows if _td_group(r) == g]
        ax.scatter([r["val"]["AP"] for r in p], [r["test-dev"]["AP"] for r in p], marker=mk, s=16, edgecolor=col, facecolor=fc,
                   lw=0.9, zorder=3, label=lbl)
    lead = dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=2)
    offs = {"R1": ("640", (6, -2)), "R": ("box fusion", (10, -26)), "M_ab": ("selected config.", (-50, 10)), "R2": ("2048", (12, -16)),
            "X2": ("2048 + tiles, SCAHI", (-62, 8))}
    for r in rows:
        if r["id"] in offs:
            lab, o = offs[r["id"]]
            ax.annotate(lab, (r["val"]["AP"], r["test-dev"]["AP"]), xytext=o, textcoords="offset points", fontsize=6, color=INK2,
                        arrowprops=None if r["id"] == "R1" else lead)
    ax.set_xlabel("AP val [%]")
    ax.set_ylabel("AP test-dev [%]")
    ax.set_title("(a) Frozen rows: val vs. test-dev")
    # (b) per-size AP on test-dev relative to the 1280 single pass
    ax = axes[1]
    by = {r["id"]: r for r in a["rows"]}
    bins = ["<8", "8-16", "16-32", "32-64", "64-128", ">128"]
    base = by["R2a"]["test-dev"]["AP_size"]
    x = np.arange(len(bins))
    ax.axhline(0, color=MUTED, lw=0.8)
    for rid, lbl, col, mk, fc in (("R2", "single pass 2048", BLUE, "o", BLUE), ("R", "1280 + tiles, box fusion", ORANGE, "s", ORANGE),
                                  ("M_ab", MAB.replace(", SCAHI", ",\nSCAHI"), AQUA, "D", AQUA),  # narrow panel: three lines
                                  ("X2", "2048 + tiles, SCAHI (expl.)", INK2, "^", "none")):
        yv = [by[rid]["test-dev"]["AP_size"][b] - base[b] for b in bins]
        ax.plot(x, yv, "-", color=col, lw=1.3, zorder=3, label=lbl)
        ax.scatter(x, yv, marker=mk, s=16, edgecolor=col, facecolor=fc, lw=0.9, zorder=4)
    sz = a["split_sizes"]
    for xi, b in zip(x, bins):
        ax.text(xi, 1.0, f"{100 * sz['test-dev'][b]:.0f}%", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=5.6, color=MUTED)
    ax.set_xticks(x, [f"{b}" for b in bins], fontsize=5.4, rotation=30)
    ax.set_ylim(-10.5, 8.6)
    ax.set_xlabel("object size σ = √(wh) [px]  (top: share of test-dev GT)")
    ax.set_ylabel("ΔAP vs. single pass 1280 [%]")
    ax.set_title("(b) Per-size effect on test-dev", pad=10)
    ax.legend(loc="lower left", fontsize=5.6, handlelength=1.4)
    # (c) Pareto on test-dev: AP vs. GFLOPs
    ax = axes[2]
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    s = sorted([r for r in rows if _td_group(r) == "single"], key=lambda r: r["gflops_per_img_val"])
    ax.plot([r["gflops_per_img_val"] for r in s], [r["test-dev"]["AP"] for r in s], "-", color=BLUE, lw=1.3, zorder=3)
    for g, (mk, col, fc, lbl) in GROUP_STYLE.items():
        p = [r for r in rows if _td_group(r) == g and r.get("gflops_per_img_val")]
        ax.scatter([r["gflops_per_img_val"] for r in p], [r["test-dev"]["AP"] for r in p], marker=mk, s=16, edgecolor=col,
                   facecolor=fc, lw=0.9, zorder=4)
    ax.set_xscale("log")
    ticks = [10, 30, 100, 300, 600]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    for r in rows:
        tag = {"R2": ("single pass 2048", (-44, 8)), "D-1536": ("1536 w., z1 + z2 tiles, SCAHI", (-92, -80)),
               "X2": ("2048 + tiles, SCAHI", (5, -2))}.get(r["id"])
        if tag and r.get("gflops_per_img_val"):
            ax.annotate(tag[0], (r["gflops_per_img_val"], r["test-dev"]["AP"]), xytext=tag[1], textcoords="offset points",
                        fontsize=6, color=INK2, ha="left",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=2) if r["id"] == "D-1536" else None)
    ax.set_xlabel("GFLOPs per image")
    ax.set_ylabel("AP test-dev [%]")
    ax.set_title("(c) Accuracy vs. compute, test-dev")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=6.2, columnspacing=1.2, handlelength=1.6)
    save(fig, "fig_testdev")


# ------------------------------------------------------------------------------------------- extension study E

OUT_EXT = REPO / "reports/figures/ext"
FAMILY_COL = {"yolov10": BLUE, "yolo11": ORANGE, "yolo12": AQUA}  # categorical slots in fixed order
MODEL_LABEL = {"yolov10s": "YOLOv10s", "yolov10m": "YOLOv10m", "yolo11s": "YOLOv11s", "yolo11m": "YOLOv11m",
               "yolo12s": "YOLOv12s", "yolo12m": "YOLOv12m"}


def _family(model: str) -> str:
    return "yolov10" if model.startswith("yolov10") else model[:6]


def _training_runs() -> list[dict]:
    """B-FULL training runs on visdrone10 (finished study: YOLOv10s; extension: runs/ext/train), with their calib curves."""
    runs = []
    for d in sorted((REPO / "runs/p2").glob("20261022_v10s_BFULL*_train_s*")):
        tag = d.name.split("_")[2]
        if not tag[5:].isdigit():  # skip person_vehicle / bridge-experiment runs
            continue
        runs.append({"model": "yolov10s", "imgsz": int(tag[5:]), "seed": int(d.name[-1]), "dir": d})
    for d in sorted((REPO / "runs/ext/train").glob("*_s*")):
        m, sz, s = d.name.rsplit("_", 2)
        runs.append({"model": m, "imgsz": int(sz), "seed": int(s[1:]), "dir": d})
    out = []
    for r in runs:
        f = r["dir"] / "ultralytics/train/weights/calib_curve.json"
        if f.exists():
            r["curve"] = jload(f.relative_to(REPO))
            r["weights"] = r["dir"] / "ultralytics/train/weights/best_calib.pt"
            r["done"] = (r["dir"] / "TRAIN_DONE").exists()
            out.append(r)
    return out


def fig_ext_calib_curves() -> None:
    """Calib AP over epochs for o2o and o2m+NMS of every B-FULL run (descriptive; recipe unchanged)."""
    order = list(MODEL_LABEL)
    runs = sorted(_training_runs(), key=lambda r: (order.index(r["model"]), r["imgsz"], r["seed"]))
    n = len(runs)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(FULL_W, 1.55 * rows + 0.4), constrained_layout=True, sharex=True,
                             squeeze=False)
    for ax, r in zip(axes.flat, runs):
        c = r["curve"]
        ep = [x["epoch"] for x in c]
        sel = c[0].get("selection_branch", "o2o")
        sel_lbl = "o2o (NMS-free)" if sel == "o2o" else "single head + NMS"
        ax.plot(ep, [x["AP"] for x in c], "-o", color=BLUE if sel == "o2o" else ORANGE, ms=2.4, lw=1.3, label=sel_lbl)
        if "AP_o2m" in c[0]:
            ax.plot(ep, [x["AP_o2m"] for x in c], "-s", color=ORANGE, ms=2.2, lw=1.1, label="o2m + NMS")
        b = max(c, key=lambda x: x["AP"])
        ax.axvline(b["epoch"], color=MUTED, lw=0.7, ls=(0, (2, 2)))
        drop = c[-1]["AP"] - b["AP"]
        state = "" if r["done"] else "  (running)"
        ax.set_title(f"{MODEL_LABEL[r['model']]} {r['imgsz']}" + (f" s{r['seed']}" if r["seed"] else "") + state, fontsize=7)
        ax.text(0.98, 0.04, f"best ep {b['epoch']}: {b['AP']:.1f}" + (f"\nep {c[-1]['epoch']}: {drop:+.1f}" if r["done"] else ""),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=INK2)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("epoch", fontsize=6.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("calib AP [%]", fontsize=6.5)
    h = [plt.Line2D([], [], color=BLUE, marker="o", ms=3, lw=1.3), plt.Line2D([], [], color=ORANGE, marker="s", ms=3, lw=1.1),
         plt.Line2D([], [], color=MUTED, lw=0.7, ls=(0, (2, 2)))]
    fig.legend(h, ["o2o (NMS-free head, YOLOv10 selection)", "o2m + NMS (YOLOv10) / single head + NMS (YOLOv11/v12)",
                   "selected epoch (best calib AP)"], loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6.2)
    fig.suptitle("Calib AP during training (toolkit port at the scheduled calib evaluations) - descriptive, recipe unchanged",
                 fontsize=7.6, x=0.01, ha="left")
    save(fig, "fig_ext_calib_curves", OUT_EXT)
    save_csv([{"model": r["model"], "imgsz": r["imgsz"], "seed": r["seed"], **x} for r in runs for x in r["curve"]],
             "fig_ext_calib_curves", OUT_EXT)


def fig_ext_scaling() -> None:
    """AP vs. GFLOPs per image, one point per (model, resolution): scaling the model vs. scaling the resolution.
    val AP of the single-pass rows once every run has one; until then a marked calib preview (best calib AP)."""
    from scahi.flops import FlopCounter, single_pass

    # The points need the weights (GFLOPs via the weights hash) and the val image sizes. Where they are missing (release
    # without weights/data) the stored points are re-plotted; never drop rows silently (release check 2026-11-01).
    runs = [r for r in _training_runs() if not r["seed"]]
    stored = REPO / "runs/ext/scaling_points.json"
    if not all(r["weights"].exists() for r in runs) or not (REPO / "data/canonical/visdrone10/coco/val.json").exists():
        if not stored.exists():
            raise SystemExit("fig_ext_scaling: weights or val.json missing and no runs/ext/scaling_points.json")
        st = json.loads(stored.read_text())
        pts = st["points"]
        _SOURCES[:0] = st["sources"]  # the files the points were computed from (as in the full run)
        runs = []
    else:
        coco = json.loads((REPO / "data/canonical/visdrone10/coco/val.json").read_text())
        sizes_wh = [(i["width"], i["height"]) for i in coco["images"]]
        pts = []
    for r in runs:
        val_ap = None
        if r["model"] == "yolov10s":
            mf = REPO / f"runs/p2/20261022_v10s_BFULL{r['imgsz']}_full{r['imgsz']}_none_s0/metrics.json"
            if r["imgsz"] in (2560, 3072):
                mf = REPO / f"runs/ext/rows/E_yolov10s_full{r['imgsz']}_o2o/metrics.json"
        else:
            br = "o2o" if r["model"].startswith("yolov10") else "o2m"
            mf = REPO / f"runs/ext/rows/E_{r['model']}_full{r['imgsz']}_{br}/metrics.json"
        if mf.exists():
            val_ap = jload(mf.relative_to(REPO))["metrics"]["AP"]
        pts.append({"model": r["model"], "imgsz": r["imgsz"], "calib_AP": max(x["AP"] for x in r["curve"]), "val_AP": val_ap,
                    "done": r["done"], "gflops": single_pass(FlopCounter(r["weights"]), sizes_wh, r["imgsz"])})
    if runs:
        stored.write_text(json.dumps({"points": pts, "sources": list(_SOURCES)}, indent=1))
    preview = not all(p["val_AP"] is not None for p in pts if p["done"])
    key = "calib_AP" if preview else "val_AP"
    fig, ax = plt.subplots(figsize=(COL_W * 1.25, 2.6), constrained_layout=True)
    for m in MODEL_LABEL:
        p = sorted([q for q in pts if q["model"] == m and q[key] is not None], key=lambda q: q["imgsz"])
        if not p:
            continue
        col = FAMILY_COL[_family(m)]
        mk = "o" if m.endswith("s") else "s"
        ax.plot([q["gflops"] for q in p], [q[key] for q in p], "-", color=col, lw=1.1, zorder=2)
        for q in p:  # model = colour (family) + marker (s / m) in the legend; the point carries its input size
            ax.scatter(q["gflops"], q[key], marker=mk, s=22, color=col if q["done"] else SURF, edgecolor=col, lw=1.0, zorder=3)
            ax.annotate(str(q["imgsz"]), (q["gflops"], q[key]), xytext=(4, -3), textcoords="offset points", fontsize=5.6,
                        color=INK2)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    ticks = [10, 30, 100, 300]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("GFLOPs per image (single pass, exact per input shape; val geometry)")
    ax.set_ylabel(("calib" if preview else "val") + " AP [%]")
    ax.set_title("PREVIEW (calib AP, selection split) - model vs. resolution scaling" if preview else
                 "Model vs. resolution scaling (val, single pass)", fontsize=7.4, color=ORANGE if preview else INK)
    h = [plt.Line2D([], [], color=FAMILY_COL[f], marker="o", lw=1.1, ms=4) for f in FAMILY_COL] + \
        [plt.Line2D([], [], color=INK2, marker=mk, lw=0, ms=4) for mk in ("o", "s")] + \
        [plt.Line2D([], [], color=INK2, marker="o", lw=0, ms=4, markerfacecolor=SURF)]
    ax.legend(h, ["YOLOv10", "YOLOv11", "YOLOv12", "s model", "m model", "training running"], fontsize=5.6, loc="lower right", ncol=2)
    if not _ext_complete():
        ax.text(0.01, 0.98, "INTERIM", transform=ax.transAxes, va="top", fontsize=6.4, color=ORANGE, weight="semibold")
    _EXTRA.update(points=pts, preview=preview)
    save(fig, "fig_ext_scaling", OUT_EXT)
    save_csv(pts, "fig_ext_scaling", OUT_EXT)


def save_csv(rows: list[dict], name: str, out: Path) -> None:
    """Plotted values as CSV next to the figure (extension figures)."""
    import pandas as pd

    pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)


def _ext_complete() -> bool:
    """All five models evaluated (E4 final output present); otherwise every extension figure is marked interim."""
    return (REPO / "runs/ext/e4/e4.json").exists()


def _interim_title(ax, title: str) -> None:
    final = _ext_complete()
    ax.set_title(title if final else f"INTERIM - {title}", fontsize=7.4, color=INK if final else ORANGE)


EXT_HYPS = (("E-H1", "Q1: single pass 2048 −\n1280 + tiles, box fusion"),
            ("E-H2 s=1280", "Q2: SCAHI − box fusion,\nsame passes, s = 1280"),
            ("E-H2 s=2048", "Q2: SCAHI − box fusion,\nsame passes, s = 2048"),
            ("E-H3", "Q3: SCAHI (tiles) −\nsingle pass, s = 2048"),
            ("E-H4", "YOLOv10 head\n(o2m vs. o2o), YOLOv10m"))
EXT_MODELS = ("yolov10m", "yolo11s", "yolo12s", "yolo11m", "yolo12m")


SPLITS = (("val", "val", "o"), ("test-dev", "test-dev", "D"))  # (key, label, marker); test-dev markers hollow


def fig_ext_forest() -> None:
    """Cross-model forest plot per hypothesis, val (filled) and test-dev (hollow) next to each other (paired bootstrap
    CI, same logic on both splits); dashed = eps (from val); diamond row = pooled (descriptive)."""
    f = "runs/ext/e4/e4.json" if _ext_complete() else "runs/ext/e4/e4_dryrun.json"
    if not (REPO / f).exists():
        print("skip fig_ext_forest (no E4 output yet)")
        return
    e4 = {"val": jload(f)}
    if (REPO / "runs/ext/e4/e4_testdev.json").exists():
        e4["test-dev"] = jload("runs/ext/e4/e4_testdev.json")
    eps = e4["val"]["epsilon"]["eps"]
    off = {"val": 0.17, "test-dev": -0.17} if len(e4) == 2 else {"val": 0.0}
    fig, axes = plt.subplots(1, len(EXT_HYPS), figsize=(FULL_W, 2.5), sharey=True, constrained_layout=True)
    y = np.arange(len(EXT_MODELS))[::-1]
    rows = []
    for ax, (h, lbl) in zip(axes, EXT_HYPS):
        ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
        if h != "E-H4":
            ax.axvline(eps, color=INK, lw=0.8, ls=(0, (2, 1.5)), zorder=1)
        for sp, _, mk in SPLITS:
            if sp not in e4:
                continue
            o = e4[sp]["hypotheses"].get(h, {})
            for yi, m in zip(y, EXT_MODELS):
                r = o.get("per_model", {}).get(m)
                if r is None or "missing" in r:
                    if h == "E-H4" and not m.startswith("yolov10"):
                        continue
                    if sp == "val":
                        ax.text(0.03, yi, "pending" if not _ext_complete() else "missing", fontsize=5.4, color=MUTED,
                                va="center", transform=ax.get_yaxis_transform())
                    rows.append({"split": sp, "hypothesis": h, "model": m, "status": "pending/missing"})
                    continue
                if "untestable" in r:  # dropped training size: conservatively "not met", no estimate
                    if sp == "val":
                        ax.text(0.03, yi, "untestable", fontsize=5.4, color=MUTED, va="center",
                                transform=ax.get_yaxis_transform())
                    rows.append({"split": sp, "hypothesis": h, "model": m, "status": f"{r['status']} ({r['untestable']})"})
                    continue
                col, yy = FAMILY_COL[_family(m)], yi + off[sp]
                ax.plot(r["ci95"], [yy, yy], color=col, lw=1.3, solid_capstyle="round", zorder=3,
                        alpha=1.0 if sp == "val" else 0.75)
                ax.scatter([r["delta"]], [yy], s=16, marker=mk, zorder=4, lw=0.8,
                           **(dict(color=col, edgecolor=SURF) if sp == "val" else dict(facecolor=SURF, edgecolor=col)))
                rows.append({"split": sp, "hypothesis": h, "model": m, "delta": r["delta"], "ci_lo": r["ci95"][0],
                             "ci_hi": r["ci95"][1], "eps": None if h == "E-H4" else eps, "status": r.get("status", "")})
            p = o.get("pooled")
            if p and p["delta"] is not None:
                yy = -1 + off[sp]
                ax.plot(p["ci95"], [yy, yy], color=INK2, lw=1.1, zorder=3)
                ax.scatter([p["delta"]], [yy], marker="D", s=14, zorder=4, lw=0.8,
                           **(dict(color=INK2) if sp == "val" else dict(facecolor=SURF, edgecolor=INK2)))
                rows.append({"split": sp, "hypothesis": h, "model": f"pooled ({p['n_models']} models, descriptive)",
                             "delta": p["delta"], "ci_lo": p["ci95"][0], "ci_hi": p["ci95"][1], "eps": eps})
        ax.set_title(lbl, fontsize=6.2)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=6)
    axes[0].set_yticks([*y, -1], [MODEL_LABEL[m] for m in EXT_MODELS] + ["pooled (descr.)"])
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], ls="-", marker="o", color=INK2, mec=SURF, ms=4, lw=1.2, label="val (verdicts as computed)")]
    if "test-dev" in e4:
        hd.append(Line2D([], [], ls="-", marker="D", color=INK2, mfc=SURF, ms=3.6, lw=1.2,
                         label="test-dev (held-out confirmation, same logic)"))
    fig.legend(handles=hd, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    fig.supxlabel(f"Δ AP [%], paired-bootstrap 95 % CI (1000 resamples); dashed = ε = {eps:.2f} (from val)",
                  fontsize=6.4)
    st = ("Questions Q1–Q3 per detector, val and test-dev" if "test-dev" in e4 else
          "Questions Q1–Q3 per detector") if _ext_complete() else \
        "INTERIM (dry run, no verdicts) - questions per detector"
    fig.suptitle(st, fontsize=7.4, x=0.01, ha="left", color=INK if _ext_complete() else ORANGE)
    _EXTRA.update(eps=e4["val"]["epsilon"], mode=e4["val"]["mode"], splits=list(e4))
    save(fig, "fig_ext_forest", OUT_EXT)
    save_csv(rows, "fig_ext_forest", OUT_EXT)


def fig_ext_waterfall() -> None:
    """Decomposition per model and s (main branch), val and test-dev: A (single pass) -> + F2 alone -> + tiles = C (F2)
    -> + X-grid = C (X-grid, o2m). Steps with the paired CIs of runs/ext/testdev_explore.json (TESTDEV_EXT.md §6;
    val identical to runs/ext/decomp.json and grid_ext.json). Exploratory, post hoc."""
    f = "runs/ext/testdev_explore.json"
    if not (REPO / f).exists():
        return
    cs = jload(f)["contrasts"]
    get = lambda fam, m, sz, br, con: next((c for c in cs if (c["family"], c["model"], c["s"], c["branch"],  # noqa: E731
                                                                c["contrast"]) == (fam, m, sz, br, con)
                                            and c.get("role", "main") == "main"), None)
    items = [(m, sz) for m in EXT_MODELS for sz in (1280, 2048) if get("decomposition", m, sz, "o2m", "B − A")]
    sizes = sorted({sz for _, sz in items})
    models = [m for m in EXT_MODELS if any(mm == m for mm, _ in items)]
    panels = [(sz, sp) for sz in sizes for sp, _, _ in SPLITS]
    fig, axes = plt.subplots(len(panels), len(models), figsize=(FULL_W, 1.45 * len(panels) + 0.3), squeeze=False,
                             constrained_layout=True)
    rows = []
    for i, (sz, sp) in enumerate(panels):
        for j, m in enumerate(models):
            ax = axes[i, j]
            ba, cb = get("decomposition", m, sz, "o2m", "B − A"), get("decomposition", m, sz, "o2m", "C − B")
            xg = get("X-grid", m, sz, "o2m", "C − F2 C")
            if ba is None or ba[sp] is None:
                ax.set_visible(False)
                continue
            assert xg is None or (xg["b"] == cb["a"] and xg["td_b"] == cb["td_a"]), (m, sz)  # same F2 C run
            ax.set_axisbelow(True)
            dba, dcb, dxg = ba[sp], cb[sp], (xg or {}).get(sp)
            a, b, c = dba["b"], dba["a"], dcb["a"]
            steps = [(a, b, dba, "+SCAHI"), (b, c, dcb, "+tiles")]
            vals = [a, b, c]
            if dxg:
                steps.append((c, dxg["a"], dxg, "+refined grid"))
                vals.append(dxg["a"])
            lo = min(vals) - (2.1 if b < a else 1.5)  # room for a negative step's label
            ax.bar(0, a - lo, bottom=lo, color=MUTED, width=0.6)
            xs = {"+SCAHI": 1, "+tiles": 2, "+refined grid": 4}
            for s0, s1, d, nm in steps:
                x = xs[nm]
                ax.bar(x, s1 - s0, bottom=s0, color=(AQUA if nm == "+refined grid" else BLUE) if s1 >= s0 else ORANGE,
                       width=0.6)
                ax.errorbar(x, s1, yerr=[[d["delta"] - d["ci95"][0]], [d["ci95"][1] - d["delta"]]], color=INK, lw=0.7,
                            capsize=1.5)
                up = s1 >= s0  # label beyond the CI end of the step, centred on the bar
                ax.annotate(f"{d['delta']:+.2f}", (x, d["ci95"][1] - d["delta"] + s1 if up else s1 - (d["delta"] - d["ci95"][0])),
                            xytext=(0, 1.5 if up else -1.5), textcoords="offset points", ha="center",
                            va="bottom" if up else "top", fontsize=4.8, color=INK2)
            ax.bar(3, c - lo, bottom=lo, color=BLUE_D, width=0.6)
            ticks = ["single pass", "+SCAHI", "+tiles", "SCAHI"]
            if dxg:
                ax.bar(5, dxg["a"] - lo, bottom=lo, color=AQUA, width=0.6, edgecolor=BLUE_D, lw=0.6)
                ticks += ["+refined grid", "SCAHI, refined grid"]
            ax.set_xticks(range(len(ticks)), ticks, fontsize=4.8, rotation=40, ha="right", rotation_mode="anchor")
            ax.set_ylim(lo, max(vals) + 1.4)
            ax.grid(axis="x", visible=False)
            ax.tick_params(labelsize=5.6)
            ax.set_title(f"{MODEL_LABEL[m]} s = {sz} · {sp}", fontsize=6.2,
                         color=INK if sp == "val" else BLUE_D)
            if j == 0:
                ax.set_ylabel(f"{sp} AP [%]", fontsize=6.0)
            rows.append({"split": sp, "model": m, "s": sz, "A": a, "B": b, "C_F2": c,
                         "C_XG": dxg["a"] if dxg else None,
                         **{f"{k}{suf}": v for k, d in (("B-A", dba), ("C-B", dcb), ("XG-C", dxg)) if d
                            for suf, v in (("", d["delta"]), ("_lo", d["ci95"][0]), ("_hi", d["ci95"][1]))}})
    st = ("Decomposition: single pass → + SCAHI on the single pass (no tiles) → + z = 2 tiles → SCAHI, refined grid "
          "(o2m head), val and test-dev; exploratory, post hoc")
    fig.suptitle(st if _ext_complete() else "INTERIM - " + st, fontsize=7.0, x=0.01, ha="left",
                 color=INK if _ext_complete() else ORANGE)
    fig.text(0.0, -0.01, "Error bars: paired-bootstrap 95 % CI of each step. YOLOv10m: the single pass uses the o2o "
             "head. YOLOv12m: s = 1280 only.", fontsize=5.4, color=MUTED)
    save(fig, "fig_ext_waterfall", OUT_EXT)
    save_csv(rows, "fig_ext_waterfall", OUT_EXT)


def fig_ext_recall() -> None:
    """Recall@0.5 of crowded vs. isolated GT (main branch): A, B, C and, where available, X-grid (row C); one row of
    panels per size (s = 2048: four models, YOLO12m 2048 dropped; s = 1280: all five models)."""
    sizes = [(sz, f) for sz, f in ((2048, "runs/ext/overmerge.json"), (1280, "runs/ext/overmerge_s1280.json"))
             if (REPO / f).exists()]
    if not sizes:
        return
    xg = jload("runs/ext/grid_ext.json")["results"] if (REPO / "runs/ext/grid_ext.json").exists() else {}
    rows = []
    fig, axes = plt.subplots(len(sizes), 2, figsize=(FULL_W, 2.0 * len(sizes)), sharey=False, constrained_layout=True,
                             squeeze=False)
    series = (("A", "single pass", MUTED), ("B", "SCAHI on the single pass (no tiles)", ORANGE), ("C", "SCAHI (tiles)", BLUE),
              ("XG", "SCAHI, refined grid (exploratory)", AQUA))
    w = 0.2
    for (sz, f), ax_row in zip(sizes, axes):
        om = jload(f)["results"]
        tags = [(m, f"{m}|B:o2m") for m in EXT_MODELS if f"{m}|B:o2m" in om]
        for ax, grp, lbl in ((ax_row[0], "crowd", "crowded GT (same-class neighbour, IoU > 0.3)"),
                             (ax_row[1], "iso", "isolated GT")):
            for k, (key, name, col) in enumerate(series):
                xs, ys = [], []
                for i, (m, tag) in enumerate(tags):
                    if key == "XG":
                        r = xg.get(f"{m}/{sz}/o2m/C")
                        v = r["val_selected"][f"R50_{grp}"] if r else None
                    else:
                        v = om[tag]["recall50_all"][grp].get(key)
                    if v is None:
                        continue
                    xs.append(i + (k - 1.5) * w)
                    ys.append(v)
                    rows.append({"s": sz, "model": m, "group": grp, "row": key, "recall50": v})
                if ys:
                    ax.bar(xs, ys, width=w * 0.92, color=col, label=name)
            ax.set_axisbelow(True)
            ax.set_xticks(range(len(tags)), [MODEL_LABEL[m] for m, _ in tags], fontsize=6.2)
            ax.set_ylim(50 if grp == "crowd" else 75, 100)
            ax.grid(axis="x", visible=False)
            _interim_title(ax, f"s = {sz}: {lbl}")
            ax.set_ylabel("recall@0.5 [%]", fontsize=6.2)
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    fig.text(0.0, -0.26 / len(sizes), "val, all detections (≤ 500 per image, as in the evaluation); YOLOv12m only at "
             "s = 1280 (2048 dropped); exploratory, post hoc", fontsize=5.6, color=MUTED)
    save(fig, "fig_ext_recall", OUT_EXT)
    save_csv(rows, "fig_ext_recall", OUT_EXT)


def fig_ext_xgrid_cost() -> None:
    """X-grid calib curve per case: calib AP vs. fusion time per image (ratio to the F2 reference timed in the same
    process), every evaluated variant incl. those above the 2x limit (exploratory, post hoc)."""
    f = REPO / "runs/ext/grid_ext.json"
    if not f.exists():
        return
    res = jload("runs/ext/grid_ext.json")["results"]
    order = {m: i for i, m in enumerate((*EXT_MODELS, "yolov10s"))}  # YOLOv10s 2048 (R2 / X2 weights) last
    keys = sorted(res, key=lambda k: (order[k.split("/")[0]], int(k.split("/")[1]), k.split("/")[2], k.split("/")[3]))
    pk = lambda r: tuple(round(float(r[p]), 12) for p in ("tau_splat", "tau_peak", "cell_factor", "bins_per_oct"))  # noqa: E731
    cf_col = {7.0: ORD4[0], 9.0: ORD4[1], 11.0: ORD4[2], 13.0: ORD4[3]}
    thr = lambda r: ("ref" if (r["tau_splat"], r["tau_peak"]) == (0.005, 0.001) else  # noqa: E731
                     "low" if r["tau_splat"] < 0.005 and r["tau_peak"] < 0.001 else "mixed")
    mk = {"ref": "o", "mixed": "s", "low": "^"}
    nc = 4
    nr = -(-len(keys) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(FULL_W, 1.45 * nr), constrained_layout=True, squeeze=False)
    rows = []
    for ax, k in zip(axes.flat, keys):
        o = res[k]
        sel, best = pk(o["calib_selected"]), pk(o["calib_best_unconstrained"])
        for r in o["calib_all"]:
            ref = r["cell_factor"] == 5.0
            ax.scatter(r["time_ratio_vs_ref"], r["AP"], s=9, marker="D" if ref else mk[thr(r)],
                       color=MUTED if ref else cf_col.get(float(r["cell_factor"]), INK2), linewidths=0, zorder=3)
            rows.append({"case": k, **{c: r[c] for c in ("tau_splat", "tau_peak", "cell_factor", "bins_per_oct", "AP",
                                                          "R50_crowd", "R50_iso", "dup_rate", "frag_rate",
                                                          "fuse_ms_per_image", "time_ratio_vs_ref", "peak_mem_mib")},
                         "selected": pk(r) == sel, "best_unconstrained": pk(r) == best})
        for key, st in ((sel, dict(s=46, facecolors="none", edgecolors=INK, linewidths=0.8)),
                        (best, dict(s=60, marker="*", facecolors="none", edgecolors=ORANGE, linewidths=0.7))):
            r = next(r for r in o["calib_all"] if pk(r) == key)
            ax.scatter(r["time_ratio_vs_ref"], r["AP"], zorder=4, **st)
        ax.axvline(2.0, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=2)
        m, sz, br, row = k.split("/")
        ax.set_title(f"{MODEL_LABEL[m]} {sz} {br} · {'whole image' if row == 'B' else '+ tiles'}", fontsize=6.2)
        ax.tick_params(labelsize=5.6)
        ax.set_axisbelow(True)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("calib AP", fontsize=6.0)
    for ax in axes[-1, :]:
        ax.set_xlabel("fusion time / pre-specified setting", fontsize=6.0)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="s", color=c, ms=3.5, label=f"cell factor {int(v)}") for v, c in cf_col.items()]
    h += [Line2D([], [], ls="", marker="D", color=MUTED, ms=3, label="SCAHI, pre-specified setting"),
          Line2D([], [], ls="", marker="o", color=INK2, ms=3.5, label="ref. thresholds"),
          Line2D([], [], ls="", marker="s", color=INK2, ms=3.5, label="one threshold lowered"),
          Line2D([], [], ls="", marker="^", color=INK2, ms=3.5, label="both lowered"),
          Line2D([], [], ls="", marker="o", mfc="none", mec=INK, ms=5, label="selected (time ≤ 2×)"),
          Line2D([], [], ls="", marker="*", mfc="none", mec=ORANGE, ms=6, label="best without limit"),
          Line2D([], [], color=INK2, lw=0.6, ls=(0, (3, 2)), label="2× time limit")]
    fig.legend(handles=h, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6)
    title = ("SCAHI, refined grid on calib: AP vs. fusion time (exploratory, post hoc; selected on calib after val results "
             "were known)")
    fig.suptitle(title if _ext_complete() else f"INTERIM - {title}", fontsize=7.0,
                 color=INK if _ext_complete() else ORANGE)
    save(fig, "fig_ext_xgrid_cost", OUT_EXT)
    save_csv(rows, "fig_ext_xgrid_cost", OUT_EXT)


CROSS_ID = "X_v10s_train2560_infer3072_o2o"


def fig_ext_r3_pareto() -> None:
    """YOLOv10s compute control (exploratory, post hoc): AP vs. GFLOPs per image and vs. E2 latency (reference path
    filled, optimised path hollow) for R2, X2, X-grid B/C (R2 / X2 weights) and R3 (single pass 2560; 3072 did not fit
    in memory); one row of panels per split (val; test-dev from runs/ext/testdev_analysis.json). Cost (GFLOPs,
    latency) is measured on val and reused for test-dev. Source runs/ext/r3_compare.json; GFLOPs from
    runs/gflops_unified.json."""
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    f = REPO / "runs/ext/r3_compare.json"
    if not f.exists():
        return
    rc = jload("runs/ext/r3_compare.json")["rows"]
    if (REPO / "runs/ext/res_crossing.json").exists():  # 2560 weights at 3072 px (val only; E2 latency, unified GFLOPs)
        x = next(q for q in jload("runs/ext/res_crossing.json")["rows"] if q["exp_id"] == CROSS_ID)
        rc = rc + [{"id": CROSS_ID, "metrics": {"AP": x["AP"]}, "gflops_per_img": x["gflops_unified_per_img"],
                    "latency": {"reference_ms": x["latency_e2_ms_mean"], "optimised_ms": x["latency_e2_ms_mean"]}}]
    ap = {"val": {r["id"]: r["metrics"]["AP"] for r in rc}}
    if (REPO / "runs/ext/testdev_analysis.json").exists():
        ap["test-dev"] = {r["id"]: r["AP"] for r in jload("runs/ext/testdev_analysis.json")["compute_control_testdev"]}
    style_of = lambda r: (("single pass (train = test size)", BLUE, "o") if r["id"] in ("R2", "E_yolov10s_full2560_o2o") else  # noqa: E731
                          ("single pass, 2560 weights at 3072 px (exploratory)", BLUE_D, "v") if r["id"] == CROSS_ID else
                          ("2048 + tiles, SCAHI (pre-specified setting)", ORANGE, "s") if r["id"] == "X2" else
                          ("SCAHI, refined grid, exploratory (whole image / + tiles)", AQUA, "^"))
    short = {"R2": "2048", "X2": "SCAHI", "E_yolov10s_full2560_o2o": "2560", CROSS_ID: "2560 w. @ 3072"}
    fig, axes = plt.subplots(len(ap), 2, figsize=(FULL_W, 2.3 * len(ap)), constrained_layout=True, squeeze=False)
    rows = []
    for ax_row, sp in zip(axes, ap):
        for ax, key, xl in ((ax_row[0], "gflops", "GFLOPs per image (exact per input shape)"),
                            (ax_row[1], "latency", "latency per image [ms]  (exclusive, RTX 5090, 575 W)")):
            seen = set()
            for r in rc:
                lbl, col, mk = style_of(r)
                lat = r.get("latency") or {}
                x, yv = (r["gflops_per_img"] if key == "gflops" else lat.get("reference_ms")), ap[sp].get(r["id"])
                if x is None or yv is None:
                    continue
                ax.scatter(x, yv, marker=mk, s=26, color=col, edgecolor=SURF, lw=0.6, zorder=4,
                           label=None if lbl in seen else lbl)
                seen.add(lbl)
                if key == "latency" and lat.get("optimised_ms") and lat["optimised_ms"] != lat["reference_ms"]:
                    ax.plot([lat["optimised_ms"], x], [yv] * 2, color=col, lw=0.7, zorder=3)
                    ax.scatter(lat["optimised_ms"], yv, marker=mk, s=22, facecolor="none", edgecolor=col, lw=0.9,
                               zorder=4)
                tag = short.get(r["id"]) or "refined grid " + r["id"].replace("XG_yolov10s_2048_", "").replace(
                    "_B", ", whole image").replace("_C", ", + tiles")
                off = {"R2": (5, -8), "XG_yolov10s_2048_o2o_B": (5, 0), "XG_yolov10s_2048_o2m_B": (5, 1),
                       CROSS_ID: (-30, 6)}.get(r["id"], (4, -3))
                if sp == "test-dev":
                    off = {"R2": (5, -12 if key == "gflops" else -8), "XG_yolov10s_2048_o2o_B": (5, -1 if key == "gflops" else -6), "XG_yolov10s_2048_o2m_B": (5, 2),
                           "XG_yolov10s_2048_o2o_C": (4, -7)}.get(r["id"], (4, -3))
                ax.annotate(tag, (x, yv), xytext=off, textcoords="offset points", fontsize=5.6, color=INK2)
                rows.append({"split": sp, "panel": key, "id": r["id"], "AP": yv, "x": x,
                             "optimised_ms": lat.get("optimised_ms") if key == "latency" else None})
            ax.set_xscale("log")
            ticks = [100, 150, 200, 300, 500, 700] if key == "gflops" else [5, 10, 20, 50, 100, 200]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel(xl, fontsize=6.4)
            ax.set_ylabel(f"{sp} AP", fontsize=6.4)
            v = [y for y in ap[sp].values()]
            ax.set_ylim(38.2, 40.9) if sp == "val" else ax.set_ylim(min(v) - 0.35, max(v) + 0.35)
            ax.set_xlim(*((90, 800) if key == "gflops" else (6, 250)))
            ax.set_axisbelow(True)
        _interim_title(ax_row[0], f"YOLOv10s, {sp}: AP vs. compute")
        _interim_title(ax_row[1], f"YOLOv10s, {sp}: AP vs. latency (hollow = optimised path)")
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    fig.text(0.0, -0.22 / len(ap), "Single pass 2048, SCAHI and refined grid on the single-pass 2048 weights; single pass "
             "2560 = own training at 2560 (same recipe); training at 3072 did not fit at the recipe batch 3. The 2560 "
             "weights at 3072 px (exploratory, post hoc) are on val only. Refined grid: exploratory, post hoc, selected "
             "on calib. GFLOPs and latency measured on val, reused for test-dev.", fontsize=5.4, color=MUTED, wrap=True)
    save(fig, "fig_ext_r3_pareto", OUT_EXT)
    save_csv(rows, "fig_ext_r3_pareto", OUT_EXT)


FIGS = {"resolution": fig_resolution, "pareto": fig_pareto, "sizebins": fig_sizebins, "fusion_gain": fig_fusion_gain,
        "degradation": fig_degradation, "band_edge": fig_band_edge, "border": fig_border, "h1": fig_h1,
        "windows": fig_windows, "qualitative": fig_qualitative, "volume": fig_volume}
if (REPO / "runs/testdev/analysis.json").exists():
    FIGS.update(prereg_forest=fig_prereg_forest, testdev=fig_testdev)
EXT_FIGS = {"ext_calib_curves": fig_ext_calib_curves, "ext_scaling": fig_ext_scaling, "ext_forest": fig_ext_forest,
            "ext_waterfall": fig_ext_waterfall, "ext_recall": fig_ext_recall,
            "ext_xgrid_cost": fig_ext_xgrid_cost, "ext_r3_pareto": fig_ext_r3_pareto}  # extension study (own folder)


def main() -> None:
    style()
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()


if __name__ == "__main__":
    main()
