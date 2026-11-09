# SPDX-License-Identifier: AGPL-3.0-or-later
"""LaTeX tables (booktabs) for the paper/appendix, generated from runs/ (no hand-typed numbers).

    python scripts/paper_tables.py

Writes reports/figures/paper/tex/tab_{main,prereg,exploratory,fusion}.tex and, once runs/testdev/analysis.json exists,
the test-dev tables tab_{testdev_main,testdev_prereg,testdev_exploratory,testdev_sizebins,consistency}.tex (+ .csv).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports/figures/paper/tex"
# Paper terms (DECISIONS 2026-10-31 ~20:40): "<pass> + tiles, <merge>"; relative-zoom stacks by absolute zoom
LMT, SNT = "loss-masked training", "scale-normalised training"
S4, S5, S1 = "z0.64 + z2 tiles", "z1 + z2 tiles", "z1/3 + z1 + z2 tiles"
F3E = "SCAHI + validity windows (estimated on calib)"
MAB = "Selected main-study configuration (scale-normalised training, SCAHI + validity windows)"


def latest(glob: str) -> dict:
    return json.loads(sorted(REPO.glob(glob))[-1].read_text())


def f1(v, nd=1, sign=False):
    if v is None:
        return "--"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def ci(d):
    return f"{d['delta']:+.2f} {{\\scriptsize[{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]}}"


def tab_main() -> None:
    ex = latest("runs/p6/*_exploratory/explore.json")
    p6 = latest("runs/p6/*_phase6/phase6.json")
    rows = {r["label"]: r for r in ex["rows"]}
    dR, dR2 = p6["deltas_vs_R"], p6["deltas_vs_R2"]
    order = [("B-FULL 640 @640", "Single pass 640"), ("B-FULL 1280 @1280", "Single pass 1280"),
             ("B-FULL 1536 @1536", "Single pass 1536"), ("B-FULL 2048 @2048", "Single pass 2048 ($R_2$)"),
             ("B-SAHI-FT", "SAHI fine-tuned"), ("B-SAHI (B-FULL-1280-Gewichte)", "SAHI (1280 weights)"),
             ("R = B-MZ-native (NMS/IOS, Predictor)", "1280 + tiles, box fusion ($R$)"),
             ("B-MZ-1536-native", "1536 + tiles, box fusion"), ("B-MZ S4", f"{S4}, box fusion")]
    meth = [k for k in rows if rows[k]["kind"] == "method"]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Method & AP & AP$_{\sigma<16}$ & MP/img & ms/img & $\Delta$ vs.\ $R$ & $\Delta$ vs.\ $R_2$ \\", r"\midrule"]

    def line(key, name):
        r = rows[key]
        e = r["exp_id"]
        a = ci(dR[e]["AP"]) if e in dR else "--"
        b = ci(dR2[e]["AP"]) if e in dR2 else "--"
        lat = r["latency_ms_mean"] if r.get("latency_status") == "final" else None
        return f"{name} & {f1(r['AP'], 2)} & {f1(r['AP_size']['<16'])} & {f1(r['input_mp_per_img'], 2)} & {f1(lat)} & {a} & {b} \\\\"

    for k, n in order:
        if k in rows:
            lines.append(line(k, n))
    lines.append(r"\midrule")
    nice = {"M_ab": MAB, "T4 beste": f"{SNT}, {S5}, SCAHI", "B-FULL 1280 beste": f"1280 weights, {S5}, SCAHI",
            "B-FULL 1536 beste": f"1536 weights, {S5}, SCAHI", "T4 + F0": f"{SNT}, {S5}, box fusion",
            "T2 + F0": f"{LMT}, {S5}, box fusion"}
    for k in meth:
        n = next((v for p, v in nice.items() if k.startswith(p)), None)
        if n:
            lines.append(line(k, n))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_main.tex").write_text("\n".join(lines) + "\n")


def tab_prereg() -> None:
    p = latest("runs/p6/*_phase6/phase6.json")["preregistration"]
    rows = [("(a) $\\Delta$AP vs.\\ $R \\geq +1.0$, CI $>0$", "a"), ("(b) $\\Delta$AP$_{\\sigma<16}$ vs.\\ $R \\geq +2.0$, CI $>0$", "b"),
            ("(c) $\\leq 0.7\\times$ cost, $|\\Delta$AP$|\\leq\\varepsilon$", "c"), ("(a) vs.\\ $R_2$ (addendum)", "a_R2"),
            ("(b) vs.\\ $R_2$ (addendum)", "b_R2"), ("H1: scale-normalised $-$ loss-masked training (box fusion) $> \\varepsilon$", "H1"),
            ("H2: best SCAHI variant $-$ box fusion $> \\varepsilon$", "H2"), ("H3: o2m vs.\\ o2o (direction)", "H3")]
    trans = {"erfüllt": "met", "nicht erfüllt": "not met", "nicht prüfbar": "not testable", "unklar": "unclear"}
    lines = [r"\begin{tabular}{lrrl}", r"\toprule", r"Criterion & $\Delta$ & 95\% CI & Verdict \\", r"\midrule"]
    for name, k in rows:
        v = p.get(k, {})
        if "delta" in v:
            verdict = trans.get(v.get("verdict"), v.get("direction", "")).replace("o2m > o2o", "o2m $>$ o2o")
            lines.append(f"{name} & {v['delta']:+.2f} & [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] & {verdict} \\\\")
        else:
            lines.append(f"{name} & -- & -- & {trans.get(v.get('verdict'), v.get('verdict', '--'))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_prereg.tex").write_text("\n".join(lines) + "\n")


def tab_exploratory() -> None:
    ex = latest("runs/p6/*_exploratory/explore.json")
    d = {x["label"]: x for x in ex["deltas_vs_single_pass"]}
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Exploratory (defined after val) & AP & $\Delta$ vs.\ single pass & $\Delta_{\sigma<16}$ & frag.\ [\%] & MP & ms \\",
             r"\midrule"]
    for r in ex["exploratory"]:
        x = d[r["label"]]
        sz = r["label"].split("B-FULL ")[1][:4]
        name = f"{sz} + tiles, " + ("SCAHI" if " F2 " in r["label"] else "box fusion")
        lines.append(f"{name} & {r['AP']:.2f} & {ci(x['dAP'])} & {ci(x['dAP_lt16'])} & {100 * r['frag_rate']:.2f} & "
                     f"{r['input_mp_per_img']:.2f} & {f1(r['latency_ms_mean'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_exploratory.tex").write_text("\n".join(lines) + "\n")


def tab_fusion() -> None:
    V = ("F0-NMS", "F0-GREEDYNMM", "F0-WBF", "F1", "F2", "F3", "F3e", "F4", "F5")
    cells = (("BFULL1280", "native", "o2m", "1280 weights, 1280 + tiles"), ("BFULL1536", "native1536", "o2m", "1536 weights, 1536 + tiles"),
             ("P3T2_s0", "S4", "o2m", f"{LMT}, {S4}"), ("P3T4_s0", "S4", "o2m", f"{SNT}, {S4} (selected cell)"),
             ("P3T4_s0", "S5", "o2m", f"{SNT}, {S5}"))
    # column terms (user, DECISIONS 2026-10-31 ~20:52): F0 box fusion, F1 uniform weights, F2 SCAHI, F3-F5 cumulative on
    # top of F2 (F3 analytic validity, F4 + uncertainty, F5 + temperature); F3e = F3 with calib-estimated validity
    hdr = ("NMS & GNMM & WBF & uniform weights & SCAHI & + validity (analytic) & + validity (calib) & + uncertainty & "
           "+ temperature")
    lines = [r"\begin{tabular}{l" + "r" * len(V) + "}", r"\toprule",
             r" & \multicolumn{3}{c}{box fusion} & \multicolumn{6}{c}{dense fusion} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-10}",
             f"Weights, passes (o2m head) & {hdr} \\\\", r"\midrule"]
    for m, s, b, name in cells:
        vals = []
        for v in V:
            f = REPO / "runs/p5" / f"20261023_v10s_{m}_{s}_{v}_{b}" / "metrics.json"
            vals.append(f"{json.loads(f.read_text())['metrics']['AP']:.1f}" if f.exists() else "--")
        best = max(float(x) for x in vals if x != "--")
        vals = [f"\\textbf{{{x}}}" if x != "--" and float(x) == best else x for x in vals]
        lines.append(f"{name} & " + " & ".join(vals) + " \\\\")
    lines += [r"\bottomrule",
              r"\multicolumn{10}{l}{\footnotesize Box fusion: class-wise NMS, GREEDYNMM, WBF. Dense fusion: uniform weights, "
              r"then SCAHI (border taper); + validity (analytic), + uncertainty and + temperature are cumulative on top of "
              r"SCAHI.} \\",
              r"\multicolumn{10}{l}{\footnotesize + validity (calib) replaces the analytic validity windows by ones "
              r"estimated on calib (= SCAHI + validity windows (estimated on calib)). AP [\%], VisDrone val, best per row "
              r"in bold.} \\",
              r"\end{tabular}"]
    (OUT / "tab_fusion.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------- test-dev

TD_JSON = REPO / "runs/testdev/analysis.json"
VERD = {"erfüllt": "met", "nicht erfüllt": "not met", "nicht prüfbar": "not testable", "unklar": "unclear",
        "o2m > o2o": "o2m $>$ o2o", "o2o ≥ o2m": "o2o $\\geq$ o2m"}
# T0 / T1 / T3 (user, DECISIONS 2026-10-31 ~20:52); T0-T3 share the same chips, they differ only in the ignore handling
T0, T1, T3 = "ignore regions dropped", "ignore regions painted grey", "loss-masked + truncation ignore"
NAMES = {"R1": "Single pass 640", "R2a": "Single pass 1280", "R2b": "Single pass 1536", "R2": "Single pass 2048 ($R_2$)",
         "R3": "SAHI fine-tuned", "R4": "SAHI (1280 weights)", "R": "1280 + tiles, box fusion ($R$)",
         "R5": "1536 + tiles, box fusion", "R6": f"{S4}, box fusion", "R7": f"{S1}, box fusion",
         "R-NMS": "1280 + tiles, box fusion (NMS, cached passes)", "R-GNMM": "1280 + tiles, box fusion (GREEDYNMM, cached passes)",
         "R-WBF": "1280 + tiles, box fusion (WBF, cached passes)",
         "T0": f"{T0}, {S5}, box fusion", "T1": f"{T1}, {S5}, box fusion", "T2": f"{LMT}, {S5}, box fusion",
         "T3": f"{T3}, {S5}, box fusion", "T4": f"{SNT}, {S5}, box fusion",
         "M_ab": MAB, "H2-F0": f"{SNT}, {S4}, box fusion (o2m)", "H3-o2o": f"{SNT}, {S4}, {F3E} (o2o)",
         "2x2-T2-F0": f"{LMT}, {S4}, box fusion (o2m)", "2x2-T2-F3e": f"{LMT}, {S4}, {F3E} (o2m)",
         "D-T4": f"{SNT}, {S5}, SCAHI", "D-1280": f"1280 weights, {S5}, SCAHI", "D-1536": f"1536 weights, {S5}, SCAHI",
         "X1": "2048 + tiles, box fusion", "X2": "2048 + tiles, SCAHI", "X3": "1536 + tiles, box fusion",
         "X4": "1536 + tiles, SCAHI"}
for _sd in ("s1", "s2"):  # seed replicates of the H1 / M_ab rows (tab_consistency)
    NAMES |= {f"seed-P3T4_{_sd}-H1": f"{SNT}, {S5}, box fusion (seed {_sd[1]})",
              f"seed-P3T4_{_sd}-M_ab": f"{MAB[0].lower()}{MAB[1:]} (seed {_sd[1]})",
              f"seed-P3T2_{_sd}-H1": f"{LMT}, {S5}, box fusion (seed {_sd[1]})"}


def _td() -> dict:
    return json.loads(TD_JSON.read_text())


def _csv(name: str, header: list[str], rows: list[list]) -> None:
    import csv

    if header[0] == "id":  # paper term next to the internal id
        header = ["id", "term", *header[1:]]
        rows = [[r[0], NAMES.get(r[0], r[0]).replace("$", ""), *r[1:]] for r in rows]
    with (OUT / f"{name}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def tab_testdev_main() -> None:
    a = _td()
    dR, dR2 = a["deltas_vs_R_test-dev"], a["deltas_vs_R2_test-dev"]
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & val & \multicolumn{4}{c}{test-dev} & & & \\ \cmidrule(lr){2-2}\cmidrule(lr){3-6}",
             r"Method & AP & AP & AP$_{plain}$ & AP$_{<8}$ & AP$_{\sigma<16}$ & ms & $\Delta$ vs.\ $R$ & $\Delta$ vs.\ $R_2$ \\",
             r"\midrule"]
    csvr = []
    groups = (("reference",), ("prereg",))
    for gi, g in enumerate(groups):
        if gi:
            lines.append(r"\midrule")
        for r in a["rows"]:
            if r["group"] not in g:
                continue
            i, v, t = r["id"], r["val"], r["test-dev"]
            ci_r = ci(dR[i]["AP"]) if i in dR else "--"
            ci_r2 = ci(dR2[i]["AP"]) if i in dR2 else "--"
            lines.append(f"{NAMES[i]} & {v['AP']:.2f} & {t['AP']:.2f} & {t['AP_plain']:.2f} & {t['AP_size']['<8']:.1f} & "
                         f"{t['AP_size']['<16']:.1f} & {f1(r['latency_ms_mean_val'])} & {ci_r} & {ci_r2} \\\\")
            csvr.append([i, r["label"], v["AP"], t["AP"], t["AP_plain"], t["AP_size"]["<8"], t["AP_size"]["<16"],
                         r["latency_ms_mean_val"], dR.get(i, {}).get("AP", {}).get("delta"), dR2.get(i, {}).get("AP", {}).get("delta")])
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_testdev_main.tex").write_text("\n".join(lines) + "\n")
    _csv("tab_testdev_main", ["id", "label", "AP_val", "AP_testdev", "AP_plain_testdev", "AP_lt8_testdev", "AP_lt16_testdev",
                              "latency_ms_val", "dAP_vs_R_testdev", "dAP_vs_R2_testdev"], csvr)


def tab_testdev_prereg() -> None:
    a = _td()
    pv, pt = a["preregistration_val"], a["preregistration_test-dev"]
    rows = [("(a) $\\Delta$AP vs.\\ $R \\geq +1.0$", "a"), ("(b) $\\Delta$AP$_{\\sigma<16}$ vs.\\ $R \\geq +2.0$", "b"),
            ("(c) $\\leq 0.7\\times$ cost, $|\\Delta|\\leq\\varepsilon$", "c"), ("(a) vs.\\ $R_2$", "a_R2"),
            ("(b) vs.\\ $R_2$", "b_R2"), ("H1: scale-normalised $-$ loss-masked $> \\varepsilon$", "H1"),
            ("H2: SCAHI variant $-$ box fusion $> \\varepsilon$", "H2"),
            ("H3: o2m vs.\\ o2o", "H3")]

    def cell(p):
        if "delta" not in p:
            return "-- & " + VERD.get(p.get("verdict"), "--")
        return f"{p['delta']:+.2f} {{\\scriptsize[{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]}} & " + \
            VERD.get(p.get("verdict", p.get("direction")), "--")

    lines = [r"\begin{tabular}{lrlrl}", r"\toprule", r" & \multicolumn{2}{c}{val} & \multicolumn{2}{c}{test-dev} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Criterion & $\Delta$ [95\% CI] & verdict & $\Delta$ [95\% CI] & verdict \\",
             r"\midrule"]
    csvr = []
    for name, k in rows:
        lines.append(f"{name} & {cell(pv[k])} & {cell(pt[k])} \\\\")
        csvr.append([k, pv[k].get("delta"), pv[k].get("verdict", pv[k].get("direction")), pt[k].get("delta"),
                     pt[k].get("verdict", pt[k].get("direction"))])
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_testdev_prereg.tex").write_text("\n".join(lines) + "\n")
    _csv("tab_testdev_prereg", ["criterion", "delta_val", "verdict_val", "delta_testdev", "verdict_testdev"], csvr)


def tab_testdev_exploratory() -> None:
    a = _td()
    rows = {r["id"]: r for r in a["rows"]}
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Exploratory (defined after val) & vs. & AP val & AP test-dev & $\Delta$ val & $\Delta$ test-dev & $\Delta_{\sigma<16}$ test-dev \\",
             r"\midrule"]
    csvr = []
    for x, ref in sorted(a["consistency_exploratory"].items()):
        dv, dt = a["exploratory_val"][x], a["exploratory_test-dev"][x]
        lines.append(f"{NAMES[x]} & {NAMES[{'X1': 'R2', 'X2': 'R2', 'X3': 'R2b', 'X4': 'R2b'}[x]].replace(' ($R_2$)', '')} & "
                     f"{rows[x]['val']['AP']:.2f} & {rows[x]['test-dev']['AP']:.2f} & {ci(dv['AP'])} & {ci(dt['AP'])} & {ci(dt['AP_lt16'])} \\\\")
        csvr.append([x, rows[x]["val"]["AP"], rows[x]["test-dev"]["AP"], dv["AP"]["delta"], dt["AP"]["delta"], *dt["AP"]["ci95"]])
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_testdev_exploratory.tex").write_text("\n".join(lines) + "\n")
    _csv("tab_testdev_exploratory", ["id", "AP_val", "AP_testdev", "dAP_val", "dAP_testdev", "ci_lo_testdev", "ci_hi_testdev"], csvr)


def tab_testdev_sizebins() -> None:
    a = _td()
    rows = {r["id"]: r for r in a["rows"]}
    bins = ("<8", "8-16", "16-32", "32-64", "64-128", ">128")
    sel = ("R2a", "R2", "R3", "R", "M_ab", "D-T4", "D-1536", "X2")
    hdr = " & ".join(f"{b.replace('-', '--')}" for b in bins)
    lines = [r"\begin{tabular}{l" + "rr" * len(bins) + "}", r"\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{b.replace('-', '--')} px}}" for b in bins) + r" \\",
             "".join(f"\\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(bins))),
             "Method (test-dev) & " + " & ".join(["AP & AP$_{50}$"] * len(bins)) + r" \\", r"\midrule"]
    sz = a["split_sizes"]
    lines.insert(-1, "share of GT, train / val / test-dev [\\%] & " + " & ".join(
        f"\\multicolumn{{2}}{{c}}{{\\scriptsize {100 * sz['train'][b]:.0f} / {100 * sz['val'][b]:.0f} / {100 * sz['test-dev'][b]:.0f}}}"
        for b in bins) + r" \\ \midrule")
    csvr = []
    for i in sel:
        t = rows[i]["test-dev"]
        lines.append(f"{NAMES[i]} & " + " & ".join(f"{t['AP_size'][b]:.1f} & {t['AP50_size'][b]:.1f}" for b in bins) + r" \\")
        csvr.append([i] + [v for b in bins for v in (t["AP_size"][b], t["AP50_size"][b])])
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_testdev_sizebins.tex").write_text("\n".join(lines) + "\n")
    _csv("tab_testdev_sizebins", ["id"] + [f"{m}_{b}" for b in bins for m in ("AP", "AP50")], csvr)


def tab_consistency() -> None:
    a = _td()
    c = a["consistency_rows"]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Row & AP val & AP test-dev & $\Delta$ & rank val & rank test-dev \\", r"\midrule"]
    csvr = []
    for pool in ("main", "exploratory"):
        if pool == "exploratory":
            lines.append(r"\midrule")
        for r in a["rows"]:
            i = r["id"]
            if c[i]["rank_pool"] != pool:
                continue
            x = c[i]
            lines.append(f"{NAMES.get(i, i.replace('_', ' '))} & {x['AP_val']:.2f} & {x['AP_testdev']:.2f} & "
                         f"{x['AP_testdev'] - x['AP_val']:+.2f} & {x['rank_val']} & {x['rank_testdev']} \\\\")
            csvr.append([i, x["AP_val"], x["AP_testdev"], x["rank_val"], x["rank_testdev"], pool])
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_consistency.tex").write_text("\n".join(lines) + "\n")
    _csv("tab_consistency", ["id", "AP_val", "AP_testdev", "rank_val", "rank_testdev", "rank_pool"], csvr)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tab_main()
    tab_prereg()
    tab_exploratory()
    tab_fusion()
    if TD_JSON.exists():
        tab_testdev_main()
        tab_testdev_prereg()
        tab_testdev_exploratory()
        tab_testdev_sizebins()
        tab_consistency()
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
