# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO26m verdicts (addendum 7c01b5d item 4): E-H1, E-H2 (per s), E-H3 as for YOLOv10m, eps = 0.30, paired bootstrap
over the images (1000, seed 0), verdict per detector; E-H4 descriptive; Holm appendix for YOLO26m alone (a family of one
model = unadjusted). val decides; test-dev is reported next to it with the same definitions (no confirmation role).
The five-model verdicts, ">= 4 of 5" and the pooled effect are untouched; YOLO26m counts in no family criterion.

    python scripts/ext26_verdicts.py [--workers 6]    # -> runs/ext/yolo26m/verdicts.json, reports/EXT26_RESULTS.md

Same building blocks as the five models (scripts/testdev_analysis._row_stats / _d, scahi.ext_verdicts.status /
p_one_sided / holm); rows from runs/ext/yolo26m/val/ and runs/ext/yolo26m/testdev/ (frozen rows of
configs/ext/frozen_rows_ext26.json).

YOLO26s (``--model yolo26s``, addendum 7bc0ace): same questions and rule -> runs/ext/yolo26s/verdicts.json,
reports/EXT26S_RESULTS.md; plus G-E2 (item 6: E-1280 o2o within +-5 AP of YOLOv10s E-1280, val 33.66; reported, changes
no verdict) and the descriptive equal-compute comparison (item 8): single pass YOLO26s 2048 vs. YOLO26m 1280, both o2o,
GFLOPs per image (scahi.flops, exact per input shape) and AP on val / test-dev with paired bootstrap CI (1000, seed 0).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from testdev_analysis import _d, _row_stats  # noqa: E402

from scahi import ext as E  # noqa: E402
from scahi import ext_verdicts as V  # noqa: E402

M = "yolo26m"
D = E.mdir(M)
ROOT = {"val": D / "val", "test-dev": D / "testdev"}
EPS = 0.30
FREEZE = REPO / "configs/ext/frozen_rows_ext26.json"
REPORT = REPO / "reports/EXT26_RESULTS.md"
ADD = E.ADDENDUM_BY_MODEL[M]
GE2_REF = 33.66  # YOLOv10s E-1280 (B-FULL 1280) val AP, addendum 7bc0ace item 6
COUNTED = ("E-H1", "E-H2 s=1280", "E-H2 s=2048", "E-H3")


def specs() -> list[tuple[str, str, str]]:
    """(label, row a, row b) as for YOLOv10m (scripts/ext_interim_ci.specs, two-branch model)."""
    r = lambda s: f"E_{M}_{s}"  # noqa: E731
    out = [("E-H1", r("full2048_o2o"), r("mz1280_nms-ios_o2o"))]
    for s in (1280, 2048):
        out.append((f"E-H2 s={s}", r(f"mz{s}_F2_o2m"), r(f"mz{s}_nms-ios_o2m")))
    out.append(("E-H3", r("mz2048_F2_o2m"), r("full2048_o2o")))
    out.append(("E-H4", r("mz2048_F2_o2m"), r("mz2048_F2_o2o")))
    out.append(("E-H4 s=1280 (descriptive)", r("mz1280_F2_o2m"), r("mz1280_F2_o2o")))
    return out


def _st(args):
    split, rid, root = args
    _, _, o = _row_stats((split, rid, str(Path(root) / rid), False))
    return rid, {"AP0": o["AP0"], "AP_vec": o["AP_vec"]}


def evaluate(split: str, workers: int) -> dict:
    sp = specs()
    rows = sorted({x for s in sp for x in s[1:]})
    with ProcessPoolExecutor(workers, mp_context=E._spawn()) as ex:
        st = dict(ex.map(_st, [(split, r, str(ROOT[split])) for r in rows]))
    res = {}
    for lab, a, b in sp:
        d = _d(st[a]["AP0"], st[b]["AP0"], st[a]["AP_vec"], st[b]["AP_vec"])
        d.pop("excludes_zero")
        dv = st[a]["AP_vec"] - st[b]["AP_vec"]
        x = {**d, "row_a": a, "row_b": b}
        if lab in COUNTED:
            h = lab.split(" ")[0]
            x["status"] = V.status(h, d["delta"], d["ci95"][0], EPS)
            x["p_one_sided"] = V.p_one_sided(dv, EPS)
            hp = V.holm({M: x["p_one_sided"]}, m=1)[M]
            if h == "E-H3":
                hp["reject"] = hp["reject"] and d["ci95"][0] > 0
            x["holm_one_model"] = hp
        res[lab] = x
    return {"split": split, "rows": {r: st[r]["AP0"] for r in rows}, "hypotheses": res}


def equal_compute(workers: int) -> dict:
    """Addendum 7bc0ace item 8 (descriptive): single pass YOLO26s 2048 vs. YOLO26m 1280, both o2o."""
    from scahi.eval.evaluate import load_coco_gt

    a, b = "E_yolo26s_full2048_o2o", "E_yolo26m_full1280_o2o"
    m26 = E.mdir("yolo26m")
    out = {"row_a": a, "row_b": b, "rule": "descriptive, no verdict, no threshold (addendum 7bc0ace item 8)",
           "gflops_method": "exact per input shape (scahi.flops: Ultralytics get_flops on the fused model at the "
                            "letterboxed shape of each image), mean over the images of the split", "splits": {}}
    for split, sub in (("val", "val"), ("test-dev", "testdev")):
        with ProcessPoolExecutor(2, mp_context=E._spawn()) as ex:
            st = dict(ex.map(_st, [(split, a, str(D / sub)), (split, b, str(m26 / sub))]))
        d = _d(st[a]["AP0"], st[b]["AP0"], st[a]["AP_vec"], st[b]["AP_vec"])
        coco = load_coco_gt(REPO / f"data/canonical/visdrone10/coco/{split}.json")
        gf = {a: E._gf_single(E.weights("yolo26s", 2048), coco, 2048), b: E._gf_single(E.weights("yolo26m", 1280), coco, 1280)}
        out["splits"][split] = {**d, "gflops_per_img": gf, "gflops_ratio_a_over_b": gf[a] / gf[b]}
    return out


def main() -> None:
    global M, D, ROOT, FREEZE, REPORT, ADD
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", choices=("yolo26m", "yolo26s"), default="yolo26m")
    a = ap.parse_args()
    if a.model != M:
        M, D = a.model, E.mdir(a.model)
        ROOT = {"val": D / "val", "test-dev": D / "testdev"}
        FREEZE, REPORT = REPO / "configs/ext/frozen_rows_ext26s.json", REPO / "reports/EXT26S_RESULTS.md"
        ADD = E.ADDENDUM_BY_MODEL[M]
    lab_m = "YOLO" + M[4:]
    fz = json.loads(FREEZE.read_text())
    out = {"model": M, "eps": EPS, "rule": f"addendum {ADD} item {4 if M == 'yolo26m' else 5} (val decides; test-dev next to it)",
           "time": datetime.now().isoformat(timespec="minutes"),
           "splits": {s: evaluate(s, a.workers) for s in ("val", "test-dev")}}
    if M == "yolo26s":
        v = json.loads((ROOT["val"] / "E_yolo26s_full1280_o2o" / "metrics.json").read_text())["metrics"]["AP"]
        out["G-E2"] = {"row": "E_yolo26s_full1280_o2o", "val_AP": v, "reference": GE2_REF, "delta": v - GE2_REF,
                       "rule": "within +-5 AP of YOLOv10s E-1280 (B-FULL 1280, val 33.66); reported, changes no verdict",
                       "passed": abs(v - GE2_REF) <= 5.0}
        out["equal_compute"] = equal_compute(a.workers)
    (D / "verdicts.json").write_text(json.dumps(out, indent=1))
    ap_ = {s: {r["id"]: json.loads((ROOT[s] / r["val_exp_id"] / "metrics.json").read_text())["metrics"]["AP"]
               for r in fz["rows"]} for s in ROOT}
    L = [f"# {lab_m}: single held-out evaluation of the frozen rows", "",
         f"Freeze {FREEZE.relative_to(REPO)} (approved {fz['approved_by'][8:18]}); val once, exact val replay, test-dev once. "
         "AP@[.5:.95], toolkit protocol (visdrone_ignore).", "",
         "## Verdicts (val decides; test-dev reported with the same definitions)", "",
         "| question | a − b | val Δ [95% CI] | val | test-dev Δ [95% CI] | test-dev |", "|---|---|---|---|---|---|"]
    for lab in [s[0] for s in specs()]:
        v, t = out["splits"]["val"]["hypotheses"][lab], out["splits"]["test-dev"]["hypotheses"][lab]
        f = lambda x: f"{x['delta']:+.2f} [{x['ci95'][0]:+.2f}; {x['ci95'][1]:+.2f}]"  # noqa: E731
        L.append(f"| {lab} | {v['row_a'].replace(f'E_{M}_', '')} − {v['row_b'].replace(f'E_{M}_', '')} | {f(v)} | "
                 f"{v.get('status', 'descriptive')} | {f(t)} | {t.get('status', 'descriptive')} |")
    L += ["", f"eps = {EPS}; E-H3 also needs the lower CI bound > 0. Holm appendix ({lab_m} alone, unadjusted): "
          + "; ".join(f"{lab}: p = {out['splits']['val']['hypotheses'][lab]['p_one_sided']:.4f}"
                      for lab in COUNTED), "", "## All frozen rows", "",
          "| row | group | val AP | test-dev AP |", "|---|---|---:|---:|"]
    for r in fz["rows"]:
        L.append(f"| {r['id']} | {r['group']} | {ap_['val'][r['id']]:.2f} | {ap_['test-dev'][r['id']]:.2f} |")
    if M == "yolo26s":
        g = out["G-E2"]
        L += ["", "## G-E2 (s model; reported, changes no verdict)", "",
              f"E-1280 (o2o) val AP {g['val_AP']:.2f} vs. YOLOv10s E-1280 {g['reference']:.2f}: Δ {g['delta']:+.2f} -> "
              f"{'within' if g['passed'] else 'outside'} ±5 AP.", "",
              "## Equal-compute comparison (descriptive): single pass YOLO26s 2048 px vs. YOLO26m 1280 px, both o2o", "",
              "| split | YOLO26s 2048 AP | YOLO26m 1280 AP | Δ [95% CI] | GFLOPs/img YOLO26s 2048 | GFLOPs/img YOLO26m 1280 | ratio |",
              "|---|---:|---:|---|---:|---:|---:|"]
        ec = out["equal_compute"]
        for sp, x in ec["splits"].items():
            gf = x["gflops_per_img"]
            L.append(f"| {sp} | {x['a']:.2f} | {x['b']:.2f} | {x['delta']:+.2f} [{x['ci95'][0]:+.2f}; {x['ci95'][1]:+.2f}] | "
                     f"{gf[ec['row_a']]:.0f} | {gf[ec['row_b']]:.0f} | {x['gflops_ratio_a_over_b']:.2f} |")
        L += ["", f"GFLOPs: {ec['gflops_method']}."]
    REPORT.write_text("\n".join(L) + "\n")
    print("\n".join(L[6:6 + len(specs())]))


if __name__ == "__main__":
    main()
