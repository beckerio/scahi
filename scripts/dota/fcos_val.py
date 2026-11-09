# SPDX-License-Identifier: AGPL-3.0-or-later
"""FCOS single DOTA-val evaluation (addendum 3 item 10; starts only after the user's approval of the freeze).

    python scripts/dota/fcos_val.py precheck       # automatic pre-check (replay, limit rule, sizes); approves nothing
    python scripts/dota/fcos_val.py approve NOTE   # ONLY on the user's explicit instruction: approved true + the note
    python scripts/dota/fcos_val.py gpu            # single-pass rows (A, X) on DOTA-val (scahi.dota.final.val_gpu)
    python scripts/dota/fcos_val.py cpu ROW_ID     # one cache row on DOTA-val, fused by image in parallel
    python scripts/dota/fcos_val.py eval           # metrics, bootstrap, verdicts -> runs/dota/val/results_fcos.json
    python scripts/dota/fcos_val.py rows           # cache row ids in run order (for scripts/dota/fcos_val_auto.sh)

``precheck`` repeats the automatic pre-check of the proposal (every row replayed on calib with |Δ| = 0, limit rule
|Δ mAP50| < 0.05 for every row, both sizes merged); it reports only. The approval is the authors' decision (user
2026-11-05, which replaced the earlier conditional advance approval before it was ever used): ``approve`` is run only
when the user says so, and the approval is committed before scripts/dota/fcos_val_auto.sh produces any detection.
``cpu``: parallelisation only - every image is fused on its own with the row's frozen config and the pipeline limit
(the per-image path of scripts/dota/fcos_limit_check.py, whose calib replay through this path was exact); the parts
are concatenated in image order and written once as runs/dota/val/dets/<id>.parquet, the file every DOTA-val row has.
Finished images are kept (runs/dota/val/parts/<id>/), so a stop loses only images in flight. Workers from env
SCAHI_VAL_IMG_WORKERS (default 20).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/dota"))

import d_freeze_proposal_fcos as FP  # noqa: E402
import fcos_limit_check as LC  # noqa: E402

from scahi.dota import final as F  # noqa: E402
from scahi.dota import pipeline as P  # noqa: E402

# run order (user 2026-11-06): longest rows first - box fusion 1024, box fusion 2048, then the SCAHI rows (D-H3 first)
FILL_OWNER = F.VAL_DIR / "parts" / "FILL_OWNER"  # rows taken over by scripts/dota/fcos_val_fill.py (user 2026-11-06)
# safeguard before the evaluation (user 2026-11-07): the freeze must be the approved, committed version - 32c81fc (approval)
# with the one documented metadata correction 60d7850 (calib_selection_limit, docs/DECISIONS.md 2026-11-06 08:29)
FREEZE_APPROVED, FREEZE_CORRECTED = "32c81fc", "60d7850"
FREEZE_SHA256 = "7349a658869b16917a3d691a01b16d6c9db3fcea0f371ae01de6bc7f2815e492"  # file at 60d7850
FREEZE_ALLOWED_DIFF = {"calib_selection_limit"}
FREEZE_GIT_PATH = "configs/dota/frozen_rows_fcos.json"  # path of F.FROZEN_FCOS in git
ORDER = ("NMS-o2m_fcos_1024", "NMS-o2m_fcos_2048", "RG_fcos_2048", "F2_fcos_2048", "B-RG_fcos_2048", "B-F2_fcos_2048",
         "RG_fcos_1024", "F2_fcos_1024", "B-RG_fcos_1024", "B-F2_fcos_1024")


def precheck() -> bool:
    """The automatic pre-check of the proposal, re-run on the current freeze file; reports only, approves nothing."""
    fz = json.loads(F.FROZEN_FCOS.read_text())
    bad = FP.precheck(fz["rows"], FP._limit_calib())
    print("[fcos-val] pre-check " + ("all met" if not bad else "NOT met:"), *bad, sep="\n  ")
    return not bad


def approve(note: str) -> None:
    """Record the user's approval (only on the user's explicit instruction; user 2026-11-05: the approval before the
    held-out evaluation is the authors' decision). Rows unchanged."""
    fz = json.loads(F.FROZEN_FCOS.read_text())
    assert not fz.get("approved"), f"already approved {fz['approved_at']}"
    assert note.strip(), "approval note (the user's words) required"
    fz.update(approved=True, approved_at=time.strftime("%Y-%m-%dT%H:%M"), approval=note)
    F.FROZEN_FCOS.write_text(json.dumps(fz, indent=1))
    print(f"[fcos-val] approved {fz['approved_at']}: {len(fz['rows'])} rows unchanged; commit before any DOTA-val run")


def cpu(rid: str) -> None:
    fz = F.frozen(F.FROZEN_FCOS)
    r = next(x for x in fz["rows"] if x["id"] == rid)
    assert r["kind"] == "cache" and fz["pipeline_limit"] == LC.LIMIT, (r["kind"], fz["pipeline_limit"])
    if (F.DETS_DIR / f"{rid}.parquet").exists():
        print(f"[fcos-val] {rid}: detections exist, kept")
        return
    if rid in FILL_OWNER.read_text().split() if FILL_OWNER.exists() else False:
        print(f"[fcos-val] {rid}: run by scripts/dota/fcos_val_fill.py (cross-row filling), skipped here")
        return
    t0 = time.time()
    w = int(os.environ.get("SCAHI_VAL_IMG_WORKERS", "20"))
    d = LC.by_image(r, "val", P.split_stems("val"), False, w, F.VAL_DIR / "parts" / rid)
    F._write_dets(rid, d, {"images_at_limit": P.at_limit(d, LC.LIMIT), "wall_s": time.time() - t0,
                           "parallel": f"by image ({w} workers; parts runs/dota/val/parts/{rid})"})


def verify_freeze() -> None:
    """Abort unless configs/dota/frozen_rows_fcos.json is byte-identical to the committed 60d7850 version (SHA-256 pinned
    here and re-checked in git) and differs from the approved 32c81fc version only in ``calib_selection_limit``."""
    def show(c: str) -> bytes:
        r = subprocess.run(["git", "show", f"{c}:{FREEZE_GIT_PATH}"], cwd=F.REPO_ROOT, capture_output=True)
        if r.returncode:
            raise SystemExit(f"[fcos-val] freeze check: cannot read {FREEZE_GIT_PATH} at {c}: {r.stderr.decode().strip()}")
        return r.stdout

    cur = F.FROZEN_FCOS.read_bytes()
    bad = []
    if hashlib.sha256(cur).hexdigest() != FREEZE_SHA256:
        bad.append(f"SHA-256 {hashlib.sha256(cur).hexdigest()} != pinned {FREEZE_SHA256}")
    if hashlib.sha256(show(FREEZE_CORRECTED)).hexdigest() != FREEZE_SHA256:
        bad.append(f"{FREEZE_CORRECTED} in git does not match the pinned SHA-256")
    a, b = json.loads(show(FREEZE_APPROVED)), json.loads(cur)
    diff = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    if not a.get("approved") or diff - FREEZE_ALLOWED_DIFF:
        bad.append(f"differs from the approved {FREEZE_APPROVED} beyond {sorted(FREEZE_ALLOWED_DIFF)}: "
                   f"{sorted(diff - FREEZE_ALLOWED_DIFF)} (approved there: {a.get('approved')})")
    if bad:
        raise SystemExit("[fcos-val] freeze check FAILED -> no evaluation:\n  " + "\n  ".join(bad))
    print(f"[fcos-val] freeze check passed: SHA-256 = {FREEZE_CORRECTED} ({FREEZE_SHA256[:12]}), differs from the "
          f"approved {FREEZE_APPROVED} only in {sorted(diff) or 'nothing'}", flush=True)


def main() -> None:
    a = sys.argv[1:]
    if a[0] == "precheck":
        sys.exit(0 if precheck() else 1)
    elif a[0] == "approve":
        approve(" ".join(a[1:]))
    elif a[0] == "gpu":
        F.val_gpu(F.FROZEN_FCOS)
    elif a[0] == "cpu":
        cpu(a[1])
    elif a[0] == "eval":
        verify_freeze()
        res = F.val_eval_fcos(int(os.environ.get("SCAHI_VAL_EVAL_WORKERS", "7")))
        for h, x in res["hypotheses"].items():
            print(f"[fcos-val] {h} {x['contrast']}: Δ {x['mAP50']['delta']:+.2f} [{x['mAP50']['ci95'][0]:+.2f}, "
                  f"{x['mAP50']['ci95'][1]:+.2f}], p {x['mAP50']['p_one_sided']:.3f} -> "
                  f"{'met' if x['met'] else 'not met'} ({x['met_of_five_descriptive']} of 5)")
    elif a[0] == "rows":
        ids = {r["id"] for r in json.loads(F.FROZEN_FCOS.read_text())["rows"] if r["kind"] == "cache"}
        assert ids <= set(ORDER), ids - set(ORDER)
        print(" ".join(i for i in ORDER if i in ids))


if __name__ == "__main__":
    main()
