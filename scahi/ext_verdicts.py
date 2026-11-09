# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verdict logic of the extension study (docs/PREREGISTRATION_EXT.md section 3 and its addenda), free of I/O.

Per model and hypothesis a status: MET, NOT_MET or UNCLEAR (rows missing / model not testable). UNCLEAR never counts
as met; "generalises" = at least 4 of the 5 models MET (E-H2 separately per s). While runs are outstanding
(``final=False``) a count that can still reach 4 is reported as OPEN, never as a verdict.

Appendix (addendum 2026-10-26 item 7): one-sided bootstrap p = (1 + #{b: delta*_b <= eps}) / (B + 1), Holm over the
models per hypothesis at alpha = 0.05; E-H3 additionally needs the lower CI bound > 0. Models without the rows enter
Holm with p = 1 (m stays the number of models: the conservative reading of "missing = not met").
Pooled effect: mean of the per-model deltas, CI from the same resamples (per resample the mean of the deltas);
descriptive only.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

MET, NOT_MET, UNCLEAR = "erfüllt", "nicht erfüllt", "unklar"
GENERALISES, NOT_GENERALISES, OPEN = "gilt familienübergreifend", "gilt nicht familienübergreifend", "offen"
N_MODELS = 5
NEED = 4
ALPHA = 0.05
EPS_MIN = 0.30


def epsilon(seed_aps: Sequence[float] | None) -> dict[str, Any]:
    """eps = max(0.30, 2 * sd(ddof 1)) over the val APs of E-2048 YOLO11s, seeds 0, 1, 2; 0.30 without the seeds."""
    if seed_aps is None or len(seed_aps) < 3:
        return {"eps": EPS_MIN, "sigma_seed": None, "rule": "0.30 (YOLO11s extra seeds not run)"}
    sd = float(np.std(np.asarray(seed_aps, float), ddof=1))
    return {"eps": max(EPS_MIN, 2 * sd), "sigma_seed": sd, "rule": "max(0.30, 2 sigma_seed)"}


def status(hyp: str, delta: float | None, ci_lo: float | None, eps: float) -> str:
    """Status of one model for E-H1 / E-H2 / E-H3 (E-H4 has no threshold and is not counted)."""
    if delta is None:
        return UNCLEAR
    ok = delta > eps
    if hyp == "E-H3":
        ok = ok and ci_lo is not None and ci_lo > 0
    return MET if ok else NOT_MET


def count(statuses: Sequence[str], final: bool, n_models: int = N_MODELS) -> dict[str, Any]:
    """'>= 4 of 5' with UNCLEAR counted as not met. ``statuses`` may be shorter than n_models (models not yet run)."""
    if len(statuses) > n_models:
        raise ValueError("more statuses than models")
    n_met = sum(s == MET for s in statuses)
    n_pending = n_models - len(statuses)
    if not final and n_met < NEED <= n_met + n_pending:
        verdict = OPEN
    else:
        verdict = GENERALISES if n_met >= NEED else NOT_GENERALISES
    return {"n_met": n_met, "n_not_met": sum(s == NOT_MET for s in statuses),
            "n_unclear": sum(s == UNCLEAR for s in statuses) + (n_pending if final else 0),
            "n_pending": 0 if final else n_pending, "verdict": verdict}


def p_one_sided(delta_vec: np.ndarray, eps: float) -> float:
    d = np.asarray(delta_vec, float)
    return float((1 + np.sum(d <= eps)) / (len(d) + 1))


def holm(pvals: dict[str, float | None], alpha: float = ALPHA, m: int = N_MODELS) -> dict[str, dict[str, Any]]:
    """Holm step-down over m tests; missing p (None) and models not in ``pvals`` count as p = 1."""
    if len(pvals) > m:
        raise ValueError("more p-values than tests")
    ps = {k: (1.0 if p is None else float(p)) for k, p in pvals.items()}
    order = sorted(ps, key=lambda k: ps[k])
    out, running, stop = {}, 0.0, False
    for i, k in enumerate(order):
        adj = min(1.0, (m - i) * ps[k])
        running = max(running, adj)
        stop = stop or ps[k] > alpha / (m - i)
        out[k] = {"p": ps[k], "p_holm": running, "reject": not stop}
    return out


def pooled(deltas: dict[str, float], vecs: dict[str, np.ndarray]) -> dict[str, Any]:
    """Mean delta over the models present; CI from the per-resample mean (same resamples for all models)."""
    ks = sorted(deltas)
    if not ks:
        return {"n_models": 0, "delta": None, "ci95": None}
    pv = np.mean([np.asarray(vecs[k], float) for k in ks], axis=0)
    return {"n_models": len(ks), "models": ks, "delta": float(np.mean([deltas[k] for k in ks])),
            "ci95": [float(np.percentile(pv, 2.5)), float(np.percentile(pv, 97.5))], "descriptive_only": True}
