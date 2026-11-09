# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sensitivity note on the 2x time rule of the refined-grid selection (user, 2026-11-02; reporting only).

For each refined-grid stage: the time ratio of the selected config, and every variant with a higher calib mAP50 than
the selected one whose time ratio lies in [1.8, 2.2] (i.e. near the cap, where the timing noise of about +-10 % could
decide admissibility). The selection rule (scahi.dota.pipeline.select, capped) is not changed.
"""

from __future__ import annotations

KEYS = ("tau_splat", "tau_peak", "cell_factor", "bins_per_oct")
BAND = (1.8, 2.2)


def sensitivity(rows: list[dict], band: tuple[float, float] = BAND) -> dict:
    from scahi.dota import pipeline as P

    sel = P.select(rows, capped=True)
    near = [r for r in rows if r["mAP50"] > sel["mAP50"] and band[0] <= r["time_ratio_vs_ref"] <= band[1]]
    f = lambda r: {**{k: r[k] for k in KEYS}, "mAP50": r["mAP50"], "time_ratio_vs_ref": r["time_ratio_vs_ref"],  # noqa: E731
                   "admissible": r["time_ratio_vs_ref"] <= P.TIME_CAP}
    return {"band": list(band), "time_cap": P.TIME_CAP, "selected": f(sel),
            "higher_mAP50_in_band": sorted((f(r) for r in near), key=lambda x: -x["mAP50"])}


def _cfg(x: dict) -> str:
    return f"τ_splat {x['tau_splat']:g}, τ_peak {x['tau_peak']:g}, Zelle {x['cell_factor']:g}, {x['bins_per_oct']} Bins"


def lines_de(stages: dict[str, dict | None]) -> list[str]:
    """Markdown lines (German, as the freeze proposals) for {stage label: sensitivity(...) or None (missing)}."""
    L = ["## Sensitivität der 2×-Zeitregel (nur Bericht; Auswahlregel unverändert)", "",
         f"Je Stufe des verfeinerten Rasters: Zeitverhältnis der gewählten Einstellung und jede Variante mit höherem "
         f"calib-mAP50 als die gewählte, deren Zeitverhältnis in [{BAND[0]}, {BAND[1]}] liegt (Zeitrauschen am Deckel "
         "≈ ±10 %, Kreuzprüfung YOLOv10s 2026-11-02).", ""]
    for label, s in stages.items():
        if s is None:
            L.append(f"- **{label}:** Stufe fehlt noch.")
            continue
        sel = s["selected"]
        L.append(f"- **{label}:** gewählt {_cfg(sel)}, mAP50 {sel['mAP50']:.3f}, Zeitverhältnis "
                 f"{sel['time_ratio_vs_ref']:.3f}; Varianten mit höherem mAP50 im Band: "
                 + ("keine." if not s["higher_mAP50_in_band"] else str(len(s["higher_mAP50_in_band"])) + ":"))
        for x in s["higher_mAP50_in_band"]:
            L.append(f"  - {_cfg(x)}: mAP50 {x['mAP50']:.3f}, Zeitverhältnis {x['time_ratio_vs_ref']:.3f} "
                     f"({'zulässig' if x['admissible'] else 'nicht zulässig'})")
    return L + [""]
