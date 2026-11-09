# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build every paper figure from cached data only (files under runs/ and cache/; no GPU, no new runs).

    python scripts/make_figures.py            # all figures
    python scripts/make_figures.py pareto h1  # selected figures
    python scripts/make_figures.py --check    # test: move all figures away, rebuild, compare with the old ones

Style: ``scripts/figstyle.py`` (single source of colours, fonts, sizes, dpi; also written as
``reports/figures/paper/tex/figstyle.tex`` for the TikZ method figure). Every figure is written as PDF + PNG +
``<name>.data.json`` (plotted values and input files), so restyling needs no re-runs.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import figstyle as FS  # noqa: E402
import paper_figures as PF  # noqa: E402

TEX = PF.OUT / "tex"


def tikz_method() -> None:
    """TikZ method overview -> tex/fig_method.pdf (lualatex) + fig_method.png."""
    FS.write_tex(TEX / "figstyle.tex")
    r = subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error", "fig_method.tex"], cwd=TEX,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"lualatex fig_method.tex failed:\n{r.stdout[-1500:]}")
    for ext in ("aux", "log"):
        (TEX / f"fig_method.{ext}").unlink(missing_ok=True)
    subprocess.run(["pdftoppm", "-r", str(FS.DPI_TIKZ_PNG), "-png", "-singlefile", "fig_method.pdf", str(PF.OUT / "fig_method")],
                   cwd=TEX, check=True, capture_output=True)
    print("wrote fig_method")


def build(names: list[str]) -> None:
    PF.style()
    allf = {**PF.FIGS, **PF.EXT_FIGS}
    for n in names or [*PF.FIGS, "method", *PF.EXT_FIGS]:
        tikz_method() if n == "method" else allf[n]()


def outputs() -> list[Path]:
    return sorted([*PF.OUT.glob("fig_*.png"), *PF.OUT.glob("fig_*.pdf"), TEX / "fig_method.pdf"])


def check() -> None:
    """Remove all figures, rebuild them, compare the PNGs pixel-wise with the previous ones (render of the PDFs)."""
    import numpy as np
    from PIL import Image

    old = Path(tempfile.mkdtemp(prefix="figs_old_"))
    for p in outputs():
        shutil.move(str(p), old / p.name)
    build([])
    rows = []
    for p in sorted(old.glob("*.png")):
        new = PF.OUT / p.name
        if not new.exists():
            rows.append((p.name, "MISSING", None))
            continue
        a, b = np.asarray(Image.open(p).convert("RGB"), int), np.asarray(Image.open(new).convert("RGB"), int)
        if a.shape != b.shape:
            rows.append((p.name, f"size {a.shape[:2]} -> {b.shape[:2]}", None))
            continue
        d = np.abs(a - b)
        rows.append((p.name, "identical" if d.max() == 0 else "differs", float((d.max(2) > 16).mean())))
    for r in rows:
        print(f"{r[0]:32s} {r[1]:22s} {'' if r[2] is None else f'{100 * r[2]:.3f} % pixels differ (>16/255)'}")
    print(f"old figures kept in {old}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        build(sys.argv[1:])
