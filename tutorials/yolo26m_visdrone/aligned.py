# SPDX-License-Identifier: AGPL-3.0-or-later
"""The repository's pipeline commands with Ultralytics' input preparation (the convention of the YOLO26m rows).

    python tutorials/yolo26m_visdrone/aligned.py <command> [arguments of python -m scahi.reproduce <command>]

e.g. ``aligned.py cache --weights W --stack native2048 --split calib``, ``aligned.py tune --weights W --stack native2048
--variant F2 --branch o2m``, ``aligned.py fuse ... --split val``. Identical to ``python -m scahi.reproduce`` except that
the caches are built with Ultralytics' size rule, letterbox factor and per-axis box back-mapping (``scahi.mapcheck``)
and stored / read under cache_ualigned/. The frozen rows of the paper's five extension detectors used the original
caches (one scale factor; the difference is below 0.05 AP for every row behind a main-paper number or a verdict, see
reports/BACKMAP_CHECK.md); YOLO26m was tuned and evaluated with this convention from its freeze on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tutorial helpers (also under python -I)
import common as C  # noqa: E402


def main() -> None:
    from scahi import reproduce

    with C.ultralytics_caches():
        sys.argv = ["scahi.reproduce", *sys.argv[1:]]
        reproduce.main()


if __name__ == "__main__":
    main()
