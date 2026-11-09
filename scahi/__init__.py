# SPDX-License-Identifier: AGPL-3.0-or-later
"""ScaleSpace SAHI."""

import os
from pathlib import Path

# Keep Ultralytics settings repo-local even when the conda env vars are not loaded
# (e.g. calling the env's python directly). See docs/DECISIONS.md, 2026-10-21.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))
