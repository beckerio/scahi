# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tutorial step 1: VisDrone2019-DET train / val in YOLO format plus COCO JSON with the ignore regions.

    python tutorials/yolo26m_visdrone/prepare_data.py [--root data/tutorial]

Downloads the original VisDrone2019-DET train and val zips (the files that Ultralytics' VisDrone.yaml downloads) and
converts them with the repository's converter (``scahi.data.convert``, class setting ``visdrone10``):
``<root>/visdrone/canonical/yolo_plain`` (YOLO labels; score-0 rows, i.e. ignored regions and "others", dropped - the
same labels as Ultralytics' conversion) and ``<root>/visdrone/canonical/coco/{train,val}.json`` (every original row,
ignored regions kept for the evaluation). Unlike VisDrone.yaml, the original annotations are kept: the toolkit-style
evaluation needs the ignored regions. No test-dev (the tutorial evaluates on val only).

Splits: training on ``train_main`` (5 877 images), checkpoint selection on ``calib`` (594 images) - the paper's
calibration split (data/splits/calib.txt): whole video sequences of VisDrone train, the best of 5 000 random sequence
draws (about 9 % of the images) whose object-size and class distribution is closest to all of train
(``scahi.data.splits.make_calib_split_stratified``). VisDrone val (548 images) is the untouched test set.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{}.zip"
SPLITS = ("train", "val")


def main() -> None:
    from ultralytics.utils.downloads import safe_download

    from scahi.config import load_class_setting
    from scahi.data.convert import convert_split
    from scahi.data.splits import read_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "data/tutorial")
    root = ap.parse_args().root.resolve()
    zips, orig, out = root / "visdrone_zips", root / "visdrone/orig", root / "visdrone/canonical"
    cs = load_class_setting("visdrone10")
    stats = {}
    for s in SPLITS:
        z = zips / f"VisDrone2019-DET-{s}.zip"
        if not z.exists():
            safe_download(URL.format(s), file=z)
        d = orig / f"VisDrone2019-DET-{s}"
        if not d.exists():
            with zipfile.ZipFile(z) as f:
                f.extractall(orig)
        stats[s] = convert_split(d, s, cs, out)
    root_y = out / "yolo_plain"
    lists = {"train_main": ("train", read_split(REPO / "data/splits/train_main.txt")),
             "calib": ("train", read_split(REPO / "data/splits/calib.txt")),
             "val": ("val", sorted(p.stem for p in (root_y / "images/val").glob("*.jpg")))}
    for name, (s, stems) in lists.items():
        (root_y / f"{name}.txt").write_text("".join(f"{root_y / 'images' / s / st}.jpg\n" for st in stems))
    # train on train_main, select the checkpoint on calib; VisDrone val is the untouched test set (not in the yaml)
    ds = {"path": str(root_y), "train": "train_main.txt", "val": "calib.txt", "nc": cs.nc,
          "names": dict(enumerate(cs.classes))}
    (root_y / "dataset.yaml").write_text(yaml.safe_dump(ds, sort_keys=False))
    (out / "convert_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({s: {k: v for k, v in st.items() if k in ("images", "annotations", "ignore")} for s, st in stats.items()}))


if __name__ == "__main__":
    main()
