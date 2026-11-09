# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path

import numpy as np
import pytest
import yaml

from scahi.config import load_class_setting
from scahi.data.boxes import yolo_to_xyxy
from scahi.data.convert import coco_to_visdrone_rows, convert_image, yolo_lines
from scahi.data.splits import (
    make_calib_split,
    make_calib_split_by_sequence,
    make_calib_split_stratified,
    sequence_id,
)
from scahi.data.visdrone import parse_visdrone_text

REPO = Path(__file__).resolve().parents[1]
ROWS = np.array(
    [
        [10, 20, 30, 40, 1, 1, 0, 0],  # pedestrian
        [100, 50, 20, 10, 1, 4, 1, 2],  # car
        [0, 0, 300, 200, 0, 0, 0, 0],  # ignored region
        [500, 400, 15, 15, 0, 11, 0, 0],  # others
        [600, 100, 12, 30, 1, 3, 0, 1],  # bicycle
        [900, 500, 70, 50, 1, 10, 0, 0],  # motor, partly outside a 960x540 image
    ]
)


def _sorted(a: np.ndarray) -> np.ndarray:
    return a[np.lexsort(a.T[::-1])]


@pytest.mark.parametrize("setting", ["visdrone10", "person_vehicle"])
def test_coco_roundtrip_synthetic(setting: str) -> None:
    cs = load_class_setting(setting)
    anns, igns, unused = convert_image(1, "x", ROWS, 960, 540, cs, 1, 1)
    coco = {"annotations": anns, "ignore": igns, "unused": unused}
    np.testing.assert_array_equal(_sorted(coco_to_visdrone_rows(coco, 1)), _sorted(ROWS))


def test_ignore_channel_masks() -> None:
    cs = load_class_setting("person_vehicle")
    anns, igns, _ = convert_image(1, "x", ROWS, 960, 540, cs, 1, 1)
    by_cat = {g["orig_category"]: g for g in igns}
    assert by_cat[0]["ignore_classes"] == [1, 2] and by_cat[11]["ignore_classes"] == [1, 2]
    assert by_cat[3]["ignore_classes"] == [2] and by_cat[10]["ignore_classes"] == [2]  # vehicle only
    assert by_cat[3]["ignore_type"] == "ignore_vehicle"
    assert sorted(a["category_id"] for a in anns) == [1, 2]
    cs10 = load_class_setting("visdrone10")
    anns10, igns10, _ = convert_image(1, "x", ROWS, 960, 540, cs10, 1, 1)
    assert len(anns10) == 4 and {g["orig_category"] for g in igns10} == {0, 11}
    assert all(g["ignore_classes"] == list(range(1, 11)) for g in igns10)


def test_yolo_ids_and_clipping() -> None:
    cs = load_class_setting("person_vehicle")
    img = {"width": 960, "height": 540}
    anns, igns, _ = convert_image(1, "x", ROWS, 960, 540, cs, 1, 1)
    lines, cnt = yolo_lines(img, anns, igns, cs, with_ignore=True)
    assert cnt == {"degenerate": 0, "duplicate": 0}
    ids = sorted(int(line.split()[0]) for line in lines)
    assert ids == [0, 1, 2, 2, 3, 3]  # person, vehicle, ignore_all x2, ignore_vehicle x2
    arr = np.array([[float(v) for v in line.split()[1:]] for line in lines])
    xyxy = yolo_to_xyxy(arr, 960, 540)
    assert xyxy.min() >= -1e-6 and xyxy[:, 2].max() <= 960 + 1e-3 and xyxy[:, 3].max() <= 540 + 1e-3
    motor = xyxy[np.argmax(xyxy[:, 2])]
    np.testing.assert_allclose(motor, [900, 500, 960, 540], atol=1e-3)  # clipped
    plain, _ = yolo_lines(img, anns, igns, cs, with_ignore=False)
    assert sorted(int(line.split()[0]) for line in plain) == [0, 1]


def test_training_cleanup_duplicates_and_degenerate() -> None:
    cs = load_class_setting("visdrone10")
    rows = np.array([[10, 10, 20, 20, 1, 4, 0, 0], [10, 10, 20, 20, 1, 4, 0, 0],  # exact duplicate car
                     [10, 10, 20, 20, 1, 1, 0, 0],  # same box, other class: kept
                     [50, 50, 4, 0, 1, 4, 0, 0]])  # zero height
    anns, igns, _ = convert_image(1, "x", rows, 960, 540, cs, 1, 1)
    assert len(anns) == 4  # evaluation GT untouched
    lines, cnt = yolo_lines({"width": 960, "height": 540}, anns, igns, cs, with_ignore=True)
    assert cnt == {"degenerate": 1, "duplicate": 1} and len(lines) == 2


def test_parse_trailing_comma() -> None:
    np.testing.assert_array_equal(parse_visdrone_text("1,2,3,4,1,4,0,0,\n\n"), [[1, 2, 3, 4, 1, 4, 0, 0]])


def test_calib_split_deterministic() -> None:
    stems = [f"{s:07d}_{i:05d}_d_{i:07d}" for s in range(30) for i in range(s + 1)]
    a, b = make_calib_split(stems), make_calib_split(stems)
    assert a == b and len(a) == round(0.1 * len(stems))
    seq = make_calib_split_by_sequence(stems)
    rest = set(stems) - set(seq)
    assert not ({sequence_id(s) for s in seq} & {sequence_id(s) for s in rest})
    rng = np.random.default_rng(0)
    hist = {s: rng.integers(0, 5, 16).astype(float) for s in stems}
    strat, info = make_calib_split_stratified(stems, hist, n_candidates=200)
    assert strat == make_calib_split_stratified(stems, hist, n_candidates=200)[0]
    assert 0.09 * len(stems) <= len(strat) <= 0.11 * len(stems)
    rest = set(stems) - set(strat)
    assert not ({sequence_id(s) for s in strat} & {sequence_id(s) for s in rest})


CANON = REPO / "data/canonical"


@pytest.mark.skipif(not (CANON / "visdrone10/coco/val.json").exists(), reason="canonical data not converted")
@pytest.mark.parametrize("setting", ["visdrone10", "person_vehicle"])
def test_roundtrip_real_val(setting: str) -> None:
    import json

    coco = json.loads((CANON / setting / "coco/val.json").read_text())
    src = Path(coco["info"]["source"]) / "annotations"
    for img in coco["images"][::25]:
        orig = parse_visdrone_text((src / img["file_name"].replace(".jpg", ".txt")).read_text())
        np.testing.assert_array_equal(_sorted(coco_to_visdrone_rows(coco, img["id"])), _sorted(orig))


@pytest.mark.skipif(not (CANON / "person_vehicle/yolo_ignore/dataset.yaml").exists(), reason="not converted")
def test_dataset_yaml() -> None:
    ds = yaml.safe_load((CANON / "person_vehicle/yolo_ignore/dataset.yaml").read_text())
    assert ds["nc"] == 4 and ds["scahi_nc_model"] == 2
    assert ds["names"] == {0: "person", 1: "vehicle", 2: "ignore_all", 3: "ignore_vehicle"}
