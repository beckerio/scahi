# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import pandas as pd
import pytest

from scahi.config import load_class_setting
from scahi.eval.diagnostics import border_distance, classify_detections
from scahi.eval.evaluate import evaluate
from scahi.eval.visdrone_protocol import (
    comp_oas,
    eval_res_sorted,
    integral_image,
    matlab_round,
    toolkit_coverage,
    toolkit_ignore_map,
    voc_ap,
)

RNG = np.random.default_rng(1)
W, H = 960, 540


def _coco(n_img: int = 3, setting: str = "visdrone10") -> dict:
    """Synthetic canonical COCO with objects, an ignore region and (person_vehicle) a bicycle."""
    cs = load_class_setting(setting)
    imgs, anns, igns = [], [], []
    for i in range(1, n_img + 1):
        imgs.append({"id": i, "file_name": f"{i:04d}.jpg", "width": W, "height": H})
        for _ in range(12):
            x, y = RNG.integers(320, 900), RNG.integers(0, 500)
            w, h = RNG.integers(8, 50), RNG.integers(8, 40)
            anns.append({"id": len(anns) + 1, "image_id": i, "bbox": [float(x), float(y), float(w), float(h)],
                         "category_id": int(RNG.integers(1, cs.nc + 1)), "score": 1, "area": float(w * h), "iscrowd": 0})
        igns.append({"id": len(igns) + 1, "image_id": i, "bbox": [0.0, 0.0, 300.0, 300.0], "orig_category": 0,
                     "score": 0, "ignore_type": "ignore_all", "ignore_classes": list(range(1, cs.nc + 1))})
        if setting == "person_vehicle":
            igns.append({"id": len(igns) + 1, "image_id": i, "bbox": [300.0, 400.0, 100.0, 100.0], "orig_category": 3,
                         "score": 1, "ignore_type": "ignore_vehicle", "ignore_classes": [2]})
    cats = [{"id": k + 1, "name": n} for k, n in enumerate(cs.classes)]
    return {"images": imgs, "annotations": anns, "ignore": igns, "categories": cats}


def _perfect(coco: dict) -> pd.DataFrame:
    return pd.DataFrame([(a["image_id"], *a["bbox"], 0.5 + 0.4 * RNG.random(), a["category_id"])
                         for a in coco["annotations"]], columns=["image_id", "x", "y", "w", "h", "score", "category_id"])


def test_matlab_round() -> None:
    np.testing.assert_array_equal(matlab_round([0.5, 1.5, 2.5, -0.5, 2.4]), [1, 2, 3, -1, 2])


def test_ignore_map_inclusive_and_coverage() -> None:
    m = toolkit_ignore_map([[3, 2, 4, 1]], 10, 10)  # MATLAB rows 2:3, cols 3:7 (1-based, inclusive)
    assert m.sum() == 2 * 5 and m[1:3, 2:7].all()
    I = integral_image(m)
    # box exactly on the region: MATLAB lookup covers rows y+1..y+h, cols x+1..x+w
    cov = toolkit_coverage([[3, 2, 4, 1], [50, 50, 2, 2]], I, 10, 10)
    assert cov[0] == pytest.approx(4 / 4) and cov[1] == 0


def test_comp_oas_and_ties() -> None:
    oa = comp_oas([[0, 0, 10, 10]], [[0, 0, 20, 20], [0, 0, 20, 20]], np.array([0, 1]))
    np.testing.assert_allclose(oa, [[0.25, 1.0]])
    # two identical GT: the toolkit keeps the *last* best match
    gm, dm = eval_res_sorted(np.array([[0.8, 0.8]]), np.array([False, False]), 0.5)
    assert gm.tolist() == [0, 1] and dm.tolist() == [1]
    # no regular match, ignore GT matches -> detection ignored
    gm, dm = eval_res_sorted(np.array([[0.1, 0.9]]), np.array([False, True]), 0.5)
    assert dm.tolist() == [-1] and gm.tolist() == [0, -1]


def test_voc_ap() -> None:
    assert voc_ap(np.array([0.5, 1.0]), np.array([1.0, 0.5])) == pytest.approx(0.75)


@pytest.mark.parametrize("setting,protocol", [("visdrone10", "visdrone_ignore"), ("visdrone10", "plain"),
                                              ("person_vehicle", "visdrone_ignore"), ("person_vehicle", "visdrone_strict")])
def test_perfect_detections_ap100(setting: str, protocol: str) -> None:
    cs = load_class_setting(setting)
    coco = _coco(setting=setting)
    r = evaluate(coco, _perfect(coco), cs, protocol)
    assert r["metrics"]["AP"] == pytest.approx(100.0)
    assert r["metrics"]["AP50"] == pytest.approx(100.0)
    if protocol != "plain":
        assert r["metrics"]["coco"]["AP"] == pytest.approx(100.0)


def test_detections_in_ignore_regions_are_not_fp() -> None:
    cs = load_class_setting("visdrone10")
    coco = _coco()
    base = _perfect(coco)
    base["score"] = 0.5  # below the junk scores: any FP would hurt precision
    junk = pd.DataFrame([(i, float(RNG.integers(0, 250)), float(RNG.integers(0, 250)), 30.0, 30.0, 0.99,
                          int(RNG.integers(1, 11))) for i in (1, 2, 3) for _ in range(20)], columns=base.columns)
    r0 = evaluate(coco, base, cs, "visdrone_ignore")
    r1 = evaluate(coco, pd.concat([base, junk]), cs, "visdrone_ignore")
    assert r1["eval_info"]["n_det_dropped"] == len(junk)
    assert r1["metrics"]["AP"] == pytest.approx(r0["metrics"]["AP"]) == pytest.approx(100.0)
    # without ignore handling the same junk is FP
    assert evaluate(coco, pd.concat([base, junk]), cs, "plain")["metrics"]["AP"] < 100.0


def test_class_specific_ignore_person_vehicle() -> None:
    cs = load_class_setting("person_vehicle")
    coco = _coco(setting="person_vehicle")
    base = _perfect(coco)
    base["score"] = 0.5
    in_bicycle = lambda c: pd.DataFrame([(1, 310.0, 410.0, 60.0, 60.0, 0.99, c)], columns=base.columns)  # noqa: E731
    veh = evaluate(coco, pd.concat([base, in_bicycle(2)]), cs, "visdrone_ignore")
    per = evaluate(coco, pd.concat([base, in_bicycle(1)]), cs, "visdrone_ignore")
    strict = evaluate(coco, pd.concat([base, in_bicycle(2)]), cs, "visdrone_strict")
    assert veh["eval_info"]["n_det_dropped"] == 1 and veh["metrics"]["AP"] == pytest.approx(100.0)
    assert per["eval_info"]["n_det_dropped"] == 0 and per["metrics"]["AP_per_class"]["person"] < 100.0
    assert strict["eval_info"]["n_det_dropped"] == 0 and strict["metrics"]["AP_per_class"]["vehicle"] < 100.0


def test_class_weighting_by_image_count() -> None:
    """Toolkit quirk: class mean weighted by number of images containing the class."""
    cs = load_class_setting("person_vehicle")
    imgs = [{"id": i, "file_name": f"{i}.jpg", "width": W, "height": H} for i in (1, 2, 3)]
    anns = [{"id": 1, "image_id": 1, "bbox": [500, 300, 20, 20], "category_id": 1, "score": 1, "area": 400, "iscrowd": 0}]
    anns += [{"id": 1 + i, "image_id": i, "bbox": [600, 300, 20, 20], "category_id": 2, "score": 1, "area": 400, "iscrowd": 0}
             for i in (1, 2, 3)]
    coco = {"images": imgs, "annotations": anns, "ignore": [], "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "vehicle"}]}
    dets = pd.DataFrame([(1, 500.0, 300.0, 20.0, 20.0, 0.9, 1)] + [(i, 900.0, 10.0, 20.0, 20.0, 0.9, 2) for i in (1, 2, 3)],
                        columns=["image_id", "x", "y", "w", "h", "score", "category_id"])
    r = evaluate(coco, dets, cs, "visdrone_ignore")
    # person AP 100 (1 image), vehicle AP 0 (3 images) -> toolkit 25, pycocotools 50
    assert r["metrics"]["AP"] == pytest.approx(25.0)
    assert r["metrics"]["coco"]["AP"] == pytest.approx(50.0)


def test_max_dets_file_order_across_classes() -> None:
    cs = load_class_setting("person_vehicle")
    imgs = [{"id": 1, "file_name": "1.jpg", "width": W, "height": H}]
    anns = [{"id": 1, "image_id": 1, "bbox": [500, 300, 20, 20], "category_id": 1, "score": 1, "area": 400, "iscrowd": 0}]
    coco = {"images": imgs, "annotations": anns, "ignore": [], "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "vehicle"}]}
    dets = pd.DataFrame([(1, 0.0, 0.0, 9.0, 9.0, 0.9, 2), (1, 500.0, 300.0, 20.0, 20.0, 0.8, 1)],
                        columns=["image_id", "x", "y", "w", "h", "score", "category_id"])
    r = evaluate(coco, dets, cs, "visdrone_ignore")
    assert r["metrics"]["AR1"] == pytest.approx(0.0)  # the top-1 detection is the vehicle FP
    assert r["metrics"]["AR10"] == pytest.approx(100.0)


def test_diagnostics_dup_frag() -> None:
    gt = np.array([[0, 0, 100, 100]], float)
    dt = np.array([[0, 0, 100, 100], [2, 2, 100, 100], [10, 10, 40, 40], [300, 300, 320, 320]], float)
    lab = classify_detections(gt, np.array([1]), dt, np.array([0.9, 0.8, 0.7, 0.6]), np.ones(4, int))
    assert lab.tolist() == [1, 2, 3, 0]


def test_border_distance() -> None:
    tiles = np.array([[0, 0, 600, 540], [400, 0, 960, 540]], float)
    d = border_distance(np.array([[410, 100], [10, 10], [700, 270]]), tiles, W, H)
    np.testing.assert_allclose(d, [10, 390, 100])


def test_cap_500_per_image_across_classes() -> None:
    from scahi.eval.detections import finalize_detections

    rows = [(1, float(i), 0.0, 5.0, 5.0, float(i) / 1000, 1 + i % 2) for i in range(600)] + [(2, 0.0, 0.0, 5.0, 5.0, 0.5, 1)]
    df = pd.DataFrame(rows, columns=["image_id", "x", "y", "w", "h", "score", "category_id"])
    out = finalize_detections(df)
    assert (out.image_id == 1).sum() == 500 and (out.image_id == 2).sum() == 1
    assert out[out.image_id == 1].score.min() == pytest.approx(0.1)  # the 500 best are kept
    assert out[out.image_id == 1].score.is_monotonic_decreasing
    cs = load_class_setting("person_vehicle")
    coco = _coco(setting="person_vehicle")
    df.loc[df.image_id == 2, "image_id"] = 3
    assert evaluate(coco, df, cs, "visdrone_ignore")["eval_info"]["n_det_capped"] == 100
