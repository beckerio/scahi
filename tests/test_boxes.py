# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import pytest

from scahi.data.boxes import (
    box_sigma,
    iof_matrix,
    iou_matrix,
    xywh_to_xyxy,
    xyxy_to_xywh,
    xyxy_to_yolo,
    yolo_to_xyxy,
)

RNG = np.random.default_rng(0)


def _random_xyxy(n: int, w: float = 1920, h: float = 1080) -> np.ndarray:
    x1 = RNG.uniform(0, w - 50, n)
    y1 = RNG.uniform(0, h - 50, n)
    return np.stack([x1, y1, x1 + RNG.uniform(1, 50, n), y1 + RNG.uniform(1, 50, n)], 1)


def test_xywh_roundtrip() -> None:
    b = _random_xyxy(100)
    np.testing.assert_allclose(xywh_to_xyxy(xyxy_to_xywh(b)), b)


def test_xywh_known_value() -> None:
    np.testing.assert_allclose(xywh_to_xyxy([10, 20, 30, 40]), [[10, 20, 40, 60]])


@pytest.mark.parametrize("wh", [(1920, 1080), (960, 540), (2000, 1500)])
def test_yolo_roundtrip(wh: tuple[int, int]) -> None:
    b = _random_xyxy(100, *wh)
    np.testing.assert_allclose(yolo_to_xyxy(xyxy_to_yolo(b, *wh), *wh), b, atol=1e-9)


def test_yolo_known_value() -> None:
    np.testing.assert_allclose(yolo_to_xyxy([0.5, 0.5, 0.1, 0.2], 1000, 500), [[450, 200, 550, 300]])


def test_sigma() -> None:
    np.testing.assert_allclose(box_sigma([[0, 0, 4, 9]]), [6.0])


def test_iou_iof() -> None:
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30], [2, 2, 4, 4]], float)
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0, 4 / 100]])
    # intersection over the area of the first argument
    np.testing.assert_allclose(iof_matrix(b, a)[:, 0], [1.0, 0.5, 0.0, 1.0])
