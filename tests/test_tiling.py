# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np

from scahi.data.tiling import boxes_in_chip, slice_boxes


def test_boxes_in_chip_clip_and_min_area() -> None:
    boxes = np.array([[10, 10, 50, 50],  # fully inside
                      [600, 100, 700, 140],  # 40 % visible -> kept, clipped
                      [635, 300, 700, 340],  # 5/65 < 10 % visible -> dropped
                      [700, 700, 720, 720]])  # outside
    keep, cb = boxes_in_chip(boxes, (0, 0, 640, 640))
    assert keep.tolist() == [True, True, False, False]
    np.testing.assert_allclose(cb[1], [600, 100, 640, 140])
    keep, cb = boxes_in_chip(boxes, (512, 0, 1152, 640))
    np.testing.assert_allclose(cb[1], [88, 100, 188, 140])  # chip-local coordinates


def test_slice_geometry_covers_image() -> None:
    s = slice_boxes(1920, 1080, 640, 0.2)
    assert min(b[0] for b in s) == 0 and max(b[2] for b in s) == 1920
    assert min(b[1] for b in s) == 0 and max(b[3] for b in s) == 1080
    assert all(b[2] - b[0] == 640 and b[3] - b[1] == 640 for b in s)
