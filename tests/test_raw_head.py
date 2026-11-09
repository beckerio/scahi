# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from scahi.infer.preprocess import full_image_shape, input_to_source, letterbox, to_tensor

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "weights/yolov10s.pt"
VAL_IMAGES = sorted((REPO / "data/canonical/visdrone10/yolo_plain/images/val").glob("*.jpg"))[:3]
TOL = 1e-4
needs_model = pytest.mark.skipif(not (WEIGHTS.exists() and VAL_IMAGES), reason="weights/data missing")


def test_letterbox_mapping_roundtrip() -> None:
    img = np.zeros((1080, 1920, 3), np.uint8)
    out, lb = letterbox(img, full_image_shape(1080, 1920, 640))
    assert out.shape[:2] == (384, 640) and lb.gain == pytest.approx(1 / 3)
    src = np.array([[300.0, 150.0, 330.0, 210.0]])
    pad = np.array([lb.pad_xy * 2]).reshape(1, 4)
    assert lb.pad_xy == (0, 12)  # 360 px image rows centred in 384
    np.testing.assert_allclose(input_to_source(src * lb.gain + pad, lb), src)
    np.testing.assert_allclose(input_to_source(src * lb.gain + pad, lb, (640, 320)), src + [640, 320, 640, 320])


def test_full_image_shape_multiple_of_32() -> None:
    for h, w in [(1080, 1920), (540, 960), (1500, 2000), (788, 1400)]:
        H, W = full_image_shape(h, w, 1280)
        assert H % 32 == 0 and W % 32 == 0 and max(H, W) == 1280


def _batch(device: str) -> torch.Tensor:
    imgs = [letterbox(cv2.imread(str(p)), (384, 640))[0] for p in VAL_IMAGES]
    return to_tensor(imgs, device)


@needs_model
def test_anchors_match_ultralytics() -> None:
    from ultralytics.utils.tal import make_anchors

    from scahi.infer.raw_head import anchors_for

    feats = [torch.zeros(1, 1, 48, 80), torch.zeros(1, 1, 24, 40), torch.zeros(1, 1, 12, 20)]
    a_ref, s_ref = make_anchors(feats, torch.tensor([8.0, 16.0, 32.0]), 0.5)
    a, s, _ = anchors_for([(48, 80), (24, 40), (12, 20)], [8, 16, 32], torch.device("cpu"))
    torch.testing.assert_close(a, a_ref)
    torch.testing.assert_close(s, s_ref.view(-1))


@needs_model
@pytest.mark.parametrize("device", ["cuda" if torch.cuda.is_available() else "cpu"])
def test_parity_o2o_topk(device: str) -> None:
    from scahi.infer.dfl_decode import decode, topk_like_ultralytics
    from scahi.infer.raw_head import forward_raw, load_unfused

    m = load_unfused(WEIGHTS, device, end2end=True)
    raw = forward_raw(m, _batch(device))
    d = decode(raw.boxes["o2o"], raw.scores["o2o"], raw.anchors, raw.strides, raw.reg_max)
    ours = topk_like_ultralytics(d.xyxy, d.probs, m.model[-1].max_det)
    ref = raw.y_ultralytics
    assert ours.shape == ref.shape
    assert (ours[..., 5] == ref[..., 5]).float().mean() > 0.999  # class ids (exact ties could swap order)
    assert (ours[..., 4] - ref[..., 4]).abs().max() < TOL
    assert (ours[..., :4] - ref[..., :4]).abs().max() < TOL  # px (measured 3e-5)


@needs_model
@pytest.mark.parametrize("device", ["cuda" if torch.cuda.is_available() else "cpu"])
def test_parity_o2m_dense(device: str) -> None:
    from scahi.infer.dfl_decode import decode
    from scahi.infer.raw_head import forward_raw, load_unfused

    m = load_unfused(WEIGHTS, device, end2end=False)
    raw = forward_raw(m, _batch(device))
    d = decode(raw.boxes["o2m"], raw.scores["o2m"], raw.anchors, raw.strides, raw.reg_max)
    y = raw.y_ultralytics  # [B, 4 + nc, A], xywh
    xywh = torch.cat([(d.xyxy[..., :2] + d.xyxy[..., 2:]) / 2, d.xyxy[..., 2:] - d.xyxy[..., :2]], -1).transpose(1, 2)
    assert (xywh - y[:, :4]).abs().max() < 2 * TOL  # xywh re-conversion in float32 at ~600 px (measured 1.2e-4)
    assert (d.probs.transpose(1, 2) - y[:, 4:]).abs().max() < TOL
    assert (d.ltrb_std >= 0).all()


@needs_model
def test_cache_whole_image_matches_ultralytics_o2o(tmp_path: Path) -> None:
    """Whole-image pass from the cache: top o2o detections equal the Ultralytics o2o predictor (same input size)."""
    from ultralytics import YOLO

    from scahi.infer.cache import cache_images, weights_id, zoom_key

    p = str(VAL_IMAGES[0])
    root = cache_images(WEIGHTS, [p], [640 / 1920], tmp_path, device="cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(root / zoom_key(640 / 1920) / f"{VAL_IMAGES[0].stem}.npz")
    assert root.name == weights_id(WEIGHTS) and len(z["tiles"]) == 1
    xy, pr = z["o2o_xyxy"], z["o2o_probs"]
    top = np.argsort(-pr.max(1))[:5]
    r = YOLO(str(WEIGHTS)).predict(p, imgsz=640, conf=0.001, max_det=300, nms=False, verbose=False)[0]
    ref = r.boxes.xyxy.cpu().numpy()[:5]
    # same letterbox as the predictor -> same input tensor; residual differences only from float ops
    np.testing.assert_allclose(np.sort(pr.max(1))[::-1][:5], r.boxes.conf.cpu().numpy()[:5], atol=1e-3)
    assert np.abs(xy[top] - ref).max() < 0.1
