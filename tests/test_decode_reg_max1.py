# SPDX-License-Identifier: AGPL-3.0-or-later
"""Direct box decoding for reg_max = 1 heads (YOLO26; addendum 2026-11-13 of docs/PREREGISTRATION_EXT.md).

Exact equality with Ultralytics' own outputs of the same forward pass, both branches (o2m: xywh + probabilities with
end2end off; o2o: top-k with end2end on), on a YOLO26 checkpoint from weights/ (skipped if none is present). The DFL path
(reg_max > 1) is unchanged; its equality with the previous implementation was checked on real head outputs of every
detector (runs/ext/decode_change_identity_20261113.json).
"""

from pathlib import Path

import pytest
import torch

from scahi.infer import dfl_decode as DD

REPO = Path(__file__).resolve().parents[1]
W26 = next((p for p in (REPO / "weights/yolo26n.pt", REPO / "weights/yolo26m.pt") if p.exists()), None)


@pytest.mark.skipif(W26 is None, reason="no YOLO26 weights in weights/")
@pytest.mark.parametrize("end2end", [False, True])
def test_reg_max1_matches_ultralytics(end2end: bool) -> None:
    from scahi.infer.raw_head import forward_raw, load_unfused

    torch.manual_seed(0)
    m = load_unfused(W26, device="cpu", end2end=end2end)
    assert m.model[-1].reg_max == 1
    raw = forward_raw(m, torch.rand(2, 3, 640, 512))
    br = "o2o" if end2end else "o2m"
    d = DD.decode(raw.boxes[br], raw.scores[br], raw.anchors, raw.strides, raw.reg_max)
    y = raw.y_ultralytics
    if end2end:
        ref = DD.topk_like_ultralytics(d.xyxy, d.probs, y.shape[1])
    else:
        x = d.xyxy
        ref = torch.cat([(x[..., :2] + x[..., 2:]) / 2, x[..., 2:] - x[..., :2], d.probs], -1).transpose(1, 2)
    assert torch.equal(ref, y)
    assert torch.count_nonzero(d.ltrb_std) == 0
