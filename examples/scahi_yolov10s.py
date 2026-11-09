# SPDX-License-Identifier: AGPL-3.0-or-later
"""Minimal SCAHI example: one image, the public COCO YOLOv10s weights, a whole-image pass plus tiles, fused with SCAHI.

    python examples/scahi_yolov10s.py IMAGE [--setting prespecified|refined] [--fusion-device cpu|gpu]
                                            [--box-fusion] [--out detections.csv]

The weights are loaded by model name through Ultralytics (``yolov10s.pt``, 80 COCO classes; Ultralytics fetches them on
first use). Passes as in the paper's VisDrone stack: one whole-image pass at long side 2048 and tiles at absolute zoom
z = 2 with network input 1024 (tile side 512 px, overlap 0.3). The dense outputs of the one-to-many head of every pass
are fused by SCAHI (variant F2) on the CPU (reference) or on the GPU. ``--box-fusion`` also runs box fusion on the same
passes for comparison (one-to-one head, class-aware NMS with intersection over the smaller box 0.8, conf 0.01, as the
frozen box-fusion row of the paper). The SCAHI parameters are the paper's VisDrone YOLOv10s settings; they were tuned
for VisDrone-trained weights, not for the COCO weights used here.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import torch  # noqa: E402

from scahi.fusion import scalespace as ss  # noqa: E402
from scahi.fusion.run import clip_xywh, f0_image  # noqa: E402
from scahi.infer.cache_mz import raw_predict_level  # noqa: E402
from scahi.infer.multizoom import Level  # noqa: E402
from scahi.infer.raw_head import load_unfused  # noqa: E402

# SCAHI settings of the paper (VisDrone, YOLOv10s, 2048 + tiles): pre-specified (F2) and the refined grid (calib)
SETTINGS = {"prespecified": dict(tau_splat=0.005, tau_peak=0.001, cell_factor=5.0, bins_per_oct=5),
            "refined": dict(tau_splat=0.00125, tau_peak=0.001, cell_factor=11.0, bins_per_oct=7)}
BOX_FUSION = {"metric": "IOS", "thr": 0.8, "conf": 0.01}
OVERLAP = 0.3


def passes(model, img, device: str) -> dict[str, list[ss.CachedLevel]]:
    """Dense outputs of the whole-image pass (long side 2048) and the z = 2 tiles (input 1024), per head."""
    out = {"o2m": [], "o2o": []}
    for lv in (Level(2048 / 1920, None, 2048), Level(2.0, 1024)):
        z = raw_predict_level(model, img, lv, OVERLAP, device=device)
        for br in out:
            out[br].append(ss.CachedLevel(f"{lv.zoom}/{lv.n_input}", z[f"{br}_xyxy"], z[f"{br}_probs"], z[f"{br}_std"],
                                          z[f"{br}_tile"], z["tiles"], z["gains"], lv.n_input is None, OVERLAP))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("image")
    ap.add_argument("--weights", default="yolov10s.pt")
    ap.add_argument("--setting", choices=tuple(SETTINGS), default="prespecified")
    ap.add_argument("--fusion-device", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--box-fusion", action="store_true", help="also box fusion on the same passes")
    ap.add_argument("--out", help="CSV with the SCAHI detections (x, y, w, h, score, class)")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_unfused(a.weights, device)
    names = model.names
    img = cv2.imread(a.image)
    H, W = img.shape[:2]
    t0 = time.perf_counter()
    lv = passes(model, img, device)
    t_pass = time.perf_counter() - t0
    p = ss.FusionParams(**SETTINGS[a.setting])
    t0 = time.perf_counter()
    if a.fusion_device == "gpu" and torch.cuda.is_available():
        from scahi.fusion.scalespace_torch import fuse_image_f2

        d = fuse_image_f2(lv["o2m"], W, H, p, nc=len(names))
    else:
        d = ss.fuse_image(lv["o2m"], W, H, "F2", p, nc=len(names))
    d = clip_xywh(d, W, H)
    t_fuse = time.perf_counter() - t0
    n_pass = sum(len(x.tiles) for x in lv["o2m"])
    print(f"{Path(a.image).name}: {W}x{H}, {n_pass} passes ({t_pass:.2f} s on {device}); SCAHI ({a.setting}, "
          f"{a.fusion_device if a.fusion_device == 'cpu' or torch.cuda.is_available() else 'cpu'}): {len(d)} detections, "
          f"{len(d[d.score >= 0.25])} with score >= 0.25 ({t_fuse:.2f} s)")
    for r in d.head(5).itertuples():
        print(f"  {names[r.category_id - 1]:>12s} {r.score:.3f}  x {r.x:.0f} y {r.y:.0f} w {r.w:.0f} h {r.h:.0f}")
    if a.box_fusion:
        t0 = time.perf_counter()
        b = clip_xywh(f0_image(lv["o2o"], W, H, "o2o", "F0-NMS", BOX_FUSION), W, H)
        print(f"box fusion (NMS, IOS 0.8, conf 0.01, same passes): {len(b)} detections, "
              f"{len(b[b.score >= 0.25])} with score >= 0.25 ({time.perf_counter() - t0:.2f} s)")
    if a.out:
        d.assign(cls=[names[c - 1] for c in d.category_id]).to_csv(a.out, index=False)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
