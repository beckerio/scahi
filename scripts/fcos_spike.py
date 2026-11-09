# SPDX-License-Identifier: AGPL-3.0-or-later
"""FCOS feasibility spike steps (host-a brief 2026-11-01, task 6) -> runs/fcos/<step>.json; report reports/FCOS_SPIKE.md.

    python scripts/fcos_spike.py parity      # decoding parity vs. torchvision's own post-processing (COCO model)
    python scripts/fcos_spike.py export      # SCAHI cache export of a few DOTA calib images + SCAHI on it (smoke)
    python scripts/fcos_spike.py memory      # training memory smoke at 1024 / 2048 on the densest DOTA train_main images
    python scripts/fcos_spike.py train       # 2-epoch training smoke on a train_main subset, selection on calib
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scahi import fcos as FC  # noqa: E402

OUT = REPO / "runs/fcos"


def parity() -> dict:
    """COCO model (91 classes), torchvision defaults except the limits; a few DOTA calib and VisDrone val images."""
    import cv2

    from scahi.dota import pipeline as P

    imgs = list(P.images("calib").path[:4])
    vd = sorted((Path(__import__("scahi.machine", fromlist=["x"]).load_data_config(REPO / "configs/data/visdrone.yaml")
                     ["original_root"]) / "VisDrone2019-DET-val/images").glob("*.jpg"))[:3]
    res = []
    for lim_name, lim in (("torchvision defaults", FC.Limits(0.2, 1000, 100)), ("raised limits", FC.Limits())):
        m = FC.load(None, 1024, lim)
        m.eval()
        for p in imgs + [str(x) for x in vd]:
            img = cv2.imread(str(p))
            H, W = img.shape[:2]
            L = 1024
            FC.set_long_side(m, L)
            with torch.no_grad():
                ref = m([FC._to_tensor(img, "cuda")])[0]
            d = FC.dense_pass(m, img, L)
            own = FC.postprocess_like_torchvision(d, m, (H, W))
            same_n = len(ref["boxes"]) == len(own["boxes"])
            e = {"limits": lim_name, "image": Path(p).name, "n_ref": len(ref["boxes"]), "n_own": len(own["boxes"]),
                 "identical": bool(same_n and torch.equal(ref["labels"], own["labels"])
                                   and torch.equal(ref["scores"], own["scores"]) and torch.equal(ref["boxes"], own["boxes"])),
                 "max_box_diff": float((ref["boxes"] - own["boxes"]).abs().max()) if same_n and len(ref["boxes"]) else 0.0,
                 "dense_positions": int(d["probs"].shape[0]), "n_classes": int(d["probs"].shape[1])}
            res.append(e)
            print(e, flush=True)
        del m
        torch.cuda.empty_cache()
    return {"rows": res, "passed": all(x["identical"] for x in res)}


# ------------------------------------------------------------------------------------------------ data (DOTA HBB)


def _labels(stem: str, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Training labels as used for the YOLO trainings (data/canonical/dota/yolo_hbb, difficult GT = positives)."""
    from scahi.dota import train as T

    f = T.YOLO_ROOT / "labels/train" / f"{stem}.txt"
    a = np.loadtxt(f, ndmin=2) if f.exists() and f.stat().st_size else np.zeros((0, 5))
    cx, cy, w, h = a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
    b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    ok = (b[:, 2] > b[:, 0] + 1e-3) & (b[:, 3] > b[:, 1] + 1e-3)
    return b[ok].astype(np.float32), a[ok, 0].astype(np.int64)


def _sample(stem: str, device: str = "cuda", flip: bool = False):
    import cv2

    from scahi.dota import data as D

    img = cv2.imread(str(D.image_path("train", stem)))
    H, W = img.shape[:2]
    b, c = _labels(stem, W, H)
    if flip:
        img = img[:, ::-1]
        b = np.stack([W - b[:, 2], b[:, 1], W - b[:, 0], b[:, 3]], 1)
    return FC._to_tensor(img, device), {"boxes": torch.from_numpy(b).to(device), "labels": torch.from_numpy(c).to(device)}


def _gt_count(stems: list[str]) -> dict[str, int]:
    from scahi.dota import train as T

    out = {}
    for s_ in stems:
        f = T.YOLO_ROOT / "labels/train" / f"{s_}.txt"
        out[s_] = sum(1 for _ in f.open()) if f.exists() else 0
    return out


def _step(m, opt, scaler, batch) -> dict[str, float]:
    imgs, tgts = zip(*batch)
    with torch.autocast("cuda", dtype=torch.float16):
        losses = m(list(imgs), list(tgts))
    loss = sum(losses.values())
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return {k: float(v) for k, v in losses.items()}


def memory() -> dict:
    """Forward + backward (AMP, SGD) on the densest train_main images, per long side and batch size."""
    from scahi.dota import data as D
    from scahi.dota import train as T

    stems = T.split_stems("train_main")
    cnt = _gt_count(stems)
    dense = sorted(stems, key=lambda s_: -cnt[s_])[:8]
    res = []
    for L in (1024, 2048):
        for bs in ((8, 4, 2, 1) if L == 1024 else (4, 3, 2, 1)):
            m = FC.load(len(D.CLASSES), L)
            m.train()
            opt = torch.optim.SGD(m.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            torch.cuda.reset_peak_memory_stats()
            e = {"long_side": L, "batch": bs, "images": dense[:max(bs, 4)], "gt_per_image": [cnt[x] for x in dense[:max(bs, 4)]]}
            try:
                t0 = time.time()
                for i in range(0, max(bs, 4), bs):  # the 4 (or bs) densest images in batches of bs
                    _step(m, opt, scaler, [_sample(x) for x in dense[i:i + bs]])
                torch.cuda.synchronize()
                e.update(fits=True, peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                         reserved_gib=torch.cuda.max_memory_reserved() / 2**30, s_per_step=(time.time() - t0) / max(1, max(bs, 4) // bs))
            except torch.OutOfMemoryError as ex:
                e.update(fits=False, error=str(ex).splitlines()[0][:200])
            res.append(e)
            print({k: v for k, v in e.items() if k != "images"}, flush=True)
            del m, opt
            torch.cuda.empty_cache()
    return {"densest_train_main": {x: cnt[x] for x in dense}, "rows": res}


# ------------------------------------------------------------------------------------------------ 2-epoch smoke

def _predict_calib(m, L: int, stems: list[str]):
    import cv2
    import pandas as pd

    from scahi.dota import data as D

    FC.set_long_side(m, L)
    m.eval()
    rows = []
    with torch.no_grad():
        for s_ in stems:
            img = cv2.imread(str(D.image_path("train", s_)))
            o = m([FC._to_tensor(img, "cuda")])[0]
            b = o["boxes"].cpu().numpy()
            rows.append(pd.DataFrame({"image": s_, "cls": o["labels"].cpu().numpy().astype(int), "score": o["scores"].cpu().numpy(),
                                      "x1": b[:, 0], "y1": b[:, 1], "x2": b[:, 2], "y2": b[:, 3]}))
    return pd.concat(rows, ignore_index=True)


def train(L: int = 1024, n_img: int = 64, epochs: int = 2, bs: int = 4, lr: float = 0.01) -> dict:
    """2 epochs on the first n_img train_main images (horizontal flip), AMP, SGD (lr scaled to the batch, linear warmup
    over epoch 1); after each epoch devkit mAP50 on the whole calib split (no cap), best epoch kept."""
    import random

    from scahi.dota import data as D
    from scahi.dota import evaluate as EV
    from scahi.dota import train as T

    torch.manual_seed(0)
    random.seed(0)
    stems = T.split_stems("train_main")[:n_img]
    calib = T.split_stems("calib")
    gt = EV.gt_arrays("train", calib)
    m = FC.load(len(D.CLASSES), L)
    params = [p_ for p_ in m.parameters() if p_.requires_grad]
    lr_b = lr * bs / 16
    opt = torch.optim.SGD(params, lr=lr_b, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    it_per_ep = (len(stems) + bs - 1) // bs
    curve, best, it = [], None, 0
    torch.cuda.reset_peak_memory_stats()
    t_train = 0.0
    for ep in range(1, epochs + 1):
        m.train()
        FC.set_long_side(m, L)
        order = stems[:]
        random.shuffle(order)
        t0 = time.time()
        losses = []
        for i in range(0, len(order), bs):
            for g in opt.param_groups:
                g["lr"] = lr_b * min(1.0, (it + 1) / it_per_ep)  # linear warmup over the first epoch
            losses.append(_step(m, opt, scaler, [_sample(x, flip=random.random() < 0.5) for x in order[i:i + bs]]))
            it += 1
        torch.cuda.synchronize()
        dt_ep = time.time() - t0
        t_train += dt_ep
        t1 = time.time()
        dets = _predict_calib(m, L, calib)
        v = EV.map50(EV.build(gt, dets, None, sizes=False))
        e = {"epoch": ep, "calib_mAP50": v, "train_s": dt_ep, "s_per_img_train": dt_ep / len(order),
             "calib_eval_s": time.time() - t1, "loss": {k: float(np.mean([x[k] for x in losses])) for k in losses[0]},
             "max_dets_per_image": int(dets.groupby("image").size().max()) if len(dets) else 0}
        curve.append(e)
        print(e, flush=True)
        if best is None or v > best["calib_mAP50"]:
            best = e
            torch.save(m.state_dict(), OUT / f"smoke_{L}_best.pt")
    return {"long_side": L, "images": len(stems), "epochs": epochs, "batch": bs, "lr": lr_b, "curve": curve,
            "best_epoch": best["epoch"], "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "note": "smoke only: 2 epochs on 64 images - the calib value says nothing about the final accuracy"}


def export(n: int = 12) -> dict:
    """SCAHI cache export (stack dota1024: whole image at 1024 + native 1024 tiles, overlap 0.3) of n calib images
    with the 1024 smoke weights, then box fusion (o2m-style NMS on the passes) and SCAHI (pre-specified F2 setting of the
    DOTA preregistration: first grid point) on the same passes; devkit mAP50 on these n calib images (pipeline smoke)."""
    import cv2

    from scahi.dota import data as D
    from scahi.dota import evaluate as EV
    from scahi.dota import pipeline as P
    from scahi.dota import train as T
    from scahi.fusion import run as R
    from scahi.fusion import scalespace as ss
    from scahi.fusion.scalespace_torch import fuse_image_f2
    from scahi.infer import cache_mz as cm

    m = FC.load(len(D.CLASSES), 1024, state_dict=torch.load(OUT / "smoke_1024_best.pt", map_location="cuda"))
    stems = T.split_stems("calib")[:n]
    gt = EV.gt_arrays("train", stems)
    prm = ss.FusionParams(tau_splat=0.005, tau_peak=0.001, cell_factor=5.0, bins_per_oct=4, top_k=200_000)
    rows, info = {"single": [], "scahi_cpu": [], "scahi_gpu": []}, []
    t_cache = 0.0
    for st in stems:
        img = cv2.imread(str(D.image_path("train", st)))
        H, W = img.shape[:2]
        t0 = time.time()
        z = {k: FC.raw_predict_level(m, img, cm.LEVELS[k], P.OVERLAP) for k in (P.WHOLE[1024], P.TILES)}
        t_cache += time.time() - t0
        levels = [ss.CachedLevel(k, v["o2m_xyxy"], v["o2m_probs"], v["o2m_std"], v["o2m_tile"], v["tiles"], v["gains"],
                                 int(v["n_input"]) == 0, P.OVERLAP) for k, v in z.items()]
        info.append({"image": st, "W": W, "H": H, "positions": {k: int(len(v["o2m_tile"])) for k, v in z.items()},
                     "passes": {k: int(len(v["tiles"])) for k, v in z.items()}})
        a = ss.fuse_image(levels, W, H, "F2", prm, nc=len(D.CLASSES))
        b = fuse_image_f2(levels, W, H, prm, nc=len(D.CLASSES))
        rows["scahi_cpu"].append(P._to_dota(R.clip_xywh(a, W, H), st))
        rows["scahi_gpu"].append(P._to_dota(R.clip_xywh(b, W, H), st))
        FC.set_long_side(m, 1024)
        with torch.no_grad():
            o = m([FC._to_tensor(img, "cuda")])[0]
        bx = o["boxes"].cpu().numpy()
        import pandas as pd

        rows["single"].append(pd.DataFrame({"image": st, "cls": o["labels"].cpu().numpy().astype(int),
                                            "score": o["scores"].cpu().numpy(), "x1": bx[:, 0], "y1": bx[:, 1],
                                            "x2": bx[:, 2], "y2": bx[:, 3]}))
    import pandas as pd

    res = {k: EV.map50(EV.build(gt, pd.concat(v, ignore_index=True), None, sizes=False)) for k, v in rows.items()}
    return {"images": info, "mAP50_on_these_calib_images": res, "cache_s_per_image": t_cache / len(stems),
            "params": prm.__dict__, "cpu_gpu_counts_equal": [len(x) == len(y) for x, y in zip(rows["scahi_cpu"], rows["scahi_gpu"])],
            "note": "pipeline smoke with 2-epoch / 64-image smoke weights; values are not accuracy estimates"}


def main() -> None:
    step = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fns = {"parity": parity, "memory": memory, "train": train, "export": export,
           "train2048": lambda: train(L=2048, n_img=48, epochs=1, bs=1)}
    r = fns[step]()
    r["wall_s"] = time.time() - t0
    (OUT / f"{step}.json").write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}))


if __name__ == "__main__":
    main()
