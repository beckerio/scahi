# SPDX-License-Identifier: AGPL-3.0-or-later
"""FCOS on DOTA (preregistration addendum 3, 2026-11-01): training, single pass, caches, parity gate, S_max smoke.

Model name ``fcos`` in the DOTA framework (``scahi.dota.train`` / ``pipeline`` / ``final`` dispatch here). Recipe as in
``reports/FCOS_SPIKE.md`` §3: torchvision ``fcos_resnet50_fpn`` (COCO_V1, new 16-class output layer), 60 epochs, whole
image at long side s, horizontal flip only, SGD (momentum 0.9, wd 1e-4), lr 0.005 at effective batch 8 (0.01 at 16,
linear scaling), linear warm-up over epoch 1, cosine to 1 % at epoch 60, AMP fp16, seed 0; batch 8 at 1024, batch 1 with
gradient accumulation over 8 images at 2048 (pre-specified FCOS-only exception, addendum 3 item 3). Selection on calib:
devkit mAP50 without cap at the YOLO-DOTA schedule (every 5th epoch and each of the last 20). Score everywhere: the
detector's own sqrt(p_c * ctr). Pipeline limit 200 000 per image and pass (addendum 3 item 6).
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scahi import fcos as FC
from scahi.dota import data as D
from scahi.dota import evaluate as EV
from scahi.dota import train as T

MODEL = "fcos"
EPOCHS = 60
BATCH = {1024: 8, 2048: 1}
ACCUM = {1024: 1, 2048: 8}  # effective batch 8 at both sizes
LR = 0.01 * 8 / 16
MOMENTUM, WEIGHT_DECAY = 0.9, 1e-4
FINAL_LR_FRAC = 0.01
LIMIT = 200_000  # pipeline limit (addendum 3 item 6)
WORKERS = 6


def weights(s: int, seed: int = 0) -> Path:
    return T.train_dir(MODEL, s, seed) / "weights/best_calib.pt"


def limits(max_det: int = LIMIT) -> FC.Limits:
    return FC.Limits(score_thresh=0.001, topk_candidates=LIMIT, detections_per_img=max_det, nms_thresh=0.6)


def load(w: str | Path, long_side: int, max_det: int = LIMIT, device: str = "cuda"):
    return FC.load(len(D.CLASSES), long_side, limits(max_det), device, torch.load(w, map_location=device))


# ------------------------------------------------------------------------------------------------ data


def labels(stem: str, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Training labels as used for the YOLO trainings (data/canonical/dota/yolo_hbb: difficult GT = positives)."""
    f = T.YOLO_ROOT / "labels/train" / f"{stem}.txt"
    a = np.loadtxt(f, ndmin=2) if f.exists() and f.stat().st_size else np.zeros((0, 5))
    cx, cy, w, h = a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
    b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    ok = (b[:, 2] > b[:, 0] + 1e-3) & (b[:, 3] > b[:, 1] + 1e-3)  # torchvision rejects degenerate boxes
    return b[ok].astype(np.float32), a[ok, 0].astype(np.int64)


class TrainSet(torch.utils.data.Dataset):
    """Whole training images (uint8, BGR -> RGB on the GPU in ``FC._to_tensor`` layout) + HBB targets; horizontal flip
    with p = 0.5 drawn from a per-(epoch, index) seed (reproducible, independent of the worker split)."""

    def __init__(self, stems: list[str]) -> None:
        self.stems, self.epoch = stems, 0

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, i: int):
        import cv2

        st = self.stems[i]
        img = cv2.imread(str(D.image_path("train", st)))
        H, W = img.shape[:2]
        b, c = labels(st, W, H)
        if random.Random(self.epoch * 1_000_003 + i).random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            b = np.stack([W - b[:, 2], b[:, 1], W - b[:, 0], b[:, 3]], 1).astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(b), torch.from_numpy(c)


def _collate(batch):
    return batch


def _to_gpu(item):
    img, b, c = item
    x = img.cuda(non_blocking=True)[:, :, [2, 1, 0]].permute(2, 0, 1).float() / 255.0  # as FC._to_tensor
    return x, {"boxes": b.cuda(), "labels": c.cuda()}


# ------------------------------------------------------------------------------------------------ predict


@torch.no_grad()
def predict(w: str | Path, split: str, stems: list[str], imgsz: int, conf: float = T.PRED_CONF, max_det: int = LIMIT,
            measure_latency: bool = False, model=None) -> tuple[pd.DataFrame, dict]:
    """Single pass at long side ``imgsz`` (torchvision predictor, detector's own score); DOTA DET_COLS."""
    import cv2

    m = model or load(w, imgsz, max_det)
    FC.set_long_side(m, imgsz)
    m.eval()
    rows, lat, n_lim = [], [], 0
    torch.cuda.reset_peak_memory_stats()
    for n, st in enumerate(stems):
        img = cv2.imread(str(D.image_path(split, st)))
        if measure_latency and n == 0:
            m([FC._to_tensor(img, "cuda")])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = m([FC._to_tensor(img, "cuda")])[0]
        torch.cuda.synchronize()
        lat.append(1000 * (time.perf_counter() - t0))
        k = o["scores"] >= conf
        bx, sc, cl = o["boxes"][k].cpu().numpy(), o["scores"][k].cpu().numpy(), o["labels"][k].cpu().numpy()
        rows.append(pd.DataFrame({"image": st, "cls": cl.astype(int), "score": sc.astype(float), "x1": bx[:, 0].astype(float),
                                  "y1": bx[:, 1].astype(float), "x2": bx[:, 2].astype(float), "y2": bx[:, 3].astype(float)}))
        n_lim += int(len(o["scores"]) >= max_det)
    info = {"peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20, "latency_ms_mean": float(np.mean(lat)),
            "latency_ms_p95": float(np.percentile(lat, 95)), "max_dets_per_image": max_det, "images_at_max_det": n_lim}
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=EV.DET_COLS)), info


# ------------------------------------------------------------------------------------------------ training


def _lr(step: int, steps_per_epoch: int, total: int) -> float:
    if step < steps_per_epoch:  # linear warm-up over epoch 1
        return LR * (step + 1) / steps_per_epoch
    t = (step - steps_per_epoch) / max(1, total - steps_per_epoch)
    return LR * (FINAL_LR_FRAC + (1 - FINAL_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def train(s: int, seed: int = 0) -> Path:
    """Train (resumable per epoch from last.pt) and return best_calib.pt (state dict)."""
    from scahi.common import rel, write_env
    from scahi.train.full import eval_schedule

    d = T.train_dir(MODEL, s, seed)
    wdir = d / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    if (d / "TRAIN_DONE").exists():
        return wdir / "best_calib.pt"
    write_env(d / "env.json")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    m = FC.load(len(D.CLASSES), s, limits())
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")
    ds = TrainSet(T.split_stems("train_main"))
    bs, acc = BATCH[s], ACCUM[s]
    gen = torch.Generator().manual_seed(seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=WORKERS, collate_fn=_collate,
                                     generator=gen, persistent_workers=False, pin_memory=True, drop_last=False)
    steps_per_epoch = math.ceil(len(dl) / acc)
    total = steps_per_epoch * EPOCHS
    start, step = 1, 0
    last = wdir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cuda")
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start, step = ck["epoch"] + 1, ck["step"]
        gen.set_state(ck["gen"].cpu())
        print(f"[fcos] resume {s} from epoch {ck['epoch']}", flush=True)
    stems_calib = T.split_stems("calib")
    gt = EV.gt_arrays("train", stems_calib)
    cp = wdir / "calib_curve.json"
    curve = json.loads(cp.read_text()) if cp.exists() else []
    log = d / "train_log.jsonl"
    for ep in range(start, EPOCHS + 1):
        m.train()
        FC.set_long_side(m, s)
        ds.epoch = ep
        t0, losses, n_micro = time.time(), [], 0
        opt.zero_grad(set_to_none=True)
        for batch in dl:
            imgs, tgts = zip(*[_to_gpu(x) for x in batch])
            with torch.autocast("cuda", dtype=torch.float16):
                ld = m(list(imgs), list(tgts))
            loss = sum(ld.values()) / acc
            scaler.scale(loss).backward()
            losses.append({k: float(v.detach()) for k, v in ld.items()})
            n_micro += 1
            if n_micro % acc == 0 or n_micro == len(dl):
                for g in opt.param_groups:
                    g["lr"] = _lr(step, steps_per_epoch, total)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
        torch.cuda.synchronize()
        rec = {"epoch": ep, "train_s": time.time() - t0, "lr_end": opt.param_groups[0]["lr"], "steps": step,
               "loss": {k: float(np.mean([x[k] for x in losses])) for k in losses[0]},
               "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "epoch": ep,
                    "step": step, "gen": gen.get_state()}, last)
        if eval_schedule(ep, EPOCHS) and not any(c["epoch"] == ep for c in curve):
            t1 = time.time()
            torch.save(m.state_dict(), wdir / "epoch_eval.pt")
            dets, info = predict(wdir / "epoch_eval.pt", "train", stems_calib, s, model=m)
            v = EV.map50(EV.build(gt, dets, None, sizes=False))
            best = max((c["mAP50"] for c in curve), default=-1.0)
            curve.append({"epoch": ep, "source": "dota_devkit_port mAP50, no cap", "selection_branch": "o2m",
                          "mAP50": v, "eval_s": time.time() - t1, "images_at_limit": info["images_at_max_det"]})
            if v > best:
                (wdir / "epoch_eval.pt").replace(wdir / "best_calib.pt")
            cp.write_text(json.dumps(curve, indent=2))
            rec["calib_mAP50"] = v
        with log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[fcos] {s} epoch {ep}: {json.dumps(rec)}", flush=True)
    (wdir / "epoch_eval.pt").unlink(missing_ok=True)
    be = max(curve, key=lambda c: c["mAP50"])
    (d / "training.json").write_text(json.dumps({
        "init": "torchvision fcos_resnet50_fpn COCO_V1 (new 16-class output layer)", "epochs": EPOCHS, "imgsz": s,
        "batch": bs, "accumulation": acc, "effective_batch": bs * acc, "lr": LR, "seed": seed,
        "selection": f"devkit mAP50 on calib (no cap), best epoch {be['epoch']} ({be['mAP50']:.2f})",
        "weights": rel(wdir / "best_calib.pt")}, indent=2))
    (d / "TRAIN_DONE").write_text("ok\n")
    return wdir / "best_calib.pt"


# ------------------------------------------------------------------------------------------------ caches, gate, smoke


def caches(s: int, splits: tuple[str, ...] = ("calib", "val")) -> None:
    """GPU: SCAHI caches of both levels of dota<s> (whole image at s + native 1024 tiles) for the splits (resumable)."""
    import cv2

    from scahi.dota import pipeline as P
    from scahi.fusion.run import CACHE_ROOT
    from scahi.infer import cache_mz as cm

    w = weights(s)
    m = load(w, s)
    for split in splits:
        imgs = P.images(split)
        for lk in (P.WHOLE[s], P.TILES):
            d = cm.cache_dir(CACHE_ROOT, w, lk, P.OVERLAP, split)
            d.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            for st, p in zip(imgs.stem, imgs.path):
                f = d / f"{st}.npz"
                if not f.exists():
                    np.savez_compressed(f, **FC.raw_predict_level(m, cv2.imread(p), cm.LEVELS[lk], P.OVERLAP))
            print(f"[fcos] cache {s} {lk} {split}: {time.time() - t0:.0f} s", flush=True)


GATE_AP_TOL = 0.01


def gate(s: int, n: int = 3) -> dict:
    """Parity gate, FCOS version (addendum 3 item 8): (a) own post-processing of the dense outputs vs. torchvision's
    own post-processing of the same forward, bit-identical, at every input geometry (whole image at s, a 1024 px tile);
    (b) calib mAP50 (no cap) of the single pass from the own post-processing vs. the torchvision predictor, |Δ| <= 0.01."""
    import cv2

    from scahi.dota import pipeline as P

    w = weights(s)
    m = load(w, s)
    imgs = P.images("calib")
    out: dict = {"gate": "D-parity (FCOS)", "model": MODEL, "s": s, "rows": []}
    for kind in ("whole", "tile"):
        for p in imgs.path.iloc[:n]:
            im = cv2.imread(p)
            crop, L = (im[:1024, :1024], 1024) if kind == "tile" else (im, s)
            d = FC.dense_pass(m, crop, L)
            own = FC.postprocess_like_torchvision(d, m, crop.shape[:2])
            ref = d["dets"]
            same = (len(own["boxes"]) == len(ref["boxes"]) and torch.equal(own["boxes"], ref["boxes"])
                    and torch.equal(own["scores"], ref["scores"]) and torch.equal(own["labels"], ref["labels"]))
            out["rows"].append({"input": kind, "image": Path(p).stem, "n": len(ref["boxes"]), "identical": bool(same)})
    gt = EV.gt_arrays("train", T.split_stems("calib"))
    own_rows = []
    for st, p in zip(imgs.stem, imgs.path):
        im = cv2.imread(p)
        d = FC.dense_pass(m, im, s)
        o = FC.postprocess_like_torchvision(d, m, im.shape[:2])
        b = o["boxes"].cpu().numpy()
        own_rows.append(pd.DataFrame({"image": st, "cls": o["labels"].cpu().numpy().astype(int),
                                      "score": o["scores"].cpu().numpy().astype(float), "x1": b[:, 0], "y1": b[:, 1],
                                      "x2": b[:, 2], "y2": b[:, 3]}))
    ref_dets, _ = predict(w, "train", list(imgs.stem), s, model=m)
    a = {"own": EV.map50(EV.build(gt, pd.concat(own_rows, ignore_index=True), None, sizes=False)),
         "torchvision": EV.map50(EV.build(gt, ref_dets, None, sizes=False))}
    out["map_identity"] = {**a, "abs_delta": abs(a["own"] - a["torchvision"])}
    out["passed_decode"] = all(r["identical"] for r in out["rows"])
    out["passed_map"] = out["map_identity"]["abs_delta"] <= GATE_AP_TOL
    out["passed"] = out["passed_decode"] and out["passed_map"]
    od = P.OUT / f"{MODEL}_{s}"
    od.mkdir(parents=True, exist_ok=True)
    (od / "gate.json").write_text(json.dumps(out, indent=1))
    print(f"[fcos] gate {s}: passed={out['passed']} {out['map_identity']}", flush=True)
    if not out["passed"]:
        raise RuntimeError(f"[fcos] parity gate failed for {s} ({od / 'gate.json'})")
    return out


def smax_smoke() -> dict:
    """S_max rule (addendum 3 item 3): forward + backward (AMP) of the recipe at batch 1 on each of the 48 densest
    train_main images at 2048, and at 3072 for documentation -> runs/dota/fcos_smoke.json."""
    stems = T.split_stems("train_main")
    cnt = {}
    for st in stems:
        f = T.YOLO_ROOT / "labels/train" / f"{st}.txt"
        cnt[st] = sum(1 for _ in f.open()) if f.exists() else 0
    dense = sorted(stems, key=lambda x: -cnt[x])[:48]
    ds = TrainSet(dense)
    out: dict = {"images": {x: cnt[x] for x in dense}, "rows": []}
    for L in (2048, 3072):
        m = FC.load(len(D.CLASSES), L, limits())
        m.train()
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-6, momentum=MOMENTUM)
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.reset_peak_memory_stats()
        e = {"long_side": L, "batch": 1}
        try:
            for i in range(len(ds)):
                x, t = _to_gpu(ds[i])
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = sum(m([x], [t]).values())
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            e.update(fits=True, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        except torch.OutOfMemoryError as ex:
            e.update(fits=False, at_image=dense[i], gt=cnt[dense[i]], error=str(ex).splitlines()[0][:160])
        out["rows"].append(e)
        print(f"[fcos] smoke {e}", flush=True)
        del m, opt
        torch.cuda.empty_cache()
    out["s_max"] = 2048 if out["rows"][0]["fits"] else 1024
    (T.DOTA_RUNS / "fcos_smoke.json").write_text(json.dumps(out, indent=1))
    return out
