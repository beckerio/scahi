# SPDX-License-Identifier: AGPL-3.0-or-later
"""FCOS input-size check at runtime (user request 2026-11-02, before the FCOS calib tuning): a forward hook on
``model.transform`` (torchvision GeneralizedRCNNTransform) records min_size / max_size and the network input size
(after the transform) while the real code paths run: (a) a training step as in ``scahi.dota.fcos_train.train`` (model
built by ``FC.load(.., s, ..)``, train mode, data from ``TrainSet`` / ``_to_gpu``), (b) single pass and (c) inference
crossing via ``fcos_train.predict``, (d) tile and whole-image cache passes via ``scahi.fcos.raw_predict_level``.
No weights are changed (the training step runs forward + loss only, no optimizer step).

    python scripts/dota/fcos_size_check_runtime.py   # -> runs/fcos/size_check_runtime.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import torch  # noqa: E402

from scahi import fcos as FC  # noqa: E402
from scahi.dota import data as D  # noqa: E402
from scahi.dota import fcos_train as FT  # noqa: E402
from scahi.dota import pipeline as P  # noqa: E402
from scahi.dota import train as T  # noqa: E402
from scahi.infer import cache_mz as cm  # noqa: E402


def main() -> None:
    stems = {"calib": P.split_stems("calib")[:3], "train": ["P1868", *T.split_stems("train_main")[:2]]}
    rows, rec = [], []

    def hook(mod, inp, out):
        rec.append({"min_size": list(mod.min_size), "max_size": mod.max_size, "training": mod.training,
                    "in_hw": [list(x.shape[-2:]) for x in inp[0]], "net_hw": [list(h) for h in out[0].image_sizes]})

    def take(setting: str, s: int, intended: int | None) -> None:
        for r in rec:
            for i, n in zip(r["in_hw"], r["net_hw"]):
                L = intended if intended is not None else max(i)  # tile passes: the pass's own letterbox long side
                rows.append({"weights": s, "setting": setting, **{k: r[k] for k in ("min_size", "max_size", "training")},
                             "in_hw": i, "net_hw": n, "intended_long_side": L, "ok": max(n) == r["max_size"]
                             and r["min_size"] == [r["max_size"]] and (intended is None or r["max_size"] == intended)})
        rec.clear()

    for s in (1024, 2048):
        w = FT.weights(s)
        # (a) training step: model as built in train(s), train mode, real data path
        m = FC.load(len(D.CLASSES), s, FT.limits(), "cuda", torch.load(w, map_location="cuda"))
        h = m.transform.register_forward_hook(hook)
        m.train()
        ds = FT.TrainSet(stems["train"])
        for i in range(len(ds)):
            x, t = FT._to_gpu(ds[i])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                m([x], [t])
        take("(a) training step", s, s)
        h.remove()
        del m
        torch.cuda.empty_cache()
        # (b) single pass, (c) crossing: the real predictor
        m = FT.load(w, s)
        h = m.transform.register_forward_hook(hook)
        for L in ((s,) if s == 1024 else (2048, 2560, 3072)):
            FT.predict(w, "train", stems["calib"], L, model=m)
            take(f"({'b) single pass' if L == s else 'c) crossing'} L={L}", s, L)
        # (d) cache passes: whole image at s and native 1024 tiles
        for lk in (P.WHOLE[s], P.TILES):
            for st in stems["calib"]:
                FC.raw_predict_level(m, cv2.imread(str(D.image_path("train", st))), cm.LEVELS[lk], P.OVERLAP)
            take(f"(d) cache passes {lk}", s, s if lk == P.WHOLE[s] else None)
        h.remove()
        del m
        torch.cuda.empty_cache()
    out = {"n": len(rows), "all_ok": all(r["ok"] for r in rows), "rows": rows}
    (REPO / "runs/fcos/size_check_runtime.json").write_text(json.dumps(out, indent=1))
    seen = set()
    for r in rows:
        k = (r["weights"], r["setting"], tuple(r["in_hw"]))
        if k in seen:
            continue
        seen.add(k)
        print(f"{r['weights']} {r['setting']:<34} min/max {r['min_size']}/{r['max_size']} train={r['training']!s:<5} "
              f"{r['in_hw']} -> {r['net_hw']} {'ok' if r['ok'] else 'MISMATCH'}")
    print(f"[size-check] {out['n']} transform calls, all intended: {out['all_ok']}")


if __name__ == "__main__":
    main()
