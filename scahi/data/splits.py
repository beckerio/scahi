# SPDX-License-Identifier: AGPL-3.0-or-later
"""Calibration split: ~10 % of train, whole video sequences, size-stratified (active).

Active files: ``calib.txt`` / ``train_main.txt``. Kept for documentation only:
``calib_imagewise.txt`` (the originally specified image-wise draw) and
``calib_by_sequence_random.txt`` (unstratified sequence draw). See docs/DECISIONS.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scahi.data.visdrone import EVAL_CATEGORIES, read_visdrone_annotation

CALIB_FRACTION = 0.10
CALIB_TOLERANCE = 0.01  # image fraction must lie in [0.09, 0.11]
CALIB_SEED = 20261021
N_CANDIDATES = 5000
SIZE_EDGES = (0, 8, 16, 32, 64, 128, np.inf)


def sequence_id(stem: str) -> str:
    """VisDrone image stems start with the source video id, e.g. ``0000002_00005_d_0000014``."""
    return stem.split("_")[0]


def make_calib_split(stems: list[str], fraction: float = CALIB_FRACTION, seed: int = CALIB_SEED) -> list[str]:
    """Image-wise random sample (originally specified; documentation only)."""
    stems = sorted(stems)
    rng = np.random.default_rng(seed)
    n = int(round(fraction * len(stems)))
    return sorted(rng.choice(stems, size=n, replace=False).tolist())


def _sequences(stems: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in sorted(stems):
        out.setdefault(sequence_id(s), []).append(s)
    return out


def make_calib_split_by_sequence(
    stems: list[str], fraction: float = CALIB_FRACTION, seed: int = CALIB_SEED
) -> list[str]:
    """Whole sequences in random order until the image fraction is reached (unstratified)."""
    by_seq = _sequences(stems)
    rng = np.random.default_rng(seed)
    target = fraction * len(stems)
    chosen: list[str] = []
    for q in rng.permutation(sorted(by_seq)):
        if len(chosen) >= target:
            break
        if len(chosen) + len(by_seq[q]) > target + len(by_seq[q]) / 2 and chosen:
            continue
        chosen += by_seq[q]
    return sorted(chosen)


def object_histograms(ann_dir: Path, stems: list[str]) -> dict[str, np.ndarray]:
    """Per image: counts over size bins (6) followed by counts over the 10 evaluated categories."""
    out = {}
    for s in stems:
        a = read_visdrone_annotation(ann_dir / f"{s}.txt")
        ok = np.isin(a.category, EVAL_CATEGORIES) & (a.score == 1)
        w, h = a.xyxy[ok, 2] - a.xyxy[ok, 0], a.xyxy[ok, 3] - a.xyxy[ok, 1]
        sizes = np.histogram(np.sqrt(np.clip(w * h, 0, None)), SIZE_EDGES)[0]
        classes = np.bincount(a.category[ok] - 1, minlength=10)
        out[s] = np.concatenate([sizes, classes]).astype(np.float64)
    return out


def _distance(h: np.ndarray, ref: np.ndarray, class_weight: float) -> float:
    """L1 between normalised size-bin distributions (+ weighted L1 of class distributions), in %."""
    d_size = np.abs(h[:6] / max(h[:6].sum(), 1) - ref[:6] / ref[:6].sum()).sum()
    d_cls = np.abs(h[6:] / max(h[6:].sum(), 1) - ref[6:] / ref[6:].sum()).sum()
    return float(100 * (d_size + class_weight * d_cls))


def make_calib_split_stratified(
    stems: list[str], hist: dict[str, np.ndarray], fraction: float = CALIB_FRACTION,
    seed: int = CALIB_SEED, n_candidates: int = N_CANDIDATES, class_weight: float = 0.5,
) -> tuple[list[str], dict[str, float]]:
    """Best of ``n_candidates`` random whole-sequence draws (image fraction within tolerance),
    minimising the distance of calib's size (and class) distribution to all of train."""
    by_seq = _sequences(stems)
    seqs = sorted(by_seq)
    seq_hist = {q: sum(hist[s] for s in by_seq[q]) for q in seqs}
    seq_n = {q: len(by_seq[q]) for q in seqs}
    ref = sum(seq_hist.values())
    n_total = len(stems)
    lo, hi = (fraction - CALIB_TOLERANCE) * n_total, (fraction + CALIB_TOLERANCE) * n_total
    rng = np.random.default_rng(seed)
    best: tuple[float, list[str]] | None = None
    for _ in range(n_candidates):
        chosen, n = [], 0
        for q in rng.permutation(seqs):
            if n + seq_n[q] <= hi:
                chosen.append(q)
                n += seq_n[q]
            if n >= lo:
                break
        if not lo <= n <= hi:
            continue
        d = _distance(sum(seq_hist[q] for q in chosen), ref, class_weight)
        if best is None or d < best[0]:
            best = (d, chosen)
    assert best is not None, "no feasible candidate"
    calib = sorted(s for q in best[1] for s in by_seq[q])
    return calib, {"distance": best[0], "n_candidates": n_candidates, "class_weight": class_weight}


def write_splits(train_stems: list[str], ann_dir: Path, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = object_histograms(ann_dir, train_stems)
    calib, info = make_calib_split_stratified(train_stems, hist)
    files = {
        "calib.txt": calib,
        "train_main.txt": sorted(set(train_stems) - set(calib)),
        "calib_imagewise.txt": make_calib_split(train_stems),
        "calib_by_sequence_random.txt": make_calib_split_by_sequence(train_stems),
    }
    for name, stems in files.items():
        (out_dir / name).write_text("\n".join(stems) + "\n")
    return {**{k: len(v) for k, v in files.items()}, "stratification": info}


def read_split(path: Path) -> list[str]:
    return [s for s in Path(path).read_text().split() if s]


SIZE_LABELS = ("<8", "8-16", "16-32", "32-64", "64-128", ">128")
CLASS_NAMES = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")


def describe_splits(original_root: Path, split_dirs: dict[str, str], splits_dir: Path) -> dict[str, object]:
    """Sizes/classes of calib vs. train_main vs. val (all calib variants) and sequence overlap of the
    official splits. Distributions in % of evaluable objects (categories 1-10, score 1)."""
    stems = {s: sorted(p.stem for p in (original_root / d / "annotations").glob("*.txt")) for s, d in split_dirs.items()}
    hist = {s: object_histograms(original_root / split_dirs[s] / "annotations", stems[s]) for s in ("train", "val")}

    def dist(split: str, subset: list[str]) -> dict[str, object]:
        h = sum(hist[split][s] for s in subset)
        return {"n_images": len(subset), "n_objects": int(h[:6].sum()),
                "size_pct": dict(zip(SIZE_LABELS, np.round(100 * h[:6] / h[:6].sum(), 2).tolist())),
                "class_pct": dict(zip(CLASS_NAMES, np.round(100 * h[6:] / h[6:].sum(), 2).tolist()))}

    train_all = stems["train"]
    variants = {"active (stratified by sequence)": "calib.txt", "image-wise (spec, inactive)": "calib_imagewise.txt",
                "sequence random (inactive)": "calib_by_sequence_random.txt"}
    out: dict[str, object] = {"variants": {}, "distributions": {}}
    for name, f in variants.items():
        calib = read_split(splits_dir / f)
        rest = sorted(set(train_all) - set(calib))
        seq_rest = {sequence_id(s) for s in rest}
        out["variants"][name] = {"file": f, "n_calib": len(calib), "n_rest": len(rest),
                                 "n_calib_sequences": len({sequence_id(s) for s in calib}),
                                 "calib_images_with_sequence_in_rest": sum(sequence_id(s) in seq_rest for s in calib)}
        out["distributions"][f"calib [{name}]"] = dist("train", calib)
    out["distributions"]["train_main (active)"] = dist("train", read_split(splits_dir / "train_main.txt"))
    out["distributions"]["train (all)"] = dist("train", train_all)
    out["distributions"]["val"] = dist("val", stems["val"])
    seqs = {s: {sequence_id(x) for x in v} for s, v in stems.items()}
    overlap = {}
    names = list(stems)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = seqs[a] & seqs[b]
            overlap[f"{a}|{b}"] = {"shared_sequence_ids": len(common), f"n_seq_{a}": len(seqs[a]), f"n_seq_{b}": len(seqs[b]),
                                   f"images_{a}": sum(sequence_id(x) in common for x in stems[a]),
                                   f"images_{b}": sum(sequence_id(x) in common for x in stems[b])}
    out["official_sequence_overlap"] = overlap
    return out
