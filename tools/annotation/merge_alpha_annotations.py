#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA_MIN, ALPHA_MAX = 0.0, 0.9                                   

def _read_annotation(path: Path):

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        need = ["frame", "alpha_l", "alpha_r"]
        missing = [c for c in need if c not in cols]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        frame = cols["frame"]
        df[frame] = pd.to_numeric(df[frame], errors="raise")
        if df[frame].isna().any() or df[frame].duplicated().any():
            raise ValueError(f"{path}: missing or duplicate frame IDs")
        df = df.sort_values(frame)
        return df[frame].to_numpy(), df[[cols["alpha_l"], cols["alpha_r"]]].to_numpy(dtype=np.float64)
    with open(path, "rb") as f:
        d = pickle.load(f)
    return None, np.stack([np.asarray(d["alpha_L"]), np.asarray(d["alpha_R"])], axis=1)

def load_annotation(path: Path) -> np.ndarray:
    return _read_annotation(path)[1]


def aligned_scores(files):
    if len(set(p.resolve() for p in files)) != len(files):
        raise ValueError("each annotator must supply a distinct file")
    reference = None
    scores = []
    for path in files:
        frames, values = _read_annotation(path)
        if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"{path}: expected finite bilateral scores for every frame")
        if scores and len(values) != len(scores[0]):
            raise ValueError(f"{path}: annotator frame counts differ")
        if frames is not None:
            if reference is not None and not np.array_equal(reference, frames):
                raise ValueError(f"{path}: annotator frame IDs differ")
            reference = frames
        scores.append(values)
    return np.stack(scores)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", nargs="+", required=True,
                    help="per-annotator score files (CSV or PKL)")
    ap.add_argument("--recording", required=True,
                    help="recording directory; writes <recording>/alpha_labels.pkl")
    ap.add_argument("--min_annotators", type=int, default=3,
                    help="require at least this many annotators (paper: 3)")
    args = ap.parse_args()

    files = [Path(p) for p in args.annotations]
    if len(files) < args.min_annotators:
        raise SystemExit(
            f"got {len(files)} annotation files, need >= {args.min_annotators} "
            f"(the paper protocol uses three independent annotators)"
        )

    stack = aligned_scores(files)
    n = stack.shape[1]

    merged = np.clip(stack.mean(axis=0), ALPHA_MIN, ALPHA_MAX)             

    A = stack.shape[0]
    pair_mad = np.mean(
        [np.abs(stack[a] - stack[b]).mean() for a in range(A) for b in range(a + 1, A)]
    )
    print(f"annotators: {A}, frames: {n}, mean pairwise |diff|: {pair_mad:.4f}")

    out = Path(args.recording) / "alpha_labels.pkl"
    labels: dict = {}
    if out.exists():
        with open(out, "rb") as f:
            labels = pickle.load(f)
        print(f"updating existing {out} (keys preserved: "
              f"{sorted(k for k in labels if k not in ('alpha_L', 'alpha_R'))})")
    labels["alpha_L"] = merged[:, 0].astype(np.float32)
    labels["alpha_R"] = merged[:, 1].astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(labels, f)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
