from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_tsne(
    df: pd.DataFrame,
    output_path: Path,
    perplexity: float,
    learning_rate: float,
    n_iter: int,
    max_points: Optional[int],
    random_state: int,
    target_error_types: Optional[list[str]] = None,
):
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError("scikit-learn is required for t-SNE plotting.") from exc

    if max_points and len(df) > max_points:
        df = df.sample(max_points, random_state=random_state)

    states = np.stack([np.asarray(s).reshape(-1) for s in df["state"].tolist()], axis=0)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
    )
    emb = tsne.fit_transform(states)
    df = df.copy()
    df["x"] = emb[:, 0]
    df["y"] = emb[:, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    sp = df[df["pair_type"] == "sp"]
    xp = df[df["pair_type"] == "xp"]
    ax.scatter(sp["x"], sp["y"], s=8, alpha=0.5, label="SP", color="#2b6cb0")
    ax.scatter(xp["x"], xp["y"], s=8, alpha=0.5, label="XP", color="#ed8936")

    if "violation" not in df.columns:
        raise ValueError("Missing 'violation' column in input data. Re-generate t-SNE data with violations enabled.")

    violations = df[df["violation"] == True]
    violation_label = "Violation"
    if target_error_types:
        violations = violations[violations["pair_type"] == "xp"]
        joined = ", ".join(target_error_types)
        violation_label = f"XP Target Error ({joined})"
    if not violations.empty:
        ax.scatter(violations["x"], violations["y"], s=20, marker="x", color="black", label=violation_label)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--learning-rate", type=float, default=200.0)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--target_error_types",
        nargs="+",
        type=str,
        default=None,
        help="List of error types used to label violation markers (e.g., COLLISION ONION_COUNTER_REGRAB).",
    )
    args = parser.parse_args()

    df = pd.read_pickle(args.input)
    output_path = Path(args.output) if args.output else Path(args.input).with_suffix(".png")
    plot_tsne(
        df,
        output_path=output_path,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        max_points=args.max_points,
        random_state=args.random_state,
        target_error_types=args.target_error_types,
    )


if __name__ == "__main__":
    main()
