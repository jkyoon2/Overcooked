from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_heatmap(df: pd.DataFrame, value_col: str, output_path: Path, cmap: str, title: Optional[str] = None):
    pivot = df.pivot(index="seed_ego", columns="seed_partner", values=value_col)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap=cmap, origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Partner Seed")
    ax.set_ylabel("Ego Seed")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--value", type=str, default="return")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    output_path = Path(args.output) if args.output else Path(args.input).with_suffix(".png")
    plot_heatmap(df, args.value, output_path, args.cmap, args.title)


if __name__ == "__main__":
    main()
