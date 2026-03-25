from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_curve(df: pd.DataFrame, output_path: Path, target_error_types: list[str] | None = None):
    df = df.sort_values("step")
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df["step"], df["sp_return"], label="SP Return", color="#2b6cb0")
    ax1.plot(df["step"], df["xp_return"], label="XP Return", color="#ed8936")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Episode Return")

    violation_label = "Violation"
    violation_ylabel = "Violation Rate"
    if target_error_types:
        joined = ", ".join(target_error_types)
        violation_label = f"Target Error ({joined})"
        violation_ylabel = "Target Error Rate"

    ax2 = ax1.twinx()
    ax2.plot(df["step"], df["sp_violation"], label=f"SP {violation_label}", color="#2b6cb0", linestyle="--")
    ax2.plot(df["step"], df["xp_violation"], label=f"XP {violation_label}", color="#ed8936", linestyle="--")
    ax2.set_ylabel(violation_ylabel)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--target_error_types",
        nargs="+",
        type=str,
        default=None,
        help="List of target errors to label violation curves (e.g., COLLISION ONION_COUNTER_REGRAB).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    output_path = Path(args.output) if args.output else Path(args.input).with_suffix(".png")
    plot_curve(df, output_path, target_error_types=args.target_error_types)


if __name__ == "__main__":
    main()
