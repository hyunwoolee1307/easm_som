#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Plot EOF PC time series.")
    p.add_argument("--scores", type=str, default="results/eof_scores.npy")
    p.add_argument("--dates", type=str, default="results/som_clusters_by_day.csv")
    p.add_argument("--out", type=str, default="results/eof_pc_timeseries.png")
    p.add_argument("--npcs", type=int, default=9)
    return p.parse_args()


def main():
    args = parse_args()
    scores = np.load(args.scores)
    df = pd.read_csv(args.dates)
    dates = pd.to_datetime(df["date"].values)

    if scores.shape[1] != len(dates):
        raise ValueError(
            f"Scores time dimension ({scores.shape[1]}) does not match dates ({len(dates)})."
        )

    npcs = min(args.npcs, scores.shape[0])
    ncols = 3
    nrows = int(np.ceil(npcs / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 3.2 * nrows), sharex=True, layout="constrained"
    )
    axes = np.atleast_1d(axes).ravel()

    for i in range(npcs):
        ax = axes[i]
        ax.plot(dates, scores[i], color="#1f77b4", linewidth=0.7)
        ax.axhline(0.0, color="#666666", linewidth=0.6, alpha=0.6)
        ax.set_title(f"PC{i+1}")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

    for j in range(npcs, len(axes)):
        axes[j].axis("off")

    locator = mdates.YearLocator(base=2)
    formatter = mdates.DateFormatter("%Y")
    for ax in axes[:npcs]:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    fig.suptitle(
        "Principal component time series for EOF modes (JJA daily, 1991-2023)", y=0.995
    )
    fig.text(0.5, 0.04, "Year", ha="center")
    fig.text(0.04, 0.5, "PC amplitude (standardized units)", va="center", rotation="vertical")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
