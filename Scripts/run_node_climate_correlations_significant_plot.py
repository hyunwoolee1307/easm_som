import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "Results"
FIGURES_DIR = RESULTS_DIR / "Figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT = RESULTS_DIR / "node_climate_correlations.csv"
OUTPUT = FIGURES_DIR / "heatmap_node_climate_correlations_significant.png"

P_THRESHOLD = 0.05


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}")

    df = pd.read_csv(INPUT)
    # Expect columns: node, index, correlation, p_value, N
    sig = df[df["p_value"] < P_THRESHOLD].copy()

    # Pivot to node x index for correlation values (NaN for non-significant)
    pivot = sig.pivot(index="node", columns="index", values="correlation")

    # Keep consistent ordering
    nodes = sorted(df["node"].unique())
    indices = list(df["index"].unique())
    pivot = pivot.reindex(index=nodes, columns=indices)

    data = pivot.values.astype(float)

    # Mask non-significant values
    masked = np.ma.masked_invalid(data)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    cmap = plt.cm.RdBu_r
    cmap.set_bad(color="white")
    norm = colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(indices)))
    ax.set_xticklabels(indices)
    ax.set_yticks(np.arange(len(nodes)))
    ax.set_yticklabels(nodes)
    ax.set_xlabel("Climate Index")
    ax.set_ylabel("SOM Node")
    ax.set_title(f"Significant Node-Index Correlations (p < {P_THRESHOLD})")

    # Annotate only significant cells
    for i in range(masked.shape[0]):
        for j in range(masked.shape[1]):
            if not np.ma.is_masked(masked[i, j]):
                ax.text(j, i, f"{masked[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")

    fig.savefig(OUTPUT, dpi=300)


if __name__ == "__main__":
    main()
