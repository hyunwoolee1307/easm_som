import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
FIG_DIR = RESULTS_DIR / "Figures"


def load_u850_preprocessed():
    ds = xr.open_dataset(DATA_DIR / "u850.nc")
    da = ds["uwnd"]

    # Domain in SOM script
    lat_slice = slice(60.0, -10.0)
    lon_slice = slice(100.0, 180.0)
    subset = da.sel(lat=lat_slice, lon=lon_slice)

    # Monthly anomaly
    monthly_clim = subset.groupby("time.month").mean("time")
    subset_anom = subset.groupby("time.month") - monthly_clim

    # 10-day low-pass BEFORE JJA selection
    subset_lp = subset_anom.rolling(time=10, center=True).mean().dropna(dim="time")

    # JJA only
    subset_jja = subset_lp.sel(time=subset_lp.time.dt.month.isin([6, 7, 8]))
    return subset_jja


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    node_csv = RESULTS_DIR / "som_neuron_indices_jja.csv"
    if not node_csv.exists():
        raise FileNotFoundError(f"Missing {node_csv}. Run run_som_cluster.py first.")

    node_df = pd.read_csv(node_csv)
    node_df["time"] = pd.to_datetime(node_df["time"])
    node_df = node_df.sort_values("time")

    times = node_df["time"].to_numpy()
    node_ids = node_df["node_id"].to_numpy()

    da = load_u850_preprocessed()
    da = da.sel(time=times)

    # Build node mean vectors
    node_means = []
    labels = []
    for node in range(1, 10):
        mask = node_ids == node
        if np.sum(mask) < 2:
            continue
        comp = da.isel(time=mask).mean(dim="time")
        vec = comp.values.flatten()
        node_means.append(vec)
        labels.append(f"Node {node}")

    X = np.vstack(node_means)

    # Hierarchical clustering (Ward)
    Z = linkage(X, method="ward")

    # Plot dendrogram
    plt.figure(figsize=(8, 4), layout="constrained")
    dendrogram(Z, labels=labels, leaf_rotation=0)
    plt.title("SOM Node Dendrogram (U850 composites)")
    plt.ylabel("Distance")
    out_fig = FIG_DIR / "som_node_dendrogram.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
