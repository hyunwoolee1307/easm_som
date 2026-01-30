import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

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

    node_means = []
    labels = []
    for node in range(1, 10):
        mask = node_ids == node
        if np.sum(mask) < 2:
            continue
        comp = da.isel(time=mask).mean(dim="time")
        vec = comp.values.flatten()
        node_means.append(vec)
        labels.append(node)

    X = np.vstack(node_means)

    # Cluster nodes (k-means)
    k = 3
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(X)

    # MDS for 2D visualization
    mds = MDS(n_components=2, metric_mds=True, random_state=42, n_init=4, init="random")
    coords = mds.fit_transform(X)

    # Input distribution (node frequency) + clustering visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

    # Left: input distribution (node counts)
    node_counts = node_df["node_id"].value_counts().sort_index()
    axes[0].bar(node_counts.index, node_counts.values, color="tab:blue", alpha=0.8)
    axes[0].set_title("Input Distribution by Node")
    axes[0].set_xlabel("Node")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Right: MDS clustering
    for cluster_id in range(k):
        idx = clusters == cluster_id
        axes[1].scatter(
            coords[idx, 0],
            coords[idx, 1],
            label=f"Cluster {cluster_id+1}",
        )

    for i, node in enumerate(labels):
        axes[1].text(coords[i, 0], coords[i, 1], f"N{node}", fontsize=9, ha="center")

    axes[1].set_title("SOM Node Clustering (MDS)")
    axes[1].set_xlabel("MDS-1")
    axes[1].set_ylabel("MDS-2")
    axes[1].legend()

    out_fig = FIG_DIR / "som_input_distribution_and_clusters.png"
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)

    # Save cluster assignments
    out_csv = RESULTS_DIR / "som_node_clusters.csv"
    df_out = pd.DataFrame({"node": labels, "cluster": clusters + 1})
    df_out.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
