import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
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
    som_nodes = node_df["node_id"].to_numpy()

    da = load_u850_preprocessed()
    da = da.sel(time=times)

    # Prepare feature matrix (time x space)
    X = da.values.reshape(da.shape[0], -1)
    X = np.nan_to_num(X)

    # KMeans clustering to 9 clusters (match SOM nodes)
    k = 9
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X) + 1  # shift to 1..9

    # MDS for 2D visualization
    mds = MDS(n_components=2, metric_mds=True, random_state=42, n_init=4, init="random")
    coords = mds.fit_transform(X)

    # Confusion matrix
    conf = pd.crosstab(pd.Series(som_nodes, name="SOM"), pd.Series(km_labels, name="KMeans"))

    # Visualization: SOM clusters, KMeans clusters, and comparison heatmap
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")

    sc1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=som_nodes, cmap="tab10", s=10)
    axes[0].set_title("SOM Node Clusters")
    axes[0].set_xlabel("MDS-1")
    axes[0].set_ylabel("MDS-2")
    plt.colorbar(sc1, ax=axes[0], label="SOM Node")

    sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=km_labels, cmap="tab10", s=10)
    axes[1].set_title("KMeans Clusters (k=9)")
    axes[1].set_xlabel("MDS-1")
    axes[1].set_ylabel("MDS-2")
    plt.colorbar(sc2, ax=axes[1], label="KMeans Cluster")

    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", ax=axes[2])
    axes[2].set_title("SOM vs KMeans")
    axes[2].set_xlabel("KMeans Cluster")
    axes[2].set_ylabel("SOM Node")

    out_fig = FIG_DIR / "compare_som_kmeans.png"
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)

    # Save confusion matrix
    out_csv = RESULTS_DIR / "compare_som_kmeans_confusion.csv"
    conf.to_csv(out_csv)


if __name__ == "__main__":
    main()
