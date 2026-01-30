import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
FIG_DIR = RESULTS_DIR / "Figures"


def plot_node_composites(data_da, node_ids, m1, m2, lats, lons, title, output_path, vmin, vmax):
    n_nodes = m1 * m2
    fig, axes = plt.subplots(
        nrows=m2,
        ncols=m1,
        figsize=(15, 12),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
    )

    total_samples = len(node_ids)
    clim_mean = np.mean(data_da, axis=0)
    levels = np.linspace(vmin, vmax, 41)

    for j in range(m2):
        for i in range(m1):
            node_idx = j * m1 + (i + 1)
            ax = axes[j, i]

            indices = np.where(node_ids == node_idx)[0]
            count = len(indices)
            freq = (count / total_samples) * 100

            if count > 1:
                local_samples = data_da[indices, :, :]
                node_mean_2d = np.mean(local_samples, axis=0)

                t_stat, p_val = stats.ttest_1samp(
                    local_samples, popmean=clim_mean, axis=0
                )
                p_val_2d = p_val

                cf = ax.contourf(
                    lons,
                    lats,
                    node_mean_2d,
                    transform=ccrs.PlateCarree(),
                    cmap="RdBu_r",
                    levels=levels,
                    extend="both",
                )

                ax.contourf(
                    lons,
                    lats,
                    p_val_2d,
                    levels=[0, 0.05, 1.0],
                    colors="none",
                    hatches=["..", ""],
                    transform=ccrs.PlateCarree(),
                )
            else:
                ax.text(0.5, 0.5, "Insufficient Data", ha="center", transform=ax.transAxes)

            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=":")

            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
            if i > 0:
                gl.left_labels = False
            if j < m2 - 1:
                gl.bottom_labels = False

            ax.set_title(f"Node {node_idx} (n={count}, {freq:.1f}%)")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cf, cax=cbar_ax, label=title)

    plt.suptitle(title, fontsize=16, y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


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

    configs = [
        (DATA_DIR / "sst_anom_regridded.nc", "SST (anom)", "som_node_composites_sst.png", -1.0, 1.0),
        (DATA_DIR / "olr_anom2.nc", "OLR (anom)", "som_node_composites_olr.png", -20.0, 20.0),
        (DATA_DIR / "uwnd850_anom.nc", "U850 (anom)", "som_node_composites_u850.png", -5.0, 5.0),
    ]

    for path, title, out_name, vmin, vmax in configs:
        ds = xr.open_dataset(path)
        var = list(ds.data_vars)[0]
        da = ds[var]

        # Domain: 0-180E, 20S-60N
        lon_min, lon_max = 0, 180
        lat_min, lat_max = -20, 60
        da = da.sortby("lat").sortby("lon")
        da = da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

        # Align to SOM times
        da_sel = da.sel(time=times)
        lats = da_sel.lat.values
        lons = da_sel.lon.values

        X = da_sel.values

        plot_node_composites(
            X,
            node_ids,
            m1=3,
            m2=3,
            lats=lats,
            lons=lons,
            title=f"SOM 3x3 Node Composites (JJA) - {title}",
            output_path=FIG_DIR / out_name,
            vmin=vmin,
            vmax=vmax,
        )

        ds.close()


if __name__ == "__main__":
    main()
