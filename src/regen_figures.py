#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.signal import lombscargle
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Regenerate all figure outputs.")
    p.add_argument("--results", type=str, default="results")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--npcs", type=int, default=9)
    return p.parse_args()


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def grid_som():
    lon = np.arange(100.0, 180.0 + 2.5, 2.5)
    lat = np.arange(-10.0, 60.0 + 2.5, 2.5)
    return lon, lat


def grid_diag():
    lon = np.arange(0.0, 180.0 + 2.5, 2.5)
    lat = np.arange(-40.0, 60.0 + 2.5, 2.5)
    return lon, lat


def plot_composites_grid(
    data,
    lon,
    lat,
    title,
    outpath,
    units="",
    pvals=None,
    node_counts=None,
    cmap="coolwarm",
):
    n = data.shape[0]
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 9),
        subplot_kw={"projection": ccrs.PlateCarree()},
        layout="constrained",
    )
    axes = np.atleast_1d(axes).ravel()

    vmax = np.nanmax(np.abs(data))
    levels = np.linspace(-vmax, vmax, 21)
    LON, LAT = np.meshgrid(lon, lat)

    for i in range(n):
        ax = axes[i]
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.gridlines(draw_labels=False, linewidth=0.2, color="gray", alpha=0.5, linestyle=":")
        cf = ax.contourf(LON, LAT, data[i], levels=levels, cmap=cmap, extend="both")
        if pvals is not None:
            mask = pvals[i] < 0.05
            ax.scatter(LON[mask], LAT[mask], s=3, c="k", alpha=0.35, linewidths=0)
        if node_counts is not None:
            ax.set_title(f"Node {i+1} (n={int(node_counts[i])})")
        else:
            ax.set_title(f"Node {i+1}")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    cbar = fig.colorbar(cf, ax=axes[:n], orientation="vertical", shrink=0.85, pad=0.02)
    if units:
        cbar.set_label(units)
    fig.suptitle(title, y=0.995)
    fig.savefig(outpath, dpi=150)


def plot_u850_mean(results_dir, outdir):
    from plot_u850_jja_mean import main as plot_mean_main

    plot_mean_main()


def plot_eof_pc_timeseries(results_dir, outdir):
    from plot_eof_pc_timeseries import main as plot_pc_main

    plot_pc_main()


def plot_eof_elbow(results_dir, outdir):
    df = pd.read_csv(Path(results_dir) / "eof_explained_variance_ratio.csv")
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(df["mode"], df["explained_variance_ratio"] * 100, marker="o", linewidth=1.0)
    ax.set_xlabel("EOF mode number")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Explained-variance spectrum of EOF modes")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    fig.savefig(Path(outdir) / "eof_elbow.png", dpi=150)


def plot_eof_spatial_patterns(results_dir, outdir, n_modes=3):
    comps = np.load(Path(results_dir) / "eof_components.npy")
    lon, lat = grid_som()
    nlat, nlon = len(lat), len(lon)
    comps = comps[:n_modes].reshape(n_modes, nlat, nlon)

    fig, axes = plt.subplots(
        1,
        n_modes,
        figsize=(4.5 * n_modes, 4.2),
        subplot_kw={"projection": ccrs.PlateCarree()},
        layout="constrained",
    )
    axes = np.atleast_1d(axes).ravel()
    vmax = np.nanmax(np.abs(comps))
    levels = np.linspace(-vmax, vmax, 21)
    LON, LAT = np.meshgrid(lon, lat)

    for i in range(n_modes):
        ax = axes[i]
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.gridlines(draw_labels=False, linewidth=0.2, color="gray", alpha=0.5, linestyle=":")
        cf = ax.contourf(LON, LAT, comps[i], levels=levels, cmap="coolwarm", extend="both")
        ax.set_title(f"EOF{i+1} spatial pattern")

    cbar = fig.colorbar(cf, ax=axes, orientation="vertical", shrink=0.85, pad=0.02)
    cbar.set_label("EOF loading (unitless)")
    fig.suptitle(
        "Leading EOF spatial patterns of 850-hPa zonal wind anomalies", y=0.995
    )
    fig.savefig(Path(outdir) / "eof_spatial_patterns.png", dpi=150)


def plot_group_counts(results_dir, outdir):
    yearly = pd.read_csv(Path(results_dir) / "som_group_yearly_counts.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for col in ["Early", "Mature", "Late"]:
        ax.plot(yearly["year"], yearly[col], label=col, linewidth=1.1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Frequency (days)")
    ax.set_title("Annual frequency of phase-group occurrence (JJA, 1991-2023)")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    fig.savefig(Path(outdir) / "som_group_yearly_counts.png", dpi=150)

    yearly_ratio = pd.read_csv(Path(results_dir) / "som_group_yearly_ratio.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for col in ["Early", "Mature", "Late"]:
        ax.plot(yearly_ratio["year"], yearly_ratio[col] * 100, label=col, linewidth=1.1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Occurrence ratio (%)")
    ax.set_title("Annual occurrence ratios by phase group (JJA, 1991-2023)")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    fig.savefig(Path(outdir) / "som_group_yearly_ratio.png", dpi=150)

    monthly = pd.read_csv(Path(results_dir) / "som_group_monthly_counts.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.0), layout="constrained")
    months = monthly["month"].values
    width = 0.25
    ax.bar(months - width, monthly["Early"], width=width, label="Early")
    ax.bar(months, monthly["Mature"], width=width, label="Mature")
    ax.bar(months + width, monthly["Late"], width=width, label="Late")
    ax.set_xticks(months)
    ax.set_xlabel("Month")
    ax.set_ylabel("Frequency (days)")
    ax.set_title("Monthly frequency of phase-group occurrence (JJA, 1991-2023)")
    ax.legend()
    fig.savefig(Path(outdir) / "som_group_monthly_counts.png", dpi=150)

    monthly_ratio = pd.read_csv(Path(results_dir) / "som_group_monthly_ratio.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.0), layout="constrained")
    ax.bar(months - width, monthly_ratio["Early"] * 100, width=width, label="Early")
    ax.bar(months, monthly_ratio["Mature"] * 100, width=width, label="Mature")
    ax.bar(months + width, monthly_ratio["Late"] * 100, width=width, label="Late")
    ax.set_xticks(months)
    ax.set_xlabel("Month")
    ax.set_ylabel("Occurrence ratio (%)")
    ax.set_title("Monthly occurrence ratios by phase group (JJA, 1991-2023)")
    ax.legend()
    fig.savefig(Path(outdir) / "som_group_monthly_ratio.png", dpi=150)


def plot_node_counts(results_dir, outdir):
    yearly = pd.read_csv(Path(results_dir) / "som_node_yearly_counts.csv")
    fig, ax = plt.subplots(figsize=(8.0, 4.8), layout="constrained")
    for node in [str(i) for i in range(1, 10)]:
        ax.plot(yearly["year"], yearly[node], linewidth=0.9, label=f"Node {node}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Frequency (days)")
    ax.set_title("Annual frequency of SOM-node occurrence (JJA, 1991-2023)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    fig.savefig(Path(outdir) / "som_node_yearly_counts.png", dpi=150)

    monthly = pd.read_csv(Path(results_dir) / "som_node_monthly_counts.csv")
    fig, ax = plt.subplots(figsize=(7.0, 4.2), layout="constrained")
    months = monthly["month"].values
    for node in [str(i) for i in range(1, 10)]:
        ax.plot(months, monthly[node], marker="o", linewidth=0.9, label=f"Node {node}")
    ax.set_xticks(months)
    ax.set_xlabel("Month")
    ax.set_ylabel("Frequency (days)")
    ax.set_title("Monthly frequency of SOM-node occurrence (JJA, 1991-2023)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    fig.savefig(Path(outdir) / "som_node_monthly_counts.png", dpi=150)


def plot_node_trends(results_dir, outdir):
    df = pd.read_csv(Path(results_dir) / "som_node_monthly_counts_trend_10yr_JJA.csv")
    fig, ax = plt.subplots(figsize=(7.0, 4.2), layout="constrained")
    ax.bar(df["node"], df["slope_per_10yr"], color="#1f77b4", alpha=0.8)
    ax.errorbar(
        df["node"],
        df["slope_per_10yr"],
        yerr=df["stderr_per_10yr"],
        fmt="none",
        ecolor="k",
        capsize=3,
        linewidth=0.7,
    )
    for _, row in df.iterrows():
        if row["p_value"] < 0.05:
            ax.text(row["node"], row["slope_per_10yr"], "*", ha="center", va="bottom", fontsize=12)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xlabel("Node")
    ax.set_ylabel("Trend (counts per 10 years)")
    ax.set_title("Linear trend in JJA monthly node occurrence (1991-2023)")
    fig.savefig(Path(outdir) / "som_node_monthly_counts_trend_10yr_JJA.png", dpi=150)


def plot_periodogram(results_dir, outdir):
    df = pd.read_csv(Path(results_dir) / "som_clusters_by_day.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.month.isin([6, 7, 8])]
    df["month"] = df["date"].dt.to_period("M")

    nodes = range(1, 10)
    monthly_counts = (
        df.groupby(["month", "node_index"]).size().unstack("node_index").fillna(0.0)
    )
    monthly_counts = monthly_counts.reindex(columns=nodes, fill_value=0.0)

    t = monthly_counts.index.to_timestamp()
    t_years = t.year + (t.month - 1) / 12.0
    t_years = t_years.values.astype(float)
    t_centered = t_years - t_years.min()

    freqs = np.linspace(1 / 10.0, 4.0, 4000)  # cycles per year
    ang_freqs = 2 * np.pi * freqs

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True, layout="constrained")
    axes = axes.ravel()
    for i, node in enumerate(nodes):
        y = monthly_counts[node].values.astype(float)
        y = y - y.mean()
        p = lombscargle(t_centered, y, ang_freqs, normalize=True)
        ax = axes[i]
        ax.plot(freqs, p, color="#1f77b4", linewidth=0.8)
        ax.set_title(f"Node {node}")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    for j in range(len(nodes), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Lomb-Scargle periodograms of JJA monthly node counts (1991-2023)", y=0.995)
    fig.text(0.5, 0.04, "Frequency (cycles per year)", ha="center")
    fig.text(0.04, 0.5, "Normalized power", va="center", rotation="vertical")
    fig.savefig(Path(outdir) / "som_node_periodogram_JJA.png", dpi=150)


def plot_stability(results_dir, outdir):
    df = pd.read_csv(Path(results_dir) / "som_stability_summary.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
    ax.errorbar(
        df["seed"],
        df["mean_similarity"],
        yerr=[df["mean_similarity"] - df["min_similarity"], df["max_similarity"] - df["mean_similarity"]],
        fmt="o",
        color="#1f77b4",
        ecolor="gray",
        capsize=3,
    )
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("SOM stability across random initializations")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    fig.savefig(Path(outdir) / "som_stability_summary.png", dpi=150)


def plot_all_maps(results_dir, outdir):
    node_counts = np.load(Path(results_dir) / "som_node_counts.npy")

    lon, lat = grid_som()
    u850 = np.load(Path(results_dir) / "som_u850_composites.npy")
    plot_composites_grid(
        u850,
        lon,
        lat,
        "Composite anomalies of 850-hPa zonal wind by SOM node (JJA, 1991-2023)",
        Path(outdir) / "som_u850_composites_coastline.png",
        units="m s-1",
        node_counts=node_counts,
    )

    lon_d, lat_d = grid_diag()
    sst = np.load(Path(results_dir) / "som_sst_composites.npy")
    sst_p = np.load(Path(results_dir) / "som_sst_pvals.npy")
    plot_composites_grid(
        sst,
        lon_d,
        lat_d,
        "Composite standardized SST anomalies by SOM node (JJA, 1991-2023)",
        Path(outdir) / "som_sst_composites_stipple.png",
        units="standard deviation units",
        pvals=sst_p,
        node_counts=node_counts,
    )

    olr = np.load(Path(results_dir) / "som_olr_composites.npy")
    olr_p = np.load(Path(results_dir) / "som_olr_pvals.npy")
    plot_composites_grid(
        olr,
        lon_d,
        lat_d,
        "Composite standardized OLR anomalies by SOM node (JJA, 1991-2023)",
        Path(outdir) / "som_olr_composites_stipple.png",
        units="standard deviation units",
        pvals=olr_p,
        node_counts=node_counts,
    )

    uwnd = np.load(Path(results_dir) / "som_uwnd_composites_ext.npy")
    uwnd_p = np.load(Path(results_dir) / "som_uwnd_pvals_ext.npy")
    plot_composites_grid(
        uwnd,
        lon_d,
        lat_d,
        "Composite anomalies of 850-hPa zonal wind by SOM node (0-180E, 40S-60N)",
        Path(outdir) / "som_uwnd_composites_ext_stipple.png",
        units="m s-1",
        pvals=uwnd_p,
        node_counts=node_counts,
    )


def main():
    args = parse_args()
    results_dir = Path(args.results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_u850_mean(results_dir, outdir)
    plot_eof_pc_timeseries(results_dir, outdir)
    plot_eof_elbow(results_dir, outdir)
    plot_eof_spatial_patterns(results_dir, outdir, n_modes=3)
    plot_group_counts(results_dir, outdir)
    plot_node_counts(results_dir, outdir)
    plot_node_trends(results_dir, outdir)
    plot_periodogram(results_dir, outdir)
    plot_stability(results_dir, outdir)
    plot_all_maps(results_dir, outdir)


if __name__ == "__main__":
    main()
