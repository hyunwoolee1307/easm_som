#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def parse_args():
    p = argparse.ArgumentParser(description="Plot JJA mean u850 (1991-2023).")
    p.add_argument("--data-root", type=str, default="/home/hyunwoo/Data/Obs_NCAR")
    p.add_argument("--var", type=str, default="uwnd")
    p.add_argument("--years", type=str, default="1991-2023")
    p.add_argument("--lon-min", type=float, default=100.0)
    p.add_argument("--lon-max", type=float, default=180.0)
    p.add_argument("--lat-min", type=float, default=-10.0)
    p.add_argument("--lat-max", type=float, default=60.0)
    p.add_argument("--out", type=str, default="results/u850_jja_mean_1991_2023.png")
    return p.parse_args()


def open_uwnd(data_root, var, years):
    y0, y1 = [int(x) for x in years.split("-")]
    files = [Path(data_root) / f"{var}.{y}.nc" for y in range(y0, y1 + 1)]
    existing = [f for f in files if f.exists()]
    if not existing:
        raise FileNotFoundError(f"No files found under {data_root} for {var}.YYYY.nc")
    ds = xr.open_mfdataset(existing, engine="h5netcdf", combine="by_coords")
    return ds


def main():
    args = parse_args()
    ds = open_uwnd(args.data_root, args.var, args.years)
    if args.var not in ds:
        raise KeyError(f"Variable '{args.var}' not found in dataset.")

    da = ds[args.var]
    # Normalize lon to [0, 360) if needed
    if (da.lon < 0).any():
        da = da.assign_coords(lon=((da.lon + 360) % 360)).sortby("lon")

    # Handle descending latitude
    lat_slice = slice(args.lat_min, args.lat_max)
    if da.lat[0] > da.lat[-1]:
        lat_slice = slice(args.lat_max, args.lat_min)

    da = da.sel(
        lon=slice(args.lon_min, args.lon_max),
        lat=lat_slice,
        level=850.0,
    )

    # JJA selection and mean
    da_jja = da.sel(time=da["time"].dt.month.isin([6, 7, 8]))
    mean_field = da_jja.mean(dim="time", skipna=True)

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 6.5), subplot_kw={"projection": ccrs.PlateCarree()}, layout="constrained")
    ax.set_extent([args.lon_min, args.lon_max, args.lat_min, args.lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.6, linestyle=":")

    levels = np.linspace(float(mean_field.min()), float(mean_field.max()), 21)
    cf = ax.contourf(mean_field.lon, mean_field.lat, mean_field, levels=levels, cmap="coolwarm", extend="both")
    cbar = plt.colorbar(cf, ax=ax, orientation="vertical", shrink=0.85, pad=0.02)
    cbar.set_label("m s-1")

    ax.set_title("Climatological mean 850-hPa zonal wind (JJA, 1991-2023)")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
