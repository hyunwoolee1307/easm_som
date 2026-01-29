"""Preprocess u850 anomalies for SOM training.

Strict constraints (see Instruction.md / README.md):
- SOM input is u850 anomalies only.
- Domain: 100E–180E, 10S–60N.
- Apply anomaly calculation and standardization before SOM.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import xarray as xr
import yaml


@dataclass
class PreprocessConfig:
    lon_min: float = 100.0
    lon_max: float = 180.0
    lat_min: float = -10.0
    lat_max: float = 60.0
    climatology_years: Tuple[int, int] = (1991, 2020)
    detrend: bool = False
    standardize: str = "zscore"
    area_weighted: bool = True


def _check_u850_only(var_name: str) -> None:
    if var_name.lower() not in {"u850", "u850_anomaly", "u"}:
        raise ValueError("Only u850 is allowed as SOM input.")


def _area_weights(lat: np.ndarray) -> np.ndarray:
    # cosine latitude weights for area weighting
    return np.cos(np.deg2rad(lat))


def compute_anomalies(x: np.ndarray, time: np.ndarray, clim_years: Tuple[int, int]) -> np.ndarray:
    """Compute anomalies relative to a fixed climatology window.

    Parameters
    ----------
    x : np.ndarray
        Array with shape (time, lat, lon).
    time : np.ndarray
        Array of years with shape (time,).
    clim_years : tuple
        (start_year, end_year) inclusive.
    """
    y0, y1 = clim_years
    mask = (time >= y0) & (time <= y1)
    if not np.any(mask):
        raise ValueError("No samples within climatology years.")
    clim = x[mask].mean(axis=0)
    return x - clim


def standardize_zscore(x: np.ndarray) -> np.ndarray:
    """Standardize per grid point (time dimension)."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (x - mean) / std


def flatten_field(x: np.ndarray, lat: np.ndarray, area_weighted: bool) -> np.ndarray:
    """Flatten (time, lat, lon) to (nd, nt) with optional area weights."""
    if area_weighted:
        w = _area_weights(lat)[:, None]
        x = x * w[None, :, :]
    nt = x.shape[0]
    nd = x.shape[1] * x.shape[2]
    return x.reshape(nt, nd).T


def preprocess_u850(
    u850: np.ndarray,
    years: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    cfg: PreprocessConfig,
) -> np.ndarray:
    """Main preprocessing pipeline for u850 anomalies.

    Returns
    -------
    np.ndarray
        Array with shape (nd, nt) suitable for SOM input.
    """
    # Domain subset
    lat_mask = (lat >= cfg.lat_min) & (lat <= cfg.lat_max)
    lon_mask = (lon >= cfg.lon_min) & (lon <= cfg.lon_max)
    u = u850[:, lat_mask][:, :, lon_mask]
    lat_sub = lat[lat_mask]
    lon_sub = lon[lon_mask]

    # Anomalies
    u_anom = compute_anomalies(u, years, cfg.climatology_years)

    # Optional detrend placeholder (explicitly not implemented)
    if cfg.detrend:
        raise NotImplementedError("Detrending must be implemented and documented explicitly.")

    # Standardize
    if cfg.standardize.lower() == "zscore":
        u_std = standardize_zscore(u_anom)
    else:
        raise ValueError("Unsupported standardization method.")

    # Flatten to (nd, nt)
    x = flatten_field(u_std, lat_sub, cfg.area_weighted)
    return x


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(preprocess_path: str, domain_path: str | None = None) -> PreprocessConfig:
    raw = {}
    if domain_path:
        raw.update(_load_yaml(domain_path))
    raw.update(_load_yaml(preprocess_path))

    # domain + preprocess keys in one place
    cfg = PreprocessConfig()
    if "lon_min" in raw:
        cfg.lon_min = float(raw["lon_min"])
    if "lon_max" in raw:
        cfg.lon_max = float(raw["lon_max"])
    if "lat_min" in raw:
        cfg.lat_min = float(raw["lat_min"])
    if "lat_max" in raw:
        cfg.lat_max = float(raw["lat_max"])
    if "climatology_years" in raw:
        y0, y1 = raw["climatology_years"].split("-")
        cfg.climatology_years = (int(y0), int(y1))
    if "detrend" in raw:
        cfg.detrend = bool(raw["detrend"])
    if "standardize" in raw:
        cfg.standardize = str(raw["standardize"])
    if "area_weighted" in raw:
        cfg.area_weighted = bool(raw["area_weighted"])

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess u850 for SOM input")
    parser.add_argument("--input", required=True, help="Path to NetCDF file")
    parser.add_argument("--var", default="u850", help="Variable name (must be u850)")
    parser.add_argument("--level", type=float, default=850.0, help="Pressure level for uwnd (hPa)")
    parser.add_argument("--lat-var", default="lat", help="Latitude coord name")
    parser.add_argument("--lon-var", default="lon", help="Longitude coord name")
    parser.add_argument("--time-var", default="time", help="Time coord name")
    parser.add_argument(
        "--season",
        default="JJA",
        help="Season to average (JJA); expects monthly data if provided",
    )
    parser.add_argument(
        "--season-mean",
        action="store_true",
        help="If set, compute seasonal mean per year. Default keeps daily samples within season.",
    )
    parser.add_argument(
        "--config",
        default="configs/preprocess.yaml",
        help="Path to preprocess config YAML",
    )
    parser.add_argument(
        "--domain-config",
        default="configs/domain.yaml",
        help="Path to domain config YAML",
    )
    parser.add_argument("--output", default="data/u850_som_input.npy", help="Output .npy path")
    args = parser.parse_args()

    _check_u850_only(args.var)

    ds = xr.open_dataset(args.input)
    da = ds[args.var]
    if "level" in da.dims or "level" in da.coords:
        da = da.sel(level=args.level)

    # normalize lon to 0..360 if needed for 100E–180E domain
    lon = ds[args.lon_var]
    if float(lon.min()) < 0 and args.lon_var in da.coords:
        da = da.assign_coords({args.lon_var: (lon + 360) % 360}).sortby(args.lon_var)

    if args.season:
        if not hasattr(da[args.time_var].dt, "month"):
            raise ValueError("Time coordinate must be datetime-like for seasonal averaging.")
        # JJA mean per year
        da = da.where(da[args.time_var].dt.month.isin([6, 7, 8]), drop=True)
        if args.season_mean:
            da = da.groupby(f"{args.time_var}.year").mean(args.time_var)

    lat = da[args.lat_var].values
    lon = da[args.lon_var].values
    years = da["year"].values if "year" in da.coords else da[args.time_var].dt.year.values
    u850 = da.values

    cfg = load_config(args.config, args.domain_config)
    x = preprocess_u850(u850, years, lat, lon, cfg)
    np.save(args.output, x)


if __name__ == "__main__":
    main()
