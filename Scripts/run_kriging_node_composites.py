import math
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy import optimize
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
FIG_DIR = RESULTS_DIR / "Figures"

# Domain
LON_MIN, LON_MAX = 0, 180
LAT_MIN, LAT_MAX = -20, 60

# Sampling / grid
SAMPLE_SIZE = 500
GRID_STEP_DEG = 5
RNG = np.random.default_rng(42)


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance (km). Inputs in degrees."""
    r = 6371.0
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return r * c


def empirical_variogram(lats, lons, values, n_bins=12):
    n = len(values)
    idx = np.triu_indices(n, k=1)
    lat1 = lats[idx[0]]
    lon1 = lons[idx[0]]
    lat2 = lats[idx[1]]
    lon2 = lons[idx[1]]
    h = haversine_km(lat1, lon1, lat2, lon2)
    gamma = 0.5 * (values[idx[0]] - values[idx[1]]) ** 2

    max_h = np.nanpercentile(h, 95)
    bins = np.linspace(0, max_h, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_idx = np.digitize(h, bins) - 1

    gamma_bin = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = bin_idx == i
        if np.any(mask):
            gamma_bin[i] = np.nanmean(gamma[mask])

    valid = np.isfinite(gamma_bin)
    return bin_centers[valid], gamma_bin[valid]


def spherical_model(h, nugget, sill, rang):
    hr = h / rang
    return np.where(
        h <= rang,
        nugget + (sill - nugget) * (1.5 * hr - 0.5 * hr**3),
        sill,
    )


def exponential_model(h, nugget, sill, rang):
    return nugget + (sill - nugget) * (1.0 - np.exp(-h / rang))


def gaussian_model(h, nugget, sill, rang):
    return nugget + (sill - nugget) * (1.0 - np.exp(-(h / rang) ** 2))


def fit_variogram(h, gamma):
    var = np.nanvar(gamma, ddof=1) if np.isfinite(np.nanvar(gamma)) else 1.0
    hmax = np.nanmax(h)
    p0 = [0.0, max(var, 1e-6), max(hmax * 0.5, 1.0)]
    bounds = ([0.0, 1e-6, 1e-3], [np.inf, np.inf, np.inf])

    models = {
        "spherical": spherical_model,
        "exponential": exponential_model,
        "gaussian": gaussian_model,
    }

    results = []
    for name, func in models.items():
        try:
            params, _ = optimize.curve_fit(func, h, gamma, p0=p0, bounds=bounds, maxfev=10000)
            fit = func(h, *params)
            sse = np.nansum((gamma - fit) ** 2)
            results.append((name, params, sse))
        except Exception:
            continue

    if not results:
        raise RuntimeError("Variogram fit failed")

    best = min(results, key=lambda r: r[2])
    return best[0], best[1], results


def variogram_func(name):
    if name == "spherical":
        return spherical_model
    if name == "exponential":
        return exponential_model
    if name == "gaussian":
        return gaussian_model
    raise ValueError("Unknown model")


def ordinary_kriging(lats, lons, values, model_name, params, target_lats, target_lons):
    n = len(values)
    nugget, sill, rang = params

    # Build covariance matrix
    lat1 = lats.reshape(-1, 1)
    lon1 = lons.reshape(-1, 1)
    lat2 = lats.reshape(1, -1)
    lon2 = lons.reshape(1, -1)
    h = haversine_km(lat1, lon1, lat2, lon2)
    gamma = variogram_func(model_name)(h, nugget, sill, rang)
    cov = sill - gamma
    cov[np.diag_indices_from(cov)] = sill - nugget

    # Kriging system
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = cov
    K[:n, -1] = 1.0
    K[-1, :n] = 1.0
    K[-1, -1] = 0.0

    # Regularize
    K[:n, :n] += np.eye(n) * 1e-6

    K_inv = np.linalg.inv(K)

    # Target grid
    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)
    preds = np.zeros_like(lon_grid, dtype=float)

    for i in range(lat_grid.shape[0]):
        for j in range(lat_grid.shape[1]):
            ht = haversine_km(lats, lons, lat_grid[i, j], lon_grid[i, j])
            gamma_t = variogram_func(model_name)(ht, nugget, sill, rang)
            cov_t = sill - gamma_t
            b = np.zeros(n + 1)
            b[:n] = cov_t
            b[-1] = 1.0
            w = K_inv @ b
            preds[i, j] = np.dot(w[:n], values)

    return lon_grid, lat_grid, preds


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
        (DATA_DIR / "sst_anom_regridded.nc", "SST", "anom"),
        (DATA_DIR / "olr_anom2.nc", "OLR", "olr"),
        (DATA_DIR / "uwnd850_anom.nc", "U850", "uwnd"),
    ]

    # First pass: fit all models and collect SSE per model
    fit_cache = {}
    sse_by_model = {"spherical": [], "exponential": [], "gaussian": []}

    for path, var_label, var_name_hint in configs:
        ds = xr.open_dataset(path)
        var = var_name_hint if var_name_hint in ds.data_vars else list(ds.data_vars)[0]
        da = ds[var]

        da = da.sortby("lat").sortby("lon")
        da = da.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
        da_sel = da.sel(time=times)

        lats = da_sel.lat.values
        lons = da_sel.lon.values

        for node in range(1, 10):
            mask = node_ids == node
            if np.sum(mask) < 2:
                continue

            comp = da_sel.isel(time=mask).mean(dim="time")
            values = comp.values

            lon_grid, lat_grid = np.meshgrid(lons, lats)
            vals = values.flatten()
            latf = lat_grid.flatten()
            lonf = lon_grid.flatten()

            valid = np.isfinite(vals)
            vals = vals[valid]
            latf = latf[valid]
            lonf = lonf[valid]

            n_points = min(SAMPLE_SIZE, len(vals))
            idx = RNG.choice(len(vals), size=n_points, replace=False)
            s_vals = vals[idx]
            s_lats = latf[idx]
            s_lons = lonf[idx]

            h, gamma = empirical_variogram(s_lats, s_lons, s_vals, n_bins=12)
            _, _, all_fits = fit_variogram(h, gamma)

            fit_cache[(var_label, node)] = {
                "s_vals": s_vals,
                "s_lats": s_lats,
                "s_lons": s_lons,
                "h": h,
                "gamma": gamma,
                "fits": {name: (params, sse) for name, params, sse in all_fits},
            }

            for name, _, sse in all_fits:
                sse_by_model[name].append(sse)

        ds.close()

    avg_sse = {
        name: (np.mean(vals) if vals else np.inf) for name, vals in sse_by_model.items()
    }
    best_model = min(avg_sse.items(), key=lambda kv: kv[1])[0]
    print(f"Selected global variogram model: {best_model}")

    summary_rows = []
    for path, var_label, var_name_hint in configs:
        ds = xr.open_dataset(path)
        var = var_name_hint if var_name_hint in ds.data_vars else list(ds.data_vars)[0]
        da = ds[var]

        da = da.sortby("lat").sortby("lon")
        da = da.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
        da_sel = da.sel(time=times)

        lats = da_sel.lat.values
        lons = da_sel.lon.values

        for node in range(1, 10):
            cache = fit_cache.get((var_label, node))
            if cache is None:
                continue

            s_vals = cache["s_vals"]
            s_lats = cache["s_lats"]
            s_lons = cache["s_lons"]
            h = cache["h"]
            gamma = cache["gamma"]
            params, _ = cache["fits"][best_model]

            # Plot variogram (global best model only)
            plt.figure(figsize=(6, 4), layout="constrained")
            plt.scatter(h, gamma, color="black", label="Empirical")
            hs = np.linspace(np.min(h), np.max(h), 200)
            fit = variogram_func(best_model)(hs, *params)
            plt.plot(hs, fit, label=best_model)
            plt.title(f"{var_label} Node {node} Variogram ({best_model})")
            plt.xlabel("Distance (km)")
            plt.ylabel("Semivariance")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.4)
            out_vario = FIG_DIR / f"variogram_{var_label.lower()}_node_{node}.png"
            plt.savefig(out_vario, dpi=300)
            plt.close()

            # Kriging (BLUE - ordinary kriging)
            tgt_lats = np.arange(LAT_MIN, LAT_MAX + 0.1, GRID_STEP_DEG)
            tgt_lons = np.arange(LON_MIN, LON_MAX + 0.1, GRID_STEP_DEG)
            lon_k, lat_k, pred = ordinary_kriging(
                s_lats, s_lons, s_vals, best_model, params, tgt_lats, tgt_lons
            )

            fig = plt.figure(figsize=(7, 4), layout="constrained")
            ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            cf = ax.pcolormesh(
                lon_k,
                lat_k,
                pred,
                transform=ccrs.PlateCarree(),
                cmap="RdBu_r",
            )
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.set_title(f"{var_label} Node {node} Kriged Field ({best_model})")
            plt.colorbar(cf, ax=ax, orientation="vertical", shrink=0.8)
            out_krig = FIG_DIR / f"kriging_{var_label.lower()}_node_{node}.png"
            plt.savefig(out_krig, dpi=300)
            plt.close(fig)

            summary_rows.append(
                {
                    "variable": var_label,
                    "node": node,
                    "model": best_model,
                    "nugget": params[0],
                    "sill": params[1],
                    "range_km": params[2],
                }
            )

        ds.close()

    out_df = pd.DataFrame(summary_rows)
    out_path = RESULTS_DIR / "kriging_variogram_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
