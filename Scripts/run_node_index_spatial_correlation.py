import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
FIG_DIR = RESULTS_DIR / "Figures"

P_THRESHOLD = 0.05


def parse_wide_year_months(path: Path, skiprows=0):
    df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=skiprows)
    df = df[df.iloc[:, 0].apply(lambda v: str(v).isdigit())]
    df = df.iloc[:, :13]
    df.columns = ["Year"] + [str(i) for i in range(1, 13)]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.set_index("Year")
    return df


def to_monthly_series_from_wide(df_wide, missing_threshold=-900):
    records = []
    for year in df_wide.index:
        for month in range(1, 13):
            val = df_wide.loc[year, str(month)]
            if pd.isna(val):
                continue
            if val <= missing_threshold:
                continue
            records.append(
                {"Date": pd.Timestamp(year=int(year), month=month, day=1), "Value": val}
            )
    return pd.DataFrame(records).set_index("Date")["Value"]


def parse_wp_index(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if re.match(r"\s*YEAR\s+MONTH\s+INDEX", line):
            header_idx = i
            break
    skiprows = header_idx + 1 if header_idx is not None else 0

    df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=skiprows)
    df = df[df.iloc[:, 0].apply(lambda v: str(v).isdigit())]
    df = df.iloc[:, :3]
    df.columns = ["Year", "Month", "Value"]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[df["Value"] > -99.0]
    df["Date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
    return df.set_index("Date")["Value"]


def parse_nino34(path: Path):
    df = pd.read_csv(path, sep=r"\s+", header=0)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"YR": "Year", "MON": "Month", "ANOM": "Value"})
    df = df[["Year", "Month", "Value"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df["Date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
    return df.set_index("Date")["Value"]


def seasonal_mean(series: pd.Series, months):
    df = series.to_frame("Value")
    df = df[df.index.month.isin(months)]
    return df.groupby(df.index.year)["Value"].mean()


def corr_map(yearly_data: np.ndarray, y: np.ndarray) -> np.ndarray:
    # yearly_data: (n_years, nlat, nlon)
    n = yearly_data.shape[0]
    x = yearly_data.reshape(n, -1)
    valid = np.all(np.isfinite(x), axis=0) & np.isfinite(y).all()
    x = x[:, valid]

    if x.size == 0:
        out = np.full(yearly_data.shape[1:], np.nan, dtype=float)
        return out

    y_mean = y.mean()
    x_mean = x.mean(axis=0)
    x_centered = x - x_mean
    y_centered = y - y_mean

    cov = (x_centered * y_centered[:, None]).sum(axis=0) / (n - 1)
    stdx = x_centered.std(axis=0, ddof=1)
    stdy = y_centered.std(ddof=1)
    denom = stdx * stdy
    r = np.full(x.shape[1], np.nan, dtype=float)
    ok = denom > 0
    r[ok] = cov[ok] / denom[ok]

    out = np.full(yearly_data.shape[1:], np.nan, dtype=float)
    out.reshape(-1)[valid] = r
    return out


def spatial_corr(a: np.ndarray, b: np.ndarray):
    a_flat = a.ravel()
    b_flat = b.ravel()
    mask = np.isfinite(a_flat) & np.isfinite(b_flat)
    if mask.sum() < 3:
        return np.nan, int(mask.sum())
    a_sel = a_flat[mask]
    b_sel = b_flat[mask]
    a_mean = a_sel.mean()
    b_mean = b_sel.mean()
    a_center = a_sel - a_mean
    b_center = b_sel - b_mean
    denom = a_center.std(ddof=1) * b_center.std(ddof=1)
    if denom == 0:
        return np.nan, int(mask.sum())
    r = (a_center * b_center).sum() / (len(a_sel) - 1) / denom
    return r, int(mask.sum())


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sig_path = RESULTS_DIR / "node_climate_correlations.csv"
    if not sig_path.exists():
        raise FileNotFoundError(f"Missing {sig_path}. Run run_node_climate_correlations.py first.")

    sig_df = pd.read_csv(sig_path)
    sig_df = sig_df[sig_df["p_value"] < P_THRESHOLD].copy()
    if sig_df.empty:
        print("No significant node-index pairs found.")
        return

    node_csv = RESULTS_DIR / "som_neuron_indices_jja.csv"
    if not node_csv.exists():
        raise FileNotFoundError(f"Missing {node_csv}. Run run_som_cluster.py first.")

    node_df = pd.read_csv(node_csv)
    node_df["time"] = pd.to_datetime(node_df["time"])
    node_df = node_df.sort_values("time")
    times = node_df["time"].to_numpy()
    node_ids = node_df["node_id"].to_numpy()

    indices = {
        "PDO": ("ersst.v5.pdo.dat", "wide", 1),
        "DMI": ("dmi.had.long.data", "wide", 1),
        "NPGO": ("npgo.data", "wide", 1),
        "WP": ("wp_index.txt", "long", 0),
        "NINO34": ("detrend.nino34.ascii", "nino", 0),
    }

    index_series = {}
    for name, (fname, fmt, skiprows) in indices.items():
        path = DATA_DIR / fname
        if not path.exists():
            continue
        if fmt == "wide":
            df_wide = parse_wide_year_months(path, skiprows=skiprows)
            series = to_monthly_series_from_wide(df_wide)
        elif fmt == "long":
            series = parse_wp_index(path)
        elif fmt == "nino":
            series = parse_nino34(path)
        else:
            continue
        index_series[name] = seasonal_mean(series, [6, 7, 8])

    configs = [
        ("U850", DATA_DIR / "uwnd850_anom.nc"),
    ]

    # Precompute node composites and index correlation maps per variable
    node_composites = {}
    index_maps = {}

    for var_name, path in configs:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        ds = xr.open_dataset(path)
        var = list(ds.data_vars)[0]
        da = ds[var]

        # Domain: 0-180E, 20S-60N
        da = da.sortby("lat").sortby("lon")
        da = da.sel(lon=slice(0, 180), lat=slice(-20, 60))

        # Node composites (daily anomalies at SOM times)
        da_sel = da.sel(time=times)
        data = da_sel.values
        comp = {}
        for node in range(1, 10):
            idx = np.where(node_ids == node)[0]
            if len(idx) == 0:
                comp[node] = np.full(data.shape[1:], np.nan, dtype=float)
            else:
                comp[node] = np.nanmean(data[idx, :, :], axis=0)
        node_composites[var_name] = comp

        # Index correlation maps (yearly JJA mean vs index)
        da_jja = da.where(da.time.dt.month.isin([6, 7, 8]), drop=True)
        yearly = da_jja.groupby("time.year").mean("time")
        years = yearly["year"].values.astype(int)
        index_maps[var_name] = {}
        for idx_name, idx_series in index_series.items():
            common_years = np.intersect1d(years, idx_series.index.values.astype(int))
            if len(common_years) < 5:
                continue
            y = idx_series.loc[common_years].to_numpy(dtype=float)
            y = (y - y.mean()) / y.std()
            yearly_sel = yearly.sel(year=common_years)
            corr = corr_map(yearly_sel.values, y)
            index_maps[var_name][idx_name] = corr

        ds.close()

    # Build summary table and per-pair plots
    rows = []
    for _, row in sig_df.iterrows():
        node = int(row["node"])
        idx_name = row["index"]

        pair_out = FIG_DIR / f"node_index_spatial_corr_u850_node{node}_{idx_name}.png"
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout="constrained")
        var_name = "U850"
        comp = node_composites[var_name][node]
        idx_map = index_maps.get(var_name, {}).get(idx_name)
        if idx_map is None:
            ax.set_title(f"{var_name}: no map")
            ax.axis("off")
            r = np.nan
            n = 0
        else:
            r, n = spatial_corr(comp, idx_map)
            ax.scatter(idx_map.ravel(), comp.ravel(), s=4, alpha=0.3, color="tab:blue")
            ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
            ax.axvline(0, color="gray", linewidth=0.8, alpha=0.6)
            ax.set_title(f"{var_name} r={r:.2f} (n={n})")
            ax.set_xlabel("Index corr map")
            ax.set_ylabel("Node composite")

        rows.append(
            {
                "node": node,
                "index": idx_name,
                "variable": var_name,
                "spatial_corr": r,
                "n_grid": n,
            }
        )

        fig.suptitle(f"Spatial Correlation: Node {node} vs {idx_name}", fontsize=12)
        fig.savefig(pair_out, dpi=300)
        plt.close(fig)

    out_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "node_index_spatial_correlations.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
