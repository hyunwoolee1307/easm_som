import logging
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import config
import analysis_utils

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_convection_index(
    ds: xr.DataArray, lat_slice: slice, lon_slice: slice
) -> pd.Series:
    """
    Calculates Area-Averaged JJA Convection (Negative OLR).
    Returns raw values (not yet normalized).
    """
    # 1. Select Region
    ds = ds.sortby("lat").sortby("lon")
    region = ds.sel(lat=lat_slice, lon=lon_slice)

    # 2. Select JJA
    jja_data = region.sel(time=region.time.dt.month.isin([6, 7, 8]))

    # 3. Area Weighted Mean
    weights = np.cos(np.deg2rad(jja_data.lat))
    weights.name = "weights"
    weighted = jja_data.weighted(weights)

    # 4. Annual Mean
    annual_val = weighted.mean(dim=["lat", "lon"]).resample(time="YE").mean()

    # Return Negative OLR (Convection)
    # The anomaly file contains OLR anomalies.
    # Negative Anomaly = Enhanced Convection.
    # So we multiply by -1 to get "Convection Anomaly".
    series = -1 * annual_val.to_series()
    series.index = series.index.year
    return series


def normalize(series: pd.Series) -> pd.Series:
    """Standardizes a series (Z-score)."""
    return (series - series.mean()) / series.std()


def main():
    logger.info("Starting SAMOI-N Rolling Correlation Analysis...")

    # 1. Load Data
    olr_path = Path(config.DATA_FILES["OLR_ANOM"])
    try:
        olr_da = analysis_utils.load_data(olr_path)
    except FileNotFoundError:
        return

    # 2. Define Domains (Based on approx of "Northern India to NE Philippines" vs "Central NIO to SE Philippines")
    # North: 15-30N, 70-130E
    # South: 0-15N, 70-130E

    conv_north = calculate_convection_index(olr_da, slice(15, 30), slice(70, 130))
    conv_south = calculate_convection_index(olr_da, slice(0, 15), slice(70, 130))

    # 3. Calculate SAMOI-N
    # Difference of Normalized values
    samoi_n = normalize(conv_north) - normalize(conv_south)

    # 4. Load U850 Index
    indices_dir = Path(config.RESULTS_DIR) / "Indices"
    u850_index = pd.read_csv(indices_dir / "u850_jja_index.csv", index_col=0).squeeze()

    # Normalize U850 for consistency logic? Correlation is scale invariant, so no need for calculation.

    # 5. Align
    common_years = np.intersect1d(samoi_n.index, u850_index.index)
    idx_samoi = samoi_n.loc[common_years]
    idx_u850 = u850_index.loc[common_years]

    # 6. Rolling Correlation (10y)
    window_years = 10
    rolling_corr = idx_u850.rolling(
        window=window_years, center=True, min_periods=window_years // 2
    ).corr(idx_samoi)

    # 7. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        rolling_corr.index,
        rolling_corr,
        color="teal",
        linewidth=2,
        marker="o",
        label="SAMOI-N vs U850 (10y)",
    )

    plt.axhline(0, color="black", linewidth=1)

    # Significance threshold (approx N=10)
    plt.axhline(0.63, color="red", linestyle=":", linewidth=1, label="Significance")
    plt.axhline(-0.63, color="red", linestyle=":", linewidth=1)

    plt.title(
        f"Running Correlation: U850 JJA Index vs SAMOI-N (Window={window_years}y)"
    )
    plt.ylabel("Pearson Correlation Coefficient")
    plt.ylim(-1, 1)

    ticks = list(range(1991, 2024, 5))
    plt.xticks(ticks)
    plt.xlim(1990, 2024)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    out_path = Path(config.FIGURES_DIR) / "running_correlation_samoi_n_u850.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot: {out_path}")

    # Save CSV
    pd.DataFrame({"Year": rolling_corr.index, "Correlation": rolling_corr}).to_csv(
        Path(config.RESULTS_DIR) / "running_correlation_samoi_n.csv", index=False
    )

    # Overall Corr
    r_total = idx_u850.corr(idx_samoi)
    logger.info(f"Overall Correlation (1991-2023): {r_total:.3f}")

    # 8. Time Series Overlay Plot (Raw Annual JJA)
    plt.figure(figsize=(12, 6))

    # Standardize U850 for comparison scale (SAMOI-N is already diff of norms, roughly scale-free but let's z-score both for visual)
    # SAMOI-N is Diff of Z-scores, so its range is roughly -3 to 3.
    # U850 Index is usually raw m/s or similar. Z-score it.
    u850_std = (idx_u850 - idx_u850.mean()) / idx_u850.std()

    # SAMOI-N is already calculated as Diff of Norms, effectively standardized difference.
    # Let's verify if we want to re-standardize the final difference or just plot.
    # Plotting standardizes series is best for "pattern" comparison.
    samoi_std = (idx_samoi - idx_samoi.mean()) / idx_samoi.std()

    plt.plot(
        samoi_std.index,
        samoi_std,
        label="SAMOI-N (Convection Shift)",
        color="green",
        linewidth=2,
        marker="^",
    )
    plt.plot(
        u850_std.index,
        u850_std,
        label=f"U850 JJA Index (r={r_total:.2f})",
        color="blue",
        linewidth=2,
        marker="o",
        linestyle="--",
    )

    plt.axhline(0, color="gray", linewidth=0.8)
    plt.title("Time Series: U850 JJA Index vs SAMOI-N (1991-2023)")
    plt.ylabel("Standardized Anomaly (σ)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Tiks
    ticks = list(range(1991, 2024, 2))  # Every 2 years for finer grain on TS
    plt.xticks(ticks, rotation=45)
    plt.xlim(1990, 2024)

    ts_out_path = Path(config.FIGURES_DIR) / "timeseries_samoi_n_u850.png"
    plt.savefig(ts_out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot: {ts_out_path}")


if __name__ == "__main__":
    main()
