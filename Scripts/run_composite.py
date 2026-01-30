import logging
import pandas as pd
import xarray as xr
from pathlib import Path
import config
import analysis_utils

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Composite Analysis...")

    # 1. Load Indices
    indices_dir = Path(config.RESULTS_DIR) / "Indices"
    cluster_index_path = indices_dir / "cluster_index.csv"
    u850_jja_index_path = indices_dir / "u850_jja_index.csv"

    if not cluster_index_path.exists() or not u850_jja_index_path.exists():
        logger.error("Indices not found. Please run create_indices.py first.")
        return

    cluster_index = pd.read_csv(
        cluster_index_path, index_col=0, parse_dates=True
    ).squeeze()
    u850_index = pd.read_csv(
        u850_jja_index_path, index_col=0, parse_dates=True
    ).squeeze()

    # Ensure indices are Series with DatetimeIndex or Year index?
    # The csv has "Year" as index probably if saved from previous steps.
    # Let's inspect how they were saved. Usually just years 1991, 1992...
    # If read_csv parses dates, it might be YYYY-01-01.
    # Let's assume the index is just the Year integer for simplicity in matching,
    # but based on previous `create_indices.py`, let's check.
    # Actually, `pd.read_csv(..., index_col=0)` with just years usually gives Int64Index.
    # Let's convert to simple Year Integer Index to be safe for matching.

    # Correction: In create_indices.py, if we saved it with Year as index.
    # Let's handle both DatetimeIndex and IntIndex.
    if isinstance(cluster_index.index, pd.DatetimeIndex):
        cluster_index.index = cluster_index.index.year
    if isinstance(u850_index.index, pd.DatetimeIndex):
        u850_index.index = u850_index.index.year

    indices = {"Cluster_Index": cluster_index, "U850_JJA_Index": u850_index}

    # 2. Define Variables
    variables = {
        "SST": config.DATA_FILES["SST_ANOM"],
        "OLR": config.DATA_FILES["OLR_ANOM"],
        "U850": config.DATA_FILES["U850_ANOM"],
    }

    # 3. Perform Composite Analysis (Focus on JJA Season)
    target_season = "JJA"
    months = [6, 7, 8]  # JJA

    if isinstance(
        months, int
    ):  # Handle single month case if config uses it, but usually list
        months = [months]

    output_dir = Path(config.RESULTS_DIR) / "Figures"

    for var_name, var_path in variables.items():
        logger.info(f"Processing Variable: {var_name}")
        try:
            da = analysis_utils.load_data(Path(var_path))
        except FileNotFoundError:
            continue

        # Select JJA months only
        da_season = da.sel(time=da.time.dt.month.isin(months))

        # We need to make sure we don't just take the mean yet if we want to retain years for matching.
        # But `calculate_composite_difference` expects (time, lat, lon).
        # We want to treat one sample = one year's JJA mean.
        # So we group by year and take mean.
        da_season_mean = da_season.resample(time="YE").mean()
        # Fix time coordinate to be just year integers to match index?
        # Or keep as datetime '1991-12-31' etc.
        # `calculate_composite_difference` logic:
        # common_years = np.intersect1d(data.time.dt.year, index_series.index)
        # So it handles datetime in DataArray vs Integer/Year in Index. Excellent.

        for idx_name, idx_series in indices.items():
            logger.info(
                f"  Calculating Composite for {idx_name} vs {var_name} ({target_season})"
            )

            diff, p_val = analysis_utils.calculate_composite_difference(
                da_season_mean,
                idx_series,
                threshold_std=1.0,  # Standard threshold
            )

            # Plot
            plot_title = f"{idx_name} vs {var_name} Composite ({target_season})"
            output_filename = (
                output_dir
                / f"composite_{idx_name.lower()}_{var_name.lower()}_{target_season}.png"
            )

            # Colorbar limits
            vmin, vmax = None, None
            if var_name == "SST":
                vmin, vmax = -0.8, 0.8
            elif var_name == "OLR":
                vmin, vmax = -10, 10
            elif var_name == "U850":
                vmin, vmax = -3, 3

            analysis_utils.plot_composite_map(
                diff,
                p_val,
                title=plot_title,
                output_path=output_filename,
                vmin=vmin,
                vmax=vmax,
            )

    logger.info("Composite Analysis Completed.")


if __name__ == "__main__":
    main()
