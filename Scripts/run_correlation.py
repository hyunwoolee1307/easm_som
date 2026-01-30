import xarray as xr
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

import config
import analysis_utils as utils

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main driver for Global Seasonal Correlation Analysis.

    Workflow:
    1. Load pre-computed climate indices.
    2. Iterate through each environmental variable (SST, OLR, U850).
    3. Resample environmental data to seasonal means.
    4. For each season (DJF, MAM, JJA, SON) and each index:
       - Align years.
       - Calculate Pearson correlation.
       - Generate and save global map plots.
    """
    logger.info("Starting Global Seasonal Correlation Analysis...")

    # 1. Load Indices
    loaded_indices: Dict[str, pd.Series] = {}

    # Load Cluster Climate Index
    if config.INDEX_FILES["CLUSTER_INDEX"].exists():
        cluster_index_df = pd.read_csv(config.INDEX_FILES["CLUSTER_INDEX"])
        # Standardize index naming
        if "year" in cluster_index_df.columns:
            cluster_index_df = cluster_index_df.set_index("year")
        elif "Unnamed: 0" in cluster_index_df.columns:
            cluster_index_df = cluster_index_df.rename(
                columns={"Unnamed: 0": "year"}
            ).set_index("year")

        loaded_indices["Cluster_Index"] = cluster_index_df["Climate_Index"]
    else:
        logger.error("Cluster Index file not found.")

    # Load U850 JJA Index
    if config.INDEX_FILES["U850_JJA_INDEX"].exists():
        u850_index_df = pd.read_csv(config.INDEX_FILES["U850_JJA_INDEX"])
        if "year" in u850_index_df.columns:
            u850_index_df = u850_index_df.set_index("year")

        loaded_indices["U850_JJA_Index"] = u850_index_df["U850_JJA_Index"]
    else:
        logger.error("U850 JJA Index file not found.")

    # 2. Iterate Environmental Variables
    # Filter config to exclude non-environmental files
    env_vars_config = {
        k: v for k, v in config.DATA_FILES.items() if k != "CLUSTER_COUNTS"
    }

    for var_key, file_path in env_vars_config.items():
        var_label = var_key.replace("_ANOM", "")  # Extract Label (e.g., SST, OLR)
        logger.info(f"Processing Variable: {var_label}")

        # Load Data
        env_data_array: xr.DataArray = utils.load_data(file_path)

        # Resample to Seasonal Means (QS-DEC = Quarters starting in Dec)
        logger.info("  Resampling to seasonal means...")
        seasonal_data_array: xr.DataArray = env_data_array.resample(
            time="QS-DEC"
        ).mean()

        # 3. Iterate Seasons
        for season_name, month_start in config.SEASONS.items():
            logger.info(f"    Season: {season_name}")

            # Select specific season months
            season_subset: xr.DataArray = seasonal_data_array.sel(
                time=seasonal_data_array.time.dt.month == month_start
            )

            # Year Alignment Logic
            # DJF logic: Dec 1999 belongs to Winter 2000.
            season_years = season_subset["time"].dt.year.values
            if season_name == "DJF":
                season_years = season_years + 1

            # Re-index DataArray with 'year' coordinate for easy matching
            season_subset_aligned = season_subset.assign_coords(
                year=("time", season_years)
            )
            season_subset_aligned = season_subset_aligned.swap_dims({"time": "year"})

            # 4. Iterate Indices and Correlate
            for index_name, index_series in loaded_indices.items():
                calc_and_plot_correlation(
                    env_data=season_subset_aligned,
                    index_series=index_series,
                    index_name=index_name,
                    var_label=var_label,
                    season_name=season_name,
                )

    logger.info("Analysis Complete.")


def calc_and_plot_correlation(
    env_data: xr.DataArray,
    index_series: pd.Series,
    index_name: str,
    var_label: str,
    season_name: str,
) -> None:
    """
    Helper function to calculate correlation and plot the result.
    Aligns data by year intersection.
    """
    # Align Years
    years_idx = index_series.index.unique()
    years_var = env_data.year.values
    common_years = np.intersect1d(years_idx, years_var)

    if len(common_years) < 5:
        logger.warning(
            f"Too few overlapping years for {index_name} in {season_name}. Skipping."
        )
        return

    # Subset Data
    env_data_subset = env_data.sel(year=common_years)
    index_subset = index_series.loc[common_years]

    # Calculate Correlation
    correlation_grid = utils.calculate_correlation(env_data_subset, index_subset)

    # Reconstruct DataArray for Plotting (Map coords)
    coords = env_data_subset.isel(year=0).coords
    correlation_da = xr.DataArray(
        correlation_grid,
        coords={k: v for k, v in coords.items() if k in ["lat", "lon"]},
        dims=["lat", "lon"],
        name="correlation",
    )

    # Define Output Path
    filename = f"corr_{index_name.lower()}_{var_label.lower()}_{season_name}.png"
    output_path = config.FIGURES_DIR / filename
    plot_title = f"{index_name.replace('_', ' ')} vs {var_label} ({season_name})"

    # Plot
    utils.plot_global_correlation(correlation_da, plot_title, output_path)


if __name__ == "__main__":
    main()
