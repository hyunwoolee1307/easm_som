import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import config

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main execution entry point for climate index creation.
    Orchestrates the generation of Cluster Climate Index and U850 JJA Index.
    """
    logger.info("Starting Climate Index Creation...")
    try:
        # 1. Create Cluster-based Climate Index
        create_cluster_index()

        # 2. Create U850 JJA Monsoon Index
        create_u850_index()

        logger.info("All indices created successfully.")
    except Exception as e:
        logger.error(f"Failed to create indices: {e}")
        raise


def create_cluster_index() -> None:
    """
    Generates and saves the Cluster Climate Index.

    Logic:
    1. Load cluster counts CSV.
    2. Normalize Cluster 1 and Cluster 5 counts (Z-score).
    3. Calculate Index = Z(Cluster 1) - Z(Cluster 5).
    4. Save to CSV and visualize.
    """
    logger.info("Creating Cluster Climate Index...")

    # 1. Data Loading
    file_path: Path = config.DATA_FILES["CLUSTER_COUNTS"]
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist.")

    cluster_counts_df: pd.DataFrame = pd.read_csv(file_path)

    # Handle Index Column (Year)
    if "year" in cluster_counts_df.columns:
        cluster_counts_df = cluster_counts_df.set_index("year")
    elif "Unnamed: 0" in cluster_counts_df.columns:
        cluster_counts_df = cluster_counts_df.rename(
            columns={"Unnamed: 0": "year"}
        ).set_index("year")

    # 2. Standardization (Z-score)
    try:
        # Cluster 1: Positive Phase Representative
        cluster1_raw: pd.Series = cluster_counts_df["1"]
        cluster1_std: pd.Series = (
            cluster1_raw - cluster1_raw.mean()
        ) / cluster1_raw.std()

        # Cluster 5: Negative Phase Representative
        cluster5_raw: pd.Series = cluster_counts_df["5"]
        cluster5_std: pd.Series = (
            cluster5_raw - cluster5_raw.mean()
        ) / cluster5_raw.std()
    except KeyError:
        logger.error("Columns '1' or '5' not found in cluster counts file.")
        raise

    # 3. Index Calculation
    climate_index: pd.Series = cluster1_std - cluster5_std

    # 4. Saving
    df_out = pd.DataFrame({"Climate_Index": climate_index})
    output_csv = config.INDEX_FILES["CLUSTER_INDEX"]
    df_out.to_csv(output_csv)
    logger.info(f"Saved to {output_csv}")

    # 5. Visualization
    output_plot = config.FIGURES_DIR / "cluster_index_timeseries.png"
    plot_timeseries(
        df=df_out,
        col_name="Climate_Index",
        title="Cluster Climate Index (C1 - C5)",
        output_path=output_plot,
    )


def create_u850_index() -> None:
    """
    Generates and saves the U850 JJA Monsoon Index.

    Logic:
    1. Load U850 Anomaly NetCDF.
    2. Filter for Summer (JJA) months.
    3. Calculate area-averaged zonal wind for North (20-40N) and South (0-20N) boxes.
    4. Index = standardized(North - South).
    """
    logger.info("Creating U850 JJA Index...")

    # 1. Load Data
    file_path: Path = config.DATA_FILES["U850_ANOM"]
    u850_dataset: xr.Dataset = xr.open_dataset(file_path)
    var_name = list(u850_dataset.data_vars)[0]
    u850_data_array: xr.DataArray = u850_dataset[var_name]

    # 2. Seasonal Formatting (JJA Selection)
    # Select June, July, August
    jja_data_array: xr.DataArray = u850_data_array.sel(
        time=u850_data_array.time.dt.month.isin([6, 7, 8])
    )

    # Calculate Yearly JJA Mean
    jja_yearly_mean: xr.DataArray = jja_data_array.resample(time="YE").mean()

    # 3. Regional Averaging
    # North Box: 110-150E, 20-40N
    north_box: xr.DataArray = jja_yearly_mean.sel(
        lon=slice(110, 150), lat=slice(20, 40)
    )
    if north_box.size == 0:
        # Handle potential lat order flip
        north_box = jja_yearly_mean.sel(lon=slice(110, 150), lat=slice(40, 20))

    # South Box: 110-150E, 0-20N
    south_box: xr.DataArray = jja_yearly_mean.sel(lon=slice(110, 150), lat=slice(0, 20))
    if south_box.size == 0:
        south_box = jja_yearly_mean.sel(lon=slice(110, 150), lat=slice(20, 0))

    # Calculate Spatial Means
    north_mean_ts: xr.DataArray = north_box.mean(dim=["lat", "lon"])
    south_mean_ts: xr.DataArray = south_box.mean(dim=["lat", "lon"])

    # 4. Index Calculation (Shear)
    raw_shear_index = north_mean_ts - south_mean_ts

    # Standardize
    standardized_index = (
        raw_shear_index - raw_shear_index.mean()
    ) / raw_shear_index.std()

    # 5. Saving
    years = standardized_index["time"].dt.year.values
    df_out = pd.DataFrame({"U850_JJA_Index": standardized_index.values}, index=years)
    df_out.index.name = "year"

    output_csv = config.INDEX_FILES["U850_JJA_INDEX"]
    df_out.to_csv(output_csv)
    logger.info(f"Saved to {output_csv}")

    # 6. Visualization
    output_plot = config.FIGURES_DIR / "u850_jja_index_timeseries.png"
    plot_timeseries(
        df=df_out,
        col_name="U850_JJA_Index",
        title="U850 JJA Index (North - South)",
        output_path=output_plot,
    )


def plot_timeseries(
    df: pd.DataFrame, col_name: str, title: str, output_path: Path
) -> None:
    """
    Plots a standardized time series bar chart.

    Args:
        df (pd.DataFrame): DataFrame containing the time series.
        col_name (str): The column name to plot.
        title (str): The title of the plot.
        output_path (Path): Destination path for the image.
    """
    plt.figure(figsize=(10, 5), layout="constrained")
    values = df[col_name]

    # Color coding: Red for positive, Blue for negative
    colors = ["red" if x >= 0 else "blue" for x in values]

    plt.bar(df.index, values, color=colors, alpha=0.7)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Standardized Anomaly (std)")

    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
