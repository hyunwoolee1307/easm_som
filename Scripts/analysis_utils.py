import logging
import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Optional, List, Tuple
from pathlib import Path

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(file_path: Path) -> xr.DataArray:
    """
    Loads a NetCDF file and returns the first variable (or 'olr' if present).

    Args:
        file_path (Path): Path to the NetCDF file.

    Returns:
        xr.DataArray: The loaded DataArray.
    """
    logger.info(f"Loading data from {file_path}...")
    try:
        ds = xr.open_dataset(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

    var_name = list(ds.data_vars)[0]
    if "olr" in ds.data_vars:
        var_name = "olr"

    logger.info(f"Using variable: {var_name}")
    return ds[var_name]


def calculate_correlation(data: xr.DataArray, index_series: pd.Series) -> np.ndarray:
    """
    Calculates Pearson correlation between a 3D DataArray (time, lat, lon) and a time series.

    Args:
        data (xr.DataArray): Spatio-temporal data (time, lat, lon).
        index_series (pd.Series): Time series data (index).

    Returns:
        np.ndarray: 2D array of correlation coefficients.
    """
    data_values = data.values
    index_values = index_series.values

    # Check dimensions
    if data_values.shape[0] != index_values.shape[0]:
        raise ValueError(
            f"Time dimension mismatch: Data={data_values.shape[0]}, Index={index_values.shape[0]}"
        )

    # Center data
    data_mean = np.mean(data_values, axis=0)
    index_mean = np.mean(index_values)

    data_centered = data_values - data_mean
    index_centered = index_values - index_mean

    # Covariance
    numerator = np.sum(data_centered * index_centered[:, None, None], axis=0)

    # Denominator
    data_sq_sum = np.sum(data_centered**2, axis=0)
    index_sq_sum = np.sum(index_centered**2)

    denominator = np.sqrt(data_sq_sum * index_sq_sum)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        r = numerator / denominator

    return r


def plot_global_correlation(
    da_corr: xr.DataArray, title: str, output_path: Path
) -> None:
    """
    Plots a global correlation map and saves it to the specified path.

    Args:
        da_corr (xr.DataArray): 2D DataArray of correlation values.
        title (str): Output plot title.
        output_path (Path): File path to save the plot.
    """
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_global()

    im = da_corr.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=-0.8,
        vmax=0.8,
        cbar_kwargs={"label": "Correlation Coefficient"},
    )

    ax.add_feature(cfeature.COASTLINE)
    ax.set_title(title)

    gl = ax.gridlines(draw_labels=True, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {output_path}")


def calculate_composite_difference(
    data: xr.DataArray, index_series: pd.Series, threshold_std: float = 1.0
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Calculates the composite difference (Positive Phase - Negative Phase) and its significance.

    Args:
        data (xr.DataArray): Spatio-temporal data (time, lat, lon).
        index_series (pd.Series): Standardized index time series.
        threshold_std (float): Threshold in standard deviations to define phases.

    Returns:
        Tuple[xr.DataArray, xr.DataArray]:
            - Composite difference (Pos - Neg)
            - P-values from two-tailed t-test
    """
    # Align data
    common_years = np.intersect1d(data.time.dt.year, index_series.index)
    data_sel = data.sel(time=data.time.dt.year.isin(common_years))
    index_sel = index_series.loc[common_years]

    # Define phases
    pos_years = index_sel[index_sel >= threshold_std].index
    neg_years = index_sel[index_sel <= -threshold_std].index

    logger.info(
        f"Composite Analysis: {len(pos_years)} Positive years, {len(neg_years)} Negative years (Threshold={threshold_std}std)"
    )

    pos_data = data_sel.sel(time=data_sel.time.dt.year.isin(pos_years))
    neg_data = data_sel.sel(time=data_sel.time.dt.year.isin(neg_years))

    # Calculate means
    pos_mean = pos_data.mean(dim="time")
    neg_mean = neg_data.mean(dim="time")

    diff = pos_mean - neg_mean

    # T-test (ind)
    # scipy.stats.ttest_ind returns t-statistic and p-value. We only need p-value.
    # We need to pass arrays shape (n_samples, lat, lon)
    t_stat, p_val = stats.ttest_ind(
        pos_data.values,
        neg_data.values,
        axis=0,
        equal_var=False,  # Welch's t-test
        nan_policy="omit",
    )

    # Convert p_val back to DataArray
    p_val_da = xr.DataArray(p_val, coords=diff.coords, dims=diff.dims)

    return diff, p_val_da


def plot_composite_map(
    diff: xr.DataArray,
    p_val: xr.DataArray,
    title: str,
    output_path: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdBu_r",
) -> None:
    """
    Plots a composite difference map with stippling for significance (p < 0.05).

    Args:
        diff (xr.DataArray): Composite difference data.
        p_val (xr.DataArray): P-values data.
        title (str): Output plot title.
        output_path (Path): File path to save the plot.
        vmin, vmax (float): Colorbar limits.
        cmap (str): Colormap name.
    """
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_global()

    # Calculate sensible vmin/vmax if not provided
    if vmin is None or vmax is None:
        abs_max = np.nanmax(np.abs(diff.values))
        vmin = -abs_max
        vmax = abs_max

    im = diff.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kwargs={"label": "Anomaly Difference"},
    )

    # Stippling for significance (p < 0.05)
    # Use hatch='...' for stippling on significant areas
    sig_mask = p_val < 0.05

    # Since cartopy and matplotlib shading can be tricky with Cyclic points,
    # we just plot the hatch on top where p < 0.05.
    # We use contourf with 'none' color and hatches.
    if sig_mask.any():
        try:
            ax.contourf(
                diff.lon,
                diff.lat,
                p_val,
                levels=[0, 0.05],
                colors="none",
                hatches=["..."],
                transform=ccrs.PlateCarree(),
            )
        except Exception as e:
            logger.warning(f"Could not plot stippling: {e}")

    ax.add_feature(cfeature.COASTLINE)
    ax.set_title(title)

    gl = ax.gridlines(draw_labels=True, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved composite plot: {output_path}")
