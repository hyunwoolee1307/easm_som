import numpy as np
import pandas as pd
import xarray as xr

from Scripts import analysis_utils


def _make_da(time, lat, lon, values):
    return xr.DataArray(
        values,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"],
    )


def test_calculate_correlation_perfect_positive():
    time = pd.date_range("2000-01-01", periods=4, freq="YS")
    lat = [0.0, 1.0]
    lon = [10.0, 20.0]

    index = pd.Series([1.0, 2.0, 3.0, 4.0], index=[2000, 2001, 2002, 2003])
    base = index.values.reshape(-1, 1, 1)
    data = base * np.ones((4, 2, 2))

    da = _make_da(time, lat, lon, data)
    corr = analysis_utils.calculate_correlation(da, index)

    assert np.allclose(corr, 1.0)


def test_calculate_correlation_time_mismatch_raises():
    time = pd.date_range("2000-01-01", periods=3, freq="YS")
    lat = [0.0]
    lon = [0.0]

    data = np.arange(3).reshape(3, 1, 1)
    da = _make_da(time, lat, lon, data)

    index = pd.Series([1.0, 2.0, 3.0, 4.0], index=[2000, 2001, 2002, 2003])

    try:
        analysis_utils.calculate_correlation(da, index)
    except ValueError as exc:
        assert "Time dimension mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError was not raised")


def test_calculate_composite_difference_basic():
    years = pd.date_range("2000-01-01", periods=6, freq="YS")
    lat = [0.0]
    lon = [0.0]

    values = np.array([1, 2, 3, 4, 5, 6], dtype=float).reshape(6, 1, 1)
    da = _make_da(years, lat, lon, values)

    # Index: positive years are 2004-2005, negative years are 2000-2001
    index = pd.Series(
        [-2.0, -1.5, 0.0, 0.5, 1.6, 2.0],
        index=[2000, 2001, 2002, 2003, 2004, 2005],
    )

    diff, p_val = analysis_utils.calculate_composite_difference(
        da, index, threshold_std=1.0
    )

    # pos mean = (5+6)/2 = 5.5; neg mean = (1+2)/2 = 1.5; diff = 4.0
    assert np.isclose(diff.values[0, 0], 4.0)
    assert diff.dims == ("lat", "lon")
    assert p_val.shape == diff.shape
