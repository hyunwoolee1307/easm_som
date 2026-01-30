from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT = DATA_DIR / "u850.nc"
OUTPUT = RESULTS_DIR / "som_input_jja.csv.gz"

if not INPUT.exists():
    raise FileNotFoundError(f"Missing {INPUT}")

ds = xr.open_dataset(INPUT)
if "uwnd" not in ds:
    raise ValueError("Variable 'uwnd' not found in u850.nc")

subset = ds["uwnd"].sel(lat=slice(60.0, -10.0), lon=slice(100.0, 180.0))

monthly_clim = subset.groupby("time.month").mean("time")
subset_anom = subset.groupby("time.month") - monthly_clim

subset_lp = subset_anom.rolling(time=10, center=True).mean().dropna(dim="time")
subset_jja = subset_lp.sel(time=subset_lp.time.dt.month.isin([6, 7, 8]))

subset_stacked = subset_jja.stack(space=("lat", "lon"))
subset_valid = subset_stacked.dropna(dim="space", how="any")

X = subset_valid.values  # (time, space)
times = pd.to_datetime(subset_valid.time.values)

# Build DataFrame with time column
cols = [f"f{i+1}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=cols)
df.insert(0, "time", times.strftime("%Y-%m-%d"))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False, compression="gzip")
print(f"Saved {OUTPUT}")
