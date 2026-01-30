import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import som

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"

SEED = 42
M1 = 3
M2 = 3
ALPHA0 = 0.5
ALPHAMIN = 0.05
TAUA = 1000.0
TAUS = 500.0

SIGMA0_GRID = [1.5, 2.0, 2.5]
NTER_GRID = [2000, 3000, 4000]


def load_preprocessed_u850():
    ds = xr.open_dataset(DATA_DIR / "u850.nc")
    da = ds["uwnd"]

    # Domain
    lat_slice = slice(60.0, -10.0)
    lon_slice = slice(100.0, 180.0)
    subset = da.sel(lat=lat_slice, lon=lon_slice)

    # Monthly anomaly
    monthly_clim = subset.groupby("time.month").mean("time")
    subset_anom = subset.groupby("time.month") - monthly_clim

    # 10-day low-pass BEFORE JJA selection
    subset_lp = subset_anom.rolling(time=10, center=True).mean().dropna(dim="time")

    # JJA only
    subset_jja = subset_lp.sel(time=subset_lp.time.dt.month.isin([6, 7, 8]))

    subset_stacked = subset_jja.stack(space=("lat", "lon"))
    subset_valid = subset_stacked.dropna(dim="space", how="any")
    X = subset_valid.values.T
    nd, nt = X.shape
    return X, nd, nt


def normalize_columns(x):
    x = x.copy()
    for t in range(x.shape[1]):
        v = x[:, t]
        n = np.linalg.norm(v)
        if n > 1e-12:
            x[:, t] = v / n
    return x


def main():
    X, nd, nt = load_preprocessed_u850()
    X_norm = normalize_columns(X)

    results = []
    for sigma0 in SIGMA0_GRID:
        for nter in NTER_GRID:
            weights = som.Som(
                x=X.copy(),
                nd=nd,
                nt=nt,
                m1=M1,
                m2=M2,
                nter=nter,
                alpha0=ALPHA0,
                taua=TAUA,
                alphamin=ALPHAMIN,
                sigma0=sigma0,
                taus=TAUS,
                normalize_weights=True,
                seed=SEED,
            )
            _, _, _, qerror, _ = som.cluster(X_norm, weights, nd, nt, M1, M2)
            results.append(
                {
                    "sigma0": sigma0,
                    "nter": nter,
                    "qerror": qerror,
                }
            )

    df = pd.DataFrame(results).sort_values("qerror")
    out_path = RESULTS_DIR / "som_param_search.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(df.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
