import numpy as np
import xarray as xr
from pathlib import Path
import pandas as pd
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
SIGMA0 = 2.0
TAUS = 500.0
NTER = 4000

N_ITER = 1000
RNG = np.random.default_rng(123)


def load_preprocessed_u850():
    ds = xr.open_dataset(DATA_DIR / "u850.nc")
    da = ds["uwnd"]

    lat_slice = slice(60.0, -10.0)
    lon_slice = slice(100.0, 180.0)
    subset = da.sel(lat=lat_slice, lon=lon_slice)

    monthly_clim = subset.groupby("time.month").mean("time")
    subset_anom = subset.groupby("time.month") - monthly_clim

    subset_lp = subset_anom.rolling(time=10, center=True).mean().dropna(dim="time")
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


def compute_qe(X, nd, nt):
    weights = som.Som(
        x=X.copy(),
        nd=nd,
        nt=nt,
        m1=M1,
        m2=M2,
        nter=NTER,
        alpha0=ALPHA0,
        taua=TAUA,
        alphamin=ALPHAMIN,
        sigma0=SIGMA0,
        taus=TAUS,
        normalize_weights=True,
        seed=SEED,
    )
    _, _, _, qerror, _ = som.cluster(X, weights, nd, nt, M1, M2)
    return qerror


def main():
    X, nd, nt = load_preprocessed_u850()
    X_norm = normalize_columns(X)

    # Observed QE on original data
    q_obs = compute_qe(X_norm, nd, nt)

    # Permutation test: shuffle time indices
    q_perm = np.zeros(N_ITER)
    for i in range(N_ITER):
        idx = RNG.permutation(nt)
        X_perm = X_norm[:, idx]
        q_perm[i] = compute_qe(X_perm, nd, nt)

    p_value = (np.sum(q_perm <= q_obs) + 1) / (N_ITER + 1)

    df = pd.DataFrame({"q_perm": q_perm})
    df.to_csv(RESULTS_DIR / "qe_permutation_distribution.csv", index=False)

    summary = {
        "q_obs": q_obs,
        "q_perm_mean": float(np.mean(q_perm)),
        "q_perm_std": float(np.std(q_perm, ddof=1)),
        "p_value": float(p_value),
        "n_iter": N_ITER,
    }

    pd.DataFrame([summary]).to_csv(
        RESULTS_DIR / "qe_significance_summary.csv", index=False
    )

    print("QE significance summary:")
    print(summary)


if __name__ == "__main__":
    main()
