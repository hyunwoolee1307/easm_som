import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
FIG_DIR = RESULTS_DIR / "Figures"

BASE_SEASON = "JJA"
LAGS = [-2, -1, 0, 1, 2]
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


def seasonal_mean(series: pd.Series, season: str) -> pd.Series:
    if season == "DJF":
        df = series.to_frame("Value")
        df = df[df.index.month.isin([12, 1, 2])].copy()
        # Assign Dec to next year to keep DJF labeled by Jan/Feb year
        years = df.index.year.to_numpy()
        years = np.where(df.index.month == 12, years + 1, years)
        df["SeasonYear"] = years
        return df.groupby("SeasonYear")["Value"].mean()

    months = {
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }[season]
    df = series.to_frame("Value")
    df = df[df.index.month.isin(months)]
    return df.groupby(df.index.year)["Value"].mean()


def lag_to_season(lag: int) -> str:
    # Relative to base JJA season of year Y
    mapping = {-2: "DJF", -1: "MAM", 0: "JJA", 1: "SON", 2: "DJF"}
    return mapping[lag]


def season_year_shift(lag: int) -> int:
    # DJF at +2 corresponds to DJF of year Y+1
    return 1 if lag == 2 else 0


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    stats_path = RESULTS_DIR / "som_yearly_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {stats_path}. Run run_som_cluster.py first.")

    df_nodes = pd.read_csv(stats_path)
    node_cols = [f"node_{i}_count" for i in range(1, 10)]
    node_df = df_nodes.set_index("year")[node_cols]

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
        index_series[name] = series

    results = []
    for idx_name, series in index_series.items():
        seasonal = {
            s: seasonal_mean(series, s) for s in ["DJF", "MAM", "JJA", "SON"]
        }
        for lag in LAGS:
            season = lag_to_season(lag)
            shift = season_year_shift(lag)
            idx_season = seasonal[season].copy()
            if shift != 0:
                idx_season.index = idx_season.index + shift

            for node in range(1, 10):
                node_series = node_df[f"node_{node}_count"]
                common_years = np.intersect1d(node_series.index, idx_season.index)
                if len(common_years) < 5:
                    continue
                x = node_series.loc[common_years]
                y = idx_season.loc[common_years]
                r, p = stats.pearsonr(x, y)
                results.append(
                    {
                        "node": node,
                        "index": idx_name,
                        "lag": lag,
                        "season": season,
                        "correlation": r,
                        "p_value": p,
                        "N": len(common_years),
                    }
                )

    out_df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "node_climate_lagged_correlations.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    # Plot: per index heatmap of lag vs node (significant only)
    for idx_name in out_df["index"].unique():
        sub = out_df[out_df["index"] == idx_name]
        pivot = sub.pivot(index="node", columns="lag", values="correlation")
        pvals = sub.pivot(index="node", columns="lag", values="p_value")

        plt.figure(figsize=(7, 4), layout="constrained")
        mask = pvals >= P_THRESHOLD
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            mask=mask,
            cbar_kws={"label": "Correlation"},
        )
        plt.title(f"{idx_name}: Lagged Correlations (JJA base, p < {P_THRESHOLD})")
        plt.xlabel("Lag (seasons)")
        plt.ylabel("Node")
        out_fig = FIG_DIR / f"heatmap_lagged_correlations_{idx_name.lower()}.png"
        plt.savefig(out_fig, dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
