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
    # Find header line containing YEAR MONTH INDEX
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
    # Columns: YR MON TOTAL ClimAdjust ANOM
    df = pd.read_csv(path, sep=r"\s+", header=0)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"YR": "Year", "MON": "Month", "ANOM": "Value"})
    df = df[["Year", "Month", "Value"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df["Date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
    return df.set_index("Date")["Value"]


def seasonal_mean(series: pd.Series, months):
    df = series.to_frame("Value")
    df = df[df.index.month.isin(months)]
    return df.groupby(df.index.year)["Value"].mean()


def main():
    stats_path = RESULTS_DIR / "som_yearly_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {stats_path}. Run run_som_cluster.py first.")

    df_nodes = pd.read_csv(stats_path)
    years = df_nodes["year"].to_numpy()

    node_cols = [f"node_{i}_count" for i in range(1, 10)]
    for col in node_cols:
        if col not in df_nodes.columns:
            raise ValueError(f"Missing {col} in {stats_path}")

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

        # JJA mean per year
        index_series[name] = seasonal_mean(series, [6, 7, 8])

    results = []
    for node in range(1, 10):
        node_series = df_nodes.set_index("year")[f"node_{node}_count"]
        for idx_name, idx_series in index_series.items():
            common_years = np.intersect1d(node_series.index, idx_series.index)
            if len(common_years) < 5:
                continue
            x = node_series.loc[common_years]
            y = idx_series.loc[common_years]
            r, p = stats.pearsonr(x, y)
            results.append(
                {
                    "node": node,
                    "index": idx_name,
                    "correlation": r,
                    "p_value": p,
                    "N": len(common_years),
                }
            )

    out_df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "node_climate_correlations.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    # Timeseries plots (one per index, 3x3 node subplots)
    fig_dir = RESULTS_DIR / "Figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    node_df = df_nodes.set_index("year")[node_cols]

    for idx_name, idx_series in index_series.items():
        common_years = node_df.index.intersection(idx_series.index)
        if len(common_years) < 5:
            continue

        # Normalize index (z-score) for comparability across plots
        idx_norm = (idx_series - idx_series.mean()) / idx_series.std()

        fig, axes = plt.subplots(
            3,
            3,
            figsize=(12, 8),
            layout="constrained",
            sharex=True,
            gridspec_kw={"bottom": 0.25},
        )
        axes = axes.ravel()
        node_vals = node_df.loc[common_years, node_cols]
        y_min = node_vals.min().min()
        y_max = node_vals.max().max()
        pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
        y_limits = (y_min - pad, y_max + pad)
        ax2_list = []
        for node in range(1, 10):
            ax = axes[node - 1]
            ax.plot(
                common_years,
                node_df.loc[common_years, f"node_{node}_count"],
                color="tab:blue",
                linewidth=1.2,
                label="Node count",
            )
            ax2 = ax.twinx()
            ax2_list.append(ax2)
            ax2.plot(
                common_years,
                idx_norm.loc[common_years],
                color="black",
                linewidth=1.2,
                linestyle="--",
                label=f"{idx_name} (JJA, z)",
            )
            ax2.axhline(0, color="black", linewidth=0.8, alpha=0.6)
            ax.set_title(f"Node {node}")
            ax.set_ylim(y_limits)
            if idx_name == "NPGO":
                ax2.set_ylim(-1, 1)
            else:
                ax2.set_ylim(-3, 3)
            ax.grid(True, linestyle="--", alpha=0.4)

        handles, labels = axes[0].get_legend_handles_labels()
        handles2, labels2 = ax2_list[0].get_legend_handles_labels()
        handles += handles2
        labels += labels2
        fig.legend(
            handles,
            labels,
            loc="upper left",
            ncol=1,
            bbox_to_anchor=(1.02, 1.0),
            bbox_transform=fig.transFigure,
        )
        fig.suptitle(f"Node Frequencies vs {idx_name} (JJA)")
        out_fig = fig_dir / f"timeseries_nodes_vs_{idx_name.lower()}.png"
        plt.savefig(out_fig, dpi=300)
        plt.close(fig)

    # Heatmap of correlations (nodes x indices)
    pivot = out_df.pivot(index="node", columns="index", values="correlation")
    plt.figure(figsize=(8, 4), layout="constrained")
    sns.heatmap(pivot, annot=True, cmap="vlag", center=0, fmt=".2f")
    plt.title("Node vs Climate Index Correlations (JJA)")
    out_fig = fig_dir / "heatmap_node_climate_correlations.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
