import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import config

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_climate_index(file_path: Path, index_name: str) -> pd.DataFrame:
    """
    Parses various climate index text files into a monthly DataFrame.
    Expected format often: Year Jan Feb ... Dec
    """
    logger.info(f"Parsing {index_name} from {file_path.name}")

    try:
        # Common format: Year + 12 columns. Delimiters vary (spaces).
        # We try reading with whitespace usage.
        # Check rows to skip. Usually headers.

        if "oni.data" in file_path.name:
            # ONI: Year DJF JFM ... - sliding 3 month windows?
            # Actually ONI data is usually Year followed by 12 months (or 3-month running).
            # The inspect showed: 1950 -1.53 -1.34 ...
            # Let's assume standard whitespace with Year col 0.
            df = pd.read_csv(file_path, sep=r"\s+", header=None, skiprows=1)
            # Just take first 13 columns (Year + 12 months) if extra cols exist
            df = df.iloc[:, :13]
            df.columns = ["Year"] + [str(i) for i in range(1, 13)]

        elif "dmi.had.long.data" in file_path.name:
            # DMI: text file.
            df = pd.read_csv(file_path, sep=r"\s+", header=None, skiprows=1)
            # First col might be year.
            # Let's check headers/skip rows dynamically?
            # Assuming simple rectangular data for now based on typical climate data.
            # Only Year + 12 months is standard.
            # Filter out rows that are not years (e.g. headers if failed to skip)
            df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notnull()]
            df = df.iloc[:, :13]
            df.columns = ["Year"] + [str(i) for i in range(1, 13)]

        elif "ersst.v5.pdo.dat" in file_path.name:
            # PDO: often has header lines.
            df = pd.read_csv(file_path, sep=r"\s+", header=0, skiprows=1)
            # Inspect showed: "Year Jan Feb..." in line 1.
            # So header=0 might be "Year".
            # Let's stick to safe parsing: remove non-numeric rows
            df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notnull()]
            df = df.iloc[:, :13]
            df.columns = ["Year"] + [str(i) for i in range(1, 13)]

        elif "npgo.data" in file_path.name:
            # NPGO often: Year Month NPGO
            # It's NOT wide format (Year, Jan...Dec). It is usually Long format.
            # file content inspect step 90:
            # 1950 2025 ... wait that output was chaotic.
            # Let's assume it might be Year Month Value or Year 12cols.
            # Actually Step 90 output looked like: "1950 -2.188 -1.446 ..." -> Looks like Wide.
            df = pd.read_csv(file_path, sep=r"\s+", comment="#", skiprows=1)
            # Filter non-year rows
            df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notnull()]
            df = df.iloc[:, :13]
            df.columns = ["Year"] + [str(i) for i in range(1, 13)]

        elif "norm.nao" in file_path.name or "norm.pna" in file_path.name:
            # NOAA CPC format: Year Month Value... No, CPC is usually Year Month Index.
            # Or "Year  Jan Feb..."
            # If "monthly.b5001.current.ascii", it's likely Year Month Value.
            # Wait, usually these files are "Year Month PNA".
            # Let's assume Long format if 3 columns, Wide if 13.
            try:
                temp = pd.read_csv(
                    file_path, sep=r"\s+", header=None, skiprows=1
                )  # check structure
                cols = temp.shape[1]
                if cols == 3:  # Year Month Value
                    # Pivot to Wide
                    temp.columns = ["Year", "Month", index_name]
                    df = temp.pivot(
                        index="Year", columns="Month", values=index_name
                    ).reset_index()
                    df.columns = ["Year"] + [str(i) for i in range(1, 13)]
                elif cols >= 13:
                    df = temp.iloc[:, :13]
                    df.columns = ["Year"] + [str(i) for i in range(1, 13)]
                else:
                    logger.warning(f"Unknown column count {cols} for {index_name}")
                    return pd.DataFrame()
            except Exception:
                # Retry with different skipping ??
                return pd.DataFrame()

        else:
            # Generic fallback
            df = pd.read_csv(file_path, sep=r"\s+", comment="#", header=None)
            df = df.iloc[:, :13]
            df.columns = ["Year"] + [str(i) for i in range(1, 13)]

        # Melt to Long format: Date, Value
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.set_index("Year")
        df = df[~df.index.isna()]

        # Stack to create Series with MultiIndex (Year, Month)
        # Or even better: create datetime index
        series_list = []
        for year in df.index:
            for month in range(1, 13):
                val = df.loc[year, str(month)]
                if isinstance(val, pd.Series):
                    val = val.mean()
                if pd.notna(val) and val > -900:  # Remove missing values often -999.9
                    series_list.append(
                        {
                            "Date": pd.Timestamp(year=int(year), month=month, day=1),
                            "Value": val,
                        }
                    )

        full_df = pd.DataFrame(series_list).set_index("Date")
        full_df.columns = [index_name]
        return full_df

    except Exception as e:
        logger.error(f"Failed to parse {index_name}: {e}")
        return pd.DataFrame()


def calculate_seasonal_means(
    df: pd.DataFrame, season_months: Dict[str, list]
) -> pd.DataFrame:
    """
    Calculates seasonal means for DJF, MAM, JJA, SON.
    DJF is constructed from Dec(prev) + Jan(curr) + Feb(curr).
    """
    seasonal_data = {}

    # Simple Resampling?
    # For JJA: Year-06-01 to Year-08-31 mean.

    for season, months in season_months.items():
        # Custom resampling
        # We can iterate years.
        means = []
        years = df.index.year.unique().sort_values()

        for y in years:
            if season == "DJF":
                # Dec (y-1), Jan (y), Feb (y)
                target_dates = [
                    pd.Timestamp(year=y - 1, month=12, day=1),
                    pd.Timestamp(year=y, month=1, day=1),
                    pd.Timestamp(year=y, month=2, day=1),
                ]
            else:
                target_dates = [pd.Timestamp(year=y, month=m, day=1) for m in months]

            vals = df.loc[df.index.isin(target_dates)]
            if len(vals) == len(months):  # Require full season
                means.append({"Year": y, f"{season}": vals.mean().item()})

        if means:
            s_df = pd.DataFrame(means).set_index("Year")
            seasonal_data[season] = s_df

    # Concat all seasons
    if not seasonal_data:
        return pd.DataFrame()

    result = pd.concat(seasonal_data.values(), axis=1)
    return result


def main():
    logger.info("Starting Teleconnection Analysis...")

    # 1. Load Our Indices
    indices_dir = Path(config.RESULTS_DIR) / "Indices"
    cluster_index = pd.read_csv(
        indices_dir / "cluster_index.csv", index_col=0
    ).squeeze()
    u850_index = pd.read_csv(indices_dir / "u850_jja_index.csv", index_col=0).squeeze()

    # Assume our indices are JJA specific, indexed by Year.
    # We want to correlate with:
    # - Concurrent JJA Climate Index
    # - Preceding DJF Climate Index (Lag 1)

    # 2. Define Climate Index Files
    data_dir = Path(config.DATA_DIR)
    climate_indices_files = {
        "ONI": data_dir / "oni.data",
        "DMI": data_dir / "dmi.had.long.data",
        "PDO": data_dir / "ersst.v5.pdo.dat",
        "NPGO": data_dir / "npgo.data",
        "NAO": data_dir / "norm.nao.monthly.b5001.current.ascii",
        "PNA": data_dir / "norm.pna.monthly.b5001.current.ascii",
    }

    results = []

    for name, path in climate_indices_files.items():
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue

        raw_df = parse_climate_index(path, name)
        if raw_df.empty:
            continue

        # Calculate Seasons
        # We need DJF (Preceding) and JJA (Concurrent)
        # config.SEASONS has months
        seasons_def = {"DJF": [12, 1, 2], "JJA": [6, 7, 8]}

        seasonal_means = calculate_seasonal_means(raw_df, seasons_def)

        # Align and Correlate
        # Common Years
        # Our Cluster Index is Year-indexed (1991-2023)

        for my_idx_name, my_idx_series in [
            ("Cluster_Index", cluster_index),
            ("U850_JJA_Index", u850_index),
        ]:
            common_years = np.intersect1d(my_idx_series.index, seasonal_means.index)

            # 1. Concurrent JJA
            if "JJA" in seasonal_means.columns:
                corr = np.corrcoef(
                    my_idx_series.loc[common_years],
                    seasonal_means.loc[common_years, "JJA"],
                )[0, 1]
                results.append(
                    {
                        "My_Index": my_idx_name,
                        "Climate_Index": name,
                        "Season": "JJA (Concurrent)",
                        "Correlation": corr,
                        "N": len(common_years),
                    }
                )

            # 2. Preceding DJF
            # DJF of Year Y in seasonal_means table is constructed from Dec(Y-1), Jan(Y), Feb(Y).
            # So it effectively represents the winter OF that year (early year).
            # This is "Concurrent Winter" to the start of the year, or "0-lag" if we think in calendar years.
            # But the Summer (JJA) happens AFTER this DJF.
            # So `seasonal_means.loc[Y, 'DJF']` IS the preceding winter for `my_idx_series.loc[Y]`.

            if "DJF" in seasonal_means.columns:
                corr = np.corrcoef(
                    my_idx_series.loc[common_years],
                    seasonal_means.loc[common_years, "DJF"],
                )[0, 1]
                results.append(
                    {
                        "My_Index": my_idx_name,
                        "Climate_Index": name,
                        "Season": "DJF (Preceding)",
                        "Correlation": corr,
                        "N": len(common_years),
                    }
                )

    # 3. Save and Plot
    results_df = pd.DataFrame(results)
    output_csv = Path(config.RESULTS_DIR) / "teleconnection_correlations.csv"
    results_df.to_csv(output_csv, float_format="%.3f")
    logger.info(f"Saved correlations to {output_csv}")

    # Plot Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=results_df, x="Climate_Index", y="Correlation", hue="Season", errorbar=None
    )
    plt.title(
        "Correlation with Climate Indices (Cluster Index)"
    )  # Just showing one? or facet?
    # Let's facet or just save separate plots
    plt.close()  # clear

    for my_idx in results_df["My_Index"].unique():
        sub_df = results_df[results_df["My_Index"] == my_idx]
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=sub_df,
            x="Climate_Index",
            y="Correlation",
            hue="Season",
            palette="vlag",
        )
        plt.axhline(0, color="k", linewidth=0.8)
        plt.ylabel("Pearson Correlation")
        plt.title(f"Teleconnections: {my_idx}")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # Add significance threshold lines (approx for N=33, p<0.05 ~ +/- 0.34)
        # N varies slightly but ~33.
        # r_crit = 0.34
        plt.axhline(0.34, color="r", linestyle=":", label="p<0.05")
        plt.axhline(-0.34, color="r", linestyle=":")
        plt.legend()

        out_fig = Path(config.FIGURES_DIR) / f"teleconnection_bar_{my_idx.lower()}.png"
        plt.savefig(out_fig, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot: {out_fig}")
        plt.close()


if __name__ == "__main__":
    main()
