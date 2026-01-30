import math
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "Results"


def mann_kendall_test(x: np.ndarray):
    """Return S, var_s, z, p (two-sided)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan

    # S statistic
    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k + 1 :] - x[k]))

    # Tie correction
    unique, counts = np.unique(x, return_counts=True)
    tie_counts = counts[counts > 1]
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    if tie_counts.size > 0:
        tie_term = np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5))
        var_s -= tie_term / 18.0

    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    # two-sided p-value from normal approximation
    # use erfc for numerical stability: p = 2 * (1 - Phi(|z|))
    p = math.erfc(abs(z) / math.sqrt(2))
    return s, var_s, z, p


def main():
    stats_path = RESULTS_DIR / "som_yearly_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {stats_path}. Run run_som_cluster.py first.")

    df = pd.read_csv(stats_path)
    if "year" not in df.columns:
        raise ValueError("som_yearly_stats.csv must contain a 'year' column")

    out_rows = []
    for node in range(1, 10):
        col = f"node_{node}_count"
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {stats_path}")
        series = df[col].to_numpy()
        s, var_s, z, p = mann_kendall_test(series)
        trend = "increasing" if z > 0 else "decreasing" if z < 0 else "no trend"
        out_rows.append(
            {
                "node": node,
                "S": s,
                "var_S": var_s,
                "Z": z,
                "p_value": p,
                "trend": trend,
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_path = RESULTS_DIR / "mann_kendall_node_trends.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
