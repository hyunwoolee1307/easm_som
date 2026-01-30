import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "Results"
FIG_DIR = RESULTS_DIR / "Figures"


def compute_periodogram(x, dt=1.0):
    """Return frequencies (1/yr) and power using FFT periodogram."""
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    n = len(x)
    # FFT
    fft_vals = np.fft.rfft(x)
    power = (np.abs(fft_vals) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, power


def red_noise_spectrum(x, freqs, dt=1.0):
    """AR(1) red-noise spectrum scaled to the variance of x."""
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    if len(x) < 2:
        return np.full_like(freqs, np.nan)

    x0 = x[:-1]
    x1 = x[1:]
    r1 = np.corrcoef(x0, x1)[0, 1]
    r1 = np.clip(r1, -0.99, 0.99)
    var = np.nanvar(x, ddof=1)
    omega = 2.0 * np.pi * freqs * dt
    denom = 1.0 + r1**2 - 2.0 * r1 * np.cos(omega)
    return var * (1.0 - r1**2) / denom


def fisher_g_test(power):
    """Fisher's g-test for periodicity (H0: no significant periodic component)."""
    power = np.asarray(power, dtype=float)
    power = power[np.isfinite(power)]
    n = len(power)
    if n < 2:
        return np.nan, np.nan

    total = np.sum(power)
    if total <= 0:
        return np.nan, np.nan

    g = np.max(power) / total

    # Fisher's exact p-value
    p = 0.0
    k_max = int(np.floor(1.0 / g))
    for k in range(1, k_max + 1):
        term = (-1) ** (k - 1)
        comb = math.comb(n, k)
        base = 1.0 - k * g
        if base <= 0:
            continue
        p += term * comb * (base ** (n - 1))

    p = max(min(p, 1.0), 0.0)
    return g, p


def main():
    stats_path = RESULTS_DIR / "som_yearly_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {stats_path}. Run run_som_cluster.py first.")

    df = pd.read_csv(stats_path)
    if "year" not in df.columns:
        raise ValueError("som_yearly_stats.csv must contain a 'year' column")

    years = df["year"].to_numpy()
    if len(years) < 5:
        raise ValueError("Not enough years for periodogram")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for node in range(1, 10):
        col = f"node_{node}_count"
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {stats_path}")

        series = df[col].to_numpy()
        freqs, power = compute_periodogram(series, dt=1.0)

        # Exclude zero frequency
        mask = freqs > 0
        freqs_nz = freqs[mask]
        power_nz = power[mask]

        # Convert to periods (years), apply Nyquist (dt=1 year => f_N = 0.5 cycles/yr => period >= 2 years)
        periods = 1.0 / freqs_nz
        nyquist_period = 2.0
        max_period = 33.0 / 2.0
        valid = (periods >= nyquist_period) & (periods <= max_period)
        periods = periods[valid]
        power_nz = power_nz[valid]
        freqs_nz = freqs_nz[valid]

        # Red-noise (AR1) significance (95%)
        red = red_noise_spectrum(series, freqs_nz, dt=1.0)
        chi2_95 = stats.chi2.ppf(0.95, df=2)
        red_sig = red * (chi2_95 / 2.0)

        # Fisher's g-test (H0: no meaningful periodic component)
        g_stat, g_p = fisher_g_test(power_nz)
        g_reject = bool(g_p < 0.05) if np.isfinite(g_p) else False

        # Top 3 peaks by power
        top_idx = np.argsort(power_nz)[-3:][::-1]
        top_periods = periods[top_idx]
        top_powers = power_nz[top_idx]
        top_sig = top_powers > red_sig[top_idx]

        summary_rows.append(
            {
                "node": node,
                "peak1_period_years": top_periods[0],
                "peak1_power": top_powers[0],
                "peak2_period_years": top_periods[1],
                "peak2_power": top_powers[1],
                "peak3_period_years": top_periods[2],
                "peak3_power": top_powers[2],
                "peak1_sig95": bool(top_sig[0]),
                "peak2_sig95": bool(top_sig[1]),
                "peak3_sig95": bool(top_sig[2]),
                "fisher_g": g_stat,
                "fisher_p": g_p,
                "fisher_reject_0p05": g_reject,
            }
        )

        # Plot periodogram
        plt.figure(figsize=(8, 4), layout="constrained")
        plt.plot(periods, power_nz, color="black", linewidth=1.2, label="Periodogram")
        plt.plot(periods, red_sig, color="red", linewidth=1.2, linestyle="--", label="Red-noise 95%")
        plt.xlabel("Period (years)")
        plt.ylabel("Power")
        plt.title(f"Periodogram - Node {node}")
        plt.gca().invert_xaxis()  # shorter periods on the right
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        out_fig = FIG_DIR / f"periodogram_node_{node}.png"
        plt.savefig(out_fig, dpi=300)
        plt.close()

    out_df = pd.DataFrame(summary_rows)
    out_path = RESULTS_DIR / "node_periodogram_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
