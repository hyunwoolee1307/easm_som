import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "Results"
FIG_DIR = RESULTS_DIR / "Figures"


def main():
    csv_path = RESULTS_DIR / "som_param_search.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run run_som_param_search.py first.")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("som_param_search.csv is empty")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Heatmap of QE across sigma0 x nter
    pivot = df.pivot(index="sigma0", columns="nter", values="qerror")
    plt.figure(figsize=(6, 4), layout="constrained")
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
    plt.title("SOM Parameter Search (QE)")
    plt.xlabel("nter")
    plt.ylabel("sigma0")
    out_fig = FIG_DIR / "som_param_search_heatmap.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
