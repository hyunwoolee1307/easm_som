from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "Results"
FIG_DIR = RESULTS_DIR / "Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

som_path = RESULTS_DIR / "som_neuron_indices_jja.csv"
koh_path = RESULTS_DIR / "kohonen_bmu.csv"

if not som_path.exists():
    raise FileNotFoundError(f"Missing {som_path}")
if not koh_path.exists():
    raise FileNotFoundError(f"Missing {koh_path}")

som_df = pd.read_csv(som_path)
som_df["time"] = pd.to_datetime(som_df["time"])

koh_df = pd.read_csv(koh_path)
koh_df["time"] = pd.to_datetime(koh_df["time"])

merged = pd.merge(
    som_df[["time", "node_id"]].rename(columns={"node_id": "som_node"}),
    koh_df[["time", "node_id"]].rename(columns={"node_id": "kohonen_node"}),
    on="time",
    how="inner",
)

if merged.empty:
    raise RuntimeError("No overlapping time samples between SOM and kohonen results.")

# Confusion matrix
conf = pd.crosstab(merged["som_node"], merged["kohonen_node"])
conf_path = RESULTS_DIR / "compare_som_kohonen_confusion.csv"
conf.to_csv(conf_path)

# Metrics
ari = adjusted_rand_score(merged["som_node"], merged["kohonen_node"])
nmi = normalized_mutual_info_score(merged["som_node"], merged["kohonen_node"])
metrics = pd.DataFrame([
    {"metric": "ARI", "value": ari},
    {"metric": "NMI", "value": nmi},
    {"metric": "n_samples", "value": len(merged)},
])
metrics_path = RESULTS_DIR / "compare_som_kohonen_metrics.csv"
metrics.to_csv(metrics_path, index=False)

# Heatmap
plt.figure(figsize=(6, 5), layout="constrained")
sns.heatmap(conf, annot=True, fmt="d", cmap="Blues")
plt.title("SOM vs Kohonen BMU Confusion")
plt.xlabel("Kohonen node")
plt.ylabel("SOM node")
fig_path = FIG_DIR / "compare_som_kohonen_confusion.png"
plt.savefig(fig_path, dpi=300)
plt.close()

print(f"Saved {conf_path}")
print(f"Saved {metrics_path}")
print(f"Saved {fig_path}")
