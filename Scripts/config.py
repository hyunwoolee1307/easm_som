from pathlib import Path
from typing import Dict, List

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
FIGURES_DIR = RESULTS_DIR / "Figures"
INDICES_DIR = RESULTS_DIR / "Indices"

# Ensure directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
INDICES_DIR.mkdir(parents=True, exist_ok=True)

# Data Files
DATA_FILES = {
    "CLUSTER_COUNTS": DATA_DIR / "u850_cluster_counts.csv",
    "SST_ANOM": DATA_DIR / "sst_anom_regridded.nc",
    "OLR_ANOM": DATA_DIR / "olr_anom2.nc",
    "U850_ANOM": DATA_DIR / "uwnd850_anom.nc",
}

# Index Files
INDEX_FILES = {
    "CLUSTER_INDEX": INDICES_DIR / "cluster_index.csv",
    "U850_JJA_INDEX": INDICES_DIR / "u850_jja_index.csv",
}

# Analysis Settings
SEASONS: Dict[str, int] = {
    "DJF": 12,
    "MAM": 3,
    "JJA": 6,
    "SON": 9
}

DOMAINS: Dict[str, List[float]] = {
    "Indian": [30, 120, -30, 30],
    "Pacific": [120, 290, -60, 60],
    "Global": [0, 360, -90, 90]
}
