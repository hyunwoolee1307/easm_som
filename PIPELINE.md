# Reproducible Pipeline (Execution Order)

This document defines the **recommended execution order** and the **minimum inputs/outputs** for a reproducible run.
The pipeline is split into **Core** (required for main results) and **Optional** analyses.

---

## 0) Environment

```bash
cd /home/hyunwoo/Projects/NWP_SOM
conda env create -f environment.yml
conda activate nwp_som
```

---

## 1) Required Inputs (Core)

These files are **required** for the core pipeline:

- `Data/u850_cluster_counts.csv`  (cluster frequency table; columns `1`..`9`)
- `Data/uwnd850_anom.nc`          (U850 anomaly field)
- `Data/sst_anom_regridded.nc`    (SST anomaly on the U850 grid)
- `Data/olr_anom2.nc`             (OLR anomaly field)

If you already have the files above, you can start at **Step 3**.

### Optional / external inputs used by additional scripts

- `Data/iitm_aismr.txt` (AISMR rainfall)
- `Data/oni.data`, `Data/dmi.had.long.data`, `Data/ersst.v5.pdo.dat`, `Data/npgo.data`,
  `Data/norm.nao.monthly.b5001.current.ascii`, `Data/norm.pna.monthly.b5001.current.ascii`
- `Data/olr_anom.nc`, `Data/sst_anom.nc` (raw anomaly files, if you plan to regrid SST)

---

## 2) (Optional) Data Preparation

These steps **prepare inputs** if you start from raw files. Skip if your `Data/` already contains the files in Step 1.

### 2.1 Build merged U850 (raw wind files)

```bash
python Scripts/preprocess_u850.py
```

- **Input**: `Data/uwnd.YYYY.nc` (1991-2023)
- **Output**: `Data/u850.nc`

### 2.2 Create U850 anomaly field (external)

`Scripts/config.py` expects `Data/uwnd850_anom.nc`. This file is **not built by any script in `Scripts/`**.
If you use a different filename, update `Scripts/config.py` accordingly.

### 2.3 Regrid SST anomalies to U850 grid

```bash
python Scripts/calc_coarson.py
```

- **Input**: `Data/sst_anom.nc`, `Data/uwnd850_anom.nc`
- **Output**: `Data/sst_anom_regridded.nc`

### 2.4 OLR anomaly field (external)

`Scripts/config.py` expects `Data/olr_anom2.nc`. Ensure it exists or update `Scripts/config.py`.

---

## 3) SOM Clustering and Cluster Counts

The core pipeline needs `Data/u850_cluster_counts.csv`.
There are two ways to generate it:

### Option A: Notebook-driven (current, used in past runs)

- `Notebooks/cluster_u850_composite_u850.ipynb` (or `Notebooks/cluster_u850_composite_sst.ipynb`)
- These notebooks save `Data/u850_cluster_counts.csv` directly.

### Option B: Script-driven (SOM only, plus manual counts extraction)

```bash
python Scripts/run_som_cluster.py
```

- **Input**: `Data/u850.nc`
- **Output**: `Results/som_neuron_indices_jja.csv`, `Results/som_yearly_stats.csv`
- **Preprocessing**: monthly anomaly + 10-day low-pass, then JJA selection (see `Scripts/run_som_cluster.py`)

To make this output usable by the index generator, build a `Data/u850_cluster_counts.csv`
with columns `year,1,2,...,9` (cluster counts per year). You can derive it from
`Results/som_yearly_stats.csv` or `Results/som_neuron_indices_jja.csv`.

Example conversion from `Results/som_yearly_stats.csv`:

```bash
python - <<'PY'
import pandas as pd

stats = pd.read_csv("Results/som_yearly_stats.csv")
cols = ["year"] + [f"node_{i}_count" for i in range(1, 10)]
df = stats[cols].rename(columns={f"node_{i}_count": str(i) for i in range(1, 10)})
df.to_csv("Data/u850_cluster_counts.csv", index=False)
print("Wrote Data/u850_cluster_counts.csv")
PY
```

---

## 4) Core Indices

```bash
python Scripts/create_indices.py
```

- **Input**: `Data/u850_cluster_counts.csv`, `Data/uwnd850_anom.nc`
- **Output**:
  - `Results/Indices/cluster_index.csv`
  - `Results/Indices/u850_jja_index.csv`
  - `Results/Figures/cluster_index_timeseries.png`
  - `Results/Figures/u850_jja_index_timeseries.png`

---

## 5) Core Analyses

### 5.1 Global correlation maps

```bash
python Scripts/run_correlation.py
```

- **Input**: indices from Step 4 + `Data/sst_anom_regridded.nc`, `Data/olr_anom2.nc`, `Data/uwnd850_anom.nc`
- **Output**: `Results/Figures/corr_*` PNGs

### 5.2 Composite difference maps (JJA)

```bash
python Scripts/run_composite.py
```

- **Input**: indices from Step 4 + same anomaly fields as above
- **Output**: `Results/Figures/composite_*` PNGs

### 5.3 Teleconnection correlations

```bash
python Scripts/run_teleconnections.py
```

- **Input**: indices from Step 4 + climate index files in `Data/`
- **Output**:
  - `Results/teleconnection_correlations.csv`
  - `Results/Figures/teleconnection_bar_*` PNGs

---

## 6) Optional Analyses (Dependent on Step 4)

### AISMR links

```bash
python Scripts/run_aismr_analysis.py
python Scripts/run_aismr_rolling_corr.py
python Scripts/run_aismr_decadal_analysis.py
```

- **Input**: indices + `Data/iitm_aismr.txt`
- **Output**: AISMR figures and stats in `Results/`

### NIO SST / OLR coupling

```bash
python Scripts/run_nio_sst_rolling_corr.py
python Scripts/run_nio_sst_lagged_analysis.py
python Scripts/run_nio_sst_ea_olr_rolling_corr.py
```

- **Input**: indices + `Data/sst_anom_regridded.nc` + `Data/olr_anom2.nc`

### Other teleconnection checks

```bash
python Scripts/run_wy_index_rolling_corr.py
python Scripts/run_samoi_n_analysis.py
```

### SOM node composites (SST/OLR/U850)

```bash
python Scripts/run_som_node_composites.py
```

- **Input**: `Results/som_neuron_indices_jja.csv` + daily anomaly fields
- **Output**: `Results/Figures/som_node_composites_*.png`

### Node trend tests (Mann–Kendall)

```bash
python Scripts/run_mk_trend_node_freq.py
python Scripts/run_mk_trend_node_freq_decadal.py
```

- **Output**: `Results/mann_kendall_node_trends.csv`, `Results/mann_kendall_node_trends_decadal.csv`

### Node periodicity (Periodogram + Red-noise + Fisher g-test)

```bash
python Scripts/run_node_periodogram.py
```

- **Output**: `Results/node_periodogram_summary.csv`, `Results/Figures/periodogram_node_*.png`

---

## 7) Reproducibility Checklist

- Update `Scripts/config.py` if any filenames differ from your local `Data/`.
- Confirm required NetCDF files exist **before** running the core steps.
- Run steps **in order**; Steps 4-6 depend on outputs from earlier steps.
- If you regenerate `Data/u850_cluster_counts.csv`, rerun Step 4 and downstream steps.

---

## 8) Fast Rebuild (Core Only)

If all required inputs already exist in `Data/`:

```bash
python Scripts/create_indices.py
python Scripts/run_correlation.py
python Scripts/run_composite.py
python Scripts/run_teleconnections.py
```
