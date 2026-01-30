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

## 3) SOM Clustering

```bash
python Scripts/run_som_cluster.py
```

- **Input**: `Data/u850.nc`
- **Output**: `Results/som_neuron_indices_jja.csv`, `Results/som_yearly_stats.csv`
- **Preprocessing**: monthly anomaly + 10-day low-pass, then JJA selection (see `Scripts/run_som_cluster.py`)

---

## 4) Core Analyses

```bash
python Scripts/run_som_node_composites.py
python Scripts/run_mk_trend_node_freq.py
python Scripts/run_mk_trend_node_freq_decadal.py
python Scripts/run_node_periodogram.py
```

- **Outputs**:
  - `Results/Figures/som_node_composites_*.png` (SST/OLR는 합성장만 생성, 이후 분석은 U850만)
  - `Results/mann_kendall_node_trends.csv`, `Results/mann_kendall_node_trends_decadal.csv`
  - `Results/node_periodogram_summary.csv`, `Results/Figures/periodogram_node_*.png`

---

## 5) Optional Analyses (Dependent on Step 4)

### Clustering diagnostics

```bash
python Scripts/run_som_dendrogram.py
python Scripts/run_som_node_clustering.py
```

### Node vs climate indices

```bash
python Scripts/run_node_climate_correlations.py
python Scripts/run_node_climate_correlations_significant_plot.py
python Scripts/run_node_climate_lagged_correlations.py
```

- **Output**:
  - `Results/node_climate_correlations.csv`
  - `Results/Figures/heatmap_node_climate_correlations_significant.png`
  - `Results/node_climate_lagged_correlations.csv`
  - `Results/Figures/heatmap_lagged_correlations_*.png`

### Kriging (SST/OLR/U850) (variogram fit + BLUE)

```bash
python Scripts/run_kriging_node_composites.py
```

- **Output**: `Results/kriging_variogram_summary.csv`,
  `Results/Figures/kriging_{sst,olr,u850}_node_*.png`

### Kohonen SOM (R) + Comparison (optional)

```bash
Rscript Scripts/run_kohonen_som.R
python Scripts/run_compare_som_kohonen.py
```

- **Output**: `Results/kohonen_bmu.csv`, `Results/kohonen_codes.csv`,
  `Results/compare_som_kohonen_confusion.csv`, `Results/compare_som_kohonen_metrics.csv`,
  `Results/Figures/compare_som_kohonen_confusion.png`

---

## 7) Reproducibility Checklist

- Update `Scripts/config.py` if any filenames differ from your local `Data/`.
- Confirm required NetCDF files exist **before** running the core steps.
- Run steps **in order**; Steps 4-6 depend on outputs from earlier steps.

---

## 6) Fast Rebuild (Core Only)

지수 기반 재빌드 단계는 현재 워크스페이스에서 제거되었습니다.
