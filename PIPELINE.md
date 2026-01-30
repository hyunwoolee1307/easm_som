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

Core 분석 단계는 필요 시 별도 스크립트로 추가하세요.

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

---

## 8) Fast Rebuild (Core Only)

지수 기반 재빌드 단계는 현재 워크스페이스에서 제거되었습니다.
