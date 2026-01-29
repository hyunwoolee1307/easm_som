# EASM_SOM

Regime-Based Analysis of the East Asian Summer Monsoon
Using 850 hPa Zonal Wind SOM (1991–2023, daily JJA)

1. Background and Motivation

The East Asian Summer Monsoon (EASM) is a dynamically complex system whose variability is influenced by tropical–extratropical interactions, particularly over the western North Pacific (WNP). Although links between the EASM, Indian Summer Monsoon (ISM), and Indo-Pacific SST variability have been widely discussed, these relationships are often nonlinear, regime-dependent, and temporally nonstationary.

Conventional analyses based on linear correlations or single climate indices are insufficient to capture such behavior. This project adopts a Self-Organizing Map (SOM) approach to objectively identify distinct low-level circulation regimes, focusing exclusively on 850 hPa zonal wind (u850) as a dynamically fundamental variable.

2. Scientific Objective

The primary objective is to diagnose how EASM-related low-level circulation regimes vary over time and how these regimes relate to tropical forcing and convection in a conditional (regime-dependent) sense, rather than through fixed linear teleconnections.

3. Data Description
3.1 Variables

u850: 850 hPa zonal wind (SOM input)

OLR: Outgoing Longwave Radiation (diagnostic only)

SST: Sea Surface Temperature (diagnostic only)

3.2 Data Source

NCAR reanalysis / observational datasets

Data root (symlinked):
- /home/hyunwoo/Data  (symlink to /mnt/d/Data)

3.3 Temporal Coverage

1991–2023 (daily)

3.4 Seasonal Definition

Boreal summer: JJA (daily samples)
(Any alternative season definition must be explicitly documented.)

4. SOM Configuration
4.1 Input Variable (Strict)

SOM training uses only u850 anomalies.

OLR and SST are not included in the SOM input space under any circumstance.

4.2 Spatial Domain

Longitude: 100°E – 180°E

Latitude: 10°S – 60°N

This domain captures:

Western North Pacific low-level circulation variability

Core EASM regions (East China, Korean Peninsula, Japan)

Tropical–extratropical interaction zones relevant to PJ-like dynamics

4.3 Preprocessing Steps

Seasonal anomaly calculation relative to a fixed climatology (1991–2020 JJA)

Daily JJA samples retained (nt=3036 for 1991–2023)

Optional linear detrending (must be reported if applied)

Area-weighted flattening of the spatial field

Standardization (z-score normalization)

5. SOM Training Design
5.1 Map Topology

Fixed SOM size: 3 × 3 (9 nodes)

Rationale:

3×3 provides a balance between:

Capturing regime diversity

Avoiding artificial fragmentation or overfitting

Each node is interpretable as a physically meaningful circulation regime

5.2 Distance Metric

Euclidean distance

5.3 Robustness

Multiple initializations are recommended

Regime patterns are evaluated for structural consistency

6. Regime Diagnostics
6.1 Circulation Interpretation

For each SOM node:

Composite u850 pattern

Identification of:

WNP anticyclonic vs cyclonic circulation

Strength and position of low-level westerlies

Dynamical relevance to EASM rainfall regions

6.2 Post-hoc Composite Analysis

(Not used for SOM training)

OLR: Convective response associated with each regime

SST: Indo-Pacific background state

Indices:

ENSO (Niño3.4, Niño4)

Indian Ocean SST indices

WPSH / PJ-related indices (definition sensitivity acknowledged)

6.3 Temporal Evolution

Time series of SOM node occupancy

Decadal modulation of dominant regimes

Changes in regime frequency rather than mean-state trends

7. Interpretation Scope and Limitations

SOM nodes represent dynamical circulation regimes, not direct forcings.

SST and convection are interpreted conditionally, within each regime.

Conclusions are framed in terms of changes in regime occurrence and structure, not causal attribution.

8. Repository Structure
.
├─ data/               # (gitignored) raw and intermediate datasets
├─ configs/            # SOM, domain, and preprocessing settings
├─ src/
│  ├─ load.py
│  ├─ preprocess.py
│  ├─ som.py
│  ├─ diagnostics.py
│  └─ plotting.py
├─ notebooks/          # analysis and figure reproduction
├─ results/
│  ├─ figures/
│  └─ tables/
└─ README.md

9. Expected Contributions

Objective classification of EASM-related low-level circulation regimes

Evidence for nonstationary monsoon coupling

A circulation-centered framework complementary to SST-based studies

10. Diagnostics Workflow (Current)

SOM input:
- Daily JJA u850 anomalies (1991–2023), climatology 1991–2020
- Domain: 100E–180E, 10S–60N

Diagnostics (not SOM inputs):
- SST and OLR domain: 0E–180E, 40S–60N
- SST regridded to uwnd grid using xesmf (bilinear) for consistent diagnostics
- Standardized anomalies (z-score) relative to 1991–2020 JJA
- Significance: one-sample t-test against 0; stipple where p < 0.05

Key outputs (results/):
- som_weights.npy
- som_cluster_i.npy, som_cluster_j.npy, som_node_counts.npy
- som_clusters_by_day.csv, som_clusters_by_year.csv
- som_u850_composites*.png
- som_sst_composites*.png, som_sst_pvals.npy
- som_olr_composites*.png, som_olr_pvals.npy
- som_uwnd_composites_ext*.png, som_uwnd_pvals_ext.npy

Environment:
- Conda env: /home/hyunwoo/miniforge3/envs/easm_som

Example paths (symlinked data root):
- /home/hyunwoo/Data/Obs_NCAR/uwnd.{year}.nc
- /home/hyunwoo/Data/Obs_NCAR/olr.day.mean.nc
- /home/hyunwoo/Data/Obs_OISST/v2.1/

11. Figure Regeneration

To regenerate all figures with publication-ready titles/labels:

```
TMPDIR=/tmp /home/hyunwoo/miniforge3/envs/easm_som/bin/python /home/hyunwoo/EASM_SOM/src/regen_figures.py
```

This script recreates all plots in `results/` from existing diagnostics arrays and tables.

Run log:
- `log.md`
