# Research Report: East Asian Summer Monsoon Variability (SOM-Based)

**Date**: January 30, 2026
**Project**: NWP_SOM

---

## 1. Executive Summary
This study investigates the variability of the East Asian Summer Monsoon (EASM) using Self-Organizing Maps (SOM) on JJA 850 hPa zonal wind anomalies. The current workspace focuses on SOM pattern discovery and diagnostics (node frequency trends, periodicity, clustering comparisons, and variogram-based kriging). Index-based analyses have been removed from this workspace per current scope.

## 2. Methodology
- **Pattern Classification**: 3x3 SOM applied to JJA 850 hPa zonal wind anomalies (1991–2023) over the Western North Pacific.
- **SOM Preprocessing**: Monthly anomaly removal → 10-day low-pass → JJA selection.
- **Node Frequency Trends**: Mann–Kendall test on annual and decadal node counts.
- **Periodicity**: FFT periodogram (2–16.5 yr), AR(1) red-noise 95% significance, Fisher g-test.
- **Clustering Diagnostics**: Dendrogram of node composites and MDS/KMeans comparison.
- **Spatial Interpolation**: Variogram-based kriging for **U850 only** (global best model; BLUE/ordinary kriging).

## 3. Key Findings

### 3.1 Node Frequency Trends (Mann–Kendall)
- Annual and decadal Mann–Kendall tests show **no statistically significant trends at p < 0.05** across nodes.

### 3.2 Climate Index Correlations (Significant Pairs Only)
- Significant (p < 0.05) node–index correlations were limited to a subset of indices and nodes.
- The strongest signals were associated with NINO34 (multiple nodes), with additional significant pairs for WP, DMI, NPGO, and PDO.
- A significant-only heatmap is provided at `Results/Figures/heatmap_node_climate_correlations_significant.png`.

### 3.3 Periodicity (Periodogram + Fisher g-test)
- Periodograms were evaluated within **2–16.5 years** (Nyquist-limited and capped at 33/2).
- **Fisher g-test** (H0: “periodic component is not meaningful”) **was not rejected at p < 0.05** for any node.

### 3.4 Node Clustering Diagnostics
- Dendrograms indicate hierarchical similarity among node composites.
- SOM vs KMeans comparison shows partial agreement with non-trivial reassignment across clusters.

### 3.5 Variogram + Kriging (U850 only)
- Global best variogram model (mean SSE) is **spherical**, used for U850 kriging runs.
- Kriged fields provide smoothed spatial estimates consistent with SOM composites.

## 4. Conclusion
The SOM-based workflow provides a consistent framework to summarize EASM variability, evaluate node behavior over time, and compare clustering structures. Index-based teleconnection interpretations are out of scope for the current workspace.

## 5. References & Data
- **Reanalysis**: NCEP/DOE Reanalysis-2 (local processed files).
- **Climate Indices**: NOAA PSL / CPC (for optional analyses).

## 6. Session Log (January 30, 2026)
- SOM stippled composite colorbar range narrowed to **-5 to +5**.
- NPGO node-frequency plot right y-axis range set to **-1 to +1**.
- Results regenerated for the affected plots.
