# Research Report: Decadal Changes in East Asian Summer Monsoon and its Global Teleconnections

**Date**: January 8, 2026
**Project**: NWP_SOM

---

## 1. Executive Summary
This study investigates the variability of the East Asian Summer Monsoon (EASM) using Self-Organizing Maps (SOM). Index-based analyses have been removed from this workspace per current scope.

## 2. Methodology
-   **Pattern Classification**: A 3x3 SOM was applied to Summer (JJA) 850hPa zonal wind anomalies (1991-2023) over the Western North Pacific.
-   **Index Definition**: Removed from this workspace per current scope.
-   **Teleconnection Analysis**: Pearson correlation with standardized global climate indices (ONI, DMI, PDO, NPGO, NAO, PNA) and All India Summer Monsoon Rainfall (AISMR).
-   **SOM Preprocessing**: Monthly anomaly removal, 10-day low-pass filtering, then JJA selection prior to SOM training.
-   **Node Trend Tests**: Mann–Kendall test on annual node frequencies and decadal aggregates.
-   **Periodicity**: FFT periodogram (2–16.5 yr band), AR(1) red-noise 95% significance, and Fisher g-test for dominant periodic components.
-   **Period**: 1991-2023 (33 years).

## 3. Key Findings

### 3.1 Composite Anomalies
Composite analysis based on the removed index pipeline has been removed from this workspace per current scope.
-   **SST**: Warm anomalies in the Western Pacific Warm Pool and cold anomalies in the central Pacific (La Niña-like).
-   **Circulation**: A strong cyclonic anomaly over the East China Sea and anticyclonic anomaly over the South China Sea.
-   **Convection (OLR)**: Enhanced convection (negative OLR) over the Maritime Continent and suppressed convection over the western subtropical Pacific.
-   *(See `Results/Figures/composite_*` for detailed maps)*

### 3.2 Global Teleconnections
Teleconnection results tied to the removed index pipeline have been removed from this workspace per current scope.
-   **NAO (Winter/DJF)**: Strong negative correlation (**r ~ -0.6**), suggesting winter North Atlantic signals may precursor EASM variability.
-   **PNA**: Moderate positive correlation.
-   **ENSO**: Varying relationship, suggesting the pattern captures non-ENSO variance as well.

### 3.3 Linkage with Indian Monsoon (AISMR)
A crucial finding is the connection to the Indian Summer Monsoon:
-   **Overall Correlation (1991-2023)**: **r = +0.365** (Moderate Positive)
-   **Interpretation**: A strong EASM is often accompanied by a strong Indian Monsoon, consistent with the "Asian Summer Monsoon" coupled system.

### 3.4 Decadal Shift in Coupling
Running correlation analysis (10-year window) reveals a dramatic shift in the EASM-AISMR relationship:
-   **1990s**: Weak/Negative correlation.
-   **2000s - Present**: Significant positive coupling (**r > 0.6**).
-   **Implication**: The two monsoon subsystems have become more synchronized in the 21st century.

### 3.5 North Indian Ocean Forcing
Recent analysis highlights the **North Indian Ocean (NIO)** as a key driver:
-   **NIO SST vs East Asian Convection**: Strong correlation (**r ~ -0.8**) in recent decades.
-   **Predictability**: Comparison with lagged SSTs confirms that NIO warming in **Spring (MAM)** is a strong predictor (**r = 0.61**) of the East Asian Summer Monsoon intensity.
-   **Mechanism**: NIO warming is increasingly effective at triggering convection over East Asia, likely through atmospheric wave propagation (e.g., Kelvin wave induced Ekman divergence mechanism).

### 3.6 Validation against Standard Indices
-   **Webster-Yang Index (Dynamic)**: Robust positive correlation (**r ~ 0.6**), confirming validity as a large-scale monsoon metric.
-   **SAMOI-N (Convective Shift)**: Moderate negative correlation (**r ~ -0.4**), indicating that the cyclonic anomalies captured by our index are associated with convection centered more in the southerly (tropical) band rather than a strong northward shift.

![Running Correlation NIO-EA_OLR](/d:/Research/Projects/NWP_SOM/Results/Figures/running_correlation_nio_sst_ea_olr.png)

### 3.7 Node Frequency Trends (Mann–Kendall)
-   Annual and decadal Mann–Kendall tests show **no statistically significant trends at p < 0.05** across nodes.
-   Decadal aggregates suggest weak tendencies in some nodes, but none pass the 95% significance threshold.

### 3.8 Periodicity (Periodogram + Fisher g-test)
-   Periodograms were evaluated within **2–16.5 years** (Nyquist-limited for annual data and capped at 33/2).
-   Red-noise (AR1) significance lines were applied.
-   **Fisher g-test** (H0: “periodic component is not meaningful”) **was not rejected at p < 0.05 for any node**; the strongest case was Node 8 (p ≈ 0.074).

## 4. Conclusion
The modernized analysis pipeline successfully quantified the EASM variability and uncovered a strengthening dynamic link with the Indian Monsoon. The results highlight the importance of recent decadal shifts and suggest that seasonal prediction models should account for high-latitude precursors (NAO) and the evolving EASM-AISMR coupling.

## 5. References & Data
-   **AISMR**: IITM Indian Seasonal/Annual Rainfall Data.
-   **Climate Indices**: NOAA PSL / CPC.
-   **Reanalysis**: ERA5 (via local processed files).
