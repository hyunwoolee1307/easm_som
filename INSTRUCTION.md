# Project Instructions (NWP_SOM)

## Environment
- Use `conda env create -f environment.yml` and `conda activate nwp_som`.
- Run scripts from repo root unless noted otherwise.
## System Resource Policy
- Check system specs with `neofetch --stdout` before heavy runs.
- Limit resource usage to **<= 80%** of available CPU, memory, and disk during processing.
- Use `Scripts/run_with_limits.sh` to run heavy commands with enforced limits.
  - Example: `Scripts/run_with_limits.sh conda run -n nwp_som python Scripts/run_som_cluster.py`

## Core Execution Order
1) `Scripts/run_som_cluster.py`

## SOM Preprocessing (Current)
- Monthly anomaly removal → 10-day low-pass → JJA selection.
- If you change filtering, rerun SOM and downstream analyses.

## Optional Analyses
- SOM node composites: `Scripts/run_som_node_composites.py`
- Mann–Kendall trends: `Scripts/run_mk_trend_node_freq.py`, `Scripts/run_mk_trend_node_freq_decadal.py`
- Periodogram + red-noise + Fisher g-test: `Scripts/run_node_periodogram.py`

## Output Notes
- Figures are written to `Results/Figures/`.
- Deleting `Results/Figures/` removes all plots; rerun scripts to regenerate.

## Plotting Standard
- Use `layout="constrained"` (or `constrained_layout=True`) instead of `plt.tight_layout()`.

## Data Assumptions
- Daily anomaly fields for SST/OLR/U850 are expected in `Data/`.
- If filenames differ, update `Scripts/config.py`.
