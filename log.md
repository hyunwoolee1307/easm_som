# Workflow Run Log (Reproduction)

## Summary
- Input generation: `data/u850_som_input.npy` created with shape (957, 3036).
- SOM training: 3x3 grid, 500 iterations; final alpha ≈ 0.0412, sigma = 0.5.
- Clustering: 9 nodes; QE ≈ 1.1392.
- u850 composites (SOM domain): `som_u850_composites.npy`, `som_u850_composites_coastline.png`.
- OLR composites + significance: `som_olr_composites.npy`, `som_olr_pvals.npy`, `som_olr_composites_stipple.png`.
- SST composites + significance (xesmf regridded): `som_sst_composites.npy`, `som_sst_pvals.npy`, `som_sst_composites_stipple.png`.
- u850 expanded-domain diagnostics: `som_uwnd_composites_ext.npy`, `som_uwnd_pvals_ext.npy`, `som_uwnd_composites_ext_stipple.png`.

## Notes
- Initial run failed during expanded-domain u850 diagnostics due to index mismatch; rerun completed successfully.
- NetCDF I/O uses `h5netcdf` engine when available.
- SST regridding uses xesmf bilinear weights (`results/sst_to_uwnd_weights.nc`).
