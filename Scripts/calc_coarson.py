# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 14:47:09 2026

@author: user
"""

import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar


# ==========================================
# 1. 설정 및 효율적인 데이터 로딩 (Dask 활용)
# ==========================================
def load_and_preprocess(ssta_path, target_grid_ds=None):
    """
    대용량 NetCDF 데이터를 로드하고, 타겟 그리드에 맞춰 해상도를 조정(Regridding)합니다.

    Args:
        ssta_path (str): 고해상도 SSTA 파일 경로
        target_grid_ds (xr.Dataset): 목표 해상도(위도, 경도)를 가진 데이터셋
    """
    print(f">>> 데이터 로딩 중 (Path: {ssta_path})...")

    # [중요] chunks 설정을 통해 메모리에 한 번에 올리지 않고 지연 연산 수행
    ds = xr.open_dataset(
        ssta_path,
        engine="netcdf4",
        chunks={"time": 100, "lat": 360, "lon": 360},
    )

    # ---------------------------------------------------------
    # [수정] Coarsen 대신 Interp(보간)를 사용하여 원하는 그리드로 직접 변환
    # ---------------------------------------------------------
    if target_grid_ds is not None:
        print(">>> 타겟 그리드로 공간 보간 (Spatial Interpolation)...")
        
        # [핵심 해결책] 
        # interp_like(target_grid_ds)를 쓰면 시간(Time)축까지 덮어씌워버려 NaN이 발생할 수 있습니다.
        # 따라서 'lat', 'lon' 좌표만 콕 집어서 보간해야 합니다.
        new_lat = target_grid_ds["lat"]
        new_lon = target_grid_ds["lon"]

        # 위도/경도 범위가 약간 달라도(예: 89.88 vs 90.0) 값을 채우도록 설정
        # method='nearest': 가장 가까운 값 사용 (NaN 전파 방지에 유리)
        # kwargs={"fill_value": "extrapolate"}: 범위 밖 좌표도 외삽 (필요시 사용)
        ds_regridded = ds.interp(
            lat=new_lat, 
            lon=new_lon, 
            method="nearest",  # [수정] nearest로 변경하여 원본 값 유지
            kwargs={"fill_value": "extrapolate"} # 90도 등 끝부분 NaN 방지
        )
    else:
        # 타겟이 없으면 기존처럼 10배 축소 (fallback)
        print(">>> 타겟 그리드가 없어 기본 10x10 Coarsen 수행...")
        ds_regridded = ds.coarsen(lat=10, lon=10, boundary="trim").mean()

    # 연산 수행 및 메모리 적재
    with ProgressBar():
        print(">>> 메모리로 데이터 적재 중 (Compute)...")
        # Dask 그래프 실행
        ds_loaded = ds_regridded.compute()

    return ds_loaded


# ==========================================
# 메인 실행 블록
# ==========================================
if __name__ == "__main__":
    ssta_file_path = "Data/sst_anom.nc"
    
    # Target 데이터 로드
    target = xr.open_dataset("Data/uwnd850_anom.nc")

    # 함수 호출
    ds_processed = load_and_preprocess(ssta_file_path, target_grid_ds=target)

    # 결과 저장
    output_path = "Data/sst_anom_regridded.nc"
    ds_processed.to_netcdf(output_path)
    
    print(f"\n[저장 완료] {output_path}")
    print("\n[결과 확인] Regridding 완료된 데이터:")
    print(ds_processed)