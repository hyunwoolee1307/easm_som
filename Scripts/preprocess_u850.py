#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import xarray as xr
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 2. 데이터 소스 경로 및 파일 리스트 생성
data_dir = BASE_DIR / "Data"
target_years = range(1991, 2024)
flist = [data_dir / f"uwnd.{year}.nc" for year in target_years]

# 파일 존재 여부 필터링
flist = [f for f in flist if f.exists()]

if not flist:
    print("Error: No files found within the specified range.")
else:
    print(f"Found {len(flist)} files. Processing...")

    # 3. 저장 경로 준비 (폴더가 없으면 생성)
    output_path = BASE_DIR / "Data" / "u850.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True) 

    # 4. 데이터 로드 및 처리
    # chunks={'time': 12}: 메모리 최적화를 위해 Dask Chunk 사용
    ds = xr.open_mfdataset(
        flist,
        combine="by_coords",
        parallel=False,
        chunks={"time": 12} 
    )
    print(" Selecting 850hPa level...")

    # 레벨 선택 및 차원 축소 (squeeze: level 차원 제거)
    # NCAR 데이터 변수명이 'uwnd'인지 확인 (가끔 'u'일 수 있음)
    var_name = 'uwnd' if 'uwnd' in ds else 'u'
    da = ds[var_name].sel(level=850.0).squeeze()

    # 메타데이터 수정 (선택 사항)
    da.attrs['description'] = 'JJA 850hPa U-wind (Merged 1991-2023)'

    print(f" Saving to {output_path} ... (This may take a while)")

    # 5. NetCDF 저장
    # compute=True: 실제로 연산을 수행하여 디스크에 씀
    da.to_netcdf(output_path, compute=True)

    print("Done! File created successfully.")


# In[ ]:



