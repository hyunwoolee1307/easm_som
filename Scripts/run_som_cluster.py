import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
import som  # 사용자가 제공한 som.py 모듈
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_som_nodes(X_raw, node_ids, m1, m2, lats, lons, output_dir):
    """
    각 SOM 노드별 원본 데이터(u850)의 평균장(Composite Map)을 시각화하고,
    기후값 대비 유의한 차이가 있는 영역(p < 0.05)을 stipple 처리합니다.
    """
    print("\n--- Plotting Node Composites with Significance Stippling ---")
    
    n_nodes = m1 * m2
    fig, axes = plt.subplots(nrows=m2, ncols=m1, figsize=(15, 12), 
                             subplot_kw={'projection': ccrs.PlateCarree()})
    
    # 전체 기간 수 및 전체 평균(Climatology) 계산
    # T-test의 귀무가설: "해당 노드의 평균은 전체 평균(Climatology)과 같다"
    total_samples = len(node_ids)
    
    # [수정] popmean broadcasting을 위해 shape을 (Space, 1)로 변환
    # scipy.stats.ttest_1samp의 axis=1 연산을 위해 두 번째 차원을 1로 맞춰줍니다.
    clim_mean = np.mean(X_raw, axis=1).reshape(-1, 1) # Shape: (Space, 1)
    
    # [수정] 컬러바 레벨 설정 (0을 중심으로 대칭)
    # -20 ~ 20 범위, 1.0 간격으로 설정하여 0이 정확히 중심에 오도록 함
    limit = 20
    levels = np.linspace(-limit, limit, 41)
    
    # Plotting loop
    for j in range(m2):
        for i in range(m1):
            # 1D node index (1-based for display, logic matches som.py output)
            node_idx = j * m1 + (i + 1)
            
            # 현재 subplot
            ax = axes[j, i]
            
            # 해당 노드에 속하는 시간 인덱스 찾기
            indices = np.where(node_ids == node_idx)[0]
            count = len(indices)
            freq = (count / total_samples) * 100
            
            if count > 1: # 샘플이 2개 이상이어야 통계 계산 가능
                # 1. 해당 노드의 샘플 추출 (Space x Local_Samples)
                local_samples = X_raw[:, indices]
                
                # 2. 평균장 계산 (Composite Mean)
                node_mean_1d = np.mean(local_samples, axis=1)
                node_mean_2d = node_mean_1d.reshape(len(lats), len(lons))
                
                # 3. T-test (One-sample t-test against population mean)
                # 각 격자점별로 t-test 수행
                # Null Hypothesis: Sample mean == Population mean (clim_mean)
                # popmean(clim_mean)은 이제 (Space, 1) 형태이므로 axis=1에 대해 올바르게 브로드캐스팅 됩니다.
                t_stat, p_val = stats.ttest_1samp(local_samples, popmean=clim_mean, axis=1)
                p_val_2d = p_val.reshape(len(lats), len(lons))
                
                # 4. Plot Mean Field
                # JJA u850 범위 고려 (Monsoon westerlies & Trade easterlies)
                cf = ax.contourf(lons, lats, node_mean_2d, 
                                 transform=ccrs.PlateCarree(),
                                 cmap='RdBu_r', 
                                 levels=levels, 
                                 extend='both')
                
                # 5. Plot Stippling (Significance)
                # p-value < 0.05 인 영역에 해치(hatch) 추가
                # levels=[0, 0.05, 1.0] -> 0~0.05 구간에만 hatch 적용
                ax.contourf(lons, lats, p_val_2d,
                            levels=[0, 0.05, 1.0],
                            colors='none', 
                            hatches=['..', ''], # '..'은 점 패턴, ''은 무늬 없음
                            transform=ccrs.PlateCarree())
                
            else:
                ax.text(0.5, 0.5, "Insufficient Data", ha='center', transform=ax.transAxes)
            
            # 지도 요소 추가
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            
            # Gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            if i > 0: gl.left_labels = False
            if j < m2 - 1: gl.bottom_labels = False
                
            ax.set_title(f'Node {node_idx} (n={count}, {freq:.1f}%)')

    # Colorbar 추가
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    # Ticks 설정을 통해 0이 명확히 보이도록 함
    cbar_ticks = np.linspace(-limit, limit, 9) # -20, -15, -10, -5, 0, 5, 10, 15, 20
    fig.colorbar(cf, cax=cbar_ax, label='u850 (m/s)', ticks=cbar_ticks)
    
    plt.suptitle('SOM 3x3 Node Composites (JJA u850)\nStippling indicates p < 0.05 (vs Climatology)', fontsize=16, y=0.98)
    
    output_path = os.path.join(output_dir, 'som_node_composites_stippled.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")

def main():
    # ---------------------------------------------------------
    # 1. 데이터 로드 및 전처리 (Data Preprocessing)
    # ---------------------------------------------------------
    file_path = '../Data/u850.nc'
    results_dir = '../Results'
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(file_path):
        print(f"Error: 파일이 존재하지 않습니다: {file_path}")
        return

    print(f"Loading data from {file_path}...")
    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        print(f"파일 열기 실패: {e}")
        return

    # 위도/경도 범위 설정
    lat_slice = slice(60.0, -10.0) 
    lon_slice = slice(100.0, 180.0)
    
    subset = ds['uwnd'].sel(lat=lat_slice, lon=lon_slice)

    # Monthly anomaly (remove monthly climatology)
    monthly_clim = subset.groupby("time.month").mean("time")
    subset_anom = subset.groupby("time.month") - monthly_clim

    # 10-day low-pass (rolling mean) BEFORE JJA selection
    subset_lp = subset_anom.rolling(time=10, center=True).mean().dropna(dim="time")

    # JJA only
    subset_jja = subset_lp.sel(time=subset_lp.time.dt.month.isin([6, 7, 8]))
    
    print(f"Subset dimensions: {subset_jja.shape}")
    
    # SOM 입력을 위한 데이터 변환 (Space x Time)
    subset_stacked = subset_jja.stack(space=('lat', 'lon'))
    subset_valid = subset_stacked.dropna(dim='space', how='any')
    
    # 원본 데이터 보존 (시각화용)
    X_raw = subset_valid.values.T.copy()
    X = subset_valid.values.T  # 학습용
    
    time_index = subset_valid.time.values
    lats = subset.lat.values 
    lons = subset.lon.values 
    
    nd, nt = X.shape
    print(f"Data reshaped for SOM: nd={nd} (grid points), nt={nt} (time steps)")

    # ---------------------------------------------------------
    # 2. SOM 파라미터 설정 및 학습 (SOM Training)
    # ---------------------------------------------------------
    m1 = 3
    m2 = 3
    nter = 5000
    alpha0 = 1.0
    alphamin = 0.05
    taua = 1000.0
    sigma0 = 2.0
    taus = 500.0
    
    print("\n--- Starting SOM Training ---")
    weights = som.Som(
        x=X, 
        nd=nd, 
        nt=nt, 
        m1=m1, 
        m2=m2, 
        nter=nter, 
        alpha0=alpha0, 
        taua=taua, 
        alphamin=alphamin, 
        sigma0=sigma0, 
        taus=taus,
        normalize_weights=True
    )
    
    # ---------------------------------------------------------
    # 3. 클러스터링 및 결과 추출 (Clustering)
    # ---------------------------------------------------------
    print("\n--- Assigning Clusters ---")
    c1, c2, qerr, total_qe, count = som.cluster(X, weights, nd, nt, m1, m2)
    
    node_ids = (c2 - 1) * m1 + c1
    
    # ---------------------------------------------------------
    # 4. 결과 저장: Neuron Indices (Time Series)
    # ---------------------------------------------------------
    df_result = pd.DataFrame({
        'time': time_index,
        'bmu_i': c1,
        'bmu_j': c2,
        'node_id': node_ids
    })
    
    df_result['year'] = pd.to_datetime(df_result['time']).dt.year
    df_result['month'] = pd.to_datetime(df_result['time']).dt.month
    
    output_indices_file = os.path.join(results_dir, 'som_neuron_indices_jja.csv')
    df_result.to_csv(output_indices_file, index=False)
    print(f"Neuron indices saved to {output_indices_file}")

    # ---------------------------------------------------------
    # 5. 통계 분석: 연도별 Cluster Counts 및 P-value
    # ---------------------------------------------------------
    print("\n--- Calculating Statistics (Counts & P-values) ---")
    
    years = sorted(df_result['year'].unique())
    n_nodes = m1 * m2
    stats_data = []
    prob_null = 1.0 / n_nodes 
    
    for year in years:
        year_data = df_result[df_result['year'] == year]
        n_samples_year = len(year_data)
        row = {'year': year}
        for node in range(1, n_nodes + 1):
            count_k = len(year_data[year_data['node_id'] == node])
            if n_samples_year > 0:
                pval = stats.binomtest(k=count_k, n=n_samples_year, p=prob_null, alternative='greater').pvalue
            else:
                pval = 1.0
            row[f'node_{node}_count'] = count_k
            row[f'node_{node}_pval'] = pval
        stats_data.append(row)
    
    df_stats = pd.DataFrame(stats_data)
    output_stats_file = os.path.join(results_dir, 'som_yearly_stats.csv')
    df_stats.to_csv(output_stats_file, index=False)
    print(f"Yearly statistics saved to {output_stats_file}")

    # ---------------------------------------------------------
    # 6. 시각화 (Visualization) with Stippling
    # ---------------------------------------------------------
    plot_som_nodes(X_raw, node_ids, m1, m2, lats, lons, results_dir)

    print("\n=== Analysis Summary ===")
    print(f"Period: JJA")
    print(f"Output Directory: {results_dir}")
    print("Files generated:")
    print(f" - {os.path.basename(output_indices_file)}")
    print(f" - {os.path.basename(output_stats_file)}")
    print(f" - som_node_composites_stippled.png")

if __name__ == "__main__":
    main()
