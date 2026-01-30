# NWP_SOM: 북서태평양 대기 순환 패턴 분석

**자체 조직화 맵(Self-Organizing Map, SOM) 기반 동아시아 여름(JJA) 850hPa 동서바람 패턴 분류 및 기후 변동성 연관 분석**

---

## 📌 프로젝트 개요

### 핵심 연구 목표
1. **패턴 인식**: 1991-2023년 JJA 기간 동안 850hPa 동서바람 패턴 분리
2. **인과 메커니즘 규명**: 분류된 패턴과 ONI, PDO, NPGO 등 대규모 기후 변동성의 물리적 연결 관계 파악
3. **활용성**: 여름철 동아시아 몬순의 변동 분석 및 기후 지수 개발

### ✨ 주요 특징
- **분석 기간**: 1991-2023년 (33년간의 여름철 JJA)
- **공간 범위**: 적도 ~ 60°N, 100°E ~ 180°E (북서태평양 중심)
- **분류 방법**: 3×3 Self-Organizing Map (SOM) 클러스터링
- **주요 데이터**: 850hPa 동서바람(U850), 해수면 온도(SST), 외향 장파 복사(OLR)

---

## 📂 프로젝트 구조 (Modernized Workspace)

```
NWP_SOM/
├── Data/                   # 분석 데이터 (NetCDF, CSV)
├── Scripts/                # 파이썬 분석 스크립트 (Modernized)
│   ├── config.py           # 중앙 설정 파일 (경로, 변수, 시즌 정의)
│   ├── analysis_utils.py   # 공통 분석 함수 (데이터 로드, 상관분석, 플롯)
├── Tests/                  # 유닛 테스트
│   └── test_analysis_utils.py
├── Results/                # 분석 결과
│   ├── Figures/            # 생성된 그래프 (PNG)
└── README.md               # 프로젝트 문서
```

---

## 🚀 사용 가이드

### ✅ 재현 가능한 실행 순서
실행 순서와 입력/출력 의존성은 `PIPELINE.md`에 정리되어 있습니다.

### 1. 환경 설정
Conda 환경을 생성/활성화합니다.
```bash
cd /home/hyunwoo/Projects/NWP_SOM
conda env create -f environment.yml
conda activate nwp_som
```
`Data/`와 `Results/`는 로컬 경로(`/home/hyunwoo/Projects/NWP_SOM/`)에 존재합니다.

### 2. 핵심 실행
```bash
python Scripts/run_som_cluster.py
```

### 3. 설정 변경 (`config.py`)
데이터 경로, 분석 시즌, 도메인 설정은 `Scripts/config.py`에서 중앙 관리됩니다.
```python
# Scripts/config.py 예시
SEASONS = {"DJF": 12, "MAM": 3, "JJA": 6, "SON": 9}
```

### 4. 테스트 실행 (`pytest`)
분석 함수의 무결성을 검증하기 위해 유닛 테스트를 실행할 수 있습니다.
```bash
pytest Tests/
```

---

## 📊 주요 산출물
- **SOM 결과**: `Results/som_neuron_indices_jja.csv`, `Results/som_yearly_stats.csv`
- **노드 합성장**: `Results/Figures/som_node_composites_{sst,olr,u850}.png` (SST/OLR는 합성장만 생성)
- **노드 빈도 추세**: `Results/mann_kendall_node_trends.csv`, `Results/mann_kendall_node_trends_decadal.csv`
- **노드 주기성**: `Results/node_periodogram_summary.csv`
- **기후지수 상관**: `Results/node_climate_correlations.csv`
- **유의한 상관만 표시한 히트맵**: `Results/Figures/heatmap_node_climate_correlations_significant.png`
- **계절 지연 상관**: `Results/node_climate_lagged_correlations.csv`, `Results/Figures/heatmap_lagged_correlations_*.png`
- **Kriging variogram (global best)**: spherical 모델 사용 (SST/OLR/U850)

---

## 🔎 추가 분석(선택)

- **SOM 노드 합성장 지도 (SST/OLR/U850)**  
  `Scripts/run_som_node_composites.py`
- **노드 빈도 추세(Mann-Kendall)**  
  `Scripts/run_mk_trend_node_freq.py`  
  `Scripts/run_mk_trend_node_freq_decadal.py`
- **노드 주기성(Periodogram + Red-noise + Fisher g-test)**  
  `Scripts/run_node_periodogram.py`
- **노드-지수 상관/지연 상관**  
  `Scripts/run_node_climate_correlations.py`  
  `Scripts/run_node_climate_correlations_significant_plot.py`  
  `Scripts/run_node_climate_lagged_correlations.py`
- **Kohonen SOM 비교 (선택)**  
  `Scripts/run_kohonen_som.R`  
  `Scripts/run_compare_som_kohonen.py`
