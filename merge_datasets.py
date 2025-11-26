import pandas as pd
from pathlib import Path

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# CSV 파일 목록
csv_files = [
    "1. Percentage-of-underweight-children-data.csv",
    "2. Prevalence of Underweight among Female Adults (Age Standardized Estimate).csv",
    "3. GDP per capita (Constant 2015 US).csv",
    "4. Domestic general government health expenditure ( of GDP).csv",
    "5. Maternal Mortality Ratio.csv",
    "6. Mean-age-at-first-birth-of-women-aged-20-50-data.csv",
    "7. School enrollment secondary female ( gross).csv"
]

CONTINENT_MAP = {
    'ABW': 'North America', 'AFG': 'Asia', 'AGO': 'Africa', 'ALB': 'Europe', 
    'AND': 'Europe', 'ARE': 'Asia', 'ARG': 'South America', 'ARM': 'Asia', 
    'ASM': 'Oceania', 'ATG': 'North America', 'AUS': 'Oceania', 'AUT': 'Europe',
    'AZE': 'Asia', 'BDI': 'Africa', 'BEL': 'Europe', 'BEN': 'Africa', 
    'BFA': 'Africa', 'BGD': 'Asia', 'BGR': 'Europe', 'BHR': 'Asia', 
    'BHS': 'North America', 'BIH': 'Europe', 'BLR': 'Europe', 'BLZ': 'North America',
    'BMU': 'North America', 'BOL': 'South America', 'BRA': 'South America', 
    'BRB': 'North America', 'BRN': 'Asia', 'BTN': 'Asia', 'BWA': 'Africa',
    'CAF': 'Africa', 'CAN': 'North America', 'CHE': 'Europe', 'CHL': 'South America',
    'CHN': 'Asia', 'CIV': 'Africa', 'CMR': 'Africa', 'COD': 'Africa', 
    'COG': 'Africa', 'COL': 'South America', 'COM': 'Africa', 'CPV': 'Africa',
    'CRI': 'North America', 'CUB': 'North America', 'CYM': 'North America',
    'CYP': 'Asia', 'CZE': 'Europe', 'DEU': 'Europe', 'DJI': 'Africa', 
    'DMA': 'North America', 'DNK': 'Europe', 'DOM': 'North America', 'DZA': 'Africa',
    'ECU': 'South America', 'EGY': 'Africa', 'ERI': 'Africa', 'ESP': 'Europe', 
    'EST': 'Europe', 'ETH': 'Africa', 'FIN': 'Europe', 'FJI': 'Oceania', 
    'FRA': 'Europe', 'FRO': 'Europe', 'FSM': 'Oceania', 'GAB': 'Africa', 
    'GBR': 'Europe', 'GEO': 'Asia', 'GHA': 'Africa', 'GIN': 'Africa', 
    'GMB': 'Africa', 'GNB': 'Africa', 'GNQ': 'Africa', 'GRC': 'Europe', 
    'GRD': 'North America', 'GRL': 'North America', 'GTM': 'North America', 
    'GUM': 'Oceania', 'GUY': 'South America', 'HKG': 'Asia', 'HND': 'North America', 
    'HRV': 'Europe', 'HTI': 'North America', 'HUN': 'Europe', 'IDN': 'Asia', 
    'IND': 'Asia', 'IRL': 'Europe', 'IRN': 'Asia', 'IRQ': 'Asia', 
    'ISL': 'Europe', 'ISR': 'Asia', 'ITA': 'Europe', 'JAM': 'North America', 
    'JOR': 'Asia', 'JPN': 'Asia', 'KAZ': 'Asia', 'KEN': 'Africa', 
    'KGZ': 'Asia', 'KHM': 'Asia', 'KIR': 'Oceania', 'KNA': 'North America', 
    'KOR': 'Asia', 'KWT': 'Asia', 'LAO': 'Asia', 'LBN': 'Asia', 
    'LBR': 'Africa', 'LBY': 'Africa', 'LCA': 'North America', 'LIE': 'Europe', 
    'LKA': 'Asia', 'LSO': 'Africa', 'LTU': 'Europe', 'LUX': 'Europe', 
    'LVA': 'Europe', 'MAC': 'Asia', 'MAR': 'Africa', 'MCO': 'Europe', 
    'MDA': 'Europe', 'MDG': 'Africa', 'MDV': 'Asia', 'MEX': 'North America', 
    'MHL': 'Oceania', 'MKD': 'Europe', 'MLI': 'Africa', 'MLT': 'Europe', 
    'MMR': 'Asia', 'MNE': 'Europe', 'MNG': 'Asia', 'MNP': 'Oceania', 
    'MOZ': 'Africa', 'MRT': 'Africa', 'MUS': 'Africa', 'MWI': 'Africa', 
    'MYS': 'Asia', 'NAM': 'Africa', 'NCL': 'Oceania', 'NER': 'Africa', 
    'NGA': 'Africa', 'NIC': 'North America', 'NLD': 'Europe', 'NOR': 'Europe', 
    'NPL': 'Asia', 'NRU': 'Oceania', 'NZL': 'Oceania', 'OMN': 'Asia', 
    'PAK': 'Asia', 'PAN': 'North America', 'PER': 'South America', 'PHL': 'Asia', 
    'PLW': 'Oceania', 'PNG': 'Oceania', 'POL': 'Europe', 'PRI': 'North America', 
    'PRK': 'Asia', 'PRT': 'Europe', 'PRY': 'South America', 'PSE': 'Asia', 
    'PYF': 'Oceania', 'QAT': 'Asia', 'ROU': 'Europe', 'RUS': 'Europe', 
    'RWA': 'Africa', 'SAU': 'Asia', 'SDN': 'Africa', 'SEN': 'Africa', 
    'SGP': 'Asia', 'SLB': 'Oceania', 'SLE': 'Africa', 'SLV': 'North America', 
    'SMR': 'Europe', 'SOM': 'Africa', 'SRB': 'Europe', 'SSD': 'Africa', 
    'STP': 'Africa', 'SUR': 'South America', 'SVK': 'Europe', 'SVN': 'Europe', 
    'SWE': 'Europe', 'SWZ': 'Africa', 'SYC': 'Africa', 'SYR': 'Asia', 
    'TCA': 'North America', 'TCD': 'Africa', 'TGO': 'Africa', 'THA': 'Asia', 
    'TJK': 'Asia', 'TKM': 'Asia', 'TLS': 'Asia', 'TON': 'Oceania', 
    'TTO': 'North America', 'TUN': 'Africa', 'TUR': 'Asia', 'TUV': 'Oceania', 
    'TWN': 'Asia', 'TZA': 'Africa', 'UGA': 'Africa', 'UKR': 'Europe', 
    'URY': 'South America', 'USA': 'North America', 'UZB': 'Asia', 'VCT': 'North America', 
    'VEN': 'South America', 'VGB': 'North America', 'VIR': 'North America', 'VNM': 'Asia', 
    'VUT': 'Oceania', 'WSM': 'Oceania', 'YEM': 'Asia', 'ZAF': 'Africa', 
    'ZMB': 'Africa', 'ZWE': 'Africa',
    'XKX': 'Europe', 'KSV': 'Europe', 'CUW': 'North America', 'SXM': 'North America', 
    'MAF': 'North America', 'CHI': 'Europe', 'IMN': 'Europe', 'GIB': 'Europe', 
    'PLW': 'Oceania', 'NIU': 'Oceania', 'COK': 'Oceania', 'SDN': 'Africa'
}

## 데이터 병합
print("### 데이터 병합 및 정리 시작 ###")
# 첫 번째 파일 로드
df_merged = pd.read_csv(csv_files[0])

# 나머지 파일들을 순차적으로 merge
for i, file in enumerate(csv_files[1:], 2):
    df_temp = pd.read_csv(file)
    
    # Country Code와 Year를 기준으로 merge
    df_merged = pd.merge(
        df_merged, 
        df_temp, 
        on=['Country Code', 'Year'], 
        how='outer',
        suffixes=('', f'_dup{i}')
        # 기준 df는 접미사 없음, 추가되는 df는 _dup2, _dup3... 붙임
    )

## 데이터 중복 컬럼 통합 및 결측치 채우기   
# 처리할 타겟 컬럼 리스트 (이름과 대륙 정보)
target_columns = ['Country Name', 'Continent']

for target in target_columns:
    # col.startswith(target) = 해당 이름으로 시작하는 모든 컬럼 찾기
    dup_cols = [col for col in df_merged.columns if col.startswith(target)]
    
    # 중복 컬럼이 있다면
    if len(dup_cols) > 1:
        # 첫 번째 컬럼(원본)의 빈 값(NaN)을 나머지 중복 컬럼들의 값으로 채움
        for col in dup_cols[1:]:
            df_merged[target] = df_merged[target].fillna(df_merged[col])
        
        # 값을 다 옮겼으면 불필요한 중복 컬럼(_dup2, _dup3 등) 삭제
        df_merged = df_merged.drop(columns=dup_cols[1:])

## 대륙 정보 매핑 및 그룹 데이터 정제
# Continent 결측치를 Country Code를 이용해 채우기
df_merged['Continent'] = df_merged['Continent'].fillna(df_merged['Country Code'].map(CONTINENT_MAP))

# 여전히 대륙 정보가 없는 행 삭제
# :world, High income 와 같은 데이터들은 특정 대륙에 속하지 않기 때문에 분석 대상에서 제외
before_len = len(df_merged)
df_merged = df_merged.dropna(subset=['Continent'])
after_len = len(df_merged)

# print(f"  -> 그룹 데이터 제거: {before_len}행 -> {after_len}행")
print("\n### 대륙 결측치 채우기 완료 ###")
remaining_nan_count = df_merged['Continent'].isna().sum()
print(f"남은 Continent 결측치 수: {remaining_nan_count}")

## 다중대치(MICE)를 통한 결측치 처리
# 수치형 컬럼 선택
numeric_cols = df_merged.select_dtypes(include=['number']).columns

print("\n### MICE 대치 대상 수치형 컬럼 목록 ###")
print(f"[{len(numeric_cols)}개]\n" + "\n".join(numeric_cols))

if len(numeric_cols) > 0:
    # 원래 데이터 타입 저장 (복원용)
    original_dtypes = df_merged[numeric_cols].dtypes
    
    # MICE 알고리즘 학습
    imputer = IterativeImputer(random_state=42, min_value=0)
    imputed_array = imputer.fit_transform(df_merged[numeric_cols])
    df_imputed = pd.DataFrame(imputed_array, columns=numeric_cols, index=df_merged.index)

    # 원본 DF 업데이트
    for col in numeric_cols:
        df_merged[col] = df_imputed[col]

        # MICE 출력 값은 대개 float 타입
        # -> 원본이 정수형이면 업데이트 된 값을 반올림하여 정수형으로 변환
        if pd.api.types.is_integer_dtype(original_dtypes[col]):
                df_merged[col] = df_merged[col].round().astype(original_dtypes[col])

## 컬럼 재정렬
cols_order = ['Country Name', 'Country Code', 'Year']

# 대륙 정보가 있다면 index=2 위치에 추가
if 'Continent' in df_merged.columns:
    cols_order.insert(2, 'Continent')

# 나머지 컬럼들 추가
other_cols = [col for col in df_merged.columns if col not in cols_order]
cols_order.extend(sorted(other_cols))

# 최종 순서 적용
df_merged = df_merged[cols_order]

# 결과 저장 폴더 생성
output_dir = Path("merged_data")
output_dir.mkdir(exist_ok=True)

# 결과 저장
output_file = output_dir / "merged_all_datasets.csv"
df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n### 병합 완료! ###")
print(f"총 행 수: {len(df_merged)}")
print(f"총 컬럼 수: {len(df_merged.columns)}")
print(f"결과 파일: {output_file}")