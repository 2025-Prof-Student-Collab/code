from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 파일 경로 설정 
INPUT_FILE = Path("merged_data/merged_all_datasets.csv")
OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 데이터 로드
df = pd.read_csv(INPUT_FILE)

## 데이터 요약
print("\n ### 데이터 요약 ###")
print(f"크기: df.shape")
print(f"결측치: \n{df.isna().sum()}\n")
print(f"기초 통계량:\n{df.describe()}")

# 3. 데이터 요약
print("\n### 데이터 요약 ###")
print(f"크기: {df.shape}")
print(f"결측치: \n{df.isna().sum()}\n")
print(f"기초 통계량:\n{df.describe()}")

## 4. 시각화
# (1) 상관관계 히트맵
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_matrix.png")
plt.close()


# (2) GDP vs 모성사망비 산점도 (가장 중요한 관계)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='GDP per capita', y='Maternal Mortality Ratio', hue='Continent', alpha=0.6)
plt.xscale('log')
plt.title('GDP vs Maternal Mortality Ratio (Log Scale)')
plt.grid(True, ls="--")
plt.savefig(OUTPUT_DIR / "scatter_gdp_mortality.png")
plt.close()

# (3) 대륙별 주요 지표 트렌드 (Line Plots)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (1) 1인당 GDP
sns.lineplot(ax=axes[0, 0], data=df, x='Year', y='GDP per capita', hue='Continent', errorbar=None)
axes[0, 0].set_title('Trend of GDP per capita by Continent')

# (2) 모성 사망비
sns.lineplot(ax=axes[0, 1], data=df, x='Year', y='Maternal Mortality Ratio', hue='Continent', errorbar=None)
axes[0, 1].set_title('Trend of Maternal Mortality Ratio by Continent')

# (3) 저체중 아동 비율
sns.lineplot(ax=axes[1, 0], data=df, x='Year', y='Percentage of Underweight Children', hue='Continent', errorbar=None)
axes[1, 0].set_title('Trend of Underweight Children (%) by Continent')

# (4) 여성 중등 교육 등록률
sns.lineplot(ax=axes[1, 1], data=df, x='Year', y='School enrollment rate, secondary, female', hue='Continent', errorbar=None)
axes[1, 1].set_title('Trend of Female Secondary School Enrollment by Continent')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Key Indicators Trends by Continent.png")
plt.close()
