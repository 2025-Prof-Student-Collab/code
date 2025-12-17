import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


df = pd.read_csv("/content/merged_all_datasets.csv")

# 열 이름 정리
df = df.rename(
    columns={
        "Govt. Health Expenditure": "govt_health_expenditure",
        "Maternal Mortality Ratio": "maternal_mortality_ratio",
        "Mean age of women at first birth": "mean_age_first_birth",
        "Percentage of Underweight Children": "pct_underweight_children",
        "Prevelance of Underweight among Adults": "pct_underweight_adults",
        "School enrollment rate, secondary, female": "school_enrollment_secondary_female",
        "GDP per capita": "gdp_per_capita",
    }
)

## 2. 데이터 준비 (Feature Selection & Imputation)

# 사용할 특성(Feature) 정의
feature_cols = [
    "Continent",
    "Year",
    "gdp_per_capita",
    "govt_health_expenditure",
    "mean_age_first_birth",
    "pct_underweight_children",
    "pct_underweight_adults",
    "school_enrollment_secondary_female",
]

X = df[feature_cols].copy()
y = df["maternal_mortality_ratio"]

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

## 3. 파이프라인 구축 (Preprocessing + Model)
categorical_cols = ["Continent"]

# feature_cols 중 categorical_cols를 제외한 나머지는 수치형으로 간주
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

# 전처리기 설정 (범주형: 원-핫 인코딩, 수치형: 그대로 통과)
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# XGBoost 모델 설정
xgb_model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=4,
)

# 파이프라인 결합
pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", xgb_model),
    ]
)

## 4. 모델 학습 및 평가
pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)

# RMSE 계산 (scikit-learn 버전에 따라 squared=False 옵션 대신 **0.5 사용)
rmse = mean_squared_error(y_test, preds) ** 0.5
print(f"테스트 RMSE: {rmse:.2f} (데이터 개수: {len(y_test)})")

# y_test(실제값)의 통계 확인
print(f"실제값 평균: {y_test.mean():.2f}")
print(f"실제값 최소~최대: {y_test.min()} ~ {y_test.max()}")

# 변동 계수(CV)로 오차율 가늠하기 (RMSE / 평균)
cv = rmse / y_test.mean()
print(f"평균 대비 오차율(CV): {cv * 100:.1f}%")

r2 = r2_score(y_test, preds)
print(f"R2 Score: {r2:.4f}")

## 5. 시각화 (Visualization)
import matplotlib.pyplot as plt
import seaborn as sns

# 그래프 스타
sns.set_theme(style="whitegrid")

# (1) 실제값 vs 예측값 산점도
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=preds, alpha=0.6, color='blue', edgecolor='w')

# 정답 기준선 (빨간 점선)
max_val = max(y_test.max(), preds.max())
plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Fit')

plt.xlabel('Actual MMR') # 실제값
plt.ylabel('Predicted MMR') # 예측값
plt.title(f'Prediction Result (R2 Score: {r2:.4f})')
plt.legend()
plt.show()

# (2) 변수 중요도 (Feature Importance)
model = pipeline.named_steps['model']
preprocessor = pipeline.named_steps['preprocess']
feature_names = preprocessor.get_feature_names_out()

# 1단계: 원래 중요도 데이터프레임 만들기
raw_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
})

# 2단계: 'Continent' 관련 변수들만 찾아서 합치기
continent_score = raw_importance_df[
    raw_importance_df['Feature'].str.contains('Continent')
]['Importance'].sum()

# 3단계: 나머지 변수들만 남기기
clean_df = raw_importance_df[
    ~raw_importance_df['Feature'].str.contains('Continent')
].copy()

# 4단계: 합친 Continent 점수 추가하기
new_row = pd.DataFrame({'Feature': ['Continent (Combined)'], 'Importance': [continent_score]})
clean_df = pd.concat([clean_df, new_row], ignore_index=True)

# "num__" 같은 접두사 제거
clean_df['Feature'] = clean_df['Feature'].str.replace('num__', '').str.replace('cat__', '')

# 상위 10개 추출
top_10_features = clean_df.sort_values(by='Importance', ascending=False).head(10)

# 막대 그래프 그리기
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=top_10_features, 
    palette='viridis', 
    hue='Feature',  
    legend=False    
)
plt.title('Top 10 Feature Importance') # 변수 중요도
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()