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