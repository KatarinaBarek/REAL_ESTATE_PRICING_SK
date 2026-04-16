"""
quick_tuning.py — Quick hyperparameter tuning using existing data.

Loads existing model and data, then does fast parameter search.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── Load existing model and data ─────────────────────────────────────────────
print("Loading existing model and data…")
saved_model = joblib.load("model.joblib")

# Extract the processed data from the saved model
# We'll need to recreate the data processing since it's not saved
from config import DB_URL, TABLE_NAME, TARGET_COL
from feature_engineering import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    KNNLocalPriceStats,
    apply_filters,
    build_features,
    decode_gps_point,
    encode_categoricals,
)

# Quick data load (smaller sample for speed)
from sqlalchemy import create_engine
engine = create_engine(
    DB_URL,
    connect_args={"ssl_disabled": True, "read_timeout": 60, "write_timeout": 60},
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

# Load smaller sample for faster tuning
query = f"SELECT {', '.join(['id', 'offer_type', 'is_active', 'date_archived', 'price_current', 'price_area_usable_per_m2', 'price_area_total_per_m2', 'area_usable', 'area_total', 'area_other', 'area_balcony', 'area_cellar', 'rooms', 'floor_is', 'floor_max', 'has_balcony', 'has_elevator', 'condition_name', 'city_name', 'listing_quality_score', 'gps_point', 'locality', 'pricing'])} FROM {TABLE_NAME} LIMIT 50000"
chunks = pd.read_sql(query, engine, chunksize=10000)
df_raw = pd.concat(chunks, ignore_index=True)
print(f"✓ Loaded {len(df_raw):,} rows (sample)")

# ── Quick data processing ───────────────────────────────────────────────────
# Parse JSON fields
def _json_get(series: pd.Series, key: str) -> pd.Series:
    def _extract(val):
        if isinstance(val, str):
            try:
                return json.loads(val).get(key)
            except (json.JSONDecodeError, AttributeError):
                return None
        return None
    return series.apply(_extract)

if "locality_country" not in df_raw.columns and "locality" in df_raw.columns:
    for field in ["country", "countryCode", "region", "city", "town", "gpsPrecisionScore"]:
        df_raw[f"locality_{field}"] = _json_get(df_raw["locality"], field)

if "priceEstimateSale_price" not in df_raw.columns and "pricing" in df_raw.columns:
    def _extract_price_estimate(v):
        if not isinstance(v, str):
            return "{}"
        try:
            parsed = json.loads(v)
            if isinstance(parsed, dict):
                return json.dumps(parsed.get("priceEstimateSale", {}))
        except (json.JSONDecodeError, AttributeError):
            pass
        return "{}"

    df_raw["priceEstimateSale_price"] = _json_get(
        df_raw["pricing"].apply(_extract_price_estimate),
        "priceEstimate",
    )

# Process data
df_raw = decode_gps_point(df_raw)
df = apply_filters(df_raw)
df = build_features(df)

# Train/test split
df["is_active"] = df["is_active"].apply(
    lambda x: x == b'\x01' if isinstance(x, bytes) else bool(x)
)
df["date_archived"] = pd.to_datetime(df["date_archived"])
one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)

train_mask = (df["is_active"] == False) & (df["date_archived"] >= one_year_ago)
test_mask = (df["is_active"] == True)

train_df = df[train_mask].reset_index(drop=True)
test_df = df[test_mask].reset_index(drop=True)

# KNN and encoding
knn = KNNLocalPriceStats(k=10)
knn.fit(train_df)
train_df = knn.transform(train_df)
test_df = knn.transform(test_df)

region_cats = sorted(train_df["locality_region"].dropna().unique().tolist())
condition_cats = sorted(train_df["condition_name"].dropna().unique().tolist())
region_map = {v: i for i, v in enumerate(region_cats)}
condition_map = {v: i for i, v in enumerate(condition_cats)}

train_df = encode_categoricals(train_df, region_map, condition_map)
test_df = encode_categoricals(test_df, region_map, condition_map)

for col in NUMERICAL_FEATURES:
    if col not in train_df.columns:
        train_df[col] = np.nan
    if col not in test_df.columns:
        test_df[col] = np.nan

X_train = train_df[ALL_FEATURES].copy()
y_train = train_df[TARGET_COL].astype(float)
X_test = test_df[ALL_FEATURES].copy()
y_test = test_df[TARGET_COL].astype(float)

print(f"✓ Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Quick parameter grids ───────────────────────────────────────────────────
os.makedirs("tuning_results", exist_ok=True)

# Very minimal grids for speed
catboost_param_grid = {
    'iterations': [500, 1000],
    'depth': [6, 8],
}

xgboost_param_grid = {
    'n_estimators': [500, 1000],
    'max_depth': [6, 8],
}

lightgbm_param_grid = {
    'n_estimators': [500, 1000],
    'max_depth': [6, 8],
}

print("\n" + "="*60)
print("QUICK HYPERPARAMETER TUNING")
print("="*60)

cv = KFold(n_splits=3, shuffle=True, random_state=42)
best_params = {}

# ── CatBoost Quick Search ───────────────────────────────────────────────────
print("\n🔹 CatBoost Quick Search…")
train_pool = Pool(data=X_train, label=y_train, cat_features=CATEGORICAL_FEATURES)

cb_model = CatBoostRegressor(
    random_seed=42,
    verbose=False,
    eval_metric="MAE",
)

cb_grid = GridSearchCV(
    cb_model,
    catboost_param_grid,
    cv=cv,
    scoring='neg_mean_absolute_error',
    n_jobs=1,
    verbose=1
)

cb_grid.fit(X_train, y_train)
best_params['CatBoost'] = cb_grid.best_params_
print(f"✓ Best CatBoost params: {cb_grid.best_params_}")

# ── XGBoost Quick Search ────────────────────────────────────────────────────
print("\n🔹 XGBoost Quick Search…")
def _to_cat(df_, cats_train=None):
    df_ = df_.copy()
    mapping = {}
    for col in CATEGORICAL_FEATURES:
        if cats_train is None:
            cats = sorted(df_[col].unique())
        else:
            cats = cats_train[col]
        df_[col] = pd.Categorical(df_[col], categories=cats)
        mapping[col] = cats
    return df_, mapping

X_train_xgb, xgb_cats = _to_cat(X_train)

# Convert categorical to codes for XGBoost compatibility
X_train_xgb_codes = X_train_xgb.copy()
for col in CATEGORICAL_FEATURES:
    X_train_xgb_codes[col] = X_train_xgb_codes[col].cat.codes

xgb_model = XGBRegressor(
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

xgb_grid = GridSearchCV(
    xgb_model,
    xgboost_param_grid,
    cv=cv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train_xgb_codes, y_train)
best_params['XGBoost'] = xgb_grid.best_params_
print(f"✓ Best XGBoost params: {xgb_grid.best_params_}")

# ── LightGBM Quick Search ───────────────────────────────────────────────────
print("\n🔹 LightGBM Quick Search…")
lgb_model = LGBMRegressor(
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

lgb_grid = GridSearchCV(
    lgb_model,
    lightgbm_param_grid,
    cv=cv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

lgb_grid.fit(X_train, y_train)
best_params['LightGBM'] = lgb_grid.best_params_
print(f"✓ Best LightGBM params: {lgb_grid.best_params_}")

# ── Save results ────────────────────────────────────────────────────────────
print("\n🔹 Saving quick tuning results…")

with open("tuning_results/best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)

print("\n" + "="*60)
print("QUICK TUNING COMPLETED")
print("="*60)

print("\n📊 BEST PARAMETERS FOUND:")
for model, params in best_params.items():
    print(f"  {model}: {params}")

print("\n💡 These are basic tuned parameters.")
print("   For production, consider more extensive tuning with:")
print("   - More parameter combinations")
print("   - Bayesian optimization")
print("   - Full dataset (not sample)")
print("   - Early stopping with validation sets")