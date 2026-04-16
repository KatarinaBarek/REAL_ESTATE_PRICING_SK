"""
retrain_with_best_params.py — Retrain models with optimal hyperparameters.

Loads the best parameters from tuning_results/best_params.json and retrains
all models with these optimal settings. Compares performance with original models.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── Load best parameters ────────────────────────────────────────────────────
print("Loading best parameters from tuning...")
with open("tuning_results/best_params.json", "r") as f:
    best_params = json.load(f)

print("✓ Loaded optimal parameters:")
for model, params in best_params.items():
    print(f"  {model}: {params}")

# ── Load original model and data ────────────────────────────────────────────
print("\nLoading original model and data...")
saved_model = joblib.load("model.joblib")

# Extract the processed data from the saved model
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

# Load full dataset for final training
from sqlalchemy import create_engine
engine = create_engine(
    DB_URL,
    connect_args={"ssl_disabled": True, "read_timeout": 60, "write_timeout": 60},
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

query = f"SELECT {', '.join(['id', 'offer_type', 'is_active', 'date_archived', 'price_current', 'price_area_usable_per_m2', 'price_area_total_per_m2', 'area_usable', 'area_total', 'area_other', 'area_balcony', 'area_cellar', 'rooms', 'floor_is', 'floor_max', 'has_balcony', 'has_elevator', 'condition_name', 'city_name', 'listing_quality_score', 'gps_point', 'locality', 'pricing'])} FROM {TABLE_NAME}"
chunks = pd.read_sql(query, engine, chunksize=10000)

df_raw = pd.DataFrame()
for chunk in chunks:
    df_raw = pd.concat([df_raw, chunk], ignore_index=True)

print(f"✓ Loaded {len(df_raw):,} rows (full dataset)")

# Process data (same order as train.py)
# First extract JSON fields
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
            pricing = json.loads(v)
            return pricing.get("priceEstimate", {})
        except (json.JSONDecodeError, AttributeError):
            return "{}"

    df_raw["priceEstimateSale_price"] = _json_get(
        df_raw["pricing"].apply(_extract_price_estimate),
        "priceEstimate",
    )

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

# Prepare features
X_train = train_df[ALL_FEATURES]
y_train = train_df[TARGET_COL]
X_test = test_df[ALL_FEATURES]
y_test = test_df[TARGET_COL]

print(f"✓ Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Retrain models with optimal parameters ──────────────────────────────────
print("\n" + "="*60)
print("RETRAINING MODELS WITH OPTIMAL PARAMETERS")
print("="*60)

tuned_models = {}
results = {}

# CatBoost
print("\n🔹 Retraining CatBoost...")
cb_params = best_params['CatBoost']
cb_model = CatBoostRegressor(
    **cb_params,
    random_seed=42,
    verbose=False,
    eval_metric="MAE",
)
cb_model.fit(X_train, y_train, cat_features=CATEGORICAL_FEATURES)
tuned_models['CatBoost'] = cb_model

# XGBoost
print("\n🔹 Retraining XGBoost...")
xgb_params = best_params['XGBoost']
xgb_model = XGBRegressor(
    **xgb_params,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
# Convert categorical to codes for XGBoost
X_train_xgb = X_train.copy()
X_test_xgb = X_test.copy()
for col in CATEGORICAL_FEATURES:
    X_train_xgb[col] = X_train_xgb[col].astype('category').cat.codes
    X_test_xgb[col] = X_test_xgb[col].astype('category').cat.codes

xgb_model.fit(X_train_xgb, y_train)
tuned_models['XGBoost'] = xgb_model

# LightGBM
print("\n🔹 Retraining LightGBM...")
lgb_params = best_params['LightGBM']
lgb_model = LGBMRegressor(
    **lgb_params,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)
lgb_model.fit(X_train, y_train)
tuned_models['LightGBM'] = lgb_model

# ── Evaluate tuned models ───────────────────────────────────────────────────
print("\n🔹 Evaluating tuned models...")

for name, model in tuned_models.items():
    print(f"\n📊 {name} Results:")

    if name == 'XGBoost':
        X_test_eval = X_test_xgb
    else:
        X_test_eval = X_test

    y_pred = model.predict(X_test_eval)

    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    }

    print(".2f")
    print(".2f")
    print(".3f")

# ── Compare with original models ────────────────────────────────────────────
print("\n" + "="*60)
print("COMPARISON: ORIGINAL VS TUNED MODELS")
print("="*60)

original_results = {}
for name, model in saved_model.items():
    if name in ['CatBoost', 'XGBoost', 'LightGBM']:
        print(f"\n📊 Original {name} Results:")

        if name == 'XGBoost':
            X_test_eval = X_test_xgb
        else:
            X_test_eval = X_test

        y_pred = model.predict(X_test_eval)

        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        original_results[name] = {
            'MAE': mae,
            'MAPE': mape,
            'R²': r2
        }

        print(".2f")
        print(".2f")
        print(".3f")

print("\n" + "="*60)
print("PERFORMANCE IMPROVEMENT SUMMARY")
print("="*60)

for name in ['CatBoost', 'XGBoost', 'LightGBM']:
    if name in original_results and name in results:
        orig_r2 = original_results[name]['R²']
        tuned_r2 = results[name]['R²']
        improvement = tuned_r2 - orig_r2

        print(f"\n{name}:")
        print(".3f")
        print(".3f")
        print(".3f")
        print(f"  {'📈 Improved' if improvement > 0 else '📉 Declined'} by {abs(improvement):.3f}")

# ── Save tuned models ───────────────────────────────────────────────────────
print("\n🔹 Saving tuned models...")
joblib.dump(tuned_models, "tuned_models.joblib")
print("✓ Saved tuned models to tuned_models.joblib")

print("\n" + "="*60)
print("RETRAINING COMPLETED SUCCESSFULLY")
print("="*60)