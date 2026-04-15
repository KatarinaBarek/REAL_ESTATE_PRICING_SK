"""
train.py — Train and compare pricing models on Slovak flat listing data.

Replicates the full pipeline from Nomerio_RealEstates_SK notebooks 1-3
in pure pandas (no Spark), then adds XGBoost and LightGBM for comparison.

Run:  python train.py
"""

import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sqlalchemy import create_engine
from xgboost import XGBRegressor

from config import DB_URL, MODEL_PATH, TABLE_NAME, TARGET_COL
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


# ── 1. Load raw data ──────────────────────────────────────────────────────────
print("Connecting to database…")
engine = create_engine(DB_URL)
LOAD_COLUMNS = [
    "id", "offer_type", "is_active", "date_archived",
    "price_current", "price_area_usable_per_m2", "price_area_total_per_m2",
    "area_usable", "area_total", "area_other", "area_balcony", "area_cellar",
    "rooms", "floor_is", "floor_max",
    "has_balcony", "has_elevator",
    "condition_name", "city_name",
    "listing_quality_score",
    "gps_point",
    "locality", "pricing",
]
query = f"SELECT {', '.join(LOAD_COLUMNS)} FROM {TABLE_NAME}"
chunks = pd.read_sql(query, engine, chunksize=5000)
df_raw = pd.concat(chunks, ignore_index=True)
print(f"Loaded {len(df_raw):,} rows, {df_raw.shape[1]} columns")

# Parse JSON locality/pricing fields (stored as JSON strings in MySQL)
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


# ── 2. Decode GPS binary → lat/lon (Notebook 2) ──────────────────────────────
print("Decoding GPS points…")
df_raw = decode_gps_point(df_raw)

# ── 3. Filter (Notebook 1) ────────────────────────────────────────────────────
df = apply_filters(df_raw)


# ── 4. Feature engineering (Notebooks 1 & 2) ─────────────────────────────────
print("Engineering features…")
df = build_features(df)


# ── 4. Train/test split — archived (≤1 year) vs active ───────────────────────
# is_active is stored as bytes in MySQL
df["is_active"] = df["is_active"].apply(
    lambda x: x == b'\x01' if isinstance(x, bytes) else bool(x)
)
df["date_archived"] = pd.to_datetime(df["date_archived"])
one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)

train_mask = (df["is_active"] == False) & (df["date_archived"] >= one_year_ago)
test_mask  = (df["is_active"] == True)

train_df = df[train_mask].reset_index(drop=True)
test_df  = df[test_mask].reset_index(drop=True)

print(f"\nTrain (archived ≤1yr): {len(train_df):,}  |  Test (active): {len(test_df):,}")


# ── 5. KNN local price stats — fit on train only, transform both ──────────────
print("Computing KNN local price statistics…")
knn = KNNLocalPriceStats(k=10)
knn.fit(train_df)
train_df = knn.transform(train_df)
test_df  = knn.transform(test_df)


# ── 6. Ordinal encoding (mirrors StringIndexer from Notebook 3) ───────────────
region_cats    = sorted(train_df["locality_region"].dropna().unique().tolist())
condition_cats = sorted(train_df["condition_name"].dropna().unique().tolist())
region_map     = {v: i for i, v in enumerate(region_cats)}
condition_map  = {v: i for i, v in enumerate(condition_cats)}

train_df = encode_categoricals(train_df, region_map, condition_map)
test_df  = encode_categoricals(test_df,  region_map, condition_map)

for col in NUMERICAL_FEATURES:
    if col not in train_df.columns:
        train_df[col] = np.nan
    if col not in test_df.columns:
        test_df[col] = np.nan

X_train = train_df[ALL_FEATURES].copy()
y_train = train_df[TARGET_COL].astype(float)
X_test  = test_df[ALL_FEATURES].copy()
y_test  = test_df[TARGET_COL].astype(float)


# ── 7. Train models ───────────────────────────────────────────────────────────

def _eval(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))
    r2   = r2_score(y_true, y_pred)
    return {"mae": mae, "mape_pct": mape, "rmse": rmse, "r2": r2}


results = {}

# ── CatBoost (replicates Notebook 3 exactly) ─────────────────────────────────
print("\n── CatBoost ─────────────────────────────────────────")
train_pool = Pool(data=X_train, label=y_train, cat_features=CATEGORICAL_FEATURES)
test_pool  = Pool(data=X_test,  label=y_test,  cat_features=CATEGORICAL_FEATURES)

cb_model = CatBoostRegressor(
    iterations=2000,
    depth=8,
    learning_rate=0.03,
    loss_function="RMSE",
    eval_metric="R2",
    custom_metric=["RMSE", "MAE"],
    random_seed=42,
    l2_leaf_reg=5,
    min_data_in_leaf=20,
    verbose=200,
)
cb_model.fit(train_pool, eval_set=test_pool, use_best_model=True)
cb_preds = cb_model.predict(X_test)
results["CatBoost"] = {"model": cb_model, "kind": "catboost", **_eval(y_test, cb_preds)}

# ── XGBoost ───────────────────────────────────────────────────────────────────
print("\n── XGBoost ──────────────────────────────────────────")
# Encode categoricals as pandas category dtype (XGBoost ≥ 1.7 native support)
def _to_cat(df_: pd.DataFrame, cats_train=None):  # type: (pd.DataFrame, dict) -> tuple
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

Xtr_xgb, xgb_cats = _to_cat(X_train)
Xte_xgb, _        = _to_cat(X_test, xgb_cats)

xgb_model = XGBRegressor(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_lambda=5,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric="mae",
    enable_categorical=True,
    verbosity=1,
)
xgb_model.fit(Xtr_xgb, y_train, eval_set=[(Xte_xgb, y_test)], verbose=200)
xgb_preds = xgb_model.predict(Xte_xgb)
results["XGBoost"] = {"model": xgb_model, "kind": "xgboost", "cats": xgb_cats, **_eval(y_test, xgb_preds)}

# ── LightGBM ──────────────────────────────────────────────────────────────────
print("\n── LightGBM ─────────────────────────────────────────")
Xtr_lgb, lgb_cats = _to_cat(X_train)
Xte_lgb, _        = _to_cat(X_test, lgb_cats)

lgb_model = LGBMRegressor(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_lambda=5,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(Xtr_lgb, y_train, eval_set=[(Xte_lgb, y_test)])
lgb_preds = lgb_model.predict(Xte_lgb)
results["LightGBM"] = {"model": lgb_model, "kind": "lightgbm", "cats": lgb_cats, **_eval(y_test, lgb_preds)}


# ── 8. Compare ────────────────────────────────────────────────────────────────
print("\n── Model comparison ─────────────────────────────────────────────────")
print(f"{'Model':<12} {'MAE (EUR)':>12} {'MAPE':>8} {'RMSE (EUR)':>12} {'R²':>8}")
print("─" * 56)
for name, r in results.items():
    print(f"{name:<12} {r['mae']:>12,.0f} {r['mape_pct']:>7.2f}% {r['rmse']:>12,.0f} {r['r2']:>8.4f}")

best_name = min(results, key=lambda n: results[n]["mape_pct"])
print(f"\nBest model: {best_name}  (MAPE {results[best_name]['mape_pct']:.2f}%)")


# ── 9. Save artifact ──────────────────────────────────────────────────────────
best = results[best_name]

artifact = {
    "model":               best["model"],
    "model_name":          best_name,
    "model_kind":          best["kind"],
    # For XGBoost/LightGBM category alignment at inference time
    "cats":                best.get("cats"),
    "knn_transformer":     knn,
    "region_map":          region_map,
    "condition_map":       condition_map,
    "numerical_features":  NUMERICAL_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "all_features":        ALL_FEATURES,
    "target_col":          TARGET_COL,
    "eval": {
        name: {k: v for k, v in r.items() if k not in ("model", "cats")}
        for name, r in results.items()
    },
}
joblib.dump(artifact, MODEL_PATH)
print(f"\nSaved → {MODEL_PATH}")
