"""
generate_results.py — Generate SHAP values and visualizations for model analysis.

Creates:
- SHAP summary plots
- Feature importance plots
- Model comparison visualizations
- SHAP values CSV for interpretation
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sqlalchemy import create_engine
import shap
import warnings
warnings.filterwarnings('ignore')

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

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

print("Loading and preparing data…")

# ── Load raw data from database ───────────────────────────────────────────────
engine = create_engine(
    DB_URL,
    connect_args={"ssl_disabled": True, "read_timeout": 60, "write_timeout": 60},
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)
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
print(f"✓ Loaded {len(df_raw):,} rows")

# ── Parse JSON locality/pricing fields ────────────────────────────────────────
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

# ── Decode GPS and filter ─────────────────────────────────────────────────────
df_raw = decode_gps_point(df_raw)
df = apply_filters(df_raw)
df = build_features(df)

# ── Train/test split ──────────────────────────────────────────────────────────
df["is_active"] = df["is_active"].apply(
    lambda x: x == b'\x01' if isinstance(x, bytes) else bool(x)
)
df["date_archived"] = pd.to_datetime(df["date_archived"])
one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)

train_mask = (df["is_active"] == False) & (df["date_archived"] >= one_year_ago)
test_mask  = (df["is_active"] == True)

train_df = df[train_mask].reset_index(drop=True)
test_df  = df[test_mask].reset_index(drop=True)

# ── KNN local price stats ─────────────────────────────────────────────────────
knn = KNNLocalPriceStats(k=10)
knn.fit(train_df)
train_df = knn.transform(train_df)
test_df  = knn.transform(test_df)

# ── Encoding ──────────────────────────────────────────────────────────────────
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

print(f"✓ Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Load trained model ────────────────────────────────────────────────────────
saved_model = joblib.load("model.joblib")
cb_model = saved_model["model"]
eval_metrics = saved_model["eval"]

print("\n" + "="*70)
print("GENERATING SHAP VALUES AND VISUALIZATIONS")
print("="*70)

# ── CatBoost: SHAP Analysis ──────────────────────────────────────────────────
print("\n🔹 CatBoost SHAP Analysis…")

# Create explainer and compute SHAP values (sample for speed)
sample_size = min(1000, len(X_test))
sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
X_test_sample = X_test.iloc[sample_idx].reset_index(drop=True)

explainer = shap.TreeExplainer(cb_model)
shap_values = explainer.shap_values(X_test_sample)

# ── SHAP Summary Plot ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("results/shap_summary_catboost.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved: results/shap_summary_catboost.png")

# ── SHAP Beeswarm Plot ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.tight_layout()
plt.savefig("results/shap_beeswarm_catboost.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved: results/shap_beeswarm_catboost.png")

# ── Feature Importance (built-in) ─────────────────────────────────────────────
feature_importance = cb_model.get_feature_importance()
feature_names = ALL_FEATURES
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importance
}).sort_values("importance", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis")
ax.set_title("CatBoost Top 15 Features by Importance", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance", fontsize=12)
plt.tight_layout()
plt.savefig("results/catboost_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved: results/catboost_feature_importance.png")

# ── Model Predictions Comparison ──────────────────────────────────────────────
print("\n🔹 Model Predictions Comparison…")
cb_preds = cb_model.predict(X_test)

# Prepare XGBoost predictions
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

Xte_xgb, _ = _to_cat(X_test, {col: sorted(X_train[col].unique().tolist()) for col in CATEGORICAL_FEATURES})

# Train XGBoost for comparison (same parameters as train.py)
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
    verbosity=0,
)
Xtr_xgb, xgb_cats = _to_cat(X_train)
Xte_xgb, _ = _to_cat(X_test, xgb_cats)
xgb_model.fit(Xtr_xgb, y_train, eval_set=[(Xte_xgb, y_test)], verbose=False)
xgb_preds = xgb_model.predict(Xte_xgb)

# Predictions scatter plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# CatBoost
axes[0].scatter(y_test, cb_preds, alpha=0.5, s=20)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel("Actual Price (EUR)", fontsize=11)
axes[0].set_ylabel("Predicted Price (EUR)", fontsize=11)
axes[0].set_title(f"CatBoost (R² = {r2_score(y_test, cb_preds):.4f})", fontweight="bold")
axes[0].grid(True, alpha=0.3)

# XGBoost
axes[1].scatter(y_test, xgb_preds, alpha=0.5, s=20, color="orange")
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel("Actual Price (EUR)", fontsize=11)
axes[1].set_ylabel("Predicted Price (EUR)", fontsize=11)
axes[1].set_title(f"XGBoost (R² = {r2_score(y_test, xgb_preds):.4f})", fontweight="bold")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/predictions_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved: results/predictions_comparison.png")

# ── Residuals Analysis ────────────────────────────────────────────────────────
residuals = y_test - cb_preds
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(cb_preds, residuals, alpha=0.5, s=20)
axes[0].axhline(y=0, color="r", linestyle="--", lw=2)
axes[0].set_xlabel("Predicted Price (EUR)", fontsize=11)
axes[0].set_ylabel("Residuals (EUR)", fontsize=11)
axes[0].set_title("CatBoost Residuals vs Predictions", fontweight="bold")
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
axes[1].set_xlabel("Residuals (EUR)", fontsize=11)
axes[1].set_ylabel("Frequency", fontsize=11)
axes[1].set_title("Distribution of Residuals", fontweight="bold")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/residuals_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved: results/residuals_analysis.png")

# ── Model Metrics Comparison ──────────────────────────────────────────────────
print("\n🔹 Model Metrics Summary…")

# Use metrics from the saved trained model
cb_mae = eval_metrics["CatBoost"]["mae"]
cb_mape = eval_metrics["CatBoost"]["mape_pct"]
cb_rmse = eval_metrics["CatBoost"]["rmse"]
cb_r2 = eval_metrics["CatBoost"]["r2"]

xgb_mae = eval_metrics["XGBoost"]["mae"]
xgb_mape = eval_metrics["XGBoost"]["mape_pct"]
xgb_rmse = eval_metrics["XGBoost"]["rmse"]
xgb_r2 = eval_metrics["XGBoost"]["r2"]

lgb_mae = eval_metrics["LightGBM"]["mae"]
lgb_mape = eval_metrics["LightGBM"]["mape_pct"]
lgb_rmse = eval_metrics["LightGBM"]["rmse"]
lgb_r2 = eval_metrics["LightGBM"]["r2"]

metrics_data = {
    "Model": ["CatBoost", "XGBoost", "LightGBM"],
    "MAE (EUR)": [cb_mae, xgb_mae, lgb_mae],
    "MAPE (%)": [cb_mape, xgb_mape, lgb_mape],
    "RMSE (EUR)": [cb_rmse, xgb_rmse, lgb_rmse],
    "R²": [cb_r2, xgb_r2, lgb_r2],
}
metrics_df = pd.DataFrame(metrics_data)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# MAE
axes[0, 0].bar(metrics_df["Model"], metrics_df["MAE (EUR)"], color=["#2ecc71", "#3498db", "#e74c3c"])
axes[0, 0].set_ylabel("MAE (EUR)", fontsize=11)
axes[0, 0].set_title("Mean Absolute Error", fontweight="bold")
axes[0, 0].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(metrics_df["MAE (EUR)"]):
    axes[0, 0].text(i, v + 500, f"{v:,.0f}", ha="center", fontweight="bold")

# MAPE
axes[0, 1].bar(metrics_df["Model"], metrics_df["MAPE (%)"], color=["#2ecc71", "#3498db", "#e74c3c"])
axes[0, 1].set_ylabel("MAPE (%)", fontsize=11)
axes[0, 1].set_title("Mean Absolute Percentage Error", fontweight="bold")
axes[0, 1].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(metrics_df["MAPE (%)"]):
    axes[0, 1].text(i, v + 0.3, f"{v:.2f}%", ha="center", fontweight="bold")

# RMSE
axes[1, 0].bar(metrics_df["Model"], metrics_df["RMSE (EUR)"], color=["#2ecc71", "#3498db", "#e74c3c"])
axes[1, 0].set_ylabel("RMSE (EUR)", fontsize=11)
axes[1, 0].set_title("Root Mean Squared Error", fontweight="bold")
axes[1, 0].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(metrics_df["RMSE (EUR)"]):
    axes[1, 0].text(i, v + 1500, f"{v:,.0f}", ha="center", fontweight="bold")

# R²
axes[1, 1].bar(metrics_df["Model"], metrics_df["R²"], color=["#2ecc71", "#3498db", "#e74c3c"])
axes[1, 1].set_ylabel("R² Score", fontsize=11)
axes[1, 1].set_title("Coefficient of Determination", fontweight="bold")
axes[1, 1].set_ylim([0, 1])
axes[1, 1].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(metrics_df["R²"]):
    axes[1, 1].text(i, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("results/model_metrics_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved: results/model_metrics_comparison.png")

# ── Save metrics to CSV ───────────────────────────────────────────────────────
metrics_df.to_csv("results/model_metrics.csv", index=False)
print("✓ Saved: results/model_metrics.csv")

# ── Save SHAP values sample ───────────────────────────────────────────────────
shap_df = pd.DataFrame(shap_values, columns=feature_names)
shap_df.to_csv("results/shap_values_sample.csv", index=False)
print("✓ Saved: results/shap_values_sample.csv")

# ── Create Excel export with evaluated records ───────────────────────────────
print("\n🔹 Creating Excel export with evaluated records…")

# Get our model's predictions for test set
test_predictions = cb_model.predict(X_test)

# Create evaluation DataFrame with key columns
evaluation_df = test_df[[
    'id', 'price_current', 'area_usable', 'area_total', 'rooms',
    'city_name', 'locality_region', 'condition_name',
    'has_balcony', 'has_elevator', 'floor_is', 'floor_max',
    'listing_quality_score', 'locality_gpsPrecisionScore'
]].copy()

# Add our model's predictions
evaluation_df['our_model_predicted_price'] = test_predictions
evaluation_df['prediction_error'] = evaluation_df['our_model_predicted_price'] - evaluation_df['price_current']
evaluation_df['prediction_error_pct'] = (evaluation_df['prediction_error'] / evaluation_df['price_current'] * 100).round(2)

# Join with original raw data to get the estimated price from the other model
# First, ensure we have the right join key (id)
raw_for_join = df_raw[[
    'id', 'priceEstimateSale_price', 'locality_country', 'locality_city',
    'locality_region', 'locality_town', 'offer_type', 'is_active', 'date_archived'
]].copy()

# Convert priceEstimateSale_price to numeric
raw_for_join['other_model_estimated_price'] = pd.to_numeric(raw_for_join['priceEstimateSale_price'], errors='coerce')

# Join the dataframes
evaluation_with_estimates = evaluation_df.merge(raw_for_join, on='id', how='left')

# Calculate comparison metrics
evaluation_with_estimates['other_model_error'] = (
    evaluation_with_estimates['other_model_estimated_price'] - evaluation_with_estimates['price_current']
).abs()
evaluation_with_estimates['our_model_error'] = evaluation_with_estimates['prediction_error'].abs()

# Calculate average errors ONLY for valid comparisons
valid_mask = evaluation_with_estimates['other_model_estimated_price'].notna()
our_avg_error = evaluation_with_estimates.loc[valid_mask, 'our_model_error'].mean()
other_avg_error = evaluation_with_estimates.loc[valid_mask, 'other_model_error'].mean()

print(f"\n📈 Average Absolute Errors (on {valid_mask.sum():,} properties with both estimates):")
print(f"   Our model: €{our_avg_error:,.0f}")
print(f"   Other model: €{other_avg_error:,.0f}")
if other_avg_error > 0:
    improvement = ((other_avg_error - our_avg_error) / other_avg_error * 100)
    print(f"   Difference: €{other_avg_error - our_avg_error:,.0f} ({improvement:+.1f}%)")

# Determine which model is better for each property
# Handle missing values properly
evaluation_with_estimates['better_model'] = 'No estimate available'

# Only compare where both estimates exist
valid_comparison = evaluation_with_estimates['other_model_estimated_price'].notna()

evaluation_with_estimates.loc[valid_comparison, 'better_model'] = np.where(
    evaluation_with_estimates.loc[valid_comparison, 'our_model_error'] < 
    evaluation_with_estimates.loc[valid_comparison, 'other_model_error'],
    'Our Model',
    np.where(
        evaluation_with_estimates.loc[valid_comparison, 'our_model_error'] > 
        evaluation_with_estimates.loc[valid_comparison, 'other_model_error'],
        'Other Model',
        'Tie (equal error)'
    )
)

# Reorder columns for better readability (only include columns that exist)
available_columns = evaluation_with_estimates.columns.tolist()
column_order = [
    'id', 'city_name', 'locality_region', 'locality_city', 'locality_town',
    'area_usable', 'area_total', 'rooms', 'floor_is', 'floor_max',
    'has_balcony', 'has_elevator', 'condition_name', 'listing_quality_score',
    'locality_gpsPrecisionScore', 'price_current',
    'other_model_estimated_price', 'our_model_predicted_price',
    'other_model_error', 'our_model_error', 'better_model',
    'prediction_error', 'prediction_error_pct',
    'offer_type', 'is_active', 'date_archived'
]

# Only keep columns that actually exist in the dataframe
column_order = [col for col in column_order if col in available_columns]

evaluation_with_estimates = evaluation_with_estimates[column_order]

# Save to Excel
evaluation_with_estimates.to_excel("results/evaluated_records_comparison.xlsx", index=False, engine='openpyxl')
print("✓ Saved: results/evaluated_records_comparison.xlsx")

# Also save as CSV for easier processing
evaluation_with_estimates.to_csv("results/evaluated_records_comparison.csv", index=False)
print("✓ Saved: results/evaluated_records_comparison.csv")

# Print summary statistics
total_properties = len(evaluation_with_estimates)
our_model_better = (evaluation_with_estimates['better_model'] == 'Our Model').sum()
other_model_better = (evaluation_with_estimates['better_model'] == 'Other Model').sum()
ties = (evaluation_with_estimates['better_model'] == 'Tie (equal error)').sum()
no_estimate = (evaluation_with_estimates['better_model'] == 'No estimate available').sum()
valid_comparisons = our_model_better + other_model_better + ties

print(f"\n📊 Model Comparison Summary:")
print(f"   Total evaluated properties: {total_properties:,}")
print(f"   Properties with valid estimates: {valid_comparisons:,} ({valid_comparisons/total_properties*100:.1f}%)")
print(f"   Properties with MISSING estimates: {no_estimate:,} ({no_estimate/total_properties*100:.1f}%)")

print(f"\n📊 Valid Comparison Breakdown ({valid_comparisons:,} properties):")
print(f"   Our model better: {our_model_better:,} ({our_model_better/valid_comparisons*100 if valid_comparisons > 0 else 0:.1f}%)")
print(f"   Other model better: {other_model_better:,} ({other_model_better/valid_comparisons*100 if valid_comparisons > 0 else 0:.1f}%)")
print(f"   Equal error (Tie): {ties:,} ({ties/valid_comparisons*100 if valid_comparisons > 0 else 0:.1f}%)")

# Calculate average errors
our_avg_error = evaluation_with_estimates['our_model_error'].mean()
other_avg_error = evaluation_with_estimates['other_model_error'].mean()

print(f"\n📈 Average Absolute Errors:")
print(f"   Our model: €{our_avg_error:,.0f}")
print(f"   Other model: €{other_avg_error:,.0f}")
print(f"   Improvement: €{other_avg_error - our_avg_error:,.0f} ({((other_avg_error - our_avg_error) / other_avg_error * 100):.1f}%)")

print("\n" + "="*70)
print("RESULTS SAVED TO results/")
print("="*70)
print("\nGenerated files:")
print("  📊 shap_summary_catboost.png — Feature importance by SHAP")
print("  📊 shap_beeswarm_catboost.png — SHAP values distribution")
print("  📊 catboost_feature_importance.png — Top 15 features")
print("  📊 predictions_comparison.png — Actual vs Predicted")
print("  📊 residuals_analysis.png — Residuals distribution")
print("  📊 model_metrics_comparison.png — Model performance metrics")
print("  📄 model_metrics.csv — Metrics table")
print("  📄 shap_values_sample.csv — SHAP values for 1000 test samples")
print("  📊 evaluated_records_comparison.xlsx — Excel with evaluated records and model comparison")
print("  📄 evaluated_records_comparison.csv — CSV with evaluated records and model comparison")
