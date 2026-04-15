"""
investigate_split.py

Investigation of the raw data to understand how to split correctly.
Focus on: active flag, status, date columns, and data quality.
"""

import sys
sys.path.insert(0, "/Users/katka/real_estate_pricing")

import pandas as pd
from sqlalchemy import create_engine
from config import DB_URL

engine = create_engine(DB_URL)

print("Loading sample data…")
df = pd.read_sql("""
    SELECT
        id, offer_type, status, is_active,
        date_published_first_time, date_last_change, date_archived, date_active_checked,
        price_current, price_historic_max, price_historic_min,
        area_usable, area_total, rooms,
        gps_lat, gps_lon,
        condition_name, city_name, listing_quality_score,
        number_of_price_changes, discount_percentage
    FROM final_flat_listings
    LIMIT 50000
""", engine)

print(f"Loaded {len(df):,} rows\n")

# ── 1. is_active distribution ─────────────────────────────────────────────────
print("=== is_active ===")
print(df["is_active"].value_counts(dropna=False))

# ── 2. status distribution ────────────────────────────────────────────────────
print("\n=== status ===")
print(df["status"].value_counts(dropna=False))

# ── 3. is_active vs status cross-tab ─────────────────────────────────────────
print("\n=== is_active × status ===")
print(pd.crosstab(df["is_active"], df["status"], dropna=False))

# ── 4. Date columns — nulls and ranges ───────────────────────────────────────
print("\n=== Date columns ===")
for col in ["date_published_first_time", "date_last_change", "date_archived", "date_active_checked"]:
    nulls = df[col].isna().sum()
    min_d = df[col].min()
    max_d = df[col].max()
    print(f"  {col}: {nulls} nulls | range {min_d} → {max_d}")

# ── 5. Is date_archived set only for archived/inactive listings? ──────────────
print("\n=== date_archived null vs is_active ===")
print(pd.crosstab(df["is_active"], df["date_archived"].isna().map({True: "no_archive_date", False: "has_archive_date"}), dropna=False))

# ── 6. GPS — are gps_lat/gps_lon already populated? ──────────────────────────
print("\n=== gps_lat / gps_lon nulls ===")
print(f"  gps_lat nulls: {df['gps_lat'].isna().sum()}")
print(f"  gps_lon nulls: {df['gps_lon'].isna().sum()}")
print(f"  both non-null: {df['gps_lat'].notna().sum()}")

# ── 7. Price sanity ───────────────────────────────────────────────────────────
print("\n=== price_current stats ===")
print(df["price_current"].describe())

# ── 8. Potential split strategy ───────────────────────────────────────────────
print("\n=== Possible split: active vs archived ===")
active   = df[df["is_active"] == True]
inactive = df[df["is_active"] == False]
print(f"  Active:   {len(active):,}")
print(f"  Inactive: {len(inactive):,}")

print("\n=== date_published_first_time distribution by year ===")
df["year_published"] = pd.to_datetime(df["date_published_first_time"]).dt.year
print(df["year_published"].value_counts().sort_index())
