"""
feature_engineering.py

Shared preprocessing logic for training and inference.
Mirrors the PySpark steps from:
  - Notebook 1: filtering and categorical standardisation
  - Notebook 2: distances, KNN local price stats, log transforms
  - Notebook 3: feature list and ordinal encoding
"""

import struct

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import BallTree

# ── Constants ─────────────────────────────────────────────────────────────────

# City coordinates for Haversine distance features
CITY_COORDS = {
    "dist_ba": (48.148598, 17.107748),  # Bratislava
    "dist_tt": (48.37741,  17.58723),   # Trnava
    "dist_tn": (48.89452,  18.04436),   # Trenčín
    "dist_nr": (48.30763,  18.08453),   # Nitra
    "dist_za": (49.22444,  18.74111),   # Žilina
    "dist_bb": (48.73628,  19.14619),   # Banská Bystrica
    "dist_po": (49.00005,  21.23309),   # Prešov
    "dist_ke": (48.71638,  21.26107),   # Košice
}

# Major employer coordinates
EMPLOYER_COORDS = {
    "dist_vw_bratislava":       (48.1969, 17.0614),
    "dist_us_steel_kosice":     (48.6544, 21.2867),
    "dist_kia_zilina":          (49.2263, 18.7550),
    "dist_psa_trnava":          (48.4004, 17.5286),
    "dist_slovnaft_ba":         (48.1180, 17.1710),
    "dist_samsung_galanta":     (48.1921, 17.7300),
    "dist_samsung_voderady":    (48.3114, 17.6834),
    "dist_continental_puchov":  (49.1323, 18.3299),
    "dist_mondi_rk":            (49.0747, 19.2989),
    "dist_schaeffler_kysuce":   (49.2980, 18.7820),
    "dist_hella_banovce":       (48.7210, 18.2590),
    "dist_whirlpool_poprad":    (49.0554, 20.2966),
    "dist_foxconn_nitra":       (48.2990, 18.1110),
    "dist_jlr_nitra":           (48.2602, 18.0877),
    "dist_zeleziarne_podbrezova": (48.8018, 19.5335),
    "dist_duslo_sala":          (48.1536, 17.8790),
    "dist_embraco_snv":         (48.9456, 20.5610),
    "dist_matador_vrable":      (48.2440, 18.3070),
    "dist_se_mochovce":         (48.2514, 18.4561),
    "dist_t_systems_kosice":    (48.7245, 21.2581),
}

# Tourist attraction coordinates
ATTRACTION_COORDS = {
    "dist_spissky_hrad":         (49.000496, 20.768625),
    "dist_vlkolinec":            (49.039268, 19.277304),
    "dist_banska_stiavnica":     (48.458652, 18.893036),
    "dist_strbske_pleso":        (49.1200,   20.0580),
    "dist_slovensky_kras":       (48.466667, 20.466667),
    "dist_muranska_planina":     (48.7360,   19.9480),
    "dist_lomnicky_stit":        (49.1951,   20.2134),
    "dist_jasna":                (48.940136, 19.723692),
    "dist_tatranska_lomnica":    (49.16616,  20.28128),
    "dist_donovaly":             (48.880403, 19.222129),
    "dist_piestany":             (48.589233, 17.834047),
    "dist_trencianske_teplice":  (48.91063,  18.16691),
    "dist_rajecke_teplice":      (49.129992, 18.679730),
    "dist_tatralandia":          (49.1020,   19.5865),
}

REGIONAL_CITIES = {
    "Bratislava", "Košice", "Trnava", "Nitra",
    "Trenčín", "Banská Bystrica", "Poprad", "Prešov", "Žilina",
}

VALID_REGIONS = {
    "Bratislavský kraj",
    "Košický kraj",
    "Trnavský kraj",
    "Nitriansky kraj",
    "Trenčiansky kraj",
    "Banskobystrický kraj",
    "Žilinský kraj",
    "Prešovský kraj",
}

VALID_CONDITIONS = {
    "Kompletná rekonštrukcia",
    "Čiastočná rekonštrukcia",
    "Pôvodný stav",
    "Novostavba",
    "Vo výstavbe / Projekt",
}

# Feature lists — must match training order exactly 
NUMERICAL_FEATURES = [
    "has_balcony_int",
    "has_elevator_int",
    "dummy_regional_city",
    "rooms",
    "floor_is",
    "floor_max",
    "lat",
    "lon",
    "area_usable",
    "area_total",
    "area_other",
    "area_balcony",
    "area_cellar",
    "log_area_usable",
    "log_area_total",
    "interaction_area_rooms",
    "local_price_area_usable_per_m2_knn_avg",
    "local_price_area_usable_per_m2_knn_median",
    "local_price_area_total_per_m2_knn_avg",
    "local_price_area_total_per_m2_knn_median",
    "knn_count",
    "is_ground_floor",
    "floor_ratio",
    "listing_quality_score",
    "area_per_room",
    "balcony_ratio",
    "dist_nearest_employer",
    "dist_nearest_attraction",
] + list(CITY_COORDS.keys()) + list(EMPLOYER_COORDS.keys()) + list(ATTRACTION_COORDS.keys())

# These are integer-encoded but declared as categorical to CatBoost
CATEGORICAL_FEATURES = ["locality_region_idx", "condition_name_idx"]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ── Distance helper ───────────────────────────────────────────────────────────

def _dist_km(lat1, lon1, lat2: float, lon2: float) -> np.ndarray:
    """
    Spherical law of cosines (identical to the Spark formula in Notebook 2):
      6371 * acos(cos(lat1)*cos(lat2)*cos(lon2-lon1) + sin(lat1)*sin(lat2))
    """
    rlat1 = np.radians(np.asarray(lat1, dtype=float))
    rlon1 = np.radians(np.asarray(lon1, dtype=float))
    rlat2 = np.radians(lat2)
    rlon2 = np.radians(lon2)
    cos_angle = (
        np.cos(rlat1) * np.cos(rlat2) * np.cos(rlon2 - rlon1)
        + np.sin(rlat1) * np.sin(rlat2)
    )
    return 6371.0 * np.arccos(np.clip(cos_angle, -1.0, 1.0))


# ── GPS decoding ─────────────────────────────────────────────────

def decode_gps_point(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decodes the binary gps_point column from MySQL into lat/lon float columns.
    Mirrors the PySpark UDF in Notebook 2:
      - Takes the last 16 bytes of the blob
      - Bytes 0–7  → lon (little-endian double)
      - Bytes 8–15 → lat (little-endian double)
    Adds columns: lat, lon, gps_valid (1 = decoded OK, 0 = failed/missing).
    Also filters to valid, non-zero GPS within Slovakia's bounding box.
    """
    def _decode(raw):
        if raw is None:
            return (None, None, 0)
        try:
            if len(raw) < 16:
                return (None, None, 0)
            last16 = raw[-16:]
            lon = struct.unpack("<d", last16[:8])[0]
            lat = struct.unpack("<d", last16[8:])[0]
            return (lat, lon, 1)
        except Exception:
            return (None, None, 0)

    decoded = df["gps_point"].apply(_decode)
    df = df.copy()
    df["lat"]       = [r[0] for r in decoded]
    df["lon"]       = [r[1] for r in decoded]
    df["gps_valid"] = [r[2] for r in decoded]

    return df.reset_index(drop=True)


# ── Filtering ────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas translation of the PySpark filters from Notebook 1.
    Expects the raw DataFrame loaded directly from final_flat_listings.
    """
    n0 = len(df)

    df = df[df["offer_type"] == 1]
    df = df[(df["area_usable"] < 500) | (df["area_total"] < 500)]
    df = df[df["price_current"] > 3000]
    df = df[(df["area_usable"] > 15) | (df["area_total"] > 15)]

    # Notebook 1: drop if both areas null/zero; Notebook 2: also drop if either is zero
    both_null_or_zero = (
        (df["area_usable"].isna() | (df["area_usable"] == 0))
        & (df["area_total"].isna() | (df["area_total"] == 0))
    )
    df = df[~both_null_or_zero]
    df = df[(df["area_usable"] != 0) & (df["area_total"] != 0)]

    # Slovakia only (mirrors notebook 1 OR logic)
    sk_mask = (
        df["locality_country"].str.contains("Slovensko", na=False)
        | df["locality_countryCode"].str.upper().isin(["SK", "SLOVENSKO"])
    )
    df = df[sk_mask]

    df = df[df["rooms"] < 10]
    df = df[df["price_current"].notna() & (df["price_current"] != 0)]

    # Filter: must be decoded and non-zero
    df = df[df["lat"].notna() & df["lon"].notna()]
    df = df[(df["lat"] != 0) & (df["lon"] != 0)]

    # Filter: within Slovakia's bounding box (Notebook 2)
    df = df[
        (df["lat"] >= 47.7) & (df["lat"] <= 49.6) &
        (df["lon"] >= 16.8) & (df["lon"] <= 22.6)
    ]

    df = df.reset_index(drop=True)
    print(f"Filtering: {n0:,} → {len(df):,} rows ({len(df)/n0*100:.1f}% kept)")
    return df


# ── Feature engineering ────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all deterministic engineered features.

    Input:  raw / filtered DataFrame with gps_lat, gps_lon columns
    Output: DataFrame with all NUMERICAL_FEATURES except the KNN stats
            (those require KNNLocalPriceStats.transform)

    Note: renames gps_lat/gps_lon → lat/lon to match the Notebook 3 feature names.
    """
    df = df.copy()

    # Standardise categoricals (Notebook 1)
    df["locality_region"] = df["locality_region"].apply(
        lambda r: r if r in VALID_REGIONS else "other"
    )
    df["condition_name"] = df["condition_name"].apply(
        lambda c: c if c in VALID_CONDITIONS else "other"
    )

    # Boolean → int (Notebook 3 step 1)
    df["has_balcony_int"]  = df["has_balcony"].fillna(False).astype(bool).astype(int)
    df["has_elevator_int"] = df["has_elevator"].fillna(False).astype(bool).astype(int)

    # Regional city indicator — use locality_city if available, else city_name
    city_col = "locality_city" if "locality_city" in df.columns else "city_name"
    df["dummy_regional_city"] = df.get(city_col, pd.Series("", index=df.index)).isin(REGIONAL_CITIES).astype(int)

    # Log transforms (Notebook 1)
    df["log_area_usable"] = np.log1p(df["area_usable"].clip(lower=0))
    area_total_filled = df["area_total"].fillna(df["area_usable"]).clip(lower=0)
    df["log_area_total"] = np.log1p(area_total_filled)

    # Interaction feature (Notebook 3 feature list)
    df["interaction_area_rooms"] = df["area_usable"] * df["rooms"]

    # Fill optional area columns with 0
    for col in ["area_total", "area_other", "area_balcony", "area_cellar"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    # Ground floor flag
    df["is_ground_floor"] = (df["floor_is"] == 0).astype(int)

    # Floor ratio — NaN when either value is missing or floor_max is 0
    floor_max = pd.to_numeric(df["floor_max"], errors="coerce").replace(0, np.nan)
    df["floor_ratio"] = pd.to_numeric(df["floor_is"], errors="coerce") / floor_max

    # Area per room
    df["area_per_room"] = df["area_usable"] / (df["rooms"].replace(0, np.nan))

    # Balcony ratio — proportion of living space that is balcony
    df["balcony_ratio"] = df["area_balcony"] / df["area_usable"].replace(0, np.nan)

    # Distance to 8 Slovak cities, 20 major employers, 14 attractions (Notebook 2)
    for dist_col, (clat, clon) in {**CITY_COORDS, **EMPLOYER_COORDS, **ATTRACTION_COORDS}.items():
        df[dist_col] = _dist_km(df["lat"].values, df["lon"].values, clat, clon)

    # Min distance to nearest major employer
    employer_cols = list(EMPLOYER_COORDS.keys())
    df["dist_nearest_employer"] = df[employer_cols].min(axis=1)

    # Min distance to nearest attraction
    attraction_cols = list(ATTRACTION_COORDS.keys())
    df["dist_nearest_attraction"] = df[attraction_cols].min(axis=1)

    return df


# ── Ordinal encoding ──────────────────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame, region_map: dict, condition_map: dict
) -> pd.DataFrame:
    """
    Convert locality_region and condition_name to integer indices.
    Mirrors StringIndexer from Notebook 3. Unknown values → -1.
    """
    df = df.copy()
    df["locality_region_idx"] = (
        df["locality_region"].map(region_map).fillna(-1).astype(int)
    )
    df["condition_name_idx"] = (
        df["condition_name"].map(condition_map).fillna(-1).astype(int)
    )
    return df


# ── KNN local price stats ───────────────────────────────────────

class KNNLocalPriceStats(BaseEstimator, TransformerMixin):
    """
    Computes local market statistics by finding K geographically nearest
    neighbors in the training data using a BallTree (haversine metric),
    then filtering to similar flats (mirrors Notebook 2):
      - area_usable within ±20% of the query flat
      - rooms within ±1 of the query flat

    fit(X)     — stores GPS, area, rooms and price/sqm arrays from training data
                 X must contain: lat, lon, area_usable, area_total, rooms, price_current
    transform(X) — for each row, finds k candidates, applies similarity filter,
                   then appends 5 columns:
                   local_price_area_usable_per_m2_knn_avg
                   local_price_area_usable_per_m2_knn_median
                   local_price_area_total_per_m2_knn_avg
                   local_price_area_total_per_m2_knn_median
                   knn_count  (number of neighbors that passed the similarity filter)
    """

    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, X: pd.DataFrame, y=None):
        coords = np.radians(X[["lat", "lon"]].values.astype(float))
        self.tree_ = BallTree(coords, metric="haversine")

        area_u = X["area_usable"].replace(0, np.nan)
        area_t = X["area_total"].replace(0, np.nan).fillna(area_u)
        price  = X["price_current"].values.astype(float)

        self.area_u_ = area_u.values.astype(float)
        self.rooms_  = X["rooms"].values.astype(float)
        self.ppu_    = (price / area_u.values).astype(float)
        self.ppt_    = (price / area_t.values).astype(float)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        coords   = np.radians(X[["lat", "lon"]].values.astype(float))
        area_u_q = X["area_usable"].values.astype(float)
        rooms_q  = X["rooms"].values.astype(float)

        # Query more candidates than k so similarity filter still leaves enough
        k_query = min(self.k * 10, len(self.ppu_))
        _, indices = self.tree_.query(coords, k=k_query)

        rows = []
        for i, idx_row in enumerate(indices):
            au_q = area_u_q[i]
            r_q  = rooms_q[i]

            # Similarity filter (mirrors Notebook 2)
            area_diff_ok  = np.abs(self.area_u_[idx_row] - au_q) <= au_q * 0.20
            rooms_diff_ok = np.abs(self.rooms_[idx_row] - r_q) <= 1
            similar = idx_row[area_diff_ok & rooms_diff_ok][:self.k]

            ppu_v = self.ppu_[similar]
            ppt_v = self.ppt_[similar]
            ppu_v = ppu_v[~np.isnan(ppu_v)]
            ppt_v = ppt_v[~np.isnan(ppt_v)]

            rows.append({
                "local_price_area_usable_per_m2_knn_avg":    float(np.nanmean(ppu_v))   if len(ppu_v) else np.nan,
                "local_price_area_usable_per_m2_knn_median": float(np.nanmedian(ppu_v)) if len(ppu_v) else np.nan,
                "local_price_area_total_per_m2_knn_avg":     float(np.nanmean(ppt_v))   if len(ppt_v) else np.nan,
                "local_price_area_total_per_m2_knn_median":  float(np.nanmedian(ppt_v)) if len(ppt_v) else np.nan,
                "knn_count": len(similar),
            })

        knn_df = pd.DataFrame(rows, index=X.index)
        return pd.concat([X.reset_index(drop=True), knn_df.reset_index(drop=True)], axis=1)
