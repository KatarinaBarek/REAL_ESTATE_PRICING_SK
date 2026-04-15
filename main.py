"""
main.py — FastAPI price estimation service for Slovak flats.

Feature set mirrors Notebook 3 (Nomerio_RealEstates_SK).
All derived features (distances, KNN stats, log transforms) are computed
server-side — the caller only needs to supply the raw flat attributes.

Run:   uvicorn main:app --reload
Docs:  http://localhost:8000/docs
"""

from typing import Optional

import joblib
import numpy as np
import pandas as pd
from catboost import Pool as CatPool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import MODEL_PATH
from feature_engineering import (
    VALID_CONDITIONS,
    VALID_REGIONS,
    build_features,
    encode_categoricals,
)

# ── Load model artifact ───────────────────────────────────────────────────────
try:
    artifact = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"Model file '{MODEL_PATH}' not found. Run  python train.py  first."
    )

model        = artifact["model"]
model_name   = artifact["model_name"]
model_kind   = artifact["model_kind"]
knn          = artifact["knn_transformer"]
region_map   = artifact["region_map"]
condition_map = artifact["condition_map"]
ALL_FEATURES = artifact["all_features"]
CATEGORICAL_FEATURES = artifact["categorical_features"]
cats         = artifact.get("cats")  # category alignment for XGB / LGB


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Slovakia Flat Pricing API",
    description=(
        "Estimates the sale price of a flat in Slovakia based on its attributes. "
        "Derived features (distances to cities, local market KNN stats, log-area) "
        "are computed automatically — you only need to provide the raw flat details."
    ),
    version="1.0.0",
)


# ── Request / Response schemas ────────────────────────────────────────────────
class FlatInput(BaseModel):
    # ── Required ──────────────────────────────────────────────────────────────
    rooms:           int   = Field(..., ge=1, le=9,       description="Number of rooms (1–9)")
    floor_is:        int   = Field(..., ge=0,             description="Floor the flat is on")
    floor_max:       int   = Field(..., ge=0,             description="Total floors in the building")
    area_usable:     float = Field(..., gt=15, lt=500,    description="Usable area in m²")
    lat:             float = Field(...,                   description="GPS latitude  (WGS84, e.g. 48.148)")
    lon:             float = Field(...,                   description="GPS longitude (WGS84, e.g. 17.107)")
    locality_region: str   = Field(...,
        description="One of the 8 official Slovak regions (or 'other'). "
                    "Example: 'Bratislavský kraj'")
    condition_name:  str   = Field(...,
        description="Flat condition. One of: Kompletná rekonštrukcia, "
                    "Čiastočná rekonštrukcia, Pôvodný stav, Novostavba, "
                    "Vo výstavbe / Projekt")

    # ── Optional ──────────────────────────────────────────────────────────────
    locality_city:   str            = Field("",   description="City name — used for regional city flag")
    area_total:      Optional[float] = Field(None, ge=0, description="Total area in m² (defaults to area_usable)")
    area_other:      Optional[float] = Field(None, ge=0, description="Other area in m²")
    area_balcony:    Optional[float] = Field(None, ge=0, description="Balcony area in m²")
    area_cellar:     Optional[float] = Field(None, ge=0, description="Cellar area in m²")
    has_balcony:     bool = Field(False, description="Flat has a balcony")
    has_elevator:    bool = Field(False, description="Building has an elevator")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "rooms": 3,
                    "floor_is": 2,
                    "floor_max": 8,
                    "area_usable": 65.0,
                    "lat": 48.148598,
                    "lon": 17.107748,
                    "locality_region": "Bratislavský kraj",
                    "condition_name": "Kompletná rekonštrukcia",
                    "locality_city": "Bratislava",
                    "area_total": 70.0,
                    "has_balcony": True,
                    "has_elevator": True,
                }
            ]
        }
    }


class PriceResponse(BaseModel):
    estimated_price_eur: float
    currency: str = "EUR"
    model_used: str


# ── Feature preparation ───────────────────────────────────────────────────────
def _prepare_input(flat: FlatInput) -> pd.DataFrame:
    """
    Convert a FlatInput into the full feature DataFrame expected by the model.
    Applies the same build_features → KNN → encode pipeline as train.py.
    """
    row = {
        # GPS (build_features renames these to lat/lon internally)
        "gps_lat":         flat.lat,
        "gps_lon":         flat.lon,
        # Structural
        "rooms":           flat.rooms,
        "floor_is":        flat.floor_is,
        "floor_max":       flat.floor_max,
        "area_usable":     flat.area_usable,
        "area_total":      flat.area_total if flat.area_total is not None else flat.area_usable,
        "area_other":      flat.area_other   or 0.0,
        "area_balcony":    flat.area_balcony or 0.0,
        "area_cellar":     flat.area_cellar  or 0.0,
        "has_balcony":     flat.has_balcony,
        "has_elevator":    flat.has_elevator,
        # Categorical (standardised in build_features)
        "locality_region": flat.locality_region,
        "condition_name":  flat.condition_name,
        "locality_city":   flat.locality_city,
        # price_current not needed at inference — knn.transform only uses lat/lon
    }

    df = pd.DataFrame([row])
    df = build_features(df)           # adds distances, log, interaction, dummy_regional_city
    df = knn.transform(df)             # adds KNN local price stats
    df = encode_categoricals(df, region_map, condition_map)  # → locality_region_idx, condition_name_idx

    return df[ALL_FEATURES]


def _predict(X: pd.DataFrame) -> float:
    """Run inference, handling per-model quirks."""
    if model_kind == "catboost":
        pool = CatPool(data=X, cat_features=CATEGORICAL_FEATURES)
        return float(model.predict(pool)[0])

    if model_kind in ("xgboost", "lightgbm") and cats:
        X = X.copy()
        for col in CATEGORICAL_FEATURES:
            X[col] = pd.Categorical(X[col], categories=cats[col])

    return float(model.predict(X)[0])


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PriceResponse, summary="Estimate flat price")
def predict(flat: FlatInput) -> PriceResponse:
    """
    Returns an estimated sale price in EUR.

    - You don't need to supply log-area, distances, or KNN stats — those are computed automatically.
    - Unknown region/condition values are treated as 'other'.
    - Missing optional area fields default to 0.
    """
    try:
        X = _prepare_input(flat)
        price = _predict(X)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return PriceResponse(
        estimated_price_eur=round(price, 2),
        model_used=model_name,
    )


@app.get("/features", summary="Accepted input fields")
def list_features():
    """Lists all fields accepted by /predict, with valid categorical values."""
    return {
        "required": [
            "rooms", "floor_is", "floor_max", "area_usable",
            "lat", "lon", "locality_region", "condition_name",
        ],
        "optional": [
            "locality_city", "area_total", "area_other",
            "area_balcony", "area_cellar", "has_balcony", "has_elevator",
        ],
        "valid_regions":    sorted(VALID_REGIONS) + ["other"],
        "valid_conditions": sorted(VALID_CONDITIONS) + ["other"],
        "note": (
            "Distances to Slovak cities, KNN local price stats, and log-area features "
            "are derived automatically from the inputs — do not supply them."
        ),
    }


@app.get("/model-info", summary="Model evaluation metrics")
def model_info():
    """Returns evaluation metrics recorded during training for all compared models."""
    return {
        "best_model":   model_name,
        "eval_metrics": artifact.get("eval", {}),
        "note": "Metrics from a held-out 30% test split (random_state=42).",
    }


@app.get("/health", summary="Liveness check")
def health():
    return {"status": "ok", "model_loaded": model_name}
