from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field


def model_dir() -> Path:
    if os.environ.get("DEMAND_MODEL_DIR"):
        return Path(os.environ["DEMAND_MODEL_DIR"])
    candidates = ("artifacts/demand_forcast",)
    for candidate in candidates:
        path = Path(candidate)
        if path.is_dir() and (path / "enhanced_demand_model.pkl").is_file():
            return path
    return Path("artifacts/demand_forcast")


REQUIRED = (
    "enhanced_demand_model.pkl",
    "enhanced_scaler.pkl",
    "enhanced_le_zone.pkl",
    "enhanced_le_product.pkl",
    "enhanced_le_season.pkl",
    "enhanced_feature_names.pkl",
)


class Bundle:
    model: Any
    scaler: Any
    le_zone: Any
    le_product: Any
    le_season: Any
    feature_names: list[str]


b: Bundle | None = None


def init_bundle() -> None:
    global b
    base = model_dir()
    missing = [f for f in REQUIRED if not (base / f).is_file()]
    if missing:
        raise RuntimeError(
            f"Missing demand artifacts under {base.resolve()}: {missing}. "
            "Set DEMAND_MODEL_DIR or copy training outputs."
        )
    loaded = Bundle()
    loaded.model = joblib.load(base / "enhanced_demand_model.pkl")
    loaded.scaler = joblib.load(base / "enhanced_scaler.pkl")
    loaded.le_zone = joblib.load(base / "enhanced_le_zone.pkl")
    loaded.le_product = joblib.load(base / "enhanced_le_product.pkl")
    loaded.le_season = joblib.load(base / "enhanced_le_season.pkl")
    loaded.feature_names = list(joblib.load(base / "enhanced_feature_names.pkl"))
    b = loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_bundle()
    yield


router = APIRouter()
app = FastAPI(title="Market Demand API", lifespan=lifespan)


class ForecastRequest(BaseModel):
    zone: str = Field(..., description="Zone name used in training data.")
    product: str = Field(..., description="Product name used in training data.")
    season_type: str = Field(..., description="Season type used in training data.")
    month: int = Field(..., ge=1, le=12)
    week: int = Field(..., ge=1, le=53)
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Optional numeric feature overrides by exact feature name.",
    )


def _encode_or_400(encoder: Any, value: str, name: str) -> int:
    try:
        return int(encoder.transform([value])[0])
    except ValueError as exc:
        known = sorted([str(v) for v in encoder.classes_])
        raise HTTPException(
            status_code=400,
            detail=f"Unknown {name} '{value}'. Allowed values: {known}",
        ) from exc


def build_input(req: ForecastRequest) -> pd.DataFrame:
    assert b is not None
    month_sin = math.sin(2 * math.pi * req.month / 12.0)
    month_cos = math.cos(2 * math.pi * req.month / 12.0)
    week_sin = math.sin(2 * math.pi * req.week / 52.0)
    week_cos = math.cos(2 * math.pi * req.week / 52.0)

    row: dict[str, float] = {
        "month": float(req.month),
        "week": float(req.week),
        "days_since_last_harvest": 0.0,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "week_sin": week_sin,
        "week_cos": week_cos,
        "season_encoded": float(_encode_or_400(b.le_season, req.season_type, "season_type")),
        "zone_encoded": float(_encode_or_400(b.le_zone, req.zone, "zone")),
        "product_encoded": float(_encode_or_400(b.le_product, req.product, "product")),
    }

    # Caller can provide additional notebook features by exact feature key.
    for key, value in req.features.items():
        row[key] = float(value)

    for feature_name in b.feature_names:
        row.setdefault(feature_name, 0.0)

    input_df = pd.DataFrame([row], columns=b.feature_names)
    scaled = b.scaler.transform(input_df)
    return pd.DataFrame(scaled, columns=b.feature_names)


@router.get("/health")
def health():
    return {"status": "ok", "service": "market_demand"}


@router.get("/model/info")
def model_info():
    assert b is not None
    return {
        "model_dir": str(model_dir().resolve()),
        "model_type": type(b.model).__name__,
        "feature_count": len(b.feature_names),
        "target": "target_demand_growth_rate",
    }


@router.get("/model/features")
def model_features():
    assert b is not None
    return {"feature_names": b.feature_names}


@router.get("/model/categories")
def model_categories():
    assert b is not None
    return {
        "zone": sorted([str(v) for v in b.le_zone.classes_]),
        "product": sorted([str(v) for v in b.le_product.classes_]),
        "season_type": sorted([str(v) for v in b.le_season.classes_]),
    }


@router.post("/forecast")
def forecast(req: ForecastRequest):
    assert b is not None
    try:
        X = build_input(req)
        pred = float(b.model.predict(X)[0])
        pred_percent = round(pred * 100, 2)
        return {
            "zone": req.zone,
            "product": req.product,
            "season_type": req.season_type,
            "month": req.month,
            "week": req.week,
            "predicted_demand_growth_rate": pred_percent,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app.include_router(router)
