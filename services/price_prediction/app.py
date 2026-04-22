from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field


def model_dir() -> Path:
    if os.environ.get("MODEL_DIR"):
        return Path(os.environ["MODEL_DIR"])
    for candidate in ("artifacts/price_prediction_rf",):
        p = Path(candidate)
        if p.is_dir() and ((p / "rf_model.pkl").is_file() or (p / "best_model.pkl").is_file()):
            return p
    return Path("artifacts/price_prediction_rf")


REQUIRED = (
    "scaler.pkl",
    "categorical_encoder.pkl",
    "feature_cols.pkl",
    "numerical_cols.pkl",
    "categorical_cols.pkl",
)


class Bundle:
    model: Any
    scaler: Any
    encoder: Any
    feature_cols: list[str]
    numerical_cols: list[str]
    categorical_cols: list[str]
    train_start_date: pd.Timestamp | None


b: Bundle | None = None

MARKET_LOCATION_ALIASES = {
    "Adama": "Adama (Nazret)",
    "Bishoftu": "Bishoftu (Debre Zeyit)",
}


def default_baseline_price_etb() -> float:
    raw = os.environ.get("BASELINE_PRICE_ETB", "100")
    try:
        value = float(raw)
    except ValueError:
        value = 100.0
    return max(value, 0.0)


def resolve_model_file(base: Path) -> Path:
    """Prefer explicit RF export, fallback to historical name."""
    for candidate in ("rf_model.pkl", "best_model.pkl"):
        p = base / candidate
        if p.is_file():
            return p
    raise RuntimeError(
        f"Missing model under {base.resolve()}: expected one of ['rf_model.pkl', 'best_model.pkl']"
    )


def init_bundle() -> None:
    global b
    base = model_dir()
    missing = [f for f in REQUIRED if not (base / f).is_file()]
    if missing:
        raise RuntimeError(
            f"Missing under {base.resolve()}: {missing}. Set MODEL_DIR or copy training artifacts."
        )
    loaded = Bundle()
    model_file = resolve_model_file(base)
    loaded.model = joblib.load(model_file)
    loaded.scaler = joblib.load(base / "scaler.pkl")
    loaded.encoder = joblib.load(base / "categorical_encoder.pkl")
    loaded.feature_cols = joblib.load(base / "feature_cols.pkl")
    loaded.numerical_cols = joblib.load(base / "numerical_cols.pkl")
    loaded.categorical_cols = joblib.load(base / "categorical_cols.pkl")
    train_start_date_path = base / "train_start_date.pkl"
    loaded.train_start_date = (
        pd.Timestamp(joblib.load(train_start_date_path))
        if train_start_date_path.is_file()
        else None
    )
    b = loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_bundle()
    yield


router = APIRouter()
app = FastAPI(title="Market Price Forecast API", lifespan=lifespan)


class HistoryRow(BaseModel):
    date: datetime
    current_price_etb: float = Field(..., ge=0)


class WeatherPayload(BaseModel):
    rainfall_anomaly_mm: Optional[float] = None
    temperature_mean_c: Optional[float] = None
    temperature_max_c: Optional[float] = None
    temperature_min_c: Optional[float] = None


class MarketPayload(BaseModel):
    supply_inflow_units: Optional[float] = None
    demand_proxy_queries: Optional[float] = None


class ForecastRequest(BaseModel):
    """Price lags and rolling stats are derived from ``history`` (same idea as the training notebook)."""

    commodity_item_type: str
    market_location: str
    predict_date: datetime
    history: Optional[list[HistoryRow]] = Field(default=None, min_length=1)
    weather: Optional[WeatherPayload] = None
    market: Optional[MarketPayload] = None


def normalize_market_location(value: str) -> str:
    return MARKET_LOCATION_ALIASES.get(value, value)


def _allowed_values_for_feature(feature_name: str) -> list[str]:
    assert b is not None
    if not hasattr(b.encoder, "categories_"):
        return []
    if feature_name not in b.categorical_cols:
        return []
    feature_index = b.categorical_cols.index(feature_name)
    categories = b.encoder.categories_[feature_index]
    return sorted([str(v) for v in categories])


def _validate_known_categories_or_400(req: ForecastRequest) -> None:
    assert b is not None
    checks = (
        ("commodity_item_type", req.commodity_item_type),
        ("market_location", normalize_market_location(req.market_location)),
    )
    for feature_name, feature_value in checks:
        allowed = _allowed_values_for_feature(feature_name)
        if allowed and str(feature_value) not in allowed:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown {feature_name} '{feature_value}'. "
                    f"Allowed values: {allowed}"
                ),
            )


def build_features_from_history(
    req: ForecastRequest,
) -> tuple[pd.DataFrame, bool]:
    assert b is not None
    weather = (
        {k: v for k, v in req.weather.model_dump().items() if v is not None}
        if req.weather
        else {}
    )
    market = (
        {k: v for k, v in req.market.model_dump().items() if v is not None}
        if req.market
        else {}
    )

    used_fallback_history = False
    if not req.history:
        used_fallback_history = True
        synthetic_date = pd.Timestamp(req.predict_date) - pd.Timedelta(days=1)
        synthetic_row = HistoryRow(
            date=synthetic_date.to_pydatetime(),
            current_price_etb=default_baseline_price_etb(),
        )
        history_df = pd.DataFrame([synthetic_row.model_dump()])
    else:
        history_df = pd.DataFrame([r.model_dump() for r in req.history])

    history_df["date"] = pd.to_datetime(history_df["date"])
    history_df = history_df.sort_values("date")
    if len(history_df) == 0:
        raise ValueError("history is empty")

    last_price = history_df["current_price_etb"].iloc[-1]
    last_7 = history_df["current_price_etb"].tail(7)
    last_14 = history_df["current_price_etb"].tail(14)
    last_30 = history_df["current_price_etb"].tail(30)

    ma7 = last_7.mean() if len(last_7) > 0 else np.nan
    ma14 = last_14.mean() if len(last_14) > 0 else np.nan
    ma30 = last_30.mean() if len(last_30) > 0 else np.nan
    vol30 = last_30.std() if len(last_30) > 1 else 0

    pct_changes = history_df["current_price_etb"].pct_change() * 100
    last_change = pct_changes.iloc[-1] if len(pct_changes.dropna()) > 0 else 0
    prev_change = pct_changes.iloc[-2] if len(pct_changes.dropna()) > 1 else 0
    acceleration = last_change - prev_change

    predict_date = pd.Timestamp(req.predict_date)
    last_history_date = pd.Timestamp(history_df["date"].max())
    if predict_date <= last_history_date:
        raise ValueError(
            f"predict_date must be after last history date ({last_history_date.date()})."
        )

    # Match training behavior when available (days since global dataset start).
    start_date = b.train_start_date if b.train_start_date is not None else history_df["date"].min()
    days_since_start = int((predict_date - pd.Timestamp(start_date)).days)

    row = {
        "commodity_item_type": req.commodity_item_type,
        "market_location": normalize_market_location(req.market_location),
        "month": predict_date.month,
        "week_of_year": int(predict_date.isocalendar()[1]),
        "day_of_week": predict_date.weekday() + 1,
        "quarter": (predict_date.month - 1) // 3 + 1,
        "day_of_month": predict_date.day,
        "days_since_start": days_since_start,
        "is_weekend": 1 if predict_date.weekday() >= 5 else 0,
        "price_lag_t1": last_price,
        "price_lag_t7": history_df["current_price_etb"].iloc[-7]
        if len(history_df) >= 7
        else last_price,
        "moving_average_7d": ma7 if not pd.isna(ma7) else last_price,
        "moving_average_14d": ma14 if not pd.isna(ma14) else last_price,
        "moving_average_30d": ma30 if not pd.isna(ma30) else last_price,
        "rolling_volatility_30d": vol30,
        "price_change_rate_percent": last_change if not pd.isna(last_change) else 0,
        "price_acceleration": acceleration if not pd.isna(acceleration) else 0,
        "price_to_ma7_ratio": last_price / ((ma7 if not pd.isna(ma7) else last_price) + 1e-6),
        "price_to_ma30_ratio": last_price
        / ((ma30 if not pd.isna(ma30) else last_price) + 1e-6),
        "ma7_to_ma30_ratio": (ma7 if not pd.isna(ma7) else last_price)
        / ((ma30 if not pd.isna(ma30) else last_price) + 1e-6),
        "rainfall_anomaly_mm": weather.get("rainfall_anomaly_mm", 0),
        "temperature_mean_c": weather.get("temperature_mean_c", 22),
        "temperature_max_c": weather.get("temperature_max_c", 26),
        "temperature_min_c": weather.get("temperature_min_c", 18),
        "supply_inflow_units": market.get("supply_inflow_units", 500),
        "demand_proxy_queries": market.get("demand_proxy_queries", 600),
    }

    if "high_low_spread" in b.feature_cols:
        row["high_low_spread"] = row["temperature_max_c"] - row["temperature_min_c"]
    if "rainfall_temp_interaction" in b.feature_cols:
        row["rainfall_temp_interaction"] = (
            row["rainfall_anomaly_mm"] * row["temperature_mean_c"]
        )
    if "price_volatility_flag" in b.feature_cols:
        row["price_volatility_flag"] = int(row["rolling_volatility_30d"] > 0)
    if "supply_demand_gap" in b.feature_cols:
        row["supply_demand_gap"] = (
            row["supply_inflow_units"] - row["demand_proxy_queries"]
        )

    input_df = pd.DataFrame([row])
    for col in b.feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[b.feature_cols].copy()

    if b.categorical_cols:
        try:
            input_df[b.categorical_cols] = b.encoder.transform(
                input_df[b.categorical_cols].astype(str)
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid categorical input. Use /price/model/features and "
                    "training categories for commodity_item_type and market_location."
                ),
            ) from exc
    if b.numerical_cols:
        input_df[b.numerical_cols] = b.scaler.transform(input_df[b.numerical_cols])

    return input_df, used_fallback_history


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/model/info")
def model_info():
    assert b is not None
    return {
        "model_dir": str(model_dir().resolve()),
        "model_type": type(b.model).__name__,
        "inference": "history_optional_with_fallback",
        "default_baseline_price_etb": default_baseline_price_etb(),
        "feature_count": len(b.feature_cols),
        "has_train_start_date": b.train_start_date is not None,
    }


@router.get("/model/features")
def model_features():
    assert b is not None
    return {
        "feature_cols": b.feature_cols,
        "numerical_cols": b.numerical_cols,
        "categorical_cols": b.categorical_cols,
    }


@router.get("/model/categories")
def model_categories():
    assert b is not None
    categories: dict[str, list[str]] = {}
    if hasattr(b.encoder, "categories_"):
        for index, feature_name in enumerate(b.categorical_cols):
            values = b.encoder.categories_[index]
            categories[feature_name] = sorted([str(v) for v in values])
    return categories


@router.post("/forecast/price")
def forecast_price(req: ForecastRequest):
    assert b is not None
    try:
        _validate_known_categories_or_400(req)
        X, used_fallback_history = build_features_from_history(req)
        pred = float(b.model.predict(X)[0])
        response = {
            "commodity_item_type": req.commodity_item_type,
            "market_location": req.market_location,
            "predict_date": req.predict_date,
            "predicted_price_etb": round(pred, 2),
            "currency": "ETB",
            "used_fallback_history": used_fallback_history,
        }
        if used_fallback_history:
            response["warning"] = (
                "No history was provided. Prediction used synthetic baseline history and "
                "should be treated as low confidence."
            )
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("services.price_prediction.app:app", host="0.0.0.0", port=port, reload=False)
