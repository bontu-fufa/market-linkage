# Market Linkage ML Gateway

Unified FastAPI gateway for multiple ML services:

- Price prediction (`/price`)
- Market demand forecasting (`/demand`)
- Credit scoring scaffold (`/credit`, coming soon)

## Project Structure

- `gateway.py` - mounts all service routers under route prefixes.
- `main.py` - production/local entrypoint for the gateway app.
- `services/price_prediction/app.py` - price forecast API.
- `services/market_demand/app.py` - demand forecast API.
- `services/credit_scoring/app.py` - placeholder credit service.
- `artifacts/` - model artifacts used at runtime.
- `scripts/` - training and experimentation notebooks.

## Requirements

- Python 3.11+ (3.14 works in this project if dependencies install successfully)
- `pip`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Start the unified gateway:

```bash
uvicorn gateway:app --host 0.0.0.0 --port 8000
```

Alternative entrypoint:

```bash
python main.py
```

## Swagger / OpenAPI

Once running:

- Unified gateway docs (all services): `http://127.0.0.1:8000/docs`
- OpenAPI spec: `http://127.0.0.1:8000/openapi.json`

In gateway mode (`uvicorn gateway:app --host 0.0.0.0 --port 8000`), there is a single docs page at `/docs` that includes all prefixed routes (for example `/price/*`, `/demand/*`, `/credit/*`).

## Public Routes

- `GET /health`
- `GET /price/health`
- `GET /price/model/info`
- `GET /price/model/features`
- `POST /price/forecast/price`
- `GET /demand/health`
- `GET /demand/model/info`
- `GET /demand/model/features`
- `POST /demand/forecast`
- `GET /credit/health`
- `POST /credit/score` (returns `501`)

## Model Artifacts

### Price Prediction

Default lookup order:

1. `MODEL_DIR` environment variable
2. `artifacts/price_prediction_rf`
3. `model/rf`
4. `models/rf`

Required files:

- `rf_model.pkl` (or `best_model.pkl`)
- `scaler.pkl`
- `categorical_encoder.pkl`
- `feature_cols.pkl`
- `numerical_cols.pkl`
- `categorical_cols.pkl`

Optional:

- `train_start_date.pkl`

### Demand Forecast

Default lookup order:

1. `DEMAND_MODEL_DIR` environment variable
2. `artifacts/demand_forcast`
3. `artifacts/demand_forecast`
4. `model/demand_forcast`
5. `model/demand_forecast`

Required files:

- `enhanced_demand_model.pkl`
- `enhanced_scaler.pkl`
- `enhanced_le_zone.pkl`
- `enhanced_le_product.pkl`
- `enhanced_le_season.pkl`
- `enhanced_feature_names.pkl`

## Environment Variables

- `PORT` - service port (used by `main.py`, default `8000`)
- `MODEL_DIR` - override price model directory
- `DEMAND_MODEL_DIR` - override demand model directory
- `BASELINE_PRICE_ETB` - fallback baseline price when no history is provided (default `100`)

## API Notes

### Price Prediction

- Endpoint: `POST /price/forecast/price`
- `history` is optional. If omitted, API uses a synthetic baseline and returns:
  - `"used_fallback_history": true`
  - warning message (low confidence)
- Invalid categories for `commodity_item_type` or `market_location` return `400` with allowed values.
- Request aliases are supported:
  - `Adama` -> `Adama (Nazret)`
  - `Bishoftu` -> `Bishoftu (Debre Zeyit)`

### Demand Forecast

- Endpoint: `POST /demand/forecast`
- Response returns `predicted_demand_growth_rate` as a percentage rounded to 2 decimals.
- Unknown `zone`, `product`, or `season_type` returns `400` with allowed values.

## Quick Test (Gateway Health)

```bash
curl http://127.0.0.1:8000/health
```

## Deployment

Recommended start command on Render:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

If model loading fails in deployment, first confirm artifact files exist in deployed paths and `MODEL_DIR` / `DEMAND_MODEL_DIR` are set correctly.
