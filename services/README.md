# Services layout

- `price_prediction` is the live price API at `services/price_prediction/app.py`, mounted by `gateway.py` under `/price`.
- `market_demand` is live at `services/market_demand/app.py`, mounted by `gateway.py` under `/demand`.
- `credit_scoring` is scaffolded and currently returns `501 coming soon`.

## Run unified gateway

```bash
uvicorn gateway:app --host 0.0.0.0 --port 8000
```

## Public routes

- `GET /health` (gateway)
- `GET /price/health`
- `POST /price/forecast/price`
- `GET /demand/health`
- `GET /demand/model/info`
- `GET /demand/model/features`
- `POST /demand/forecast`
- `GET /credit/health`
- `POST /credit/score` (coming soon)

## Price API details (`/price`)

### Endpoint

- `POST /price/forecast/price`

### Request body

- `commodity_item_type` (string, required)
- `market_location` (string, required)
- `predict_date` (datetime string, required)
- `history` (optional array of rows)
  - `date` (datetime string)
  - `current_price_etb` (number, `>= 0`)
- `weather` (optional object)
  - `rainfall_anomaly_mm` (number)
  - `temperature_mean_c` (number)
  - `temperature_max_c` (number)
  - `temperature_min_c` (number)
- `market` (optional object)
  - `supply_inflow_units` (number)
  - `demand_proxy_queries` (number)

### Behavior and validation

- If `history` is omitted, API uses synthetic fallback history and returns:
  - `"used_fallback_history": true`
  - a low-confidence warning
- `predict_date` must be strictly after the latest history date.
- Invalid `commodity_item_type` or `market_location` returns `400` with allowed values.
- Input aliases are accepted and normalized before model encoding:
  - `Adama` -> `Adama (Nazret)`
  - `Bishoftu` -> `Bishoftu (Debre Zeyit)`

### Example request

```json
{
  "commodity_item_type": "Tomatoes",
  "market_location": "Adama",
  "predict_date": "2026-04-25T00:00:00",
  "history": [
    { "date": "2026-04-20T00:00:00", "current_price_etb": 97.2 },
    { "date": "2026-04-21T00:00:00", "current_price_etb": 98.0 },
    { "date": "2026-04-22T00:00:00", "current_price_etb": 99.1 }
  ],
  "weather": {
    "rainfall_anomaly_mm": 2.5,
    "temperature_mean_c": 23.5
  },
  "market": {
    "supply_inflow_units": 520,
    "demand_proxy_queries": 610
  }
}
```

### Example error response (invalid category)

```json
{
  "detail": "Unknown market_location 'Amo'. Allowed values: ['Adama (Nazret)', 'Ambo', 'Asella', 'Bale Robe', 'Bishoftu (Debre Zeyit)', 'Burayu', 'Dukem', 'Jimma', 'Meki', 'Mojo', 'Nekemte', 'Sebeta', 'Shashemene', 'Woliso']"
}
```

## Demand API details (`/demand`)

### Endpoint

- `POST /demand/forecast`

### Request body

- `zone` (string, required, must exist in trained labels)
- `product` (string, required, must exist in trained labels)
- `season_type` (string, required, must exist in trained labels)
- `month` (integer, required, `1..12`)
- `week` (integer, required, `1..53`)
- `features` (object, optional)
  - key-value numeric overrides for model features
  - keys should match feature names from `GET /demand/model/features`

### Behavior and validation

- Unknown `zone`, `product`, or `season_type` returns `400` with allowed values.
- Response field `predicted_demand_growth_rate` is returned as a percentage rounded to 2 decimals.

### Example request

```json
{
  "zone": "Adama",
  "product": "Tomato",
  "season_type": "Wet",
  "month": 4,
  "week": 16,
  "features": {
    "days_since_last_harvest": 12,
    "rainfall_anomaly_mm": -1.8
  }
}
```

### Example response

```json
{
  "zone": "Adama",
  "product": "Tomato",
  "season_type": "Wet",
  "month": 4,
  "week": 16,
  "predicted_demand_growth_rate": 39.58
}
```
