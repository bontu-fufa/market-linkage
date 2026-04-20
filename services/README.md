# Services layout

- `price_prediction` is the live price API at `services/price_prediction/app.py`, mounted by `gateway.py` under `/price`.
- `market_demand` is scaffolded and currently returns `501 coming soon`.
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
- `POST /demand/forecast` (coming soon)
- `GET /credit/health`
- `POST /credit/score` (coming soon)
