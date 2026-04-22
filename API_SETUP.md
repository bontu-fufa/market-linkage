# API setup for Random Forest model

Your FastAPI app expects model artifacts inside `model/rf/`.

## 1) Add this export block to the end of `price_prediction.ipynb`

```python
from pathlib import Path
import joblib
import pandas as pd

save_dir = Path("model/rf")
save_dir.mkdir(parents=True, exist_ok=True)

# Force Random Forest export (even if another model wins in evaluation)
rf_model = trained_models["Random Forest"]

joblib.dump(rf_model, save_dir / "rf_model.pkl")
joblib.dump(scaler, save_dir / "scaler.pkl")
joblib.dump(encoder, save_dir / "categorical_encoder.pkl")
joblib.dump(feature_cols, save_dir / "feature_cols.pkl")
joblib.dump(numerical_cols, save_dir / "numerical_cols.pkl")
joblib.dump(categorical_cols, save_dir / "categorical_cols.pkl")
joblib.dump(pd.Timestamp(df["date"].min()), save_dir / "train_start_date.pkl")

print("Saved API artifacts to:", save_dir.resolve())
```

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 3) Run the API

```bash
python main.py
```

Or run the unified multi-model gateway:

```bash
uvicorn gateway:app --host 0.0.0.0 --port 8000
```

## 4) Quick checks

- `GET /health`
- `GET /model/info`
- `POST /forecast/price`

Gateway checks:

- `GET /health`
- `GET /price/health`
- `POST /price/forecast/price`
- `GET /demand/health`
- `GET /demand/model/info`
- `GET /demand/model/features`
- `POST /demand/forecast`
- `GET /credit/health` (coming soon)
- `POST /credit/score` (returns 501)

## 5) No-history fallback (optional)

`history` is now optional. If omitted, the API builds one synthetic history point and returns a low-confidence warning.

Optional env var:

```bash
set BASELINE_PRICE_ETB=100
```

## 6) Demand model artifacts

Place these files in `artifacts/demand_forcast/` (or set `DEMAND_MODEL_DIR`):

- `enhanced_demand_model.pkl`
- `enhanced_scaler.pkl`
- `enhanced_le_zone.pkl`
- `enhanced_le_product.pkl`
- `enhanced_le_season.pkl`
- `enhanced_feature_names.pkl`
