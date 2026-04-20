# Price Prediction RF - Model Card

## Model summary

- Model type: Random Forest Regressor
- Target: Next-day market price in ETB
- Serving interface: FastAPI (`/forecast/price`)
- Artifacts: `best_model.pkl`, preprocessors, and feature metadata in this folder

## Training data note

This model is trained on synthetic data. Predictions are useful for prototyping and API integration tests, but they should not be treated as production-grade market intelligence without validation on real observations.

## Intended use

- Demo applications
- Internal workflow testing
- Baseline experimentation

## Out-of-scope use

- Financial or procurement decisions without human review
- Public-facing claims of real market forecasting accuracy
- Policy or safety-critical decisions

## Known limitations

- Synthetic patterns may not represent real market shocks, seasonality drift, or regime changes.
- Long-horizon forecasts can degrade quickly when history is short or missing.
- Category combinations not represented in training may produce unstable outputs.

## API fallback behavior

If `history` is omitted in `/forecast/price`, the API creates one synthetic history row using `BASELINE_PRICE_ETB` (default: `100`) and returns:

- `used_fallback_history: true`
- `warning` message indicating low confidence

Prefer sending recent real history (7-30 days) whenever possible.

## Validation recommendations before production

- Backtest on real historical data by commodity/market segment
- Track MAE/MAPE drift weekly
- Add prediction intervals and confidence scoring
- Define a human-approval workflow for low-confidence scenarios
