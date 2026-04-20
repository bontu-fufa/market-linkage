from fastapi import FastAPI

from services.credit_scoring.app import app as credit_app
from services.market_demand.app import app as demand_app
from services.price_prediction.app import app as price_app

app = FastAPI(
    title="Market Linkage ML Gateway",
    description="Unified gateway for price prediction, market demand, and credit scoring APIs.",
)


@app.get("/health")
def gateway_health():
    return {"status": "ok", "service": "gateway"}


app.mount("/price", price_app)
app.mount("/demand", demand_app)
app.mount("/credit", credit_app)
