from fastapi import FastAPI

from services.credit_scoring.app import router as credit_router
from services.market_demand.app import init_bundle as init_demand_bundle
from services.market_demand.app import router as demand_router
from services.price_prediction.app import init_bundle as init_price_bundle
from services.price_prediction.app import router as price_router

app = FastAPI(
    title="Market Linkage ML Gateway",
    description="Unified gateway for price prediction, market demand, and credit scoring APIs.",
)


@app.get("/health")
def gateway_health():
    return {"status": "ok", "service": "gateway"}


@app.on_event("startup")
async def startup_event() -> None:
    init_price_bundle()
    init_demand_bundle()


app.include_router(price_router, prefix="/price", tags=["price"])
app.include_router(demand_router, prefix="/demand", tags=["demand"])
app.include_router(credit_router, prefix="/credit", tags=["credit"])
