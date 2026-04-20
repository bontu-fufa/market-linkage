from fastapi import FastAPI, HTTPException

app = FastAPI(title="Market Demand API")


@app.get("/health")
def health():
    return {"status": "ok", "service": "market_demand", "stage": "coming_soon"}


@app.post("/forecast")
def forecast():
    raise HTTPException(
        status_code=501,
        detail="Market demand model is coming soon.",
    )
