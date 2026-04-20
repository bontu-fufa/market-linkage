from fastapi import FastAPI, HTTPException

app = FastAPI(title="Credit Scoring API")


@app.get("/health")
def health():
    return {"status": "ok", "service": "credit_scoring", "stage": "coming_soon"}


@app.post("/score")
def score():
    raise HTTPException(
        status_code=501,
        detail="Credit scoring model is coming soon.",
    )
