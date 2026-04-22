from fastapi import APIRouter, FastAPI, HTTPException

router = APIRouter()
app = FastAPI(title="Credit Scoring API")


@router.get("/health")
def health():
    return {"status": "ok", "service": "credit_scoring", "stage": "coming_soon"}


@router.post("/score")
def score():
    raise HTTPException(
        status_code=501,
        detail="Credit scoring model is coming soon.",
    )


app.include_router(router)
