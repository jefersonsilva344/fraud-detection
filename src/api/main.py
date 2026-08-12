from fastapi import FastAPI

from src.api.routes.prediction import router as prediction_router


# ==========================================================
# APPLICATION
# ==========================================================

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "API de detecção de fraude utilizando XGBoost."
    ),
    version="1.0.0",
)


# ==========================================================
# ROUTES
# ==========================================================

app.include_router(
    prediction_router
)


# ==========================================================
# ROOT
# ==========================================================

@app.get(
    "/",
    tags=["System"],
)
def root():
    return {
        "service": "Fraud Detection API",
        "status": "online",
        "version": "1.0.0",
    }


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get(
    "/health",
    tags=["System"],
)
def health():
    return {
        "status": "healthy",
    }