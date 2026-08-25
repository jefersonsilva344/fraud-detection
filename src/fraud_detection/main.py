from fastapi import FastAPI

from fraud_detection.interfaces.routes.prediction import (
    router as prediction_router,
)


app = FastAPI(
    title="FraudShield AI",
    description="Financial Fraud Detection API",
    version="2.0.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "FraudShield AI API",
        "status": "online",
        "version": "2.0.0",
    }


app.include_router(
    prediction_router,
)