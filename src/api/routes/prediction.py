from fastapi import APIRouter, HTTPException

from src.inference.predictor import Predictor
from src.inference.schemas import (
    FraudPredictionRequest,
    FraudPredictionResponse,
)


router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
)


# ==========================================================
# PREDICTOR
# ==========================================================

predictor = Predictor()


# ==========================================================
# HEALTH / STATUS DO MODELO
# ==========================================================

@router.get(
    "/health",
    summary="Verifica o modelo de fraude",
)
def prediction_health():
    """
    Verifica se o modelo está carregado.
    """

    return {
        "status": "ok",
        "model": "XGBoost",
        "target": predictor.target,
        "features": len(predictor.features),
        "threshold": predictor.threshold,
    }


# ==========================================================
# PREDICTION
# ==========================================================

@router.post(
    "/",
    response_model=FraudPredictionResponse,
    summary="Classifica uma transação",
)
def predict(
    request: FraudPredictionRequest,
) -> FraudPredictionResponse:

    try:

        transaction = request.model_dump()

        result = predictor.predict(
            transaction
        )

        return FraudPredictionResponse(
            **result
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Erro interno durante a inferência.",
        ) from exc