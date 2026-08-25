from fastapi import APIRouter, Depends

from fraud_detection.application.dto.prediction_request import (
    PredictionRequest,
)
from fraud_detection.application.dto.prediction_response import (
    PredictionResponse,
)
from fraud_detection.application.use_cases.predict_fraud import (
    PredictFraudUseCase,
)
from fraud_detection.composition.container import (
    create_predict_fraud_use_case,
)
from fraud_detection.interfaces.schemas.prediction import (
    PredictionRequestSchema,
)


router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
)


@router.post(
    "/fraud",
    response_model=PredictionResponse,
)
def predict_fraud(
    request: PredictionRequestSchema,
    use_case: PredictFraudUseCase = Depends(
        create_predict_fraud_use_case,
    ),
) -> PredictionResponse:

    application_request = PredictionRequest(
        features=request.features.model_dump(),
    )

    return use_case.execute(
        application_request,
    )