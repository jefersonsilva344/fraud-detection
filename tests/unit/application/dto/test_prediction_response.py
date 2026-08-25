from fraud_detection.application.dto.prediction_response import (
    PredictionResponse,
)
from fraud_detection.domain.entities.prediction import Prediction
from fraud_detection.domain.value_objects.risk_score import RiskScore


def test_prediction_response_projects_domain_prediction():
    prediction = Prediction(
        is_fraud=True,
        risk_score=RiskScore(0.85),
        threshold=0.66275984,
    )

    response = PredictionResponse.from_prediction(prediction)

    assert response == PredictionResponse(
        is_fraud=True,
        risk_score=0.85,
        risk_percentage=85.0,
        threshold=0.66275984,
    )
