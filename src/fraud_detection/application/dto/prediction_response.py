from dataclasses import dataclass

from fraud_detection.domain.entities.prediction import Prediction


@dataclass(frozen=True)
class PredictionResponse:
    """
    Resultado apresentado pela camada de aplicação.
    """

    is_fraud: bool
    risk_score: float
    risk_percentage: float
    threshold: float

    @classmethod
    def from_prediction(
        cls,
        prediction: Prediction,
    ) -> "PredictionResponse":
        return cls(
            is_fraud=prediction.is_fraud,
            risk_score=prediction.risk_score.value,
            risk_percentage=prediction.risk_score.percentage,
            threshold=prediction.threshold,
        )
