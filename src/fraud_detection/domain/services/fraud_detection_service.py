from fraud_detection.domain.entities.prediction import Prediction
from fraud_detection.domain.value_objects.risk_score import RiskScore


class FraudDetectionService:
    """
    Serviço responsável pelas regras de decisão de fraude.
    """

    def evaluate(
        self,
        probability: float,
        threshold: float,
    ) -> Prediction:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        risk_score = RiskScore(probability)

        return Prediction(
            is_fraud=probability >= threshold,
            risk_score=risk_score,
            threshold=threshold,
        )
