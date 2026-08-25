from dataclasses import dataclass

from fraud_detection.domain.value_objects.risk_score import RiskScore


@dataclass(frozen=True)
class Prediction:
    """
    Resultado de uma análise de fraude.
    """

    is_fraud: bool
    risk_score: RiskScore
    threshold: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")
