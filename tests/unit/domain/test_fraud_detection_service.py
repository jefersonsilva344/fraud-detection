import pytest

from fraud_detection.domain.entities.prediction import Prediction
from fraud_detection.domain.services.fraud_detection_service import (
    FraudDetectionService,
)
from fraud_detection.domain.value_objects.risk_score import RiskScore


THRESHOLD = 0.66275984


def test_probability_above_threshold_is_fraud():
    service = FraudDetectionService()

    prediction = service.evaluate(
        probability=0.80,
        threshold=THRESHOLD,
    )

    assert isinstance(prediction, Prediction)
    assert prediction.is_fraud is True
    assert prediction.risk_score.value == 0.80
    assert prediction.threshold == THRESHOLD


def test_probability_below_threshold_is_not_fraud():
    service = FraudDetectionService()

    prediction = service.evaluate(
        probability=0.20,
        threshold=THRESHOLD,
    )

    assert prediction.is_fraud is False
    assert prediction.risk_score.value == 0.20
    assert prediction.threshold == THRESHOLD


def test_probability_equal_to_threshold_is_fraud():
    service = FraudDetectionService()

    prediction = service.evaluate(
        probability=THRESHOLD,
        threshold=THRESHOLD,
    )

    assert prediction.is_fraud is True


def test_probability_is_converted_to_risk_score():
    service = FraudDetectionService()

    prediction = service.evaluate(
        probability=0.85,
        threshold=THRESHOLD,
    )

    assert isinstance(prediction.risk_score, RiskScore)
    assert prediction.risk_score.value == 0.85
    assert prediction.risk_score.percentage == 85.0


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_probability_outside_probability_range_is_rejected(
    probability: float,
):
    service = FraudDetectionService()

    with pytest.raises(ValueError, match="Risk score"):
        service.evaluate(
            probability=probability,
            threshold=THRESHOLD,
        )


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_threshold_outside_probability_range_is_rejected(
    threshold: float,
):
    service = FraudDetectionService()

    with pytest.raises(ValueError, match="Threshold"):
        service.evaluate(
            probability=0.5,
            threshold=threshold,
        )
