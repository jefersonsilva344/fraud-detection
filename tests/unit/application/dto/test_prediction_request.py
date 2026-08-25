import pytest

from fraud_detection.application.dto.prediction_request import (
    PredictionRequest,
)


def test_prediction_request_normalizes_and_freezes_features():
    source_features = {
        "Time": 10,
        "Amount": 25.5,
        "V1": -0.3,
    }

    request = PredictionRequest(features=source_features)

    source_features["Amount"] = 999.0

    assert request.features == {
        "Time": 10.0,
        "Amount": 25.5,
        "V1": -0.3,
    }

    with pytest.raises(TypeError):
        request.features["Amount"] = 10.0


@pytest.mark.parametrize("missing_feature", ["Time", "Amount"])
def test_prediction_request_requires_transaction_fields(
    missing_feature: str,
):
    features = {"Time": 10.0, "Amount": 25.5}
    del features[missing_feature]

    with pytest.raises(ValueError, match="Required transaction features"):
        PredictionRequest(features=features)


@pytest.mark.parametrize("value", [True, "0.5", float("inf")])
def test_prediction_request_rejects_non_finite_or_non_numeric_features(
    value: object,
):
    with pytest.raises((TypeError, ValueError)):
        PredictionRequest(
            features={
                "Time": 10.0,
                "Amount": 25.5,
                "V1": value,
            }
        )
