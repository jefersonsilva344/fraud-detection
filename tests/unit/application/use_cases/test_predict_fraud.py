from unittest.mock import Mock

import pytest

from fraud_detection.application.dto.prediction_request import (
    PredictionRequest,
)
from fraud_detection.application.dto.prediction_response import (
    PredictionResponse,
)
from fraud_detection.application.use_cases.predict_fraud import (
    PredictFraudUseCase,
)
from fraud_detection.domain.services.fraud_detection_service import (
    FraudDetectionService,
)
from fraud_detection.application.ports.prediction_repository import (
    PredictionRepository,
)
from fraud_detection.application.ports.model_predictor import (
    ModelPredictor,
)

def create_request() -> PredictionRequest:
    return PredictionRequest(
        features={
            "Time": 100.0,
            "Amount": 250.0,
            "V1": 0.1,
            "V2": -0.2,
        }
    )


def create_use_case(
    probability: float = 0.90,
) -> tuple[PredictFraudUseCase, Mock, Mock]:
    predictor = Mock(spec=ModelPredictor)
    predictor.predict_probability.return_value = probability
    prediction_repository = Mock(spec=PredictionRepository)

    use_case = PredictFraudUseCase(
        predictor=predictor,
        fraud_service=FraudDetectionService(),
        prediction_repository=prediction_repository,
        threshold=0.66275984,
    )

    return use_case, predictor, prediction_repository


def test_predict_fraud_use_case_returns_prediction_response():
    use_case, predictor, prediction_repository = create_use_case()

    request = create_request()

    response = use_case.execute(request)

    assert isinstance(response, PredictionResponse)
    assert response.is_fraud is True
    assert response.risk_score == 0.90
    assert response.risk_percentage == 90.0
    assert response.threshold == 0.66275984

    predictor.predict_probability.assert_called_once_with(request.features)
    prediction_repository.save.assert_called_once()


def test_predict_fraud_use_case_identifies_low_risk_transaction():
    use_case, _, _ = create_use_case(probability=0.20)

    request = create_request()

    response = use_case.execute(request)

    assert response.is_fraud is False
    assert response.threshold == 0.66275984


def test_predict_fraud_use_case_passes_transaction_features_to_predictor():
    use_case, predictor, _ = create_use_case(probability=0.20)

    request = create_request()

    use_case.execute(request)

    predictor.predict_probability.assert_called_once_with(
        request.features
    )


def test_predict_fraud_use_case_persists_prediction():
    use_case, _, prediction_repository = create_use_case()

    request = create_request()

    use_case.execute(request)

    prediction_repository.save.assert_called_once()
    transaction = prediction_repository.save.call_args.kwargs["transaction"]
    prediction = prediction_repository.save.call_args.kwargs["prediction"]

    assert transaction.time == request.features["Time"]
    assert transaction.amount == request.features["Amount"]
    assert transaction.features is request.features
    assert prediction.is_fraud is True
    assert prediction.risk_score.value == 0.90


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_predict_fraud_use_case_rejects_invalid_threshold(
    threshold: float,
):
    predictor = Mock(spec=ModelPredictor)
    prediction_repository = Mock(spec=PredictionRepository)

    with pytest.raises(ValueError, match="Threshold"):
        PredictFraudUseCase(
            predictor=predictor,
            fraud_service=FraudDetectionService(),
            prediction_repository=prediction_repository,
            threshold=threshold,
        )

    predictor.predict_probability.assert_not_called()
    prediction_repository.save.assert_not_called()


def test_predict_fraud_use_case_does_not_persist_when_prediction_fails():
    use_case, predictor, prediction_repository = create_use_case()
    predictor.predict_probability.side_effect = RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="model unavailable"):
        use_case.execute(create_request())

    prediction_repository.save.assert_not_called()
