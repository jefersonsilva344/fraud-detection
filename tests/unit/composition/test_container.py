from unittest.mock import Mock

from fraud_detection.application.ports.model_predictor import ModelPredictor
from fraud_detection.application.ports.prediction_repository import (
    PredictionRepository,
)
from fraud_detection.composition import container
from fraud_detection.domain.services.fraud_detection_service import (
    FraudDetectionService,
)
from fraud_detection.infrastructure.ml.model_loader import ModelLoader


def test_composition_root_wires_and_caches_application_dependencies(
    monkeypatch,
):
    loader = Mock(spec=ModelLoader)
    loader.load.return_value = loader
    loader.get_threshold.return_value = 0.66275984
    predictor = Mock(spec=ModelPredictor)
    repository = Mock(spec=PredictionRepository)

    model_loader_factory = Mock(return_value=loader)
    predictor_factory = Mock(return_value=predictor)
    repository_factory = Mock(return_value=repository)

    monkeypatch.setattr(container, "ModelLoader", model_loader_factory)
    monkeypatch.setattr(container, "XGBoostPredictor", predictor_factory)
    monkeypatch.setattr(
        container,
        "SQLitePredictionRepository",
        repository_factory,
    )
    container.create_predict_fraud_use_case.cache_clear()

    try:
        first_use_case = container.create_predict_fraud_use_case()
        second_use_case = container.create_predict_fraud_use_case()

        assert first_use_case is second_use_case
        assert first_use_case.predictor is predictor
        assert isinstance(first_use_case.fraud_service, FraudDetectionService)
        assert first_use_case.prediction_repository is repository
        assert first_use_case.threshold == 0.66275984
        model_loader_factory.assert_called_once_with()
        loader.load.assert_called_once_with()
        predictor_factory.assert_called_once_with(model_loader=loader)
        repository_factory.assert_called_once_with(
            database_path=container.PREDICTIONS_DATABASE_PATH,
        )
    finally:
        container.create_predict_fraud_use_case.cache_clear()
