from fraud_detection.application.ports.model_predictor import (
    ModelPredictor,
)
from fraud_detection.infrastructure.ml.model_loader import ModelLoader
from fraud_detection.infrastructure.ml.xgboost_predictor import (
    XGBoostPredictor,
)

def test_xgboost_predictor_implements_model_predictor():
    loader = ModelLoader().load()

    predictor = XGBoostPredictor(loader)

    assert isinstance(
        predictor,
        ModelPredictor,
    )


def test_xgboost_predictor_returns_probability():
    loader = ModelLoader().load()

    predictor = XGBoostPredictor(loader)

    features = {
        feature: 0.0
        for feature in loader.get_features()
    }

    probability = predictor.predict_probability(
        features
    )

    assert isinstance(
        probability,
        float,
    )

    assert 0.0 <= probability <= 1.0


def test_xgboost_predictor_rejects_features_outside_model_contract():
    loader = ModelLoader().load()
    predictor = XGBoostPredictor(loader)
    features = {
        feature: 0.0
        for feature in loader.get_features()
    }
    del features["V1"]
    features["unknown"] = 0.0

    with pytest.raises(ValueError, match="Features obrigatórias ausentes"):
        predictor.predict_probability(features)


def test_xgboost_predictor_exposes_only_probability_prediction():
    predictor = XGBoostPredictor(ModelLoader().load())

    assert not hasattr(predictor, "predict")
    assert not hasattr(predictor, "threshold")
import pytest
