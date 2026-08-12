from pathlib import Path

from xgboost import XGBClassifier

from src.inference.model_loader import ModelLoader


def test_model_loader_loads_model():
    """
    Verifica se o ModelLoader consegue carregar
    o modelo XGBoost sem realizar treinamento.
    """

    loader = ModelLoader()

    result = loader.load()

    assert result is loader

    model = loader.get_model()

    assert isinstance(
        model,
        XGBClassifier
    )


def test_model_loader_loads_features():
    """
    Verifica se as 30 features foram carregadas
    corretamente do metadata.json.
    """

    loader = ModelLoader().load()

    features = loader.get_features()

    assert isinstance(
        features,
        list
    )

    assert len(features) == 30

    assert "Time" in features
    assert "Amount" in features

    assert "Class" not in features


def test_model_loader_loads_threshold():
    """
    Verifica o threshold congelado utilizado
    pelo modelo em produção.
    """

    loader = ModelLoader().load()

    threshold = loader.get_threshold()

    assert threshold == 0.66275984

    assert 0 < threshold < 1


def test_model_loader_loads_target():
    """
    Verifica o target armazenado no metadata.
    """

    loader = ModelLoader().load()

    assert loader.get_target() == "Class"


def test_artifacts_exist():
    """
    Verifica se os artifacts necessários existem.
    """

    root = Path(__file__).resolve().parents[2]

    model_path = (
        root
        / "artifacts"
        / "xgboost"
        / "model.json"
    )

    metadata_path = (
        root
        / "artifacts"
        / "xgboost"
        / "metadata.json"
    )

    assert model_path.exists()
    assert metadata_path.exists()

    assert model_path.stat().st_size > 0
    assert metadata_path.stat().st_size > 0