from pathlib import Path

import pytest
from xgboost import XGBClassifier

from src.inference.model_loader import ModelLoader


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "xgboost"

MODEL_PATH = ARTIFACT_DIR / "model.json"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"


EXPECTED_FEATURES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]

EXPECTED_THRESHOLD = 0.66275984


# ============================================================
# TESTES DE ARQUIVOS
# ============================================================

def test_model_artifact_exists():
    """
    Verifica se o artifact do modelo existe.
    """

    assert MODEL_PATH.exists(), (
        f"Modelo não encontrado: {MODEL_PATH}"
    )


def test_metadata_artifact_exists():
    """
    Verifica se o metadata.json existe.
    """

    assert METADATA_PATH.exists(), (
        f"Metadata não encontrado: {METADATA_PATH}"
    )


# ============================================================
# TESTE DE CARREGAMENTO
# ============================================================

def test_model_loader_loads_artifact():
    """
    Verifica se o ModelLoader consegue carregar
    model.json + metadata.json.
    """

    loader = ModelLoader()

    result = loader.load()

    assert result is loader


# ============================================================
# TESTE DO MODELO
# ============================================================

def test_loaded_model_is_xgboost_classifier():
    """
    Verifica se o modelo carregado é realmente
    uma instância de XGBClassifier.
    """

    loader = ModelLoader().load()

    model = loader.get_model()

    assert model is not None

    assert isinstance(
        model,
        XGBClassifier,
    )


# ============================================================
# TESTE DAS FEATURES
# ============================================================

def test_model_features():
    """
    Verifica se o metadata contém as 30 features
    esperadas pelo modelo.
    """

    loader = ModelLoader().load()

    features = loader.get_features()

    assert isinstance(features, list)

    assert len(features) == 30

    assert features == EXPECTED_FEATURES


# ============================================================
# TESTE DO TARGET
# ============================================================

def test_model_target():
    """
    Verifica se o target registrado no metadata
    é Class.
    """

    loader = ModelLoader().load()

    assert loader.get_target() == "Class"


# ============================================================
# TESTE DO THRESHOLD
# ============================================================

def test_model_threshold():
    """
    Verifica se o threshold congelado utilizado
    durante a avaliação final foi preservado.
    """

    loader = ModelLoader().load()

    threshold = loader.get_threshold()

    assert isinstance(
        threshold,
        float,
    )

    assert threshold == pytest.approx(
        EXPECTED_THRESHOLD,
    )


# ============================================================
# TESTE DE PREDIÇÃO DO ARTIFACT
# ============================================================

def test_loaded_model_can_predict():
    """
    Verifica se o modelo carregado consegue
    executar predict_proba sem realizar treinamento
    e sem depender do dataset original.
    """

    import pandas as pd

    loader = ModelLoader().load()

    features = loader.get_features()

    data = {
        feature: [0.0] * 5
        for feature in features
    }

    df = pd.DataFrame(data)

    probabilities = (
        loader
        .get_model()
        .predict_proba(df[features])[:, 1]
    )

    assert len(probabilities) == 5

    assert all(
        0.0 <= probability <= 1.0
        for probability in probabilities
    )


# ============================================================
# TESTE DE CONSISTÊNCIA DO ARTIFACT
# ============================================================

def test_model_feature_count_matches_metadata():
    """
    Verifica se o número de features registrado
    no metadata é compatível com o modelo.
    """

    loader = ModelLoader().load()

    model = loader.get_model()

    features = loader.get_features()

    assert len(features) == model.n_features_in_


# ============================================================
# TESTE DE CARREGAMENTO REPETIDO
# ============================================================

def test_model_can_be_loaded_multiple_times():
    """
    Verifica se o artifact pode ser carregado
    repetidamente sem depender de treinamento.
    """

    loader_1 = ModelLoader().load()
    loader_2 = ModelLoader().load()

    assert loader_1.get_features() == (
        loader_2.get_features()
    )

    assert loader_1.get_threshold() == (
        loader_2.get_threshold()
    )

    assert loader_1.get_target() == (
        loader_2.get_target()
    )