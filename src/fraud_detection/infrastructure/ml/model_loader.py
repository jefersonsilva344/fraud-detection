from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
from typing import Any

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[4]

ARTIFACTS_DIR = BASE_DIR / "artifacts" / "xgboost"

MODEL_PATH = ARTIFACTS_DIR / "model.json"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"


class ModelLoader:
    """
    Responsável por carregar o modelo XGBoost
    e seus metadados.

    Não realiza treinamento.
    """

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        metadata_path: str | Path = METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        self.model: XGBClassifier | None = None
        self.metadata: dict[str, Any] | None = None

    def _validate_artifacts(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Modelo XGBoost não encontrado: {self.model_path}"
            )

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Metadata do modelo não encontrada: "
                f"{self.metadata_path}"
            )

    def _load_model(self) -> XGBClassifier:
        model = XGBClassifier()
        model.load_model(self.model_path)
        return model

    def _load_metadata(self) -> dict[str, Any]:
        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def load(self) -> "ModelLoader":
        self._validate_artifacts()

        self.metadata = self._load_metadata()
        self._validate_metadata()
        self.model = self._load_model()

        return self

    def _validate_metadata(self) -> None:
        if not isinstance(self.metadata, dict):
            raise ValueError(
                "metadata.json possui formato inválido."
            )

        required_fields = {
            "model_type",
            "model_format",
            "target",
            "features",
            "threshold",
        }

        missing_fields = required_fields - self.metadata.keys()

        if missing_fields:
            raise ValueError(
                "Campos obrigatórios ausentes no metadata.json: "
                f"{sorted(missing_fields)}"
            )

        if self.metadata["model_type"] != "XGBClassifier":
            raise ValueError("model_type deve ser 'XGBClassifier'.")

        if self.metadata["model_format"] != "xgboost_json":
            raise ValueError("model_format deve ser 'xgboost_json'.")

        target = self.metadata["target"]
        if not isinstance(target, str) or not target:
            raise ValueError("Target inválido no metadata.json.")

        features = self.metadata["features"]
        if not isinstance(features, list) or not features:
            raise ValueError(
                "Nenhuma feature foi definida no metadata.json."
            )

        if (
            not all(isinstance(feature, str) and feature for feature in features)
            or len(features) != len(set(features))
        ):
            raise ValueError(
                "Features do metadata.json devem ser strings únicas."
            )

        if target in features:
            raise ValueError("Target não pode fazer parte das features.")

        threshold = self.metadata["threshold"]

        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(
                "Threshold inválido no metadata.json."
            )

        if not isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(
                "Threshold deve estar entre 0 e 1."
            )

    def get_model(self) -> XGBClassifier:
        if self.model is None:
            raise RuntimeError(
                "Modelo ainda não foi carregado. "
                "Execute load() primeiro."
            )

        return self.model

    def get_metadata(self) -> dict[str, Any]:
        if self.metadata is None:
            raise RuntimeError(
                "Metadata ainda não foi carregado. "
                "Execute load() primeiro."
            )

        return deepcopy(self.metadata)

    def get_features(self) -> list[str]:
        return list(self.get_metadata()["features"])

    def get_threshold(self) -> float:
        return float(self.get_metadata()["threshold"])

    def get_target(self) -> str:
        return str(self.get_metadata()["target"])
