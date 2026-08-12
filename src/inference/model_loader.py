from pathlib import Path
import json

from xgboost import XGBClassifier


# ==========================================================
# CONFIGURAÇÃO DOS CAMINHOS
# ==========================================================

# Estrutura:
#
# fraud-detection/
# ├── artifacts/
# │   └── xgboost/
# │       ├── model.json
# │       └── metadata.json
# │
# └── src/
#     └── inference/
#         └── model_loader.py
#
# parent                -> inference
# parent.parent         -> src
# parent.parent.parent  -> fraud-detection

BASE_DIR = Path(__file__).resolve().parent.parent.parent


ARTIFACTS_DIR = (
    BASE_DIR
    / "artifacts"
    / "xgboost"
)


MODEL_PATH = (
    ARTIFACTS_DIR
    / "model.json"
)


METADATA_PATH = (
    ARTIFACTS_DIR
    / "metadata.json"
)


# ==========================================================
# MODEL LOADER
# ==========================================================

class ModelLoader:
    """
    Responsável exclusivamente por carregar
    o modelo XGBoost e seus metadados.

    Este componente NÃO treina o modelo.

    Responsabilidades:

    1. Validar existência dos artifacts.
    2. Carregar model.json.
    3. Carregar metadata.json.
    4. Disponibilizar modelo.
    5. Disponibilizar features.
    6. Disponibilizar threshold.
    """


    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH
    ):
        self.model_path = Path(model_path)

        self.metadata_path = Path(
            metadata_path
        )

        self.model = None

        self.metadata = None


    # ======================================================
    # VALIDAÇÃO DOS ARTIFACTS
    # ======================================================

    def _validate_artifacts(self):
        """
        Verifica se os arquivos necessários existem.
        """

        if not self.model_path.exists():

            raise FileNotFoundError(
                "Modelo XGBoost não encontrado:\n"
                f"{self.model_path}"
            )


        if not self.metadata_path.exists():

            raise FileNotFoundError(
                "Metadata do modelo não encontrada:\n"
                f"{self.metadata_path}"
            )


    # ======================================================
    # CARREGAMENTO DO MODELO
    # ======================================================

    def _load_model(self):
        """
        Carrega o modelo XGBoost salvo em JSON.
        """

        model = XGBClassifier()

        model.load_model(
            self.model_path
        )

        return model


    # ======================================================
    # CARREGAMENTO DOS METADADOS
    # ======================================================

    def _load_metadata(self):
        """
        Carrega os metadados do modelo.
        """

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)


        return metadata


    # ======================================================
    # LOAD
    # ======================================================

    def load(self):
        """
        Carrega modelo e metadata.

        Retorna o próprio ModelLoader
        para permitir uso encadeado.
        """

        self._validate_artifacts()

        self.model = self._load_model()

        self.metadata = self._load_metadata()

        self._validate_metadata()

        return self


    # ======================================================
    # VALIDAÇÃO DOS METADADOS
    # ======================================================

    def _validate_metadata(self):
        """
        Valida se os metadados possuem
        informações essenciais.
        """

        if not isinstance(
            self.metadata,
            dict
        ):

            raise ValueError(
                "metadata.json possui formato inválido."
            )


        required_fields = [
            "model_type",
            "model_format",
            "target",
            "features",
            "threshold",
        ]


        missing_fields = [
            field
            for field in required_fields
            if field not in self.metadata
        ]


        if missing_fields:

            raise ValueError(
                "Campos obrigatórios ausentes "
                "no metadata.json: "
                f"{missing_fields}"
            )


        if not self.metadata["features"]:

            raise ValueError(
                "Nenhuma feature foi definida "
                "no metadata.json."
            )


        threshold = self.metadata["threshold"]


        if not (
            isinstance(
                threshold,
                (int, float)
            )
            and 0 <= threshold <= 1
        ):

            raise ValueError(
                "Threshold inválido no metadata.json."
            )


    # ======================================================
    # GET MODEL
    # ======================================================

    def get_model(self):
        """
        Retorna o modelo XGBoost carregado.
        """

        if self.model is None:

            raise RuntimeError(
                "Modelo ainda não foi carregado. "
                "Execute load() primeiro."
            )

        return self.model


    # ======================================================
    # GET METADATA
    # ======================================================

    def get_metadata(self):
        """
        Retorna os metadados do modelo.
        """

        if self.metadata is None:

            raise RuntimeError(
                "Metadata ainda não foi carregado. "
                "Execute load() primeiro."
            )

        return self.metadata


    # ======================================================
    # GET FEATURES
    # ======================================================

    def get_features(self):
        """
        Retorna a lista de features utilizadas
        pelo modelo durante o treinamento.
        """

        metadata = self.get_metadata()

        return metadata["features"]


    # ======================================================
    # GET THRESHOLD
    # ======================================================

    def get_threshold(self):
        """
        Retorna o threshold congelado
        utilizado pelo modelo.
        """

        metadata = self.get_metadata()

        return float(
            metadata["threshold"]
        )


    # ======================================================
    # GET TARGET
    # ======================================================

    def get_target(self):
        """
        Retorna o nome da variável target.
        """

        metadata = self.get_metadata()

        return metadata["target"]