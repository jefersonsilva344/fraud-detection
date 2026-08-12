from typing import Any

import pandas as pd

from src.inference.model_loader import ModelLoader


class Predictor:
    """
    Responsável por executar inferência utilizando
    o modelo XGBoost previamente treinado.

    Este componente NÃO treina o modelo.

    Responsabilidades:

    1. Receber uma transação.
    2. Validar as features.
    3. Organizar as features na ordem correta.
    4. Executar predict_proba().
    5. Aplicar o threshold congelado.
    6. Retornar o resultado da predição.
    """

    def __init__(
        self,
        model_loader: ModelLoader | None = None
    ):
        self.model_loader = (
            model_loader
            if model_loader is not None
            else ModelLoader()
        )

        self.model_loader.load()

        self.model = (
            self.model_loader.get_model()
        )

        self.features = (
            self.model_loader.get_features()
        )

        self.threshold = (
            self.model_loader.get_threshold()
        )

        self.target = (
            self.model_loader.get_target()
        )

    # ======================================================
    # VALIDAÇÃO DAS FEATURES
    # ======================================================

    def _validate_features(
        self,
        transaction: dict[str, Any]
    ) -> None:
        """
        Valida se a transação possui exatamente
        as features esperadas pelo modelo.
        """

        if not isinstance(
            transaction,
            dict
        ):
            raise TypeError(
                "A transação deve ser um dicionário."
            )

        received_features = set(
            transaction.keys()
        )

        expected_features = set(
            self.features
        )

        missing_features = (
            expected_features
            - received_features
        )

        extra_features = (
            received_features
            - expected_features
        )

        if missing_features:

            raise ValueError(
                "Features obrigatórias ausentes: "
                f"{sorted(missing_features)}"
            )

        if extra_features:

            raise ValueError(
                "Features não reconhecidas: "
                f"{sorted(extra_features)}"
            )

    # ======================================================
    # PREPARAÇÃO DOS DADOS
    # ======================================================

    def _prepare_input(
        self,
        transaction: dict[str, Any]
    ) -> pd.DataFrame:
        """
        Converte a transação em DataFrame
        respeitando exatamente a ordem das
        features utilizadas no treinamento.
        """

        self._validate_features(
            transaction
        )

        data = {
            feature: [
                transaction[feature]
            ]
            for feature in self.features
        }

        dataframe = pd.DataFrame(
            data,
            columns=self.features
        )

        return dataframe

    # ======================================================
    # PREDIÇÃO
    # ======================================================

    def predict(
        self,
        transaction: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Executa uma predição para uma única transação.
        """

        dataframe = self._prepare_input(
            transaction
        )

        probability = float(
            self.model.predict_proba(
                dataframe
            )[0, 1]
        )

        is_fraud = (
            probability >= self.threshold
        )

        return {
            "fraud_probability": probability,
            "is_fraud": is_fraud,
            "threshold": self.threshold,
        }