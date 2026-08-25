from collections.abc import Mapping
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from fraud_detection.application.ports.model_predictor import (
    ModelPredictor,
)

from fraud_detection.infrastructure.ml.model_loader import ModelLoader


class XGBoostPredictor(ModelPredictor):
    """
    Adapter responsável por executar inferência
    utilizando um modelo XGBoost carregado.

    Implementa o contrato ModelPredictor definido
    pela camada de aplicação.
    """

    def __init__(
        self,
        model_loader: ModelLoader,
    ) -> None:
        self.model_loader = model_loader

        self.model: XGBClassifier = (
            model_loader.get_model()
        )

        self.features = (
            model_loader.get_features()
        )

    def _validate_features(
        self,
        transaction: Mapping[str, Any],
    ) -> None:
        """
        Valida se a transação possui exatamente
        as features esperadas pelo modelo.
        """

        if not isinstance(transaction, Mapping):
            raise TypeError(
                "A transação deve ser um mapeamento."
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

    def _prepare_input(
        self,
        transaction: Mapping[str, Any],
    ) -> pd.DataFrame:
        """
        Converte a transação em DataFrame,
        respeitando a ordem das features.
        """

        self._validate_features(transaction)

        data = {
            feature: [transaction[feature]]
            for feature in self.features
        }

        return pd.DataFrame(
            data,
            columns=self.features,
        )

    def predict_probability(
        self,
        features: Mapping[str, float],
    ) -> float:
        """
        Executa a inferência e retorna somente
        a probabilidade de fraude.

        Este método implementa o contrato
        ModelPredictor.
        """

        dataframe = self._prepare_input(
            features
        )

        probability = float(
            self.model.predict_proba(
                dataframe
            )[0, 1]
        )

        return probability

