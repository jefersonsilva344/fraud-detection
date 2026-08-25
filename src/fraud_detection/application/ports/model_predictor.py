from abc import ABC, abstractmethod
from collections.abc import Mapping


class ModelPredictor(ABC):
    """
    Contrato da aplicação para execução de predições
    utilizando um modelo de Machine Learning.
    """

    @abstractmethod
    def predict_probability(
        self,
        features: Mapping[str, float],
    ) -> float:
        raise NotImplementedError
