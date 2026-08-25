from abc import ABC, abstractmethod

from fraud_detection.domain.entities.prediction import Prediction
from fraud_detection.domain.entities.transaction import Transaction


class PredictionRepository(ABC):
    """
    Contrato para persistência de predições.
    """

    @abstractmethod
    def save(
        self,
        transaction: Transaction,
        prediction: Prediction,
    ) -> None:
        raise NotImplementedError