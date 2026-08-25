from fraud_detection.application.dto.prediction_request import (
    PredictionRequest,
)
from fraud_detection.application.dto.prediction_response import (
    PredictionResponse,
)
from fraud_detection.application.ports.model_predictor import (
    ModelPredictor,
)
from fraud_detection.application.ports.prediction_repository import (
    PredictionRepository,
)
from fraud_detection.domain.entities.transaction import Transaction
from fraud_detection.domain.services.fraud_detection_service import (
    FraudDetectionService,
)


class PredictFraudUseCase:
    """
    Orquestra o caso de uso de detecção de fraude.
    """

    def __init__(
        self,
        predictor: ModelPredictor,
        fraud_service: FraudDetectionService,
        prediction_repository: PredictionRepository,
        threshold: float,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        self.predictor = predictor
        self.fraud_service = fraud_service
        self.prediction_repository = prediction_repository
        self.threshold = threshold

    def execute(
        self,
        request: PredictionRequest,
    ) -> PredictionResponse:

        transaction = Transaction(
            time=request.features["Time"],
            amount=request.features["Amount"],
            features=request.features,
        )

        probability = self.predictor.predict_probability(
            transaction.features
        )

        prediction = self.fraud_service.evaluate(
            probability=probability,
            threshold=self.threshold,
        )

        self.prediction_repository.save(
            transaction=transaction,
            prediction=prediction,
        )

        return PredictionResponse.from_prediction(prediction)
