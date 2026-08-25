from functools import lru_cache
from pathlib import Path

from fraud_detection.application.use_cases.predict_fraud import (
    PredictFraudUseCase,
)
from fraud_detection.domain.services.fraud_detection_service import (
    FraudDetectionService,
)
from fraud_detection.infrastructure.ml.model_loader import (
    ModelLoader,
)
from fraud_detection.infrastructure.ml.xgboost_predictor import (
    XGBoostPredictor,
)
from fraud_detection.infrastructure.persistence.sqlite_prediction_repository import (
    SQLitePredictionRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS_DATABASE_PATH = PROJECT_ROOT / "data" / "predictions.db"


@lru_cache(maxsize=1)
def create_predict_fraud_use_case() -> PredictFraudUseCase:
    """
    Monta todas as dependências necessárias para
    executar o caso de uso de detecção de fraude.

    Este módulo funciona como Composition Root da aplicação.
    """

    model_loader = ModelLoader().load()

    predictor = XGBoostPredictor(
        model_loader=model_loader,
    )

    fraud_service = FraudDetectionService()

    prediction_repository = SQLitePredictionRepository(
        database_path=PREDICTIONS_DATABASE_PATH,
    )

    return PredictFraudUseCase(
        predictor=predictor,
        fraud_service=fraud_service,
        threshold=model_loader.get_threshold(),
        prediction_repository=prediction_repository,
    )
from functools import lru_cache
from pathlib import Path
