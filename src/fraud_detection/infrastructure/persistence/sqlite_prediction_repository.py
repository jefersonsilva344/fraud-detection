import sqlite3
from pathlib import Path

from fraud_detection.application.ports.prediction_repository import (
    PredictionRepository,
)
from fraud_detection.domain.entities.prediction import Prediction
from fraud_detection.domain.entities.transaction import Transaction


class SQLitePredictionRepository(PredictionRepository):
    """
    Implementação SQLite do contrato PredictionRepository.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def _create_table(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time REAL NOT NULL,
                    amount REAL NOT NULL CHECK (amount >= 0),
                    risk_score REAL NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
                    risk_percentage REAL NOT NULL
                        CHECK (risk_percentage BETWEEN 0 AND 100),
                    threshold REAL NOT NULL CHECK (threshold BETWEEN 0 AND 1),
                    is_fraud INTEGER NOT NULL CHECK (is_fraud IN (0, 1))
                )
                """
            )

    def save(
        self,
        transaction: Transaction,
        prediction: Prediction,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO predictions (
                    time,
                    amount,
                    risk_score,
                    risk_percentage,
                    threshold,
                    is_fraud
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction.time,
                    transaction.amount,
                    prediction.risk_score.value,
                    prediction.risk_score.percentage,
                    prediction.threshold,
                    int(prediction.is_fraud),
                ),
            )
