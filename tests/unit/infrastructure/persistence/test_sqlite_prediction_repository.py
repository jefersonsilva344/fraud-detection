import sqlite3

from fraud_detection.domain.entities.prediction import Prediction
from fraud_detection.domain.entities.transaction import Transaction
from fraud_detection.domain.value_objects.risk_score import RiskScore
from fraud_detection.infrastructure.persistence.sqlite_prediction_repository import (
    SQLitePredictionRepository,
)


def test_sqlite_prediction_repository_saves_prediction(
    tmp_path,
):
    database_path = tmp_path / "predictions.db"

    repository = SQLitePredictionRepository(
        str(database_path)
    )

    transaction = Transaction(
        time=100.0,
        amount=250.0,
        features={
            "Time": 100.0,
            "Amount": 250.0,
        },
    )

    prediction = Prediction(
        is_fraud=True,
        risk_score=RiskScore(0.85),
        threshold=0.66275984,
    )

    repository.save(
        transaction,
        prediction,
    )

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT
                time,
                amount,
                risk_score,
                risk_percentage,
                threshold,
                is_fraud
            FROM predictions
            """
        ).fetchone()

    assert row is not None
    assert row[0] == 100.0
    assert row[1] == 250.0
    assert row[2] == 0.85
    assert row[3] == 85.0
    assert row[4] == 0.66275984
    assert row[5] == 1


def test_sqlite_prediction_repository_saves_non_fraud_prediction(
    tmp_path,
):
    database_path = tmp_path / "predictions.db"

    repository = SQLitePredictionRepository(
        str(database_path)
    )

    transaction = Transaction(
        time=200.0,
        amount=100.0,
        features={
            "Time": 200.0,
            "Amount": 100.0,
        },
    )

    prediction = Prediction(
        is_fraud=False,
        risk_score=RiskScore(0.20),
        threshold=0.66275984,
    )

    repository.save(
        transaction,
        prediction,
    )

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT
                time,
                amount,
                risk_score,
                risk_percentage,
                threshold,
                is_fraud
            FROM predictions
            """
        ).fetchone()

    assert row is not None
    assert row[0] == 200.0
    assert row[1] == 100.0
    assert row[2] == 0.20
    assert row[3] == 20.0
    assert row[4] == 0.66275984
    assert row[5] == 0


def test_sqlite_prediction_repository_creates_parent_directory_and_schema(
    tmp_path,
):
    database_path = tmp_path / "nested" / "predictions.db"

    SQLitePredictionRepository(database_path)

    assert database_path.exists()

    with sqlite3.connect(database_path) as connection:
        schema = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'predictions'"
        ).fetchone()[0]

    assert "CHECK (amount >= 0)" in schema
    assert "CHECK (risk_score BETWEEN 0 AND 1)" in schema
    assert "CHECK (is_fraud IN (0, 1))" in schema
