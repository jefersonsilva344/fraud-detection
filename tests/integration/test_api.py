from fastapi.testclient import TestClient

from fraud_detection.main import app


client = TestClient(app)


def create_transaction():
    """
    Cria uma transação válida com as 30 features
    esperadas pelo modelo.
    """

    return {
        "Time": 0.0,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.470400,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 149.62,
    }


def test_root():
    """
    Testa o endpoint principal da nova API.
    """

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "online"


def test_prediction_endpoint():
    """
    Testa o endpoint completo de inferência.
    """

    transaction = create_transaction()

    response = client.post(
        "/predictions/fraud",
        json={
            "features": transaction,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "is_fraud" in data
    assert "risk_score" in data
    assert "risk_percentage" in data
    assert "threshold" in data

    assert isinstance(
        data["is_fraud"],
        bool,
    )

    assert 0 <= data["risk_score"] <= 1

    assert 0 <= data["risk_percentage"] <= 100

    assert 0 < data["threshold"] < 1


def test_prediction_rejects_missing_feature():
    """
    Testa rejeição de uma transação sem uma
    feature obrigatória.
    """

    transaction = create_transaction()

    del transaction["V1"]

    response = client.post(
        "/predictions/fraud",
        json={
            "features": transaction,
        },
    )

    assert response.status_code == 422


def test_prediction_rejects_extra_feature():
    """
    Testa rejeição de uma feature desconhecida.
    """

    transaction = create_transaction()

    transaction["feature_inexistente"] = 123

    response = client.post(
        "/predictions/fraud",
        json={
            "features": transaction,
        },
    )

    assert response.status_code == 422


def test_prediction_rejects_negative_amount():
    """
    Testa rejeição de Amount negativo.
    """

    transaction = create_transaction()

    transaction["Amount"] = -100

    response = client.post(
        "/predictions/fraud",
        json={
            "features": transaction,
        },
    )

    assert response.status_code == 422