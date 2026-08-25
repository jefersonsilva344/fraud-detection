import os
import time

import pytest
import requests


BASE_URL = os.getenv(
    "DOCKER_API_URL",
    "http://localhost:8000",
)


@pytest.fixture(scope="module")
def api_url() -> str:
    """
    URL base da API executando no Docker.
    """
    return BASE_URL


def valid_transaction() -> dict[str, dict[str, float]]:
    """
    Retorna uma transação válida para os testes da API.
    """

    return {
        "features": {
            "Time": 100.0,
            "V1": -1.3598071336738,
            "V2": -0.0727811733098497,
            "V3": 2.53634673796914,
            "V4": 1.37815522427443,
            "V5": -0.338320769942518,
            "V6": 0.462387777762292,
            "V7": 0.239598554061257,
            "V8": 0.0986979012610507,
            "V9": 0.363786969611213,
            "V10": 0.0907941719789316,
            "V11": -0.551599533260813,
            "V12": -0.617800855762348,
            "V13": -0.991389847235408,
            "V14": -0.311169353699879,
            "V15": 1.46817697209427,
            "V16": -0.470400525259478,
            "V17": 0.207971241929242,
            "V18": 0.0257905801985591,
            "V19": 0.403992960255733,
            "V20": 0.251412098239705,
            "V21": -0.018306777944153,
            "V22": 0.277837575558899,
            "V23": -0.110473910188767,
            "V24": 0.0669280749146731,
            "V25": 0.128539358273528,
            "V26": -0.189114843888824,
            "V27": 0.133558376740387,
            "V28": -0.0210530534538215,
            "Amount": 149.62,
        }
    }


def wait_for_api(
    url: str,
    timeout: int = 30,
    interval: float = 1.0,
) -> requests.Response:
    """
    Aguarda a API ficar disponível.
    """

    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            response = requests.get(
                f"{url}/",
                timeout=2,
            )

            if response.status_code == 200:
                return response

        except requests.RequestException:
            pass

        time.sleep(interval)

    pytest.fail(
        f"API Docker não ficou disponível em {url}"
    )


def test_docker_api_is_available(
    api_url: str,
) -> None:
    """
    Verifica se a API está acessível.
    """

    response = wait_for_api(api_url)

    assert response.status_code == 200


def test_docker_root(
    api_url: str,
) -> None:
    """
    Verifica o endpoint raiz da API.
    """

    response = requests.get(
        f"{api_url}/",
        timeout=5,
    )

    assert response.status_code == 200

    data = response.json()

    assert data == {
        "service": "FraudShield AI API",
        "status": "online",
        "version": "2.0.0",
    }


def test_docker_prediction(
    api_url: str,
) -> None:
    """
    Verifica uma predição real utilizando o modelo
    carregado dentro do container Docker.
    """

    response = requests.post(
        f"{api_url}/predictions/fraud",
        json=valid_transaction(),
        timeout=10,
    )

    assert response.status_code == 200

    data = response.json()

    assert set(data) == {
        "is_fraud",
        "risk_score",
        "risk_percentage",
        "threshold",
    }

    assert isinstance(
        data["is_fraud"],
        bool,
    )

    assert isinstance(
        data["risk_score"],
        float,
    )

    assert isinstance(
        data["risk_percentage"],
        float,
    )

    assert isinstance(
        data["threshold"],
        float,
    )

    assert 0.0 <= data["risk_score"] <= 1.0

    assert 0.0 <= data["risk_percentage"] <= 100.0

    assert 0.0 < data["threshold"] < 1.0


def test_docker_rejects_invalid_transaction(
    api_url: str,
) -> None:
    """
    Garante que a API Docker mantém a validação
    do contrato Pydantic.
    """

    transaction = valid_transaction()

    transaction["features"]["Amount"] = -100.0

    response = requests.post(
        f"{api_url}/predictions/fraud",
        json=transaction,
        timeout=10,
    )

    assert response.status_code == 422