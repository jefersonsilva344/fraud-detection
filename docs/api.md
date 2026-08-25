# API

## Base URL

Em execução local, a API está disponível em `http://localhost:8000`. A
documentação interativa OpenAPI fica em `GET /docs`.

## Estado do serviço

### `GET /`

Retorna informações básicas da aplicação.

**Resposta `200 OK`**

```json
{
  "service": "FraudShield AI API",
  "status": "online",
  "version": "2.0.0"
}
```

## Analisar transação

### `POST /predictions/fraud`

Executa a inferência do modelo, aplica a regra de decisão de domínio e
persiste o resultado no banco SQLite.

#### Corpo da requisição

O corpo deve conter `features`, com exatamente 30 campos:

- `Time`: número que identifica o instante da transação no dataset;
- `V1` a `V28`: variáveis numéricas anonimizadas;
- `Amount`: valor da transação, maior ou igual a zero.

```json
{
  "features": {
    "Time": 100.0,
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
    "V11": -0.5516,
    "V12": -0.617801,
    "V13": -0.99139,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.4704,
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
    "Amount": 149.62
  }
}
```

#### Resposta

**Resposta `200 OK`**

```json
{
  "is_fraud": false,
  "risk_score": 0.12,
  "risk_percentage": 12.0,
  "threshold": 0.66275984
}
```

| Campo | Tipo | Descrição |
| --- | --- | --- |
| `is_fraud` | boolean | `true` quando `risk_score` é maior ou igual ao threshold. |
| `risk_score` | number | Probabilidade de fraude retornada pelo modelo, entre 0 e 1. |
| `risk_percentage` | number | `risk_score` representado como percentual, entre 0 e 100. |
| `threshold` | number | Limiar operacional versionado no metadata do modelo. |

#### Erros de validação

A API responde `422 Unprocessable Entity` quando o corpo não respeita o
contrato. Exemplos:

- ausência de qualquer campo entre `Time`, `V1` a `V28` ou `Amount`;
- presença de uma feature não reconhecida;
- `Amount` negativo;
- valor que não pode ser convertido para número.

Exemplo para um `Amount` inválido:

```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "features", "Amount"],
      "msg": "Input should be greater than or equal to 0",
      "input": -10.0,
      "ctx": {"ge": 0.0}
    }
  ]
}
```

## Exemplos de uso

### cURL

```bash
curl -X POST http://localhost:8000/predictions/fraud \
  -H "Content-Type: application/json" \
  -d @transaction.json
```

O arquivo `transaction.json` deve conter o corpo de requisição mostrado acima.

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predictions/fraud",
    json={"features": features},
    timeout=10,
)
response.raise_for_status()
prediction = response.json()
```

## Execução

```bash
uvicorn fraud_detection.main:app --app-dir src --reload
```

Com Docker:

```bash
docker compose up --build
```

O serviço da API usa a porta `8000`; o dashboard está disponível na porta
`8501`. O arquivo SQLite é persistido no volume Docker `fraud-data`.
