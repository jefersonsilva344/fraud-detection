# Arquitetura

## Visão geral

O FraudShield AI segue uma arquitetura em camadas. O domínio contém as regras
de decisão; a aplicação orquestra o caso de uso; a infraestrutura adapta modelo
e banco de dados; e as interfaces expõem o contrato HTTP. A composição das
dependências ocorre em um único ponto.

```text
HTTP / Dashboard
        |
        v
Interfaces (FastAPI, schemas e rotas)
        |
        v
Application (DTOs, use case e ports)
        |
        +-------------------+
        |                   |
        v                   v
Domain              Infrastructure
                         |       |
                         v       v
                    XGBoost    SQLite
```

As dependências apontam para dentro: infraestrutura implementa os ports da
aplicação, mas domínio e aplicação não importam interfaces ou infraestrutura.

## Estrutura

```text
src/fraud_detection/
├── domain/
│   ├── entities/            Prediction e Transaction
│   ├── services/            FraudDetectionService
│   └── value_objects/       RiskScore
├── application/
│   ├── dto/                 PredictionRequest e PredictionResponse
│   ├── ports/               ModelPredictor e PredictionRepository
│   └── use_cases/           PredictFraudUseCase
├── infrastructure/
│   ├── ml/                  ModelLoader e XGBoostPredictor
│   └── persistence/         SQLitePredictionRepository
├── interfaces/
│   ├── routes/              rota de predição FastAPI
│   └── schemas/             contrato Pydantic HTTP
├── composition/             container de dependências
└── main.py                  aplicação FastAPI
```

Os scripts de exploração, treinamento e avaliação vivem em `experiments/` e
não são dependências do serviço de inferência.

## Fluxo de uma predição

1. `POST /predictions/fraud` recebe `features` validadas pelo schema Pydantic.
2. A rota converte a entrada HTTP em `PredictionRequest` e invoca
   `PredictFraudUseCase`.
3. O caso de uso cria uma `Transaction` e chama o port `ModelPredictor`.
4. `XGBoostPredictor` prepara as features na ordem do metadata e retorna
   somente a probabilidade.
5. `FraudDetectionService` transforma a probabilidade em `RiskScore` e decide
   `is_fraud` de acordo com o threshold.
6. O caso de uso persiste `Transaction` e `Prediction` pelo port
   `PredictionRepository`.
7. `PredictionResponse` projeta o resultado para a resposta HTTP.

```text
Request -> PredictionRequest -> PredictFraudUseCase
                                   |            |
                                   v            v
                          ModelPredictor   FraudDetectionService
                                   |            |
                                   +----> Prediction
                                               |
                                               v
                                    PredictionRepository -> SQLite
```

## Regras e responsabilidades

| Componente | Responsabilidade |
| --- | --- |
| `RiskScore` | Garante que o risco esteja entre 0 e 1. |
| `FraudDetectionService` | Aplica o threshold e produz `Prediction`. |
| `ModelLoader` | Carrega e valida `model.json` e `metadata.json`. |
| `XGBoostPredictor` | Valida/ordena features e retorna probabilidade. Não decide fraude. |
| `SQLitePredictionRepository` | Cria o schema e persiste previsões de forma transacional. |
| `container.py` | Monta e cacheia as dependências da aplicação. |

## Artefatos e persistência

O modelo e o metadata versionados ficam em `artifacts/xgboost/`. O metadata
define a ordem das features e o threshold operacional. O banco SQLite é criado
em `data/predictions.db`; em Docker, esse diretório é montado no volume nomeado
`fraud-data` para preservar registros entre recriações de contêiner.

## Contrato público

O endpoint público é `POST /predictions/fraud`. A entrada contém um objeto
`features` com `Time`, `V1` a `V28` e `Amount`. A resposta contém:

```json
{
  "is_fraud": false,
  "risk_score": 0.12,
  "risk_percentage": 12.0,
  "threshold": 0.66275984
}
```

Campos de feature ausentes, extras ou `Amount` negativo são rejeitados com
HTTP 422. O endpoint raiz (`GET /`) informa o estado do serviço.

## Inicialização

`create_predict_fraud_use_case()` em `composition/container.py` é o Composition
Root. Ele carrega os artefatos, monta `XGBoostPredictor`,
`FraudDetectionService` e `SQLitePredictionRepository`, e injeta tudo em
`PredictFraudUseCase`. O resultado é cacheado, portanto o modelo não é
recarregado em cada requisição.
