# 🛡️ FraudShield AI

> **Financial Fraud Detection with Machine Learning, Explainable AI and REST API**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-orange)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-59%20passed-brightgreen)](#-testes)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Sistema de detecção de transações fraudulentas desenvolvido em **Python**, combinando **Machine Learning aplicado, arquitetura em camadas, API REST, Explainable AI, persistência SQLite, Docker e testes automatizados**.

O projeto demonstra, de ponta a ponta, como transformar um modelo de Machine Learning em uma aplicação backend estruturada, separando **predição, regras de negócio, infraestrutura e exposição via API**.

---

## 📊 Resultados

O modelo final utiliza **XGBoost** com threshold otimizado, avaliado em um conjunto de teste separado.

| Modelo            |  Precision |     Recall |   F1-Score |    ROC-AUC |     PR-AUC |
| ----------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| LOF               |      0,11% |      0,70% |      0,19% |     52,46% |     0,0022 |
| Isolation Forest  |     22,16% |     26,06% |     23,95% |     94,07% |          — |
| XGBoost           |     94,83% |     77,46% |     85,27% |     96,51% |     81,48% |
| **XGBoost Tuned** | **89,06%** | **80,28%** | **84,44%** | **96,28%** | **82,02%** |

### XGBoost Tuned

```text
Precision        89,06%
Recall           80,28%
F1-Score         84,44%
ROC-AUC          96,28%
PR-AUC           82,02%
False Positive   0,0165%
False Negative   19,72%
Threshold        0,66275984
```

### Matriz de confusão

```text
                     Predição
                  Legítima  Fraude
Real legítima       42.481       7
Real fraude             14      57
```

O modelo identificou **57 de 71 transações fraudulentas** no conjunto de teste, correspondendo a um **Recall de 80,28%**.

A otimização do threshold aumentou o Recall de **77,46% para 80,28%**, reduzindo o False Negative Rate de **22,54% para 19,72%**.

> ⚠️ As métricas acima representam uma avaliação offline durante o desenvolvimento. Elas não garantem o mesmo desempenho em dados reais ou em produção.

Detalhes sobre metodologia, artefatos, threshold e limitações estão em [`docs/model.md`](docs/model.md).

---

## 🎥 Demonstração

O FraudShield AI expõe o modelo através de uma API REST construída com FastAPI.

```text
Cliente
   │
   │ POST /predictions/fraud
   ▼
FastAPI
   │
   ▼
PredictionRequest
   │
   ▼
PredictFraudUseCase
   │
   ▼
XGBoost Predictor
   │
   │ probabilidade
   ▼
FraudDetectionService
   │
   │ decisão de domínio
   ▼
Prediction + RiskScore
   │
   ├──────────────► SQLite
   │
   ▼
PredictionResponse
   │
   ▼
JSON
```

### Endpoint principal

```http
POST /predictions/fraud
```

Exemplo de requisição:

```json
{
  "features": {
    "Time": 100.0,
    "V1": -1.359807,
    "V2": -0.072782,
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
    "V16": -0.470401,
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

Exemplo de resposta:

```json
{
  "is_fraud": false,
  "risk_score": 0.12,
  "risk_percentage": 12.0,
  "threshold": 0.66275984
}
```

A API também disponibiliza documentação interativa através do Swagger/OpenAPI em:

```text
http://localhost:8000/docs
```

A especificação completa do contrato HTTP está em [`docs/api.md`](docs/api.md).

---

## 📌 Sobre o projeto

O **FraudShield AI** foi desenvolvido para demonstrar como integrar **Machine Learning e engenharia de software backend** em uma aplicação estruturada.

O projeto não se limita ao treinamento do modelo. A solução inclui:

* Machine Learning supervisionado e não supervisionado;
* comparação entre XGBoost, Isolation Forest e LOF;
* otimização de threshold;
* avaliação com métricas adequadas para dados desbalanceados;
* arquitetura em camadas;
* Domain-Driven Design aplicado de forma pragmática;
* API REST com FastAPI;
* persistência de predições em SQLite;
* Docker e Docker Compose;
* testes unitários, de integração e de Docker;
* validação de contratos;
* Explainable AI;
* documentação técnica.

---

## 🎯 Objetivo

O objetivo é identificar transações com maior probabilidade de fraude e disponibilizar essa capacidade através de uma API REST.

A solução separa explicitamente duas responsabilidades:

```text
XGBoost
   │
   └──► produz probabilidade
              │
              ▼
FraudDetectionService
   │
   └──► aplica threshold e decide risco
```

Essa separação evita acoplar a regra de decisão ao adaptador do modelo.

**O modelo informa a probabilidade; o domínio decide como essa probabilidade deve ser interpretada.**

---

## 🏗️ Arquitetura

O projeto utiliza uma arquitetura em camadas, com separação entre regras de negócio, casos de uso, infraestrutura e interfaces externas.

```text
src/fraud_detection/
│
├── application/
│   ├── dto/
│   ├── ports/
│   └── use_cases/
│
├── domain/
│   ├── entities/
│   ├── services/
│   └── value_objects/
│
├── infrastructure/
│   ├── ml/
│   └── persistence/
│
├── interfaces/
│   ├── routes/
│   └── schemas/
│
├── composition/
│   └── container.py
│
└── main.py
```

### Responsabilidades

| Camada           | Responsabilidade                            |
| ---------------- | ------------------------------------------- |
| `domain`         | Regras e conceitos centrais do negócio      |
| `application`    | Casos de uso, DTOs e contratos da aplicação |
| `infrastructure` | XGBoost, SQLite e detalhes técnicos         |
| `interfaces`     | HTTP, FastAPI e schemas                     |
| `composition`    | Montagem das dependências                   |
| `main.py`        | Inicialização da aplicação                  |

A arquitetura detalhada está em [`docs/architecture.md`](docs/architecture.md).

---

## 🔎 Explainable AI

Além da probabilidade de fraude, o sistema possui explicações locais baseadas na contribuição das features para a predição do XGBoost.

O objetivo é fornecer não apenas:

```text
Fraud probability: 0.87
```

mas também informações sobre quais características tiveram maior influência na decisão.

Conceitualmente:

```text
Feature      Contribution       Direction
------------------------------------------------
V14          +0.82              increases_fraud_risk
V10          +0.51              increases_fraud_risk
V4           -0.27              decreases_fraud_risk
```

Isso permite investigar **quais características contribuíram para o aumento ou redução do risco** de uma transação específica.

---

## 🤖 Abordagens avaliadas

O dataset possui forte desbalanceamento entre transações legítimas e fraudulentas. Por isso, accuracy isoladamente não é uma métrica adequada.

Foram avaliadas três abordagens:

### Local Outlier Factor

Detecção de anomalias baseada na densidade local dos dados.

### Isolation Forest

Algoritmo de detecção de anomalias baseado no isolamento de observações.

### XGBoost

Modelo supervisionado de classificação utilizado como solução final.

O XGBoost apresentou o melhor equilíbrio entre **Precision, Recall, F1-Score e PR-AUC**, sendo selecionado para a API.

---

## 🎚️ Threshold de decisão

O modelo produz uma probabilidade entre `0` e `1`.

O threshold final foi definido separadamente da etapa de treinamento:

```text
Threshold = 0,66275984

probabilidade >= 0,66275984
        │
        ├──► fraude
        │
        └──► não fraude
```

Essa separação permite ajustar a política de decisão sem modificar o modelo treinado.

---

## 🚀 Executando o projeto

### Pré-requisitos

* Python 3.11+
* pip

### Instalação

```bash
git clone https://github.com/jefersonsilva344/fraud-detection.git

cd fraud-detection

python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
source .venv/bin/activate
```

### Instalar dependências

```bash
pip install -e .
```

### Executar a API

```bash
uvicorn fraud_detection.main:app --reload
```

A API estará disponível em:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

---

## 🐳 Docker

O projeto também pode ser executado através do Docker Compose.

```bash
docker compose up --build
```

Verificar os containers:

```bash
docker compose ps
```

Encerrar:

```bash
docker compose down
```

A API ficará disponível em:

```text
http://localhost:8000
```

O Docker Compose possui healthcheck configurado para a API e volume persistente para o banco SQLite.

---

## 🧪 Testes

A suíte de testes está organizada em:

```text
tests/
├── unit/
├── integration/
└── docker/
```

Executar todos os testes:

```bash
python -m pytest -v
```

Resultado atual:

```text
59 passed
1 warning
```

Os testes cobrem:

* regras de domínio;
* `RiskScore`;
* DTOs;
* casos de uso;
* contratos;
* predictor XGBoost;
* carregamento dos artefatos;
* validação de metadata;
* persistência SQLite;
* API;
* validações HTTP;
* integração;
* execução via Docker;
* wiring do Composition Root.

> O único warning atual é externo ao código de negócio e está relacionado à depreciação do `TestClient`/Starlette.

---

## 📁 Estrutura do projeto

```text
fraud-detection/
│
├── data/
│
├── docs/
│   ├── api.md
│   ├── architecture.md
│   └── model.md
│
├── experiments/
│
├── src/
│   └── fraud_detection/
│       ├── application/
│       ├── composition/
│       ├── domain/
│       ├── infrastructure/
│       ├── interfaces/
│       └── main.py
│
├── tests/
│   ├── docker/
│   ├── integration/
│   └── unit/
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 📚 Documentação

A documentação detalhada foi separada do README para manter o projeto organizado.

| Documento                                      | Conteúdo                                             |
| ---------------------------------------------- | ---------------------------------------------------- |
| [`docs/architecture.md`](docs/architecture.md) | Arquitetura, camadas, responsabilidades e fluxo      |
| [`docs/api.md`](docs/api.md)                   | Endpoints, contratos HTTP, validações e exemplos     |
| [`docs/model.md`](docs/model.md)               | Modelos, artefatos, threshold, métricas e limitações |

---

## 🛠️ Tecnologias

### Backend

* Python
* FastAPI
* Pydantic
* Uvicorn

### Machine Learning

* XGBoost
* Scikit-learn
* Pandas
* NumPy

### Persistência

* SQLite

### Infraestrutura

* Docker
* Docker Compose

### Qualidade

* Pytest
* Testes unitários
* Testes de integração
* Testes de Docker

---

## 🔐 Princípios técnicos

O projeto foi estruturado priorizando:

* Separation of Concerns;
* baixo acoplamento;
* Dependency Inversion;
* contratos explícitos;
* domínio independente de infraestrutura;
* validação em múltiplas camadas;
* DTOs imutáveis;
* persistência transacional;
* composição centralizada de dependências;
* testabilidade;
* separação entre predição e decisão.

Um dos principais princípios arquiteturais é:

```text
Model Predictor
      │
      │ probability
      ▼
Application
      │
      ▼
Domain Service
      │
      │ decision
      ▼
Prediction
```

O predictor não decide se uma transação é fraudulenta. Ele apenas fornece a probabilidade produzida pelo modelo.

---

## ⚠️ Limitações

Este projeto é uma implementação de demonstração e **não deve ser considerado um sistema antifraude pronto para produção financeira**.

Principais limitações:

* dataset histórico;
* possível mudança na distribuição dos dados;
* ausência de monitoramento de model drift;
* ausência de retreinamento automático;
* SQLite como mecanismo de persistência;
* ausência de autenticação e autorização;
* ausência de rate limiting;
* observabilidade limitada;
* métricas obtidas em ambiente offline.

Essas limitações são intencionais no contexto do projeto e representam possíveis etapas de evolução.

---

## 🔮 Próximos passos

* autenticação e autorização da API;
* migração para PostgreSQL;
* observabilidade e métricas de produção;
* monitoramento de drift;
* pipeline de retreinamento;
* versionamento de modelos;
* CI/CD;
* testes de carga;
* deployment em cloud;
* dashboard de risco;
* monitoramento contínuo da performance do modelo.

---

## 📄 Licença

Este projeto está sob a licença MIT.

Consulte [`LICENSE`](LICENSE) para mais detalhes.

---

## 👨‍💻 Sobre o projeto

**FraudShield AI** é um projeto de portfólio desenvolvido com foco em:

**Python Backend + Machine Learning aplicado + Engenharia de Software + IA aplicada.**

O objetivo é demonstrar não apenas a construção de um modelo preditivo, mas sua integração em uma aplicação backend com **arquitetura estruturada, API REST, persistência, testes automatizados, Docker e documentação técnica**.

⭐ Se o projeto foi útil ou interessante, considere deixar uma estrela no repositório.
