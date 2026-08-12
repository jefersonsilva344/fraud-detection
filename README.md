# 🛡️ FraudShield AI — Financial Fraud Detection

<p align="center">

**Machine Learning • Risk Analytics • FastAPI • Docker • Automated Testing**

Sistema completo de detecção de transações fraudulentas desenvolvido em Python, combinando **Machine Learning, experimentação rigorosa, otimização de threshold, análise de risco, API REST, testes automatizados e containerização**.

O projeto foi desenvolvido com foco não apenas no treinamento de modelos, mas também na construção de uma solução de software capaz de levar um modelo de Machine Learning até uma camada de **inferência através de API**.

</p>

---

## 📌 Visão geral

A detecção de fraude financeira é um problema de classificação altamente desbalanceado: a quantidade de transações legítimas é muito superior à quantidade de transações fraudulentas.

Nesse cenário, uma métrica como **Accuracy** pode produzir uma percepção enganosa do desempenho do modelo.

Por isso, este projeto utiliza uma estratégia de avaliação baseada principalmente em:

* Precision
* Recall
* F1-Score
* ROC-AUC
* PR-AUC
* False Positive Rate
* False Negative Rate
* Precision-Recall Curve
* Confusion Matrix
* Feature Importance
* Threshold Optimization

Após a etapa experimental, o modelo selecionado é disponibilizado através de uma **API REST utilizando FastAPI** e pode ser executado em ambiente containerizado com **Docker**.

---

# 🎯 Objetivos

O projeto foi desenvolvido para demonstrar um fluxo completo de Machine Learning aplicado a um problema de negócio.

Principais objetivos:

* Construir um pipeline completo de Machine Learning.
* Realizar preparação e análise dos dados.
* Comparar diferentes estratégias de detecção.
* Avaliar modelos em um cenário de forte desbalanceamento.
* Experimentar abordagens supervisionadas e não supervisionadas.
* Otimizar o threshold de classificação.
* Evitar seleção de modelo baseada apenas em resultados intermediários.
* Avaliar o modelo final exclusivamente no conjunto TEST.
* Disponibilizar o modelo através de uma API REST.
* Implementar testes automatizados.
* Validar a aplicação em ambiente Docker.
* Manter cobertura de testes acima de 80%.

---

# 🧠 Pipeline de Machine Learning

O fluxo experimental foi estruturado da seguinte forma:

```text
                         DATASET
                            │
                            ▼
                  ┌──────────────────┐
                  │ Pré-processamento│
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Análise dos dados│
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Separação dos    │
                  │ conjuntos        │
                  └────────┬─────────┘
                           │
                           ▼
                    EXPERIMENTAÇÃO
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
      Isolation Forest     LOF         XGBoost
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                  Comparação dos modelos
                           │
                           ▼
                   Threshold Optimization
                           │
                           ▼
                  Avaliação rigorosa
                           │
                           ▼
                      TEST SET
                           │
                           ▼
                     Modelo final
                           │
                           ▼
                       Predictor
                           │
                           ▼
                       FastAPI
                           │
                           ▼
                        Docker
```

---

# 🔬 Metodologia de avaliação

Como o dataset apresenta forte desbalanceamento entre as classes, o projeto não utiliza Accuracy como principal critério de avaliação.

| Métrica       | Objetivo                                                    |
| ------------- | ----------------------------------------------------------- |
| **Precision** | Proporção das previsões positivas que realmente são fraude  |
| **Recall**    | Capacidade de identificar fraudes reais                     |
| **F1-Score**  | Equilíbrio entre Precision e Recall                         |
| **ROC-AUC**   | Capacidade discriminativa geral do modelo                   |
| **PR-AUC**    | Desempenho considerando especialmente a classe positiva     |
| **FPR**       | Proporção de transações legítimas classificadas como fraude |
| **FNR**       | Proporção de fraudes não detectadas                         |

## Por que não utilizar apenas Accuracy?

Em um dataset altamente desbalanceado, um modelo pode classificar quase todas as transações como legítimas e ainda apresentar uma Accuracy elevada.

Isso não significa que o modelo seja útil para detectar fraude.

Por esse motivo, o projeto prioriza métricas que permitem analisar diretamente o comportamento da classe minoritária.

---

# 🤖 Modelos avaliados

Foram avaliadas abordagens supervisionadas e não supervisionadas.

## Isolation Forest

Utilizado como abordagem de **detecção de anomalias**.

O algoritmo busca identificar observações que apresentam comportamento significativamente diferente do padrão predominante dos dados.

---

## Local Outlier Factor — LOF

O LOF avalia a densidade local das observações e identifica pontos que apresentam comportamento anômalo em relação aos seus vizinhos.

Foi utilizado como segunda abordagem de detecção de anomalias.

---

## XGBoost

O XGBoost foi utilizado como modelo supervisionado de classificação.

Foram realizados experimentos envolvendo:

* ajuste de hiperparâmetros;
* avaliação de métricas;
* otimização de threshold;
* Feature Importance;
* curvas ROC;
* curvas Precision-Recall;
* comparação com métodos de detecção de anomalias;
* avaliação final no conjunto TEST.

---

# ⚙️ Threshold Optimization

A decisão final do classificador depende de um threshold aplicado à probabilidade estimada pelo modelo.

Em vez de assumir automaticamente o threshold padrão de `0.5`, foram avaliados diferentes pontos de operação.

Foram observados os impactos sobre:

* Precision;
* Recall;
* F1-Score;
* False Positive Rate;
* False Negative Rate;
* PR-AUC.

O threshold representa um trade-off operacional:

```text
Threshold menor
       │
       ├── maior sensibilidade potencial
       ├── maior Recall
       └── possibilidade de mais falsos positivos


Threshold maior
       │
       ├── maior seletividade
       ├── possibilidade de maior Precision
       └── possibilidade de mais fraudes não detectadas
```

### Threshold selecionado

```text
0.66275984
```

O threshold selecionado foi utilizado na avaliação final do modelo.

---

# 🧪 Avaliação rigorosa

Os experimentos foram organizados para evitar selecionar o modelo simplesmente com base em resultados intermediários.

O fluxo final foi:

```text
Treinamento
     │
     ▼
Validação
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Threshold Selection
     │
     ▼
Modelo Final
     │
     ▼
Avaliação no TEST
```

Os resultados apresentados na comparação final pertencem **exclusivamente ao conjunto TEST**.

Isso permite avaliar o comportamento do modelo em dados que não foram utilizados diretamente para a seleção final do modelo.

---

# 📊 Resultados

## 🏆 Comparação final

Os resultados abaixo foram obtidos através do script:

```text
src/comparacao_final.py
```

Todos os valores pertencem exclusivamente ao conjunto **TEST**.

O ranking considera principalmente:

1. PR-AUC
2. Recall
3. F1-Score

| Ranking | Modelo            |  Precision |     Recall |   F1-Score |    ROC-AUC |     PR-AUC |         FPR |        FNR | Detection Rate |
| ------- | ----------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ----------: | ---------: | -------------: |
| 🥇      | **XGBoost Tuned** | **89.06%** | **80.28%** |     84.44% |     96.28% | **82.02%** |     0.0165% | **19.72%** |     **80.28%** |
| 🥈      | XGBoost           |     94.83% |     77.46% | **85.27%** | **96.51%** |     81.48% | **0.0071%** |     22.54% |         77.46% |
| 🥉      | Isolation Forest  |     22.16% |     26.06% |     23.95% |     94.07% |     10.83% |     0.1530% |     73.94% |         26.06% |
| 4       | LOF               |      0.11% |      0.70% |      0.19% |     52.46% |      0.22% |     1.0544% |     99.30% |          0.70% |

> **Nota:** o XGBoost Tuned foi selecionado como modelo final por apresentar o maior PR-AUC e maior Recall entre os modelos avaliados, de acordo com os critérios definidos para este projeto.

---

# 🥇 Modelo final — XGBoost Tuned

O modelo selecionado foi o **XGBoost Tuned**.

### Threshold

```text
0.66275984
```

### Métricas no conjunto TEST

| Métrica                  |   Resultado |
| ------------------------ | ----------: |
| **Precision**            |  **89.06%** |
| **Recall**               |  **80.28%** |
| **F1-Score**             |  **84.44%** |
| **ROC-AUC**              |  **96.28%** |
| **PR-AUC**               |  **82.02%** |
| **False Positive Rate**  | **0.0165%** |
| **False Negative Rate**  |  **19.72%** |
| **Fraud Detection Rate** |  **80.28%** |

---

# 🔎 Matriz de confusão

O modelo final apresentou:

|                   | Predito legítimo | Predito fraude |
| ----------------- | ---------------: | -------------: |
| **Real legítimo** |           42.481 |              7 |
| **Real fraude**   |               14 |             57 |

### Interpretação

No conjunto TEST:

* **42.481** transações legítimas foram classificadas corretamente.
* **7** transações legítimas foram classificadas incorretamente como fraude.
* **57** fraudes foram detectadas corretamente.
* **14** fraudes não foram detectadas.

Assim:

```text
57 / 71 = 80.28%
```

O modelo detectou **80.28% das fraudes presentes no conjunto TEST**.

Ao mesmo tempo, apresentou apenas:

```text
7 falsos positivos
```

entre as transações legítimas avaliadas.

---

# ⚖️ XGBoost vs. XGBoost Tuned

A comparação entre os modelos supervisionados demonstra um trade-off importante.

| Métrica        |     XGBoost | XGBoost Tuned |
| -------------- | ----------: | ------------: |
| Precision      |  **94.83%** |        89.06% |
| Recall         |      77.46% |    **80.28%** |
| F1-Score       |  **85.27%** |        84.44% |
| ROC-AUC        |  **96.51%** |        96.28% |
| PR-AUC         |      81.48% |    **82.02%** |
| FPR            | **0.0071%** |       0.0165% |
| FNR            |      22.54% |    **19.72%** |
| Detection Rate |      77.46% |    **80.28%** |

### XGBoost tradicional

Apresentou:

* maior Precision;
* maior F1-Score;
* maior ROC-AUC;
* menor False Positive Rate.

### XGBoost Tuned

Apresentou:

* maior PR-AUC;
* maior Recall;
* maior taxa de detecção;
* menor False Negative Rate.

O modelo Tuned foi selecionado porque o projeto prioriza a capacidade de identificar a classe fraudulenta.

Esse resultado demonstra um trade-off real:

> **Maior capacidade de detecção → maior exposição a falsos positivos.**

---

# 📈 Evidências experimentais

Os artefatos gerados durante os experimentos estão disponíveis em:

```text
results/
```

Entre eles:

* comparação de F1-Score;
* comparação de Recall;
* comparação de PR-AUC;
* comparação de False Positive Rate;
* matriz de confusão;
* Feature Importance;
* curva ROC;
* curva Precision-Recall;
* análise de threshold.

A curva **Precision-Recall** possui importância especial neste projeto devido ao forte desbalanceamento entre as classes.

---

# 💡 Principais conclusões

Os experimentos demonstraram diferenças significativas entre as abordagens avaliadas.

## Detecção de anomalias

O LOF apresentou desempenho particularmente baixo:

```text
PR-AUC:        0.0022
Recall:        0.70%
F1-Score:      0.19%
Detection Rate: 0.70%
```

O Isolation Forest apresentou desempenho superior ao LOF:

```text
PR-AUC:        0.1083
Recall:        26.06%
F1-Score:      23.95%
Detection Rate: 26.06%
```

Entretanto, ambos ficaram significativamente abaixo das abordagens supervisionadas.

## XGBoost Tuned

O modelo final alcançou:

```text
PR-AUC:             82.02%
Recall:             80.28%
F1-Score:           84.44%
Precision:          89.06%
ROC-AUC:            96.28%
False Positive Rate: 0.0165%
```

Além disso:

```text
57 / 71 fraudes detectadas
```

representando uma taxa de detecção de:

```text
80.28%
```

---

# 🏗️ Arquitetura da aplicação

O projeto separa as responsabilidades de **treinamento, inferência e exposição da API**.

```text
                    ┌──────────────────┐
                    │      Cliente     │
                    └────────┬─────────┘
                             │ HTTP
                             ▼
                    ┌──────────────────┐
                    │     FastAPI      │
                    │     REST API     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Prediction Route │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Predictor     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Model Loader   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Machine Learning │
                    │      Model       │
                    └──────────────────┘
```

Essa separação permite que o modelo treinado seja utilizado pela API sem executar novamente o pipeline completo de treinamento.

---

# 📁 Estrutura do projeto

```text
fraud-detection/
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
│
├── results/
│   ├── final_comparison/
│   ├── rigorous/
│   ├── threshold/
│   ├── xgboost/
│   ├── xgboost_final/
│   └── xgboost_tuning/
│
├── scripts/
│   └── test_docker.ps1
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── routes/
│   │       └── prediction.py
│   │
│   ├── inference/
│   │   ├── model_loader.py
│   │   ├── predictor.py
│   │   └── schemas.py
│   │
│   ├── training/
│   │   ├── xgboost_final.py
│   │   └── xgboost_tuning.py
│   │
│   ├── analise_exploratoria.py
│   ├── avaliacao.py
│   ├── avaliacao_rigorosa.py
│   ├── coleta_dados.py
│   ├── comparacao.py
│   ├── comparacao_final.py
│   ├── experimentos.py
│   ├── lof.py
│   ├── modelo_anomalias.py
│   ├── preprocessamento.py
│   ├── threshold_isolation_forest.py
│   ├── treinamento.py
│   ├── xgboost_model.py
│   └── xgboost_tuning.py
│
└── tests/
    ├── docker/
    │   └── test_docker_api.py
    │
    ├── integration/
    │   └── test_api.py
    │
    └── unit/
        ├── test_model_loader.py
        └── test_predictor.py
```

---

# 🧩 Principais componentes

| Componente                      | Responsabilidade          |
| ------------------------------- | ------------------------- |
| `preprocessamento.py`           | Preparação dos dados      |
| `analise_exploratoria.py`       | Análise exploratória      |
| `treinamento.py`                | Treinamento dos modelos   |
| `modelo_anomalias.py`           | Modelagem de anomalias    |
| `lof.py`                        | Experimentos com LOF      |
| `xgboost_model.py`              | Modelo XGBoost            |
| `xgboost_tuning.py`             | Ajuste de hiperparâmetros |
| `threshold_isolation_forest.py` | Análise de threshold      |
| `avaliacao.py`                  | Avaliação dos modelos     |
| `avaliacao_rigorosa.py`         | Avaliação rigorosa        |
| `comparacao.py`                 | Comparação experimental   |
| `comparacao_final.py`           | Comparação final          |
| `model_loader.py`               | Carregamento do modelo    |
| `predictor.py`                  | Execução da inferência    |
| `prediction.py`                 | Endpoint de previsão      |
| `main.py`                       | Inicialização da API      |

---

# 🚀 Como executar

## Pré-requisitos

* Python 3.12+
* Git
* Docker Desktop — opcional

---

## 1. Clone o repositório

```bash
git clone https://github.com/jefersonsilva344/fraud-detection.git

cd fraud-detection
```

---

## 2. Crie o ambiente virtual

### Windows

```powershell
python -m venv venv
```

Ative:

```powershell
.\venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv venv
```

Ative:

```bash
source venv/bin/activate
```

---

## 3. Instale as dependências

```bash
pip install -r requirements.txt
```

Para desenvolvimento e testes:

```bash
pip install -r requirements-dev.txt
```

---

# 🌐 Executando a API

Inicie a aplicação:

```bash
uvicorn src.api.main:app --reload
```

A API estará disponível em:

```text
http://localhost:8000
```

## Swagger UI

Documentação interativa:

```text
http://localhost:8000/docs
```

## OpenAPI

Schema da API:

```text
http://localhost:8000/openapi.json
```

---

# 🐳 Docker

A aplicação pode ser executada em ambiente containerizado.

## Construir a imagem

```bash
docker build -t fraud-detection .
```

## Executar o container

```bash
docker run -p 8000:8000 fraud-detection
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

# 🐳 Docker Compose

Também é possível utilizar Docker Compose:

```bash
docker compose up --build
```

Para executar em segundo plano:

```bash
docker compose up --build -d
```

Para interromper:

```bash
docker compose down
```

---

# 🧪 Testes automatizados

O projeto utiliza **Pytest** para validar a aplicação.

Execute:

```bash
python -m pytest
```

### Resultado atual

```text
25 passed
```

---

# 📊 Cobertura de testes

Para executar os testes com cobertura:

```bash
python -m pytest --cov --cov-report=term-missing
```

Resultado atual:

```text
===================================== tests coverage =====================================

Name                           Stmts   Miss Branch BrPart  Cover
----------------------------------------------------------------
src\api\main.py                   10      1      0      0    90%
src\api\routes\prediction.py      18      4      0      0    78%
----------------------------------------------------------------
TOTAL                             28      5      0      0    82%

Required test coverage of 80.0% reached.
Total coverage: 82.14%

25 passed
```

Também é possível validar diretamente o limite mínimo:

```bash
python -m coverage report --fail-under=80
```

---

# 🧪 Status dos testes

| Indicador         |  Resultado |
| ----------------- | ---------: |
| Testes executados |         25 |
| Testes aprovados  |         25 |
| Testes falhos     |          0 |
| Cobertura         | **82.14%** |
| Cobertura mínima  |        80% |
| Status            |     ✅ PASS |

---

# 🧩 Categorias de testes

## Unitários

Localizados em:

```text
tests/unit/
```

Incluem:

```text
test_model_loader.py
test_predictor.py
```

Validam componentes individuais da camada de inferência.

## Integração

Localizados em:

```text
tests/integration/
```

Incluem:

```text
test_api.py
```

Validam a integração da aplicação e seus endpoints.

## Docker

Localizados em:

```text
tests/docker/
```

Incluem:

```text
test_docker_api.py
```

Validam o comportamento da aplicação em ambiente containerizado.

---

# 📦 Artefatos experimentais

O diretório `results/` contém os artefatos gerados durante os experimentos.

```text
results/
│
├── final_comparison/
│   ├── final_comparison_report.txt
│   ├── final_model_comparison.csv
│   ├── final_f1_comparison.png
│   ├── final_fpr_comparison.png
│   ├── final_pr_auc_comparison.png
│   └── final_recall_comparison.png
│
├── rigorous/
│   ├── rigorous_evaluation_report.txt
│   ├── rigorous_metrics.csv
│   ├── validation_results.csv
│   ├── validation_test_comparison.csv
│   ├── precision_recall_test.png
│   └── validation_threshold_f1.png
│
├── threshold/
│   ├── optimized_metrics.csv
│   ├── precision_recall_curve.png
│   ├── threshold_analysis.csv
│   ├── threshold_f1_curve.png
│   └── threshold_report.txt
│
├── xgboost/
│   ├── xgboost_confusion_matrix.png
│   ├── xgboost_evaluation_report.txt
│   ├── xgboost_feature_importance.csv
│   ├── xgboost_feature_importance.png
│   ├── xgboost_metrics.csv
│   ├── xgboost_precision_recall.png
│   ├── xgboost_roc_curve.png
│   └── xgboost_threshold_f1.png
│
├── xgboost_final/
│   ├── xgboost_final_confusion_matrix.png
│   ├── xgboost_final_evaluation_report.txt
│   ├── xgboost_final_feature_importance.csv
│   ├── xgboost_final_feature_importance.png
│   ├── xgboost_final_metrics.csv
│   ├── xgboost_final_precision_recall.png
│   └── xgboost_final_roc_curve.png
│
└── xgboost_tuning/
    ├── tuning_validation_results.csv
    ├── xgboost_tuned_evaluation_report.txt
    ├── xgboost_tuned_feature_importance.csv
    ├── xgboost_tuned_feature_importance.png
    ├── xgboost_tuned_metrics.csv
    └── xgboost_tuned_threshold_f1.png
```

---

# 🧪 Reprodutibilidade

Os experimentos foram separados em scripts específicos.

### Pré-processamento

```bash
python src/preprocessamento.py
```

### Treinamento

```bash
python src/treinamento.py
```

### Avaliação

```bash
python src/avaliacao.py
```

### Comparação dos modelos

```bash
python src/comparacao.py
```

### Avaliação rigorosa

```bash
python src/avaliacao_rigorosa.py
```

### Comparação final

```bash
python src/comparacao_final.py
```

Os scripts relacionados ao XGBoost também podem ser executados individualmente conforme a etapa experimental.

---

# 📌 Decisões técnicas

## Por que utilizar múltiplos modelos?

A comparação entre algoritmos permite avaliar diferentes estratégias:

```text
Detecção de anomalias
        +
Densidade local
        +
Classificação supervisionada
```

Isso permite verificar empiricamente qual abordagem apresenta melhor desempenho no conjunto de dados utilizado.

---

## Por que utilizar XGBoost?

O XGBoost foi utilizado como abordagem supervisionada devido à sua capacidade de modelar relações não lineares e apresentar bom desempenho em problemas tabulares de classificação.

---

## Por que otimizar o threshold?

O threshold padrão de classificação nem sempre representa o melhor ponto operacional para um problema de fraude.

A alteração do threshold permite controlar o equilíbrio entre:

```text
Recall
  ↕
Maior capacidade de detectar fraudes

Precision / FPR
  ↕
Maior controle sobre falsos positivos
```

---

## Por que utilizar PR-AUC?

Em problemas fortemente desbalanceados, ROC-AUC pode fornecer uma percepção otimista do desempenho.

A **PR-AUC** concentra-se na relação entre Precision e Recall, oferecendo uma visão especialmente relevante para avaliar a classe positiva.

---

# 🔐 Boas práticas

O projeto utiliza `.gitignore` para evitar o versionamento de arquivos sensíveis ou gerados localmente.

Entre os arquivos protegidos estão:

```text
.env
venv/
.pytest_cache/
.coverage
data/raw/
data/processed/
models/
*.pkl
*.pickle
*.joblib
*.onnx
```

Os artefatos presentes em `results/` são mantidos no repositório porque fazem parte da documentação e evidência dos experimentos.

---

# 🧰 Stack tecnológica

| Categoria        | Tecnologias            |
| ---------------- | ---------------------- |
| Linguagem        | Python                 |
| Machine Learning | Scikit-learn, XGBoost  |
| Data             | Pandas, NumPy          |
| API              | FastAPI                |
| Validação        | Pydantic               |
| Servidor         | Uvicorn                |
| Testes           | Pytest, pytest-cov     |
| Containerização  | Docker, Docker Compose |
| Versionamento    | Git, GitHub            |

---

# 🚧 Limitações

Apesar dos resultados obtidos, existem limitações importantes para uma utilização em ambiente real de produção.

Atualmente o projeto não implementa:

* monitoramento contínuo de data drift;
* monitoramento de model drift;
* gerenciamento formal de versões de modelos;
* pipeline completo de CI/CD;
* observabilidade de produção;
* autenticação da API;
* autorização baseada em usuários;
* monitoramento de latência;
* testes de carga;
* pipeline automatizado de retreinamento;
* infraestrutura cloud.

Além disso, resultados obtidos em um dataset histórico **não garantem o mesmo desempenho em um ambiente de produção real**.

---

# 🔮 Roadmap

* [ ] Implementar CI/CD completo.
* [ ] Adicionar GitHub Actions para testes automáticos.
* [ ] Implementar versionamento de modelos.
* [ ] Integrar MLflow.
* [ ] Adicionar monitoramento de data drift.
* [ ] Adicionar monitoramento de model drift.
* [ ] Implementar logging estruturado.
* [ ] Adicionar métricas de observabilidade da API.
* [ ] Implementar autenticação e autorização.
* [ ] Criar testes de carga.
* [ ] Criar pipeline automatizado de treinamento.
* [ ] Implementar deploy em cloud.
* [ ] Implementar monitoramento do modelo em produção.

---

# 📊 Status do projeto

| Etapa                  | Status      |
| ---------------------- | ----------- |
| Machine Learning       | ✅ Concluído |
| Experimentação         | ✅ Concluído |
| Comparação de modelos  | ✅ Concluído |
| Threshold Optimization | ✅ Concluído |
| Avaliação rigorosa     | ✅ Concluído |
| Modelo final           | ✅ Concluído |
| API REST               | ✅ Concluído |
| Docker                 | ✅ Concluído |
| Testes automatizados   | ✅ Concluído |
| Cobertura > 80%        | ✅ Concluído |
| CI/CD                  | ⚪ Planejado |
| Model Monitoring       | ⚪ Planejado |
| Cloud Deploy           | ⚪ Planejado |

---

# 👨‍💻 Autor

**Jeferson Alves da Silva**

Python Developer com foco em **Machine Learning, Inteligência Artificial, Dados e Engenharia de Software**.

Este projeto demonstra, na prática, a integração entre:

```text
Machine Learning
      +
Experimentação
      +
Avaliação
      +
Threshold Optimization
      +
Model Selection
      +
Inference
      +
REST API
      +
Automated Testing
      +
Docker
```

---

# 🔗 Links

* GitHub: https://github.com/jefersonsilva344
* LinkedIn: https://www.linkedin.com/in/jeferson-silva-30136a422/

---

## ⭐ Sobre o projeto

O objetivo deste projeto não foi apenas treinar um classificador.

A proposta foi construir um fluxo completo:

```text
Dados
  ↓
Pré-processamento
  ↓
Experimentação
  ↓
Avaliação
  ↓
Hyperparameter Tuning
  ↓
Threshold Optimization
  ↓
Validação
  ↓
Modelo Final
  ↓
Inferência
  ↓
REST API
  ↓
Docker
  ↓
Testes
```

O resultado é uma aplicação que conecta **Machine Learning e Engenharia de Software**, demonstrando desde a experimentação e avaliação do modelo até sua disponibilização para inferência através de uma API.

⭐ Se este projeto foi útil ou interessante, considere deixar uma estrela no repositório.
