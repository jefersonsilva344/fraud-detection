# 🛡️ Fraud Detection — Machine Learning API

[![CI](https://github.com/jefersonsilva344/fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/jefersonsilva344/fraud-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.14-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST-009688)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![Coverage](https://img.shields.io/badge/Coverage-82.14%25-brightgreen)

Sistema de detecção de transações fraudulentas desenvolvido em Python, combinando **Machine Learning, experimentação rigorosa, otimização de threshold, API REST, testes automatizados e containerização com Docker**.

O projeto foi desenvolvido com foco não apenas no treinamento de modelos, mas também em **engenharia de software e disponibilização do modelo para inferência através de uma API**.

---

## 📌 Sobre o projeto

A detecção de fraude é um problema de classificação altamente desbalanceado, no qual a quantidade de transações legítimas é muito superior à quantidade de transações fraudulentas.

Nesse cenário, métricas como **Accuracy** podem fornecer uma visão enganosa do desempenho do modelo.

Por isso, o projeto utiliza uma abordagem baseada em múltiplas métricas e etapas de validação, incluindo:

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- False Positive Rate
- False Negative Rate
- Curva Precision-Recall
- Otimização de threshold
- Avaliação em conjunto TEST
- Comparação entre diferentes algoritmos
- Avaliação de matriz de confusão
- Feature Importance

Após a etapa experimental, o modelo selecionado é disponibilizado através de uma **API REST utilizando FastAPI**.

A aplicação também pode ser executada em ambiente containerizado utilizando **Docker**.

---

# 🎯 Objetivos

Os principais objetivos do projeto são:

- Construir um pipeline completo de Machine Learning para detecção de fraude.
- Realizar análise e pré-processamento dos dados.
- Avaliar diferentes abordagens de detecção.
- Comparar modelos utilizando métricas adequadas para dados desbalanceados.
- Avaliar modelos supervisionados e não supervisionados.
- Otimizar o threshold de classificação.
- Realizar avaliação rigorosa utilizando dados de teste.
- Selecionar um modelo final baseado em critérios definidos.
- Disponibilizar o modelo através de uma API REST.
- Criar testes unitários, de integração e Docker.
- Containerizar a aplicação.
- Manter cobertura de testes superior a 80%.

---

# 🧠 Pipeline de Machine Learning

O pipeline experimental segue o seguinte fluxo:

```text
                    DATASET
                       │
                       ▼
              Pré-processamento
                       │
                       ▼
              Análise exploratória
                       │
                       ▼
              Separação dos dados
                       │
                       ▼
              Experimentação
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
        Isolation     LOF      XGBoost
         Forest
             │         │         │
             └─────────┼─────────┘
                       ▼
              Comparação dos modelos
                       │
                       ▼
              Otimização de threshold
                       │
                       ▼
              Validação rigorosa
                       │
                       ▼
                 Conjunto TEST
                       │
                       ▼
                Modelo final
                       │
                       ▼
                  API REST
                       │
                       ▼
                    Docker
🔬 Metodologia de avaliação

Como o problema apresenta forte desbalanceamento entre as classes, a avaliação não é baseada apenas em Accuracy.

Foram utilizadas as seguintes métricas:

Métrica	Objetivo
Precision	Mede a proporção de previsões positivas que realmente são fraude
Recall	Mede a capacidade de identificar fraudes reais
F1-Score	Equilibra Precision e Recall
ROC-AUC	Mede a capacidade discriminativa geral
PR-AUC	Avalia o desempenho considerando especialmente a classe positiva em problemas desbalanceados
FPR	Mede a proporção de transações legítimas classificadas incorretamente como fraude
FNR	Mede a proporção de fraudes que não foram detectadas
Por que não utilizar apenas Accuracy?

Em um dataset altamente desbalanceado, um modelo poderia classificar quase todas as transações como legítimas e ainda obter uma Accuracy elevada.

Por exemplo, se apenas uma pequena parcela das transações for fraudulenta, um modelo que classifique praticamente tudo como legítimo poderá apresentar uma Accuracy aparentemente excelente, mesmo sendo inútil para detectar fraude.

Por isso, o projeto prioriza métricas capazes de avaliar diretamente o comportamento sobre a classe minoritária.

🤖 Modelos avaliados

Foram avaliadas abordagens supervisionadas e não supervisionadas.

Isolation Forest

Utilizado como abordagem de detecção de anomalias.

O algoritmo busca identificar observações que apresentam comportamento significativamente diferente do padrão predominante dos dados.

Local Outlier Factor — LOF

O LOF avalia a densidade local das observações e identifica pontos que apresentam comportamento anômalo em relação aos seus vizinhos.

Foi utilizado como segunda abordagem de detecção de anomalias.

XGBoost

O XGBoost foi utilizado como modelo supervisionado de classificação.

Além do modelo base, foram realizados experimentos envolvendo:

ajuste de hiperparâmetros;
avaliação de métricas;
análise de threshold;
feature importance;
curvas ROC;
curvas Precision-Recall;
comparação com métodos de detecção de anomalias;
avaliação final no conjunto TEST.
⚙️ Otimização de Threshold

A decisão final do classificador depende de um threshold aplicado à probabilidade estimada pelo modelo.

Em vez de assumir automaticamente o threshold padrão de 0.5, o projeto avaliou diferentes pontos de operação.

Foram observados os impactos sobre:

Precision;
Recall;
F1-Score;
False Positive Rate;
False Negative Rate;
PR-AUC.

O threshold representa um trade-off operacional:

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

Para o modelo final, o threshold selecionado foi:

0.66275984
🧪 Avaliação rigorosa

Os experimentos foram organizados para evitar selecionar o modelo simplesmente com base em resultados intermediários.

O fluxo final considera:

Treinamento
     │
     ▼
Validação
     │
     ▼
Tuning
     │
     ▼
Seleção de Threshold
     │
     ▼
Modelo Final
     │
     ▼
Avaliação no TEST

Os resultados apresentados na comparação final pertencem exclusivamente ao conjunto TEST.

Isso permite avaliar o comportamento do modelo em dados que não foram utilizados diretamente para a seleção final do modelo.

📊 Resultados
🏆 Comparação final dos modelos

Os resultados abaixo foram obtidos através do script:

src/comparacao_final.py

Todos os valores pertencem exclusivamente ao conjunto TEST.

O ranking foi definido priorizando:

PR-AUC
F1-Score
Recall
Ranking	Modelo	Precision	Recall	F1-Score	ROC-AUC	PR-AUC	FPR	FNR	Detection Rate
🥇	XGBoost Tuned	89.06%	80.28%	84.44%	96.28%	82.02%	0.0165%	19.72%	80.28%
🥈	XGBoost	94.83%	77.46%	85.27%	96.51%	81.48%	0.0071%	22.54%	77.46%
🥉	Isolation Forest	22.16%	26.06%	23.95%	94.07%	10.83%	0.1530%	73.94%	26.06%
4	LOF	0.11%	0.70%	0.19%	52.46%	0.22%	1.0544%	99.30%	0.70%

O XGBoost Tuned foi selecionado como modelo final por apresentar o maior PR-AUC e maior Recall entre os modelos avaliados, seguindo os critérios definidos para este projeto.

🥇 Modelo final — XGBoost Tuned

O modelo selecionado foi o XGBoost Tuned.

Threshold
0.66275984
Métricas no conjunto TEST
Métrica	Resultado
Precision	89.06%
Recall	80.28%
F1-Score	84.44%
ROC-AUC	96.28%
PR-AUC	82.02%
False Positive Rate	0.0165%
False Negative Rate	19.72%
Fraud Detection Rate	80.28%
🔎 Matriz de confusão

O modelo final apresentou:

	Predito legítimo	Predito fraude
Real legítimo	42.481	7
Real fraude	14	57
Interpretação

No conjunto TEST:

42.481 transações legítimas foram classificadas corretamente;
7 transações legítimas foram classificadas incorretamente como fraude;
57 fraudes foram detectadas corretamente;
14 fraudes não foram detectadas.

Assim, o modelo detectou:

57 / 71 = 80.28%

das fraudes presentes no conjunto TEST.

Ao mesmo tempo, apresentou apenas:

7 falsos positivos

entre as transações legítimas avaliadas.

⚖️ XGBoost vs. XGBoost Tuned

A comparação entre os modelos supervisionados evidencia um trade-off importante.

Métrica	XGBoost	XGBoost Tuned
Precision	94.83%	89.06%
Recall	77.46%	80.28%
F1-Score	85.27%	84.44%
ROC-AUC	96.51%	96.28%
PR-AUC	81.48%	82.02%
FPR	0.0071%	0.0165%
FNR	22.54%	19.72%
Detection Rate	77.46%	80.28%

O XGBoost tradicional apresentou:

maior Precision;
maior F1-Score;
maior ROC-AUC;
menor False Positive Rate.

O XGBoost Tuned apresentou:

maior PR-AUC;
maior Recall;
maior taxa de detecção de fraude;
menor False Negative Rate.

Portanto, o modelo Tuned foi selecionado porque o projeto prioriza a capacidade de identificar a classe fraudulenta.

Esse resultado também evidencia um trade-off real de negócio: aumentar a detecção de fraudes resultou em uma pequena redução de Precision e aumento de falsos positivos.

📈 Evidências dos experimentos

Os gráficos gerados durante a avaliação são mantidos no diretório results/.

Comparação de F1-Score

Comparação de Recall

Comparação de PR-AUC

Comparação de False Positive Rate

🔍 Avaliação do modelo final
Matriz de confusão

Feature Importance

A análise de feature importance permite observar quais variáveis apresentaram maior contribuição para as decisões do modelo.

Curva ROC

Curva Precision-Recall

A curva Precision-Recall é particularmente relevante neste projeto devido ao forte desbalanceamento entre as classes.

💡 Principais conclusões

Os experimentos demonstraram uma diferença significativa entre as abordagens avaliadas.

As técnicas de detecção de anomalias apresentaram desempenho inferior aos modelos supervisionados neste conjunto de dados.

O LOF apresentou desempenho particularmente baixo:

PR-AUC: 0.0022
Recall: 0.70%
F1-Score: 0.19%
Detection Rate: 0.70%

O Isolation Forest apresentou desempenho superior ao LOF, mas ainda limitado:

PR-AUC: 0.1083
Recall: 26.06%
F1-Score: 23.95%
Detection Rate: 26.06%

As abordagens supervisionadas baseadas em XGBoost apresentaram desempenho significativamente superior.

O XGBoost Tuned alcançou:

82.02% de PR-AUC
80.28% de Recall
84.44% de F1-Score
89.06% de Precision
96.28% de ROC-AUC
0.0165% de False Positive Rate

Além disso, o modelo detectou 57 das 71 fraudes presentes no conjunto TEST.

Em comparação ao XGBoost tradicional:

Recall:
77.46% → 80.28%

False Negative Rate:
22.54% → 19.72%

Fraud Detection Rate:
77.46% → 80.28%

Por outro lado:

Precision:
94.83% → 89.06%

False Positive Rate:
0.0071% → 0.0165%

Esse comportamento demonstra o trade-off esperado entre aumentar a capacidade de detecção e controlar falsos positivos.

🏗️ Arquitetura da aplicação

A aplicação foi organizada separando treinamento, inferência e exposição da API.

                    ┌──────────────────┐
                    │      Cliente     │
                    └────────┬─────────┘
                             │ HTTP
                             ▼
                    ┌──────────────────┐
                    │     FastAPI      │
                    │    REST API      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Prediction Route │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Predictor    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Model Loader   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Machine Learning│
                    │      Model       │
                    └──────────────────┘

A separação entre treinamento e inferência permite que o modelo seja utilizado pela API sem executar novamente todo o pipeline de treinamento.

📁 Estrutura do projeto
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
🚀 Instalação
Pré-requisitos
Python 3.12+
Git
Docker Desktop — opcional
1. Clonar o repositório
git clone https://github.com/jefersonsilva344/fraud-detection.git
cd fraud-detection
🐍 Execução local
Windows

Criar o ambiente virtual:

python -m venv venv

Ativar:

.\venv\Scripts\Activate.ps1
Linux / macOS

Criar o ambiente virtual:

python3 -m venv venv

Ativar:

source venv/bin/activate
Instalar dependências
pip install -r requirements.txt

Para desenvolvimento e testes:

pip install -r requirements-dev.txt
🌐 Executando a API

A API pode ser iniciada utilizando Uvicorn:

uvicorn src.api.main:app --reload

Por padrão, a aplicação estará disponível em:

http://localhost:8000
Swagger UI

A documentação interativa do FastAPI pode ser acessada em:

http://localhost:8000/docs
OpenAPI

O schema OpenAPI está disponível em:

http://localhost:8000/openapi.json
🐳 Docker

A aplicação também pode ser executada em container.

Construir a imagem
docker build -t fraud-detection .
Executar o container
docker run -p 8000:8000 fraud-detection

A API estará disponível em:

http://localhost:8000

Swagger:

http://localhost:8000/docs
🐳 Docker Compose

Também é possível utilizar Docker Compose:

docker compose up --build

Para executar em segundo plano:

docker compose up --build -d

Para interromper:

docker compose down
🧪 Testes

O projeto utiliza Pytest para testes automatizados.

Executar todos os testes:

python -m pytest

Resultado atual:

25 passed
📊 Cobertura de testes

Executar os testes com cobertura:

python -m pytest --cov --cov-report=term-missing

Resultado atual:

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

Também é possível validar diretamente o limite mínimo:

python -m coverage report --fail-under=80
🧪 Status dos testes
Indicador	Resultado
Testes executados	25
Testes aprovados	25
Testes falhos	0
Cobertura	82.14%
Cobertura mínima	80%
Status	✅ PASS
🧩 Categorias de testes
Unitários

Localizados em:

tests/unit/

Incluem:

test_model_loader.py
test_predictor.py

Responsáveis por validar componentes individuais da camada de inferência.

Integração

Localizados em:

tests/integration/

Incluem:

test_api.py

Validam a integração da aplicação e seus endpoints.

Docker

Localizados em:

tests/docker/

Incluem:

test_docker_api.py

Validam o comportamento da aplicação em ambiente containerizado.

📦 Resultados experimentais

O diretório results/ contém os artefatos gerados durante os experimentos.

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
    ├── xgboost_tuned_precision_recall.png
    └── xgboost_tuned_threshold_f1.png

Esses artefatos permitem consultar os resultados intermediários e finais dos experimentos.

🧪 Reprodutibilidade

Os experimentos foram separados em scripts específicos para facilitar a execução das diferentes etapas.

Pré-processamento
python src/preprocessamento.py
Treinamento
python src/treinamento.py
Avaliação
python src/avaliacao.py
Comparação dos modelos
python src/comparacao.py
Avaliação rigorosa
python src/avaliacao_rigorosa.py
Comparação final
python src/comparacao_final.py

Os scripts relacionados ao XGBoost também podem ser executados individualmente conforme a etapa experimental.

📋 Principais componentes
Componente	Responsabilidade
preprocessamento.py	Preparação dos dados
analise_exploratoria.py	Análise exploratória
treinamento.py	Treinamento dos modelos
modelo_anomalias.py	Modelagem de anomalias
lof.py	Experimentos com LOF
xgboost_model.py	Modelo XGBoost
xgboost_tuning.py	Ajuste de hiperparâmetros
threshold_isolation_forest.py	Análise de threshold
avaliacao.py	Avaliação dos modelos
avaliacao_rigorosa.py	Avaliação rigorosa
comparacao.py	Comparação experimental
comparacao_final.py	Comparação final
model_loader.py	Carregamento do modelo
predictor.py	Execução da inferência
prediction.py	Endpoint de previsão
main.py	Inicialização da API
🔐 Boas práticas

O projeto utiliza .gitignore para evitar o versionamento de arquivos sensíveis ou gerados localmente.

Entre os arquivos e diretórios protegidos estão:

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

Os artefatos presentes em results/ são mantidos no repositório porque fazem parte da documentação e evidência dos experimentos.

🧰 Tecnologias utilizadas
Categoria	Tecnologias
Linguagem	Python
Machine Learning	Scikit-learn, XGBoost
Dados	Pandas, NumPy
API	FastAPI
Validação	Pydantic
Servidor	Uvicorn
Testes	Pytest, pytest-cov
Containerização	Docker, Docker Compose
Versionamento	Git, GitHub
📌 Decisões técnicas
Por que utilizar múltiplos modelos?

A comparação entre algoritmos permite avaliar diferentes estratégias de detecção:

detecção de anomalias;
densidade local;
classificação supervisionada.

Isso permite verificar empiricamente qual abordagem é mais adequada ao conjunto de dados.

Por que utilizar XGBoost?

O XGBoost foi utilizado como abordagem supervisionada devido à sua capacidade de modelar relações não lineares e apresentar bom desempenho em problemas tabulares de classificação.

Por que otimizar o threshold?

O threshold padrão de classificação nem sempre representa o melhor ponto operacional para um problema de fraude.

A alteração do threshold permite controlar o trade-off entre:

Recall
   ↕
detecção de fraudes

Precision / FPR
   ↕
controle de falsos positivos
Por que utilizar PR-AUC?

Em problemas fortemente desbalanceados, ROC-AUC pode apresentar uma percepção otimista do desempenho.

A PR-AUC concentra-se na relação entre Precision e Recall e, portanto, fornece uma visão particularmente relevante sobre o desempenho da classe positiva.

🚧 Limitações

Apesar dos resultados obtidos, existem limitações importantes para utilização em um ambiente real de produção.

Atualmente o projeto não implementa:

monitoramento contínuo de data drift;
monitoramento de model drift;
gerenciamento formal de versões de modelos;
pipeline completo de CI/CD;
observabilidade de produção;
autenticação da API;
autorização baseada em usuários;
monitoramento de latência;
testes de carga;
pipeline automatizado de retreinamento;
infraestrutura cloud.

Além disso, resultados obtidos em um dataset histórico não garantem o mesmo desempenho em um ambiente de produção real.

🔮 Próximos passos
 Implementar CI/CD completo.
 Adicionar GitHub Actions para testes automáticos.
 Implementar versionamento de modelos.
 Integrar MLflow.
 Adicionar monitoramento de data drift.
 Adicionar monitoramento de model drift.
 Implementar logging estruturado.
 Adicionar métricas de observabilidade da API.
 Implementar autenticação e autorização.
 Criar testes de carga.
 Criar pipeline automatizado de treinamento.
 Implementar deploy em cloud.
 Implementar monitoramento do modelo em produção.
📊 Status do projeto
Etapa	Status
Machine Learning	✅ Concluído
Experimentação	✅ Concluído
Comparação de modelos	✅ Concluído
Threshold Optimization	✅ Concluído
Avaliação rigorosa	✅ Concluído
Modelo final	✅ Concluído
API REST	✅ Concluído
Docker	✅ Concluído
Testes automatizados	✅ Concluído
Cobertura > 80%	✅ Concluído
CI/CD	⚪ Planejado
Model Monitoring	⚪ Planejado
Cloud Deploy	⚪ Planejado
👨‍💻 Autor
Jeferson Silva

Desenvolvedor Python com foco em Machine Learning, Inteligência Artificial e Engenharia de Software.

Este projeto demonstra, na prática, a construção de uma solução completa de Machine Learning, abrangendo:

preparação e análise dos dados;
experimentação com diferentes algoritmos;
avaliação para dados desbalanceados;
otimização de threshold;
seleção de modelo;
inferência;
desenvolvimento de API REST;
testes automatizados;
cobertura de código;
Docker e containerização.
🔗 Links
GitHub: https://github.com/jefersonsilva344
LinkedIn: adicione seu perfil aqui
⭐ Sobre este projeto

Este projeto foi desenvolvido para demonstrar a integração entre:

Machine Learning
       +
Engenharia de Software
       +
Experimentação
       +
API REST
       +
Testes
       +
Docker

Mais do que construir um classificador, o projeto busca demonstrar um fluxo completo de desenvolvimento:

Dados
  ↓
Pré-processamento
  ↓
Experimentação
  ↓
Avaliação
  ↓
Tuning
  ↓
Threshold
  ↓
Validação
  ↓
Modelo final
  ↓
Inferência
  ↓
API
  ↓
Docker

⭐ Se este projeto foi útil ou interessante, considere deixar uma estrela no repositório
