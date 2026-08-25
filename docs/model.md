# Modelo de detecção de fraude

## Finalidade

O serviço utiliza um classificador binário XGBoost para estimar a probabilidade
de uma transação pertencer à classe de fraude. O modelo não toma a decisão de
negócio diretamente: ele retorna uma probabilidade entre 0 e 1, e
`FraudDetectionService` compara esse valor ao threshold operacional.

## Artefatos versionados

Os artefatos necessários para inferência estão em `artifacts/xgboost/`:

| Arquivo | Responsabilidade |
| --- | --- |
| `model.json` | Modelo `XGBClassifier` serializado no formato JSON do XGBoost. |
| `metadata.json` | Contrato do modelo: tipo, formato, target, ordem das features e threshold. |

`ModelLoader` valida a existência dos dois arquivos e a estrutura do metadata
antes de carregar o modelo. O serviço não treina modelos em tempo de execução.

## Metadata atual

```json
{
  "model_type": "XGBClassifier",
  "model_format": "xgboost_json",
  "target": "Class",
  "threshold": 0.66275984
}
```

Além desses campos, o metadata contém a lista ordenada de features e os
parâmetros usados no treinamento. O carregador rejeita metadata com campos
obrigatórios ausentes, tipo/formato incompatíveis, features vazias ou
duplicadas, target presente entre as features ou threshold inválido.

## Features de entrada

O modelo espera exatamente 30 features, nesta ordem:

```text
Time, V1, V2, V3, V4, V5, V6, V7, V8, V9,
V10, V11, V12, V13, V14, V15, V16, V17, V18, V19,
V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount
```

`XGBoostPredictor` valida a presença de todas as features e rejeita campos
extras. Em seguida, monta o `DataFrame` na ordem estabelecida pelo metadata,
independentemente da ordem recebida na requisição.

- `Time` representa o instante relativo da transação no dataset original.
- `Amount` representa o valor monetário da transação e deve ser não negativo
  no contrato HTTP.
- `V1` a `V28` são componentes anonimizados do dataset. Não devem ser
  interpretadas individualmente como variáveis causais de fraude.

## Decisão de fraude

O fluxo de decisão é:

```text
features -> XGBoostPredictor -> probability
                                  |
                                  v
                        FraudDetectionService
                                  |
                probability >= 0.66275984 ?
                    |                    |
                  fraude             não fraude
```

O threshold atual é `0.66275984`. Ele faz parte do metadata do modelo e é
repassado ao domínio pelo Composition Root. Dessa forma, o adaptador XGBoost
não contém regras de classificação de negócio.

## Saída do modelo e da aplicação

O XGBoost produz uma probabilidade de fraude. A aplicação a apresenta como:

| Campo | Significado |
| --- | --- |
| `risk_score` | Probabilidade retornada pelo modelo, entre 0 e 1. |
| `risk_percentage` | `risk_score × 100`. |
| `is_fraud` | Resultado da comparação do risco com o threshold. |
| `threshold` | Limiar usado naquela decisão. |

## Métricas de avaliação

O modelo XGBoost Tuned foi selecionado no conjunto de teste com as seguintes
métricas registradas no projeto:

| Métrica | Resultado |
| --- | ---: |
| Precision | 89.06% |
| Recall | 80.28% |
| F1-score | 84.44% |
| ROC-AUC | 96.28% |
| PR-AUC | 82.02% |

Relatórios, curvas e comparações de treinamento ficam em `results/`. Essas
métricas descrevem a avaliação offline registrada no projeto; não substituem
monitoramento contínuo em produção.

## Limites

- O modelo depende do mesmo esquema e significado das 30 features usadas no
  treinamento.
- O threshold é uma decisão operacional e pode precisar ser recalibrado para
  outro contexto, custo de erro ou distribuição de dados.
- Componentes anonimizados não fornecem explicação causal para uma decisão.
- O projeto não implementa retreinamento automático, monitoramento de drift,
  autenticação ou controle de acesso.
