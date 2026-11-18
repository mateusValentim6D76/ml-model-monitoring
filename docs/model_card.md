# Model Card: `credit-scoring-xgb`

| Campo | Valor |
|---|---|
| Tarefa | Classificação binária: probabilidade de inadimplência (PD) no mês seguinte |
| Algoritmo | XGBoost (`XGBClassifier`, 400 árvores, profundidade 4, lr 0,05) |
| Baseline | Regressão Logística padronizada (benchmark interpretável) |
| Dados de treino | Dataset de Referência, com 13.484 clientes (60% × 75% do dataset UCI após deduplicação) |
| Holdout | 4.495 clientes (mesma distribuição da referência) |
| Threshold | escolhido no holdout por máximo F1 (0,305) e versionado com o modelo |
| Rastreabilidade | MLflow: experimento `credit-scoring-training`, Model Registry `credit-scoring-xgb` |
| Artefatos | `models/credit_model.joblib`, `models/model_metadata.json`, `models/feature_importance.csv` |

## Performance de referência (holdout)

| Modelo | AUC | Gini | KS |
|---|---|---|---|
| Regressão Logística (baseline) | 0,714 | 0,428 | 0,375 |
| **XGBoost (produção)** | **0,770** | **0,539** | **0,413** |

Valores exatos da última execução: `models/model_metadata.json`.

## Features

* **Usadas (21):** `limit_bal`, `age`, `education`, `pay_status_1..6`, `bill_amt_1..6`, `pay_amt_1..6`.
* **Excluídas por desenho (Fairness by Design):** `sex` e `marriage`, retidas apenas para auditoria de viés.
* **Identificador:** `customer_id` pseudonimizado (HMAC-SHA256), nunca usado como feature.

## Uso pretendido e limitações

* Apoio à decisão de concessão/manutenção de limite de cartão, e **não** uma decisão 100% automática
  sem possibilidade de revisão (LGPD art. 20).
* Dataset de Taiwan (2005): a distribuição não representa o público brasileiro atual; o projeto
  usa-o como *proxy* didático para demonstrar sustentação e governança.
* O modelo não observa renda real nem variáveis macroeconômicas, por isso é **cego a concept drift**
  provocado por inflação/desemprego (ver [causal_analysis.md](causal_analysis.md)).

## Fairness (holdout)

| Atributo | Disparate Impact | Leitura |
|---|---|---|
| Sexo | 0,94 | dentro da regra dos 4/5 |
| Faixa etária | 0,88 | dentro da regra (grupos com n ≥ 100); monitorado a cada lote |

## Critérios de retreino

Retreinar (com nova janela de referência e novo ciclo de validação/aprovação) quando:
1. alerta `performance_degradation` crítico em 2 lotes consecutivos; **ou**
2. `dataset_drift` crítico + `prediction_drift` crítico no mesmo lote; **ou**
3. Disparate Impact < 0,80 em qualquer atributo protegido.
