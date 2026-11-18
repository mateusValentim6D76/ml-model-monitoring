# Guia de execução passo a passo

Guia prático e detalhado para **instalar, executar, inspecionar, visualizar e experimentar**
com o projeto. Os comandos estão em **PowerShell (Windows)**; quando o equivalente em
Linux/macOS (bash) é diferente, ele aparece logo abaixo.

> **Atalho:** se o ambiente já estiver instalado, só faltam três comandos:
> ative o venv (§2.3), rode `credit-monitor all` (§3) e abra
> `reports\monitoring_dashboard.html` (§5).

---

## Índice

1. [O que você vai obter](#1-o-que-você-vai-obter)
2. [Instalação (primeira vez)](#2-instalação-primeira-vez)
3. [Executar o pipeline completo](#3-executar-o-pipeline-completo)
4. [Executar etapa por etapa](#4-executar-etapa-por-etapa)
5. [Ver os gráficos e relatórios](#5-ver-os-gráficos-e-relatórios)
6. [Ver os dados](#6-ver-os-dados)
7. [MLflow: rastreamento do modelo e do monitoramento](#7-mlflow-rastreamento-do-modelo-e-do-monitoramento)
8. [Simular outros cenários (experimentos)](#8-simular-outros-cenários-experimentos)
9. [Testar o data contract com seus próprios dados](#9-testar-o-data-contract-com-seus-próprios-dados)
10. [Testes automatizados e lint](#10-testes-automatizados-e-lint)
11. [Recomeçar do zero](#11-recomeçar-do-zero)
12. [Solução de problemas](#12-solução-de-problemas)
13. [Roteiro sugerido para demonstrar o projeto](#13-roteiro-sugerido-para-demonstrar-o-projeto)
14. [Mapa: onde fica cada coisa](#14-mapa-onde-fica-cada-coisa)

---

## 1. O que você vai obter

Um único comando (`credit-monitor all`, cerca de 90 segundos) executa:

| # | Etapa | O que acontece | Onde ver o resultado |
|---|---|---|---|
| 1 | `prepare` | Baixa o dataset (30 mil clientes), limpa, pseudonimiza o ID e separa **Referência** × **Produção** | `data/reference/`, `data/production/pool.parquet` |
| 2 | `train` | Valida a referência no contrato, treina XGBoost + Regressão Logística e registra no MLflow | `models/`, MLflow |
| 3 | `contract-demo` | Valida um lote limpo (aprova) e um lote corrompido (**bloqueia**) | `reports/data_contracts/`, `data/quarantine/` |
| 4 | `simulate` | Cria 6 lotes mensais com drift crescente (2026-01 a 2026-06) | `data/production/batches/` |
| 5 | `monitor` | Para cada lote: contrato, predição, drift (Evidently + PSI/KS), performance, fairness e alertas. No 3º mês chega um lote corrompido, que é bloqueado | `reports/drift/`, `logs/`, MLflow |
| 6 | `dashboard` | Consolida tudo em uma página HTML | `reports/monitoring_dashboard.html` |

---

## 2. Instalação (primeira vez)

### 2.1 Pré-requisitos

* **Python 3.11 ou superior** (o projeto foi testado no 3.13). Para conferir:

```powershell
python --version
```

* **Internet** na primeira execução, para baixar o dataset (cerca de 3 MB, OpenML) e as bibliotecas.
* Uns 1,5 GB livres em disco (bibliotecas do MLflow/Evidently + dados + relatórios).

### 2.2 Abrir o terminal na pasta do projeto

```powershell
cd "C:\Users\Mateus\Documents\Development\Machine Learning\ml-model-monitoring"
```

### 2.3 Criar e ativar o ambiente virtual

Crie o ambiente (só na primeira vez):

```powershell
python -m venv .venv
```

Ative o ambiente (em **todo terminal novo**):

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
source .venv/bin/activate
```

O prompt passa a começar com `(.venv)`. Se o PowerShell bloquear o script de ativação,
veja o [§12](#12-solução-de-problemas).

### 2.4 Instalar as dependências

Use **sempre** o `requirements.txt`, que tem as versões fixadas. Instalar sem versões faz o
pip passar muito tempo resolvendo conflitos entre MLflow e Evidently, e o Evidently 0.7 não
funciona com pandas 3.

```powershell
python -m pip install --upgrade pip
```

```powershell
pip install -r requirements.txt
```

Depois, instale o próprio projeto em modo editável. Isso cria o comando `credit-monitor`:

```powershell
pip install -e . --no-deps
```

A instalação leva de 3 a 10 minutos. Para conferir se deu certo:

```powershell
credit-monitor --help
```

A saída deve listar os comandos `prepare, train, contract-demo, simulate, monitor, dashboard, all`.

> Se o comando `credit-monitor` não for encontrado, use a forma equivalente
> `python -m credit_monitoring <comando>`; ela funciona em todos os exemplos deste guia.

---

## 3. Executar o pipeline completo

```powershell
credit-monitor all
```

O que você verá no terminal (resumo):

```
Etapa: prepare
[ingest_download] SUCCESS ...            (download do dataset, só na 1ª vez)
Referência (treino)=13484 | Referência (teste)=4495 | Pool de produção=11986 | taxa de default=22.12%
Etapa: train
Lote 'reference_train' aprovado no contrato (13484 registros).
Baseline LogReg  AUC=0.7138 KS=0.3752
XGBoost (prod.)  AUC=0.7697 KS=0.4132 F1=0.5400 threshold=0.305
Etapa: contract-demo
Lote 'demo_clean_batch' aprovado no contrato (1000 registros).
ALERTA [CRITICAL] data_contract_violation: Lote 'demo_corrupted_batch' BLOQUEADO: 869 violações ...
Ingestão interrompida como esperado: ...
Etapa: simulate
Lote 2026-01: intensidade=0.00 | default=22.13% ...
...
Lote 2026-06: intensidade=1.00 | default=45.47% ...
Etapa: monitor
ALERTA [CRITICAL] feature_drift: ...  pay_status_3 (0.26)
ALERTA [WARNING] performance_degradation: AUC caiu 0.053 ...
ALERTA [CRITICAL] data_contract_violation: Lote '2026-03-incident' BLOQUEADO ...
...
Causa raiz (2026-06): ΔAUC total=0.130 | covariate=-0.010 | concept=0.140
Etapa: dashboard
Pipeline completo em 90s. Abra reports/monitoring_dashboard.html
```

Observações:

* **Mensagens `CRITICAL` e `WARNING` são esperadas.** Elas são os alertas que o projeto
  precisa demonstrar, não erros do programa. Erro de verdade aparece como `Traceback`.
* A **primeira execução é mais lenta**: download do dataset (uns 10 s) e criação do banco do
  MLflow (uns 20 s). Nas seguintes, o download é pulado (o arquivo fica em `data/raw/`).
* O comando `all` **zera os logs** (`logs/*.jsonl`) no início, então o dashboard sempre
  reflete uma execução limpa. O MLflow **não** é zerado: cada execução vira um novo run.
* Os resultados são **reprodutíveis** (seed 42): rodar de novo produz os mesmos números.

---

## 4. Executar etapa por etapa

Útil para entender ou demonstrar cada parte. **A ordem importa**, porque cada etapa usa as
saídas da anterior:

```powershell
credit-monitor prepare
```

```powershell
credit-monitor train
```

```powershell
credit-monitor contract-demo
```

```powershell
credit-monitor simulate
```

```powershell
credit-monitor monitor
```

```powershell
credit-monitor dashboard
```

| Comando | Depende de | Gera |
|---|---|---|
| `prepare` | internet (1ª vez) | `data/raw`, `data/processed`, `data/reference`, `data/production/pool.parquet` |
| `train` | `prepare` | `models/*`, `data/reference/reference_scored.parquet`, run no MLflow |
| `contract-demo` | `prepare` | `reports/data_contracts/contract_demo_*.html`, `data/quarantine/demo_*` |
| `simulate` | `prepare` | `data/production/batches/<mês>.parquet` e `<mês>_labels.parquet`, `manifest.json` |
| `monitor` | `train` + `simulate` | `reports/drift/*`, `reports/monitoring_*`, `data/production/predictions/`, run no MLflow |
| `dashboard` | `monitor` | `reports/monitoring_dashboard.html` |

> ⚠️ Comandos individuais **acrescentam** linhas aos logs, sem zerá-los. Se você rodar `monitor`
> duas vezes, o dashboard vai somar as duas execuções (contagens de tarefas e alertas
> dobradas). Para um dashboard limpo, use `credit-monitor all` ou apague `logs\*.jsonl`
> antes (§11).

---

## 5. Ver os gráficos e relatórios

Todos são arquivos HTML **autocontidos**: abrem offline, direto no navegador.

### 5.1 Dashboard de observabilidade (o principal)

```powershell
start reports\monitoring_dashboard.html
```

```bash
# macOS: open reports/monitoring_dashboard.html  |  Linux: xdg-open reports/monitoring_dashboard.html
```

O que ver, de cima para baixo:

| Bloco | O que mostra | O que observar |
|---|---|---|
| **KPIs** (topo) | tarefas executadas, taxa de sucesso, lotes bloqueados, alertas, AUC atual × referência, features com drift | AUC caiu de 0,770 para 0,640; 1 lote bloqueado |
| Saúde do pipeline | status de cada tarefa (verde = sucesso, vermelho = bloqueada) | `data_contract_validation` tem 2 bloqueios: a demo e o incidente |
| Latência das tarefas | tempo médio e máximo por tarefa | as tarefas de drift (Evidently) são as mais lentas |
| Volume processado | registros recebidos × pontuados por lote | `2026-03-incident`: recebido, **0 pontuados** |
| Infraestrutura | memória (MB) e CPU (s) por execução | estabilidade do processo |
| Data Contracts | tabela por lote, com link para o relatório de cada contrato | clique em "relatório" no incidente |
| **PSI por feature × mês** (heatmap) | intensidade do drift em cada variável ao longo do tempo | `pay_status_2/3` e `limit_bal` ficam vermelhos |
| Dataset drift | % de features com drift (barras) e PSI do score (linha) | passa do limite de 30% em 2026-05 |
| **Degradação do modelo** | AUC e KS por mês, com linhas de referência e limite crítico | a queda é contínua |
| Risco previsto × realizado | inadimplência real × PD média prevista | as curvas se separam: é o *concept drift* |
| Fairness | Disparate Impact por sexo e faixa etária, com a linha da regra 4/5 (0,8) | fica acima de 0,8 |
| **Análise causal** | efeito de cada feature no score e decomposição da queda de AUC | concept drift explica 0,140 de 0,130 |
| Alertas | lista completa, com severidade e mensagem | 22 alertas |
| Relatórios Evidently | links para os HTML do Evidently | - |

Os gráficos são interativos (Plotly): passe o mouse para ver valores, arraste para dar zoom,
dê duplo clique para voltar e clique na legenda para esconder/mostrar uma série.

### 5.2 Relatórios do Evidently AI

```powershell
start reports\drift\evidently_drift_2026-06.html
```

```powershell
start reports\drift\evidently_full_2026-06.html
```

* `evidently_drift_<mês>.html` (um por mês): Data Drift com **PSI** (threshold 0,10). Mostra
  a tabela "Data Drift Summary" com a distribuição de referência × atual de cada coluna, o
  teste usado e o *drift score*. Clique em uma linha para ver o histograma comparativo.
  Compare `2026-01` (nada derivou) com `2026-06` (várias colunas).
* `evidently_full_2026-06.html`: drift com o teste padrão do Evidently (KS / χ² /
  Wasserstein) **mais a qualidade da classificação** (métricas como acurácia, precisão,
  recall, F1 e ROC AUC) da produção × referência. Essa é a evidência visual do concept drift.
* Os arquivos têm cerca de 5 MB e podem levar alguns segundos para abrir.

### 5.3 Relatórios dos Data Contracts

```powershell
start reports\data_contracts\contract_demo_corrupted_batch.html
```

```powershell
start reports\data_contracts\contract_2026-03-incident.html
```

Cada página mostra o selo **APROVADO/BLOQUEADO**, as 10 regras com status ✔/✘, a contagem de
violações e o detalhe por coluna e verificação, com exemplos dos valores inválidos. Há também
`contract_<mês>.html` dos lotes aprovados e `contract_reference_*.html` do treino.

### 5.4 Alternativa: servir os relatórios por HTTP

Se o navegador bloquear algo ao abrir arquivos locais:

```powershell
python -m http.server 8000 --directory reports
```

Depois acesse http://localhost:8000/monitoring_dashboard.html. Para parar o servidor, use `Ctrl+C`.

---

## 6. Ver os dados

### 6.1 Explorador pronto (recomendado)

O script [`scripts/explore_data.py`](../scripts/explore_data.py) imprime tudo formatado no terminal:

```powershell
python scripts\explore_data.py
```

Esse comando mostra a visão geral mês a mês (status, drift, AUC, inadimplência, fairness,
alertas). Os subcomandos:

| Comando | Mostra |
|---|---|
| `python scripts\explore_data.py referencia` | amostra e estatísticas do dataset de treino e do holdout |
| `python scripts\explore_data.py producao` | parâmetros de cada lote simulado e **médias por mês × referência** (efeito dos choques) |
| `python scripts\explore_data.py lote 2026-06` | features, rótulos e predições de um lote |
| `python scripts\explore_data.py drift 2026-06` | tabela PSI/KS/χ² por feature (troque o mês à vontade) |
| `python scripts\explore_data.py quarentena` | lotes bloqueados: violações por coluna e verificação, com exemplos |
| `python scripts\explore_data.py alertas` | todos os alertas, agrupados por severidade e tipo |
| `python scripts\explore_data.py etapas` | status, latência, volume, CPU e memória de cada tarefa |
| `python scripts\explore_data.py modelo` | métricas XGBoost × LogReg, threshold, fairness, importância das features |
| `python scripts\explore_data.py causa` | análise causal do último lote |

### 6.2 Onde está cada dado

| Pasta / arquivo | Conteúdo | Formato |
|---|---|---|
| `data/raw/credit_default_raw.csv` | dataset original (com o ID original, **nunca versionado**) | CSV |
| `data/processed/credit_clean.parquet` | dataset limpo e pseudonimizado | Parquet |
| `data/reference/reference.parquet` | **Referência**: dados de treino (13.484) | Parquet |
| `data/reference/reference_test.parquet` | holdout da referência (4.495) | Parquet |
| `data/reference/reference_scored.parquet` | holdout + predições: base de comparação do monitoramento | Parquet |
| `data/production/pool.parquet` | pool de produção, que alimenta a simulação (11.986) | Parquet |
| `data/production/batches/<mês>.parquet` | lote mensal **sem** rótulo (o que chega para inferência) | Parquet |
| `data/production/batches/<mês>_labels.parquet` | rótulos do lote (chegam depois: *label delay*) | Parquet |
| `data/production/batches/manifest.json` | choques aplicados em cada mês | JSON |
| `data/production/predictions/<mês>.parquet` | predições do modelo (customer_id, PD, decisão, versão) | Parquet |
| `data/quarantine/*.csv` | lotes bloqueados e suas falhas | CSV |
| `reports/drift/<mês>_drift_table.csv` | PSI, KS/χ², p-valor e médias por feature | CSV (abre no Excel) |
| `reports/drift/psi_matrix.csv` | matriz feature × mês de PSI (base do heatmap) | CSV |
| `reports/drift/<mês>_fairness.csv` | aprovação, TPR/FPR e DI por grupo | CSV |
| `reports/drift/2026-06_root_cause.csv` | efeito causal por feature | CSV |
| `reports/monitoring_history.csv` | uma linha por lote com todas as métricas | CSV |
| `reports/monitoring_summary.json` | resumo completo do monitoramento | JSON |
| `logs/stages.jsonl` · `metrics.jsonl` · `alerts.jsonl` · `pipeline.jsonl` | telemetria (1 evento JSON por linha) | JSON Lines |
| `models/model_metadata.json` · `feature_importance.csv` · `fairness_reference.csv` | metadados do modelo | JSON / CSV |

### 6.3 Explorar no Python interativo

Os arquivos Parquet não abrem no Excel, mas abrem facilmente no pandas:

```powershell
python
```

```python
import pandas as pd
ref = pd.read_parquet("data/reference/reference.parquet")
jun = pd.read_parquet("data/production/batches/2026-06.parquet")
ref.describe().T                       # estatísticas da referência
jun["limit_bal"].mean() / ref["limit_bal"].mean() - 1   # variação do limite (perto de -45%)
ref["pay_status_1"].value_counts(normalize=True).sort_index()
jun["pay_status_1"].value_counts(normalize=True).sort_index()
pd.read_csv("reports/drift/2026-06_drift_table.csv").head(10)
```

Para converter qualquer Parquet em CSV (e abrir no Excel):

```powershell
python -c "import pandas as pd; pd.read_parquet('data/production/batches/2026-06.parquet').to_csv('lote_2026-06.csv', index=False)"
```

### 6.4 Dicionário das colunas

| Coluna | Significado |
|---|---|
| `customer_id` | pseudônimo HMAC-SHA256 do cliente (`cus_` + 24 hex) |
| `limit_bal` | limite de crédito (NT$) |
| `sex` | 1 = masculino, 2 = feminino (**não é feature**: só auditoria) |
| `education` | 1 = pós-graduação, 2 = universidade, 3 = ensino médio, 4 = outros |
| `marriage` | 1 = casado, 2 = solteiro, 3 = outros (**não é feature**) |
| `age` | idade em anos |
| `pay_status_1..6` | situação de pagamento nos últimos 6 meses (1 = mais recente): -2/-1/0 = em dia, 1..8 = meses de atraso |
| `bill_amt_1..6` | valor da fatura em cada mês |
| `pay_amt_1..6` | valor pago em cada mês |
| `default` | **alvo**: 1 = inadimplente no mês seguinte |
| `prediction_proba` | PD prevista pelo modelo (0 a 1) |
| `prediction` | decisão: 1 = negar (PD ≥ threshold 0,305), 0 = aprovar |

---

## 7. MLflow: rastreamento do modelo e do monitoramento

### 7.1 Abrir a interface

Com o venv ativo, na pasta do projeto:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Acesse **http://127.0.0.1:5000**. O terminal fica ocupado enquanto a UI estiver aberta;
use `Ctrl+C` para encerrar.

### 7.2 O que ver

**Experimento `credit-scoring-training`** (menu lateral, opção Experiments):

1. Clique no run `train-xgb-<data>`.
2. **Parameters**: hiperparâmetros do XGBoost e o threshold.
3. **Metrics**: `test_auc`, `test_ks`, `test_gini`, `baseline_logreg_*` e `fairness_di_*`.
4. **Tags**: as tags de governança (`governance.lgpd_legal_basis`,
   `governance.protected_attributes_excluded`, `governance.pii`, `data_contract`).
5. **Artifacts**: o modelo (`model/`), `model_metadata.json`, `feature_importance.csv` e
   `fairness_reference.csv`.
6. Menu **Models**, item `credit-scoring-xgb`: o modelo no **Model Registry**, com uma versão
   por execução.

**Experimento `credit-scoring-monitoring`**:

1. Clique no run `monitoring-<data>`.
2. Aba **Model metrics** (ou "Metrics"): cada métrica é uma **série temporal por lote**
   (eixo x = *step*: 0 = 2026-01, 1 = 2026-02, 2 = 2026-03, 3 = incidente, 4 = 2026-04 e assim por diante).
   * `perf_auc`, `perf_ks`: queda de performance.
   * `drift_share`, `prediction_psi`: drift crescente.
   * `psi_pay_status_3`, `psi_limit_bal` etc.: uma curva por feature.
   * `contract_passed`: vai a 0 no step 3 (lote bloqueado).
   * `alerts_critical`, `alerts_warning`.
3. **Artifacts**: todos os relatórios Evidently, os relatórios de contrato, as tabelas de drift
   e os logs da execução, versionados junto do run.

Dica: em Experiments, selecione dois runs de monitoramento (por exemplo, de cenários
diferentes, §8) e clique em **Compare** para sobrepor as curvas.

---

## 8. Simular outros cenários (experimentos)

Todo o cenário fica em [`config/config.yaml`](../config/config.yaml). Edite, salve e rode de novo.

### 8.1 Quais etapas rodar depois de cada alteração

| O que você alterou | Rode |
|---|---|
| `simulation.*` (choques, meses, intensidades) | `credit-monitor simulate`, `credit-monitor monitor`, `credit-monitor dashboard` (ou `all`) |
| `monitoring.*` (thresholds de alerta) | `credit-monitor monitor`, `credit-monitor dashboard` |
| `model.*` (hiperparâmetros, threshold) | `credit-monitor all` |
| `dataset.*` / `features.*` | `credit-monitor all` |

Para o dashboard sair limpo, prefira `credit-monitor all` (ele zera os logs).

### 8.2 Parâmetros do cenário

```yaml
simulation:
  n_batches: 6                                     # quantidade de meses
  batch_size: 2000                                 # registros por mês (limitado ao tamanho do pool)
  intensity_schedule: [0.0, 0.1, 0.3, 0.5, 0.75, 1.0]  # um valor por mês (0 = sem choque)
  shocks:
    inflation_bill_increase: 0.45    # +45% nas faturas
    limit_tightening: 0.30           # -30% nos limites
    payment_delay_share: 0.35        # 35% dos clientes com +1/+2 meses de atraso
    young_customers_share: 0.40      # 40% de clientes novos de 21 a 30 anos
    payment_capacity_drop: 0.35      # -35% nos pagamentos
  concept_drift:
    flip_share: 0.25                 # força do concept drift
    utilization_threshold: 0.5
```

Regras:

* `intensity_schedule` precisa ter **exatamente** `n_batches` valores.
* `n_batches × batch_size` não pode passar do tamanho do pool (11.986). Se passar, o
  tamanho do lote é reduzido automaticamente.

### 8.3 Experimentos prontos (bons para a apresentação)

**A) Só data drift, sem concept drift.** Mostra que drift nos dados nem sempre derruba o modelo.

```yaml
  concept_drift:
    flip_share: 0.0
```

O que esperar: os alertas de `feature_drift`, `dataset_drift` e `prediction_drift` continuam iguais
(11 features com drift em 2026-06), mas a **AUC cai só de 0,770 para 0,748**, contra 0,640 no
cenário padrão. Conclusão: drift em P(X) é um sinal de atenção; a mudança em P(y|X) é o que
destrói o modelo.

**B) Só concept drift, sem data drift.** É a degradação mais silenciosa possível.

```yaml
  shocks:
    inflation_bill_increase: 0.0
    limit_tightening: 0.0
    payment_delay_share: 0.0
    young_customers_share: 0.0
    payment_capacity_drop: 0.0
  concept_drift:
    flip_share: 0.25
```

O que esperar: **nenhuma feature com drift e PSI do score praticamente zero** (o monitoramento de dados não vê
nada), mas a AUC cai para 0,67 e a inadimplência sobe para 34%. Só o monitoramento de performance com rótulos detecta o
problema. Isso mostra por que é preciso monitorar as duas coisas.

**C) Choque abrupto, em vez de gradual.**

```yaml
  intensity_schedule: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
```

O que esperar: tudo saudável por 3 meses e, de repente, todas as regras disparam (crise súbita).

**D) Economia estável (controle).** Prova que o sistema não gera falso alarme.

```yaml
  intensity_schedule: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

O que esperar: todos os meses `healthy`, PSI < 0,10 e AUC estável. Só o lote do incidente
continua bloqueado, porque ele não depende do cenário.

**E) Inflação forte.** Para ver o `bill_amt` passar do limite de PSI:

```yaml
    inflation_bill_increase: 1.20
```

O que esperar: `bill_amt_1..6` chegam a PSI 0,18-0,23 em 2026-06 (alerta de atenção), e o total
de features com drift sobe de 11 para 17.

**F) Alertas mais sensíveis ou mais tolerantes** (seção `monitoring`):

```yaml
monitoring:
  psi_warning: 0.05        # padrão 0.10
  psi_critical: 0.15       # padrão 0.25
  auc_drop_warning: 0.01   # padrão 0.03
```

> 💡 Antes de experimentar, guarde uma cópia do config original
> (`copy config\config.yaml config\config.original.yaml`) e restaure no final
> (`copy config\config.original.yaml config\config.yaml`). Assim os relatórios versionados no
> Git voltam a corresponder ao cenário padrão: rode `credit-monitor all` depois de restaurar.

### 8.4 Mudar o momento do incidente

O lote corrompido chega depois do 3º mês. Isso é definido pela constante `INCIDENT_AFTER = 2`
em [`src/credit_monitoring/monitoring/monitor.py`](../src/credit_monitoring/monitoring/monitor.py)
(o índice começa em 0). Mude para `0` e ele chega depois de 2026-01.

---

## 9. Testar o data contract com seus próprios dados

### 9.1 Pela demo pronta

```powershell
credit-monitor contract-demo
```

### 9.2 Validando um arquivo seu

1. Exporte um lote para CSV (§6.3), por exemplo `lote_2026-06.csv`.
2. Abra no Excel e "estrague" alguns valores: apague uma célula, coloque idade 15, limite
   negativo, `education` = 9, duplique uma linha inteira ou adicione uma coluna `nome`.
3. Salve como CSV e valide:

```powershell
python -c "import pandas as pd; from credit_monitoring.contracts.validate import validate_batch; r = validate_batch(pd.read_csv('lote_2026-06.csv'), batch='meu_teste'); print('APROVADO' if r.passed else 'BLOQUEADO', r.rules_violated)"
```

```powershell
start reports\data_contracts\contract_meu_teste.html
```

Se o lote for bloqueado, ele também vai para `data\quarantine\meu_teste.csv`, junto com o
arquivo de falhas.

> O Excel pode trocar o separador ou o formato dos números (vírgula decimal). Se tudo falhar
> na regra R9 (tipos), salve como "CSV UTF-8 (delimitado por vírgulas)", ou edite o arquivo
> no VS Code.

---

## 10. Testes automatizados e lint

```powershell
pytest -v
```

São 28 testes, que rodam em uns 30 s e não precisam do dataset baixado. Eles cobrem:
* `tests/test_contracts.py`: cada regra R1-R10 isoladamente, duplicatas e o lote corrompido
  (bloqueio, quarentena e relatório);
* `tests/test_drift.py`: PSI (zero para distribuições iguais, alto para deslocadas), níveis,
  efeito de cada choque da simulação e o fato de que dado com drift **não** viola o contrato;
* `tests/test_privacy_and_fairness.py`: pseudonimização (determinística, depende da chave),
  atributos protegidos fora das features e Disparate Impact.

Para rodar só um arquivo ou um teste:

```powershell
pytest tests\test_contracts.py -v
```

```powershell
pytest -k "duplicates" -v
```

Para checar o estilo do código:

```powershell
ruff check src tests scripts
```

O mesmo lint e os mesmos testes rodam no GitHub Actions a cada push (`.github/workflows/ci.yml`).
Veja o resultado na aba **Actions** do repositório.

---

## 11. Recomeçar do zero

Apaga dados gerados, relatórios, logs e MLflow, mas **mantém** o dataset baixado em `data/raw`:

```powershell
Remove-Item -Recurse -Force data\processed, data\reference, data\production, data\quarantine, reports, mlflow.db, mlartifacts -ErrorAction SilentlyContinue
```

```powershell
Remove-Item logs\*.jsonl -ErrorAction SilentlyContinue
```

```powershell
credit-monitor all
```

Para forçar também um novo download, apague `data\raw`. Para resetar só o MLflow (os runs
se acumulam a cada execução), apague `mlflow.db` e `mlartifacts`.

---

## 12. Solução de problemas

| Sintoma | Causa | Solução |
|---|---|---|
| `Activate.ps1 não pode ser carregado porque a execução de scripts foi desabilitada` | política de execução do PowerShell | libere só para o terminal atual com `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` e ative de novo; ou dispense a ativação e chame `.\.venv\Scripts\python -m credit_monitoring all` |
| `credit-monitor` não é reconhecido | venv não ativado, ou o projeto não foi instalado | ative o venv (§2.3) e rode `pip install -e . --no-deps`; ou use `python -m credit_monitoring <comando>` |
| `ModuleNotFoundError: credit_monitoring` | projeto não instalado no venv atual | `pip install -e . --no-deps` |
| `ImportError: cannot import name 'infer_dtype_from_object'` | pandas 3 instalado (incompatível com o Evidently 0.7) | `pip install "pandas>=2.2,<3"` |
| `pip install` demora muito | resolução de dependências sem versões fixas | use `pip install -r requirements.txt` |
| erro de rede no `prepare` | sem internet ou OpenML fora do ar | tente de novo mais tarde; o download é feito uma vez só |
| `Arquivo não encontrado ... Rode antes: credit-monitor all` | o explorador precisa das saídas do pipeline | rode `credit-monitor all` |
| `FileNotFoundError` ao rodar `monitor` | faltou rodar `train` ou `simulate` | siga a ordem do §4 |
| primeira execução "parada" em `Creating initial MLflow database tables` | criação do banco do MLflow | aguarde de 20 a 60 s; só acontece na primeira vez |
| `MLflow indisponível (...)` no log | banco travado ou corrompido | o pipeline continua sem tracking; feche a `mlflow ui` e apague `mlflow.db` e `mlartifacts` |
| porta 5000 ocupada na `mlflow ui` | outro programa usa a porta | use `--port 5001` e acesse http://127.0.0.1:5001 |
| dashboard com números "dobrados" | rodou etapas isoladas sem zerar os logs | rode `credit-monitor all` |
| `ValueError: zip() argument ... shorter` no `simulate` | `intensity_schedule` com tamanho diferente de `n_batches` | ajuste para ter um valor por mês |
| acentos estranhos no terminal | console em cp1252 | a CLI já força UTF-8; em outros scripts rode `$env:PYTHONIOENCODING="utf-8"` antes |

---

## 13. Roteiro sugerido para demonstrar o projeto

Um roteiro de uns 5 minutos (vídeo ou apresentação), no formato STAR:

1. **Situação** (30 s): mostre o README §1. A fintech suspeita de degradação silenciosa do
   modelo de crédito.
2. **Tarefa** (30 s): mostre a arquitetura (README §2) e as 4 etapas.
3. **Ação** (3 min):
   1. Rode `credit-monitor all` no terminal e comente as mensagens de bloqueio e de alerta.
   2. Abra `reports\data_contracts\contract_2026-03-incident.html`: o lote foi bloqueado,
      inclusive pela coluna `cpf` vazada (LGPD).
   3. Abra o dashboard. Mostre os KPIs, o heatmap de PSI (as variáveis degradadas), o gráfico
      de AUC caindo e o de risco previsto × realizado (concept drift).
   4. Abra `evidently_drift_2026-01.html` e `evidently_drift_2026-06.html` lado a lado (antes ×
      depois).
   5. Mostre o MLflow: o modelo registrado com as tags LGPD e as curvas de `perf_auc` e
      `psi_*` por step.
4. **Resultado** (1 min): no dashboard, mostre a análise causal. A queda de AUC é quase toda
   concept drift (renda real não observada), então retreinar sem novas variáveis não basta.
   Feche com a governança LGPD (README §7): pseudonimização, base legal (art. 7º, X),
   retenção e monitoramento de vieses.

---

## 14. Mapa: onde fica cada coisa

| Quero | Onde |
|---|---|
| mudar o cenário econômico ou os thresholds | `config/config.yaml` |
| ver ou alterar as regras do contrato | `src/credit_monitoring/contracts/schemas.py` |
| ver como o lote corrompido é criado | `src/credit_monitoring/contracts/corrupt.py` |
| ver como o drift é injetado | `src/credit_monitoring/drift/simulate.py` |
| ver o cálculo de PSI/KS/χ² | `src/credit_monitoring/drift/stats.py` |
| ver a integração com o Evidently | `src/credit_monitoring/drift/detect.py` |
| ver o loop de monitoramento | `src/credit_monitoring/monitoring/monitor.py` |
| ver as regras de alerta | `src/credit_monitoring/monitoring/alerts.py` |
| ver o código do dashboard | `src/credit_monitoring/monitoring/dashboard.py` |
| ver a análise causal | `src/credit_monitoring/analysis/root_cause.py` · `docs/causal_analysis.md` |
| ver a pseudonimização | `src/credit_monitoring/data/privacy.py` |
| ler a governança LGPD | `README.md` §7 |
| ler a definição das métricas | `docs/observability_metrics.md` |
| ler o model card | `docs/model_card.md` |
