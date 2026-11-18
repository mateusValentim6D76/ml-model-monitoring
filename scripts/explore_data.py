"""Explorador dos dados e resultados do projeto (uso pessoal / apresentação).

Uso:
    python scripts/explore_data.py                 # visão geral de tudo
    python scripts/explore_data.py referencia      # dataset de referência (treino)
    python scripts/explore_data.py producao        # lotes mensais simulados
    python scripts/explore_data.py lote 2026-06    # um lote específico (features + rótulos + predições)
    python scripts/explore_data.py drift 2026-06   # tabela de drift (PSI/KS) do lote
    python scripts/explore_data.py quarentena      # lotes bloqueados e o motivo
    python scripts/explore_data.py alertas         # alertas disparados
    python scripts/explore_data.py etapas          # status/latência das tarefas do pipeline
    python scripts/explore_data.py modelo          # métricas, threshold, importância das features
    python scripts/explore_data.py causa           # análise causal (root cause) do último lote

Pré-requisito: ter rodado `credit-monitor all` ao menos uma vez.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS, LOGS, MODELS = ROOT / "data", ROOT / "reports", ROOT / "logs", ROOT / "models"

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 80)


def title(text: str) -> None:
    print(f"\n{'=' * 90}\n{text}\n{'=' * 90}")


def need(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"Arquivo não encontrado: {path.relative_to(ROOT)}\n"
                 "Rode antes: credit-monitor all")
    return path


def jsonl(name: str) -> pd.DataFrame:
    with open(need(LOGS / name), encoding="utf-8") as fh:
        return pd.DataFrame([json.loads(line) for line in fh if line.strip()])


def referencia() -> None:
    for name in ("reference.parquet", "reference_test.parquet"):
        df = pd.read_parquet(need(DATA / "reference" / name))
        title(f"REFERÊNCIA: {name}  ({len(df):,} linhas × {df.shape[1]} colunas)")
        print(df.head(5).to_string())
        print(f"\nTaxa de default: {df['default'].mean():.2%}")
        print("\nEstatísticas das principais variáveis:")
        print(df[["limit_bal", "age", "bill_amt_1", "pay_amt_1", "pay_status_1"]]
              .describe().round(1).to_string())


def producao() -> None:
    batches = DATA / "production" / "batches"
    manifest = json.loads(need(batches / "manifest.json").read_text(encoding="utf-8"))
    title("PRODUÇÃO: lotes mensais simulados (manifest.json)")
    m = pd.DataFrame(manifest)
    cols = [c for c in ["batch", "rows", "intensity", "default_rate", "bill_increase_mean",
                        "limit_tightening_mean", "payment_delay_share", "young_customers_share",
                        "concept_flips", "delay_induced_defaults"] if c in m]
    print(m[cols].round(3).to_string(index=False))

    ref = pd.read_parquet(need(DATA / "reference" / "reference.parquet"))
    title("MÉDIAS POR LOTE × REFERÊNCIA (efeito dos choques)")
    rows = [{"lote": "referência", **ref[["limit_bal", "age", "bill_amt_1", "pay_amt_1",
                                          "pay_status_1"]].mean().round(2).to_dict()}]
    for b in m["batch"]:
        df = pd.read_parquet(batches / f"{b}.parquet")
        rows.append({"lote": b, **df[["limit_bal", "age", "bill_amt_1", "pay_amt_1",
                                      "pay_status_1"]].mean().round(2).to_dict()})
    print(pd.DataFrame(rows).to_string(index=False))


def lote(batch: str) -> None:
    batches = DATA / "production" / "batches"
    feats = pd.read_parquet(need(batches / f"{batch}.parquet"))
    labels = pd.read_parquet(need(batches / f"{batch}_labels.parquet"))
    title(f"LOTE {batch}: features recebidas para inferência ({len(feats):,} linhas)")
    print(feats.head(5).to_string())
    title(f"LOTE {batch}: rótulos (chegam com atraso)")
    print(labels.head(5).to_string())
    print(f"Taxa de default: {labels['default'].mean():.2%}")
    pred_path = DATA / "production" / "predictions" / f"{batch}.parquet"
    if pred_path.exists():
        preds = pd.read_parquet(pred_path)
        title(f"LOTE {batch}: predições do modelo")
        print(preds.head(5).to_string())
        print(f"\nPD média prevista: {preds['prediction_proba'].mean():.2%} | "
              f"aprovados: {(preds['prediction'] == 0).mean():.2%}")


def drift(batch: str) -> None:
    t = pd.read_csv(need(REPORTS / "drift" / f"{batch}_drift_table.csv"))
    title(f"DRIFT: {batch} (ordenado por PSI)")
    print(t[["feature", "psi", "psi_level", "test", "statistic", "p_value", "ref_mean",
             "cur_mean", "mean_change_pct", "drift_detected"]].round(4).to_string(index=False))
    print(f"\nFeatures com drift (PSI ≥ 0,10): {int(t['drift_detected'].sum())} de {len(t)}")


def quarentena() -> None:
    q = DATA / "quarantine"
    files = sorted(need(q).glob("*_failures.csv"))
    for f in files:
        fail = pd.read_csv(f)
        title(f"QUARENTENA: {f.name.replace('_failures.csv', '')}  ({len(fail)} violações)")
        print(fail.groupby(["column", "check"], dropna=False).size()
              .rename("violações").reset_index().to_string(index=False))
        print("\nExemplos:")
        print(fail[["column", "check", "failure_case", "index"]].head(10).to_string(index=False))


def alertas() -> None:
    a = jsonl("alerts.jsonl")
    title(f"ALERTAS ({len(a)}): logs/alerts.jsonl")
    print(a.groupby(["severity", "kind"]).size().rename("qtd").reset_index().to_string(index=False))
    print()
    print(a[["batch", "severity", "kind", "message"]].to_string(index=False))


def etapas() -> None:
    s = jsonl("stages.jsonl")
    title(f"ETAPAS DO PIPELINE ({len(s)} execuções): logs/stages.jsonl")
    print(s.groupby(["stage", "status"]).agg(execucoes=("status", "size"),
                                             latencia_media_ms=("duration_ms", "mean"),
                                             linhas=("rows_in", "sum")).round(1).to_string())
    print()
    print(s[["timestamp", "stage", "batch", "status", "duration_ms", "rows_in", "rows_out",
             "cpu_time_s", "memory_rss_mb"]].to_string(index=False))


def modelo() -> None:
    meta = json.loads(need(MODELS / "model_metadata.json").read_text(encoding="utf-8"))
    title("MODELO: models/model_metadata.json")
    print(f"Algoritmo: {meta['algorithm']} | versão (run): {meta['training_run_id']} | "
          f"threshold: {meta['threshold']:.3f}")
    print(f"Treino: {meta['train_rows']:,} linhas | Holdout: {meta['test_rows']:,} linhas")
    print(f"Atributos protegidos excluídos: {meta['protected_attributes_excluded']}")
    comp = pd.DataFrame({"XGBoost": meta["reference_metrics"],
                         "LogReg (baseline)": meta["baseline_logreg_metrics"]}).round(4)
    print("\n" + comp.to_string())
    print("\nFairness:", {k: round(v, 3) for k, v in meta["fairness"].items()})
    imp = pd.read_csv(need(MODELS / "feature_importance.csv"), index_col=0)
    title("IMPORTÂNCIA DAS FEATURES")
    print(imp.round(4).to_string())


def causa() -> None:
    s = json.loads(need(REPORTS / "monitoring_summary.json").read_text(encoding="utf-8"))
    rc = s["root_cause"]
    title(f"ANÁLISE CAUSAL: lote {rc['batch']}")
    for k, v in rc["summary"].items():
        print(f"{k:28s} {v:.4f}")
    print()
    print(pd.DataFrame(rc["features"]).round(4).to_string(index=False))


def visao_geral() -> None:
    h = pd.read_csv(need(REPORTS / "monitoring_history.csv"))
    title("VISÃO GERAL: reports/monitoring_history.csv")
    cols = [c for c in ["batch", "status", "contract_passed", "n_drifted", "prediction_psi",
                        "perf_auc", "perf_ks", "perf_default_rate", "mean_score", "fair_di_sex",
                        "fair_di_age_band", "alerts_critical", "alerts_warning"] if c in h]
    print(h[cols].round(3).to_string(index=False))
    print("\nDetalhe: python scripts/explore_data.py [referencia|producao|lote <mês>|drift <mês>|"
          "quarentena|alertas|etapas|modelo|causa]")


COMMANDS = {"referencia": referencia, "producao": producao, "quarentena": quarentena,
            "alertas": alertas, "etapas": etapas, "modelo": modelo, "causa": causa}

if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    if not args:
        visao_geral()
    elif args[0] in ("lote", "drift"):
        (lote if args[0] == "lote" else drift)(args[1] if len(args) > 1 else "2026-06")
    elif args[0] in COMMANDS:
        COMMANDS[args[0]]()
    else:
        sys.exit(__doc__)
