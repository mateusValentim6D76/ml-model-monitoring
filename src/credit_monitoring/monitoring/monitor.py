"""Etapas 2 e 3: execução do monitoramento contínuo, lote a lote.

Para cada mês de produção:

    1. data contract: bloqueia lotes inválidos (quarentena + alerta)
    2. inferência: score + decisão, log de predições
    3. data drift: Evidently (PSI/KS por feature) + cross-check PSI/KS/χ² próprio
    4. prediction drift (PSI do score): sinal antecipado, sem precisar de rótulo
    5. performance: quando os rótulos chegam (label delay): AUC, KS, F1, default rate
    6. fairness: disparate impact por sexo e faixa etária
    7. alertas: regras de config/config.yaml
    8. MLflow: séries temporais por lote (step) + relatórios como artefatos

Ao final: análise causal (root cause) do lote mais recente, resumo JSON e dashboard.
"""

from __future__ import annotations

import json

import pandas as pd

from credit_monitoring.analysis.fairness import fairness_report
from credit_monitoring.analysis.root_cause import root_cause_analysis
from credit_monitoring.config import load_settings
from credit_monitoring.contracts.corrupt import make_corrupted_batch
from credit_monitoring.contracts.validate import validate_batch
from credit_monitoring.drift.detect import run_evidently_drift, run_evidently_full_report
from credit_monitoring.drift.simulate import month_names
from credit_monitoring.drift.stats import drift_table, psi
from credit_monitoring.models.metrics import classification_metrics
from credit_monitoring.models.train import load_model, predict
from credit_monitoring.monitoring.alerts import evaluate_batch_alerts
from credit_monitoring.monitoring.mlflow_tracking import setup_mlflow
from credit_monitoring.monitoring.telemetry import RUN_ID, log, record_metric, track_stage

INCIDENT_AFTER = 2  # o lote corrompido "chega" depois do 3º mês


def run_monitoring(evidently_every_batch: bool = True) -> dict:
    s = load_settings()
    mcfg = s.section("monitoring")
    sim = s.section("simulation")
    bundle = load_model()
    ref_perf = bundle["metadata"]["reference_metrics"]
    threshold = bundle["threshold"]

    reference = pd.read_parquet(s.path("reference_test").with_name("reference_scored.parquet"))
    batch_dir = s.path("production_dir")
    pred_dir = batch_dir.parent / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    drift_dir = s.path("reports_dir") / "drift"
    drift_dir.mkdir(parents=True, exist_ok=True)

    months = month_names(sim["start_month"], sim["n_batches"])
    history: list[dict] = []
    drift_tables: dict[str, pd.DataFrame] = {}
    last_scored: pd.DataFrame | None = None

    queue: list[tuple[str, pd.DataFrame]] = []
    for i, m in enumerate(months):
        queue.append((m, pd.read_parquet(batch_dir / f"{m}.parquet")))
        if i == INCIDENT_AFTER:
            corrupted = make_corrupted_batch(pd.read_parquet(batch_dir / f"{m}.parquet"), seed=s.seed)
            queue.append((f"{m}-incident", corrupted))

    for step, (batch, df) in enumerate(queue):
        row: dict = {"batch": batch, "step": step, "rows_received": len(df)}

        # 1) Data contract ------------------------------------------------------
        result = validate_batch(df, batch=batch, require_target=False)
        row.update(contract_passed=result.passed, contract_violations=len(result.failure_cases),
                   contract_report=result.report_path.name if result.report_path else None)
        if not result.passed:
            row["status"] = "blocked"
            history.append(row)
            continue

        # 2) Inferência ------------------------------------------------------------
        with track_stage("inference", batch=batch) as st:
            st.rows_in = len(df)
            proba, pred = predict(bundle, df)
            scored = df.copy()
            scored["prediction_proba"], scored["prediction"] = proba, pred
            scored[["customer_id", "prediction_proba", "prediction"]].assign(
                model_version=bundle["metadata"]["training_run_id"], batch=batch
            ).to_parquet(pred_dir / f"{batch}.parquet", index=False)
            st.rows_out = len(scored)
        record_metric("rows_scored", len(scored), batch=batch)
        record_metric("approval_rate", float(1 - pred.mean()), batch=batch)

        # 3) Data drift: cross-check estatístico próprio ---------------------------
        with track_stage("drift_detection", batch=batch) as st:
            st.rows_in = len(scored)
            table = drift_table(reference, scored, s.numerical, s.categorical,
                                mcfg["psi_warning"], mcfg["psi_critical"], mcfg["ks_pvalue"])
            table.to_csv(drift_dir / f"{batch}_drift_table.csv", index=False)
            drift_tables[batch] = table
            share = float(table["drift_detected"].mean())
            pred_psi = psi(reference["prediction_proba"], scored["prediction_proba"])

            # Evidently AI (relatório HTML oficial)
            ev = run_evidently_drift(reference, scored, s.numerical, s.categorical,
                                     drift_dir / f"evidently_drift_{batch}.html",
                                     method=mcfg["drift_method_numerical"],
                                     threshold=mcfg["psi_warning"],
                                     save_html=evidently_every_batch or step == len(queue) - 1)
            st.rows_out = len(scored)
            st.detail.update(drifted_features=int(table["drift_detected"].sum()),
                             evidently_drifted=ev.get("n_drifted"))
        row.update(drift_share=share, n_drifted=int(table["drift_detected"].sum()),
                   drifted_features=table.loc[table["drift_detected"], "feature"].tolist(),
                   evidently_drift_share=ev.get("share"), evidently_drifted=ev.get("drifted", []),
                   prediction_psi=pred_psi, mean_score=float(proba.mean()))
        for r in table.itertuples():
            record_metric("psi", r.psi, batch=batch, feature=r.feature)
        record_metric("dataset_drift_share", share, batch=batch)
        record_metric("prediction_psi", pred_psi, batch=batch)

        # 4) Rótulos chegam (label delay) e a performance é calculada------------------------
        perf = None
        labels_path = batch_dir / f"{batch}_labels.parquet"
        if labels_path.exists():
            with track_stage("performance_evaluation", batch=batch) as st:
                labels = pd.read_parquet(labels_path)
                scored = scored.merge(labels, on="customer_id", how="left", validate="one_to_one")
                st.rows_in = len(scored)
                perf = classification_metrics(scored[s.target], scored["prediction_proba"], threshold)
                st.rows_out = int(scored[s.target].notna().sum())
            row.update({f"perf_{k}": v for k, v in perf.items()})
            for k in ("auc", "ks", "f1", "recall", "precision", "default_rate", "brier"):
                record_metric(f"perf_{k}", perf[k], batch=batch)

        # 5) Fairness ---------------------------------------------------------------
        fair_table, fair = fairness_report(scored, proba, threshold,
                                           target=s.target if perf else None)
        fair_table.assign(batch=batch).to_csv(drift_dir / f"{batch}_fairness.csv", index=False)
        row.update({f"fair_{k}": v for k, v in fair.items()})
        for k, v in fair.items():
            record_metric(f"fairness_{k}", v, batch=batch)

        # 6) Alertas ----------------------------------------------------------------
        alerts = evaluate_batch_alerts(batch, table, share, pred_psi, perf, ref_perf, fair, mcfg)
        row["alerts_critical"] = sum(a["severity"] == "critical" for a in alerts)
        row["alerts_warning"] = sum(a["severity"] == "warning" for a in alerts)
        row["status"] = "degraded" if row["alerts_critical"] else (
            "attention" if row["alerts_warning"] else "healthy")
        history.append(row)
        last_scored, last_batch = scored, batch

    # 7) Relatório Evidently completo (drift + qualidade de classificação) ----------
    if last_scored is not None:
        with track_stage("evidently_full_report", batch=last_batch):
            run_evidently_full_report(reference, last_scored, s.numerical, s.categorical, s.target,
                                      drift_dir / f"evidently_full_{last_batch}.html")

    # 8) Análise causal do lote mais recente ---------------------------------------
    rca_table, rca_summary = pd.DataFrame(), {}
    if last_scored is not None:
        with track_stage("root_cause_analysis", batch=last_batch):
            drifted = drift_tables[last_batch].query("drift_detected")["feature"].tolist()
            rca_table, rca_summary = root_cause_analysis(bundle, reference, last_scored, drifted,
                                                          target=s.target)
            rca_table.to_csv(drift_dir / f"{last_batch}_root_cause.csv", index=False)
            log.info(f"Causa raiz ({last_batch}): ΔAUC total={rca_summary['auc_drop_total']:.3f} | "
                     f"covariate={rca_summary['auc_drop_covariate']:.3f} | "
                     f"concept={rca_summary['auc_drop_concept']:.3f}")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(s.path("reports_dir") / "monitoring_history.csv", index=False)
    psi_matrix = pd.DataFrame({b: t.set_index("feature")["psi"] for b, t in drift_tables.items()})
    psi_matrix.to_csv(drift_dir / "psi_matrix.csv")

    summary = {
        "run_id": RUN_ID,
        "model_version": bundle["metadata"]["training_run_id"],
        "reference_metrics": ref_perf,
        "batches": history,
        "root_cause": {"batch": last_batch if last_scored is not None else None,
                       "summary": rca_summary, "features": rca_table.to_dict(orient="records")},
    }
    (s.path("reports_dir") / "monitoring_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    _log_monitoring_to_mlflow(hist_df, psi_matrix)
    return summary


def _log_monitoring_to_mlflow(hist: pd.DataFrame, psi_matrix: pd.DataFrame) -> None:
    s = load_settings()
    try:
        mlflow = setup_mlflow("monitoring_experiment")
        with mlflow.start_run(run_name=f"monitoring-{RUN_ID}"):
            mlflow.set_tags({"pipeline_run_id": RUN_ID, "stage": "monitoring",
                             "batches": ",".join(hist["batch"])})
            for r in hist.to_dict(orient="records"):
                step = int(r["step"])
                mlflow.log_metric("contract_passed", float(r["contract_passed"]), step=step)
                mlflow.log_metric("rows_received", float(r["rows_received"]), step=step)
                for k in ("drift_share", "prediction_psi", "mean_score", "perf_auc", "perf_ks",
                          "perf_f1", "perf_default_rate", "fair_di_sex", "fair_di_age_band",
                          "alerts_critical", "alerts_warning"):
                    v = r.get(k)
                    if v is not None and pd.notna(v):
                        mlflow.log_metric(k, float(v), step=step)
            for batch in psi_matrix.columns:
                step = int(hist.loc[hist["batch"] == batch, "step"].iloc[0])
                for feat, v in psi_matrix[batch].items():
                    mlflow.log_metric(f"psi_{feat}", float(v), step=step)
            reports = s.path("reports_dir")
            mlflow.log_artifact(str(reports / "monitoring_history.csv"))
            mlflow.log_artifact(str(reports / "monitoring_summary.json"))
            mlflow.log_artifacts(str(reports / "drift"), artifact_path="drift")
            mlflow.log_artifacts(str(reports / "data_contracts"), artifact_path="data_contracts")
            for f in ("stages.jsonl", "alerts.jsonl", "metrics.jsonl"):
                p = s.path("logs_dir") / f
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="logs")
        log.info("Monitoramento registrado no MLflow (experimento credit-scoring-monitoring).")
    except Exception as exc:
        log.warning(f"MLflow indisponível ({type(exc).__name__}: {exc}). Seguindo sem tracking.")
