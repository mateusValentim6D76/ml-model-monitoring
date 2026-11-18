"""Etapa 1: treino do classificador base (Dataset de Referência).

* O dataset de referência passa pelo MESMO data contract da produção antes do treino.
* Treinamos um baseline interpretável (Regressão Logística) e o campeão (XGBoost).
* O threshold de decisão é escolhido no holdout (máximo F1) e versionado junto ao modelo.
* Tudo é rastreado no MLflow (params, métricas, fairness, artefatos e Model Registry).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from credit_monitoring.analysis.fairness import fairness_report
from credit_monitoring.config import load_settings
from credit_monitoring.contracts.validate import validate_batch
from credit_monitoring.models.metrics import best_f1_threshold, classification_metrics
from credit_monitoring.monitoring.mlflow_tracking import setup_mlflow
from credit_monitoring.monitoring.telemetry import RUN_ID, log, record_metric, track_stage

MODEL_FILE = "credit_model.joblib"


def train() -> dict:
    s = load_settings()
    train_df = pd.read_parquet(s.path("reference_data"))
    test_df = pd.read_parquet(s.path("reference_test"))

    # Quality gate também no treino: referência "limpa" é pré-requisito
    for name, frame in [("reference_train", train_df), ("reference_test", test_df)]:
        validate_batch(frame, batch=name, require_target=True, raise_on_failure=True)

    X_train, y_train = train_df[s.features], train_df[s.target]
    X_test, y_test = test_df[s.features], test_df[s.target]

    with track_stage("model_training") as st:
        st.rows_in = len(train_df)
        # Baseline interpretável
        logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        logreg.fit(X_train, y_train)
        p_lr = logreg.predict_proba(X_test)[:, 1]

        # Campeão
        params = dict(s.section("model")["params"])
        pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
        xgb = XGBClassifier(**params, random_state=s.seed, n_jobs=-1, tree_method="hist")
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        p_xgb = xgb.predict_proba(X_test)[:, 1]

        thr_cfg = s.section("model")["decision_threshold"]
        threshold = best_f1_threshold(y_test, p_xgb) if thr_cfg == "auto" else float(thr_cfg)
        m_lr = classification_metrics(y_test, p_lr, best_f1_threshold(y_test, p_lr))
        m_xgb = classification_metrics(y_test, p_xgb, threshold)
        st.rows_out = len(test_df)
        st.detail.update(auc=round(m_xgb["auc"], 4), threshold=threshold)

    fair_table, fair_summary = fairness_report(test_df, p_xgb, threshold)
    importance = (pd.Series(xgb.feature_importances_, index=s.features)
                  .sort_values(ascending=False))

    log.info(f"Baseline LogReg  AUC={m_lr['auc']:.4f} KS={m_lr['ks']:.4f}")
    log.info(f"XGBoost (prod.)  AUC={m_xgb['auc']:.4f} KS={m_xgb['ks']:.4f} "
             f"F1={m_xgb['f1']:.4f} threshold={threshold:.3f}")
    log.info(f"Fairness: DI sexo={fair_summary['di_sex']:.3f} | DI faixa etária="
             f"{fair_summary['di_age_band']:.3f}")
    for k, v in m_xgb.items():
        record_metric(f"reference_{k}", v, batch="reference")

    # ------------------------------------------------------------------ artefatos
    model_dir = s.path("model_dir")
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": s.section("mlflow")["registered_model_name"],
        "algorithm": "XGBClassifier",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "training_run_id": RUN_ID,
        "features": s.features,
        "protected_attributes_excluded": s.protected,
        "threshold": threshold,
        "reference_metrics": m_xgb,
        "baseline_logreg_metrics": m_lr,
        "fairness": fair_summary,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "scale_pos_weight_observed": pos_weight,
    }
    bundle = {"model": xgb, "features": s.features, "threshold": threshold, "metadata": metadata}
    joblib.dump(bundle, model_dir / MODEL_FILE)
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    importance.rename("importance").to_csv(model_dir / "feature_importance.csv")
    fair_table.to_csv(model_dir / "fairness_reference.csv", index=False)

    # Referência pontuada (holdout), usada como base de comparação do monitoramento
    scored = test_df.copy()
    scored["prediction_proba"] = p_xgb
    scored["prediction"] = (p_xgb >= threshold).astype(int)
    scored.to_parquet(s.path("reference_test").with_name("reference_scored.parquet"), index=False)

    _log_to_mlflow(xgb, X_train, params, m_xgb, m_lr, fair_summary, threshold, model_dir)
    return metadata


def _log_to_mlflow(model, X_train, params, m_xgb, m_lr, fair_summary, threshold, model_dir) -> None:
    s = load_settings()
    try:
        mlflow = setup_mlflow("training_experiment")
        from mlflow import xgboost as mlflow_xgb
        from mlflow.models import infer_signature

        with mlflow.start_run(run_name=f"train-xgb-{RUN_ID}"):
            mlflow.set_tags({
                "pipeline_run_id": RUN_ID, "stage": "training", "dataset": "UCI default credit card",
                "data_contract": "credit_scoring_contract_v1",
                "governance.lgpd_legal_basis": "art. 7º, X (proteção do crédito)",
                "governance.protected_attributes_excluded": ",".join(s.protected),
                "governance.pii": "customer_id pseudonimizado (HMAC-SHA256)",
            })
            mlflow.log_params({**params, "threshold": round(threshold, 4),
                               "n_features": len(s.features)})
            mlflow.log_metrics({f"test_{k}": v for k, v in m_xgb.items()})
            mlflow.log_metrics({f"baseline_logreg_{k}": v for k, v in m_lr.items()})
            mlflow.log_metrics({f"fairness_{k}": v for k, v in fair_summary.items()})
            for f in ["model_metadata.json", "feature_importance.csv", "fairness_reference.csv"]:
                mlflow.log_artifact(str(model_dir / f))
            example = X_train.head(5)
            signature = infer_signature(example, model.predict_proba(example)[:, 1])
            kwargs = dict(input_example=example, signature=signature,
                          registered_model_name=s.section("mlflow")["registered_model_name"])
            try:
                mlflow_xgb.log_model(model, name="model", **kwargs)          # MLflow >= 3
            except TypeError:
                mlflow_xgb.log_model(model, artifact_path="model", **kwargs)  # MLflow 2.x
        log.info("Treino registrado no MLflow (experimento credit-scoring-training).")
    except Exception as exc:  # observabilidade nunca derruba o pipeline principal
        log.warning(f"MLflow indisponível ({type(exc).__name__}: {exc}). Seguindo sem tracking.")


def load_model() -> dict:
    return joblib.load(load_settings().path("model_dir") / MODEL_FILE)


def predict(bundle: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    proba = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
    return proba, (proba >= bundle["threshold"]).astype(int)
