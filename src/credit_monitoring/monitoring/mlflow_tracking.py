"""Integração com MLflow: rastreamento do ciclo de vida do modelo e do monitoramento.

* Experimento `credit-scoring-training`: params, métricas, artefatos e o modelo
  (registrado no Model Registry com tags de governança).
* Experimento `credit-scoring-monitoring`: um run por execução do monitoramento,
  com métricas por lote em `step` (série temporal: drift share, PSI, AUC, latência etc.)
  e os relatórios Evidently/contratos anexados como artefatos.

Backend: SQLite local (`mlflow.db`). Para visualizar:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import os
import warnings

from credit_monitoring.config import PROJECT_ROOT, load_settings


def setup_mlflow(experiment_key: str):
    import mlflow

    warnings.filterwarnings("ignore", module="mlflow")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    cfg = load_settings().section("mlflow")
    uri = cfg["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        db_path = PROJECT_ROOT / uri.removeprefix("sqlite:///")
        uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(uri)
    name = cfg[experiment_key]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(PROJECT_ROOT / "mlartifacts" / name).as_uri())
    mlflow.set_experiment(name)
    return mlflow
