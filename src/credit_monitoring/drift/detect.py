"""Detecção de drift com Evidently AI (API 0.7+).

* `run_evidently_drift`: relatório de Data Drift por lote (PSI por feature + drift do score).
* `run_evidently_full_report`: relatório completo do lote mais recente: Data Drift com o
  teste estatístico padrão do Evidently (KS / χ² / Wasserstein, escolhido por tipo e tamanho
  da amostra) + qualidade da classificação (Concept Drift: AUC, matriz de confusão, calibração).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", module="evidently")

PRED_PROBA = "prediction_proba"
PRED_LABEL = "prediction"


def _definition(numerical: list[str], categorical: list[str], target: str | None = None):
    from evidently import BinaryClassification, DataDefinition

    classification = None
    if target:
        classification = [BinaryClassification(target=target, prediction_labels=PRED_LABEL,
                                                prediction_probas=PRED_PROBA)]
    return DataDefinition(numerical_columns=numerical + [PRED_PROBA],
                          categorical_columns=categorical + ([PRED_LABEL] if target else []),
                          classification=classification)


def _datasets(reference: pd.DataFrame, current: pd.DataFrame, cols: list[str], definition):
    from evidently import Dataset

    return (Dataset.from_pandas(current[cols].reset_index(drop=True), data_definition=definition),
            Dataset.from_pandas(reference[cols].reset_index(drop=True), data_definition=definition))


def _parse(snapshot) -> dict:
    """Extrai do snapshot: share/contagem de colunas com drift e o valor do teste por coluna."""
    out: dict = {"columns": {}}
    for m in snapshot.dict()["metrics"]:
        cfg, value = m.get("config", {}), m.get("value")
        if cfg.get("type", "").endswith("DriftedColumnsCount"):
            out["n_drifted"] = int(value["count"])
            out["share"] = float(value["share"])
        elif cfg.get("type", "").endswith("ValueDrift"):
            out["columns"][cfg["column"]] = {"value": float(value), "method": cfg.get("method"),
                                             "threshold": cfg.get("threshold")}
    return out


def run_evidently_drift(reference: pd.DataFrame, current: pd.DataFrame, numerical: list[str],
                        categorical: list[str], out_path: Path, method: str = "psi",
                        threshold: float = 0.1, save_html: bool = True) -> dict:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    cols = numerical + categorical + [PRED_PROBA]
    definition = _definition(numerical, categorical)
    cur, ref = _datasets(reference, current, cols, definition)
    report = Report([DataDriftPreset(columns=cols, method=method, threshold=threshold,
                                     drift_share=0.3)])
    snapshot = report.run(current_data=cur, reference_data=ref)
    if save_html:
        snapshot.save_html(str(out_path))

    parsed = _parse(snapshot)
    feats = [c for c in cols if c != PRED_PROBA]
    drifted = [c for c in feats if parsed["columns"].get(c, {}).get("value", 0) >= threshold]
    parsed.update(drifted=drifted, n_drifted=len(drifted), share=len(drifted) / len(feats),
                  prediction_drift=parsed["columns"].get(PRED_PROBA, {}).get("value"))
    return parsed


def run_evidently_full_report(reference: pd.DataFrame, current: pd.DataFrame, numerical: list[str],
                              categorical: list[str], target: str, out_path: Path) -> dict:
    from evidently import Report
    from evidently.presets import ClassificationPreset, DataDriftPreset

    cols = numerical + categorical + [PRED_PROBA, PRED_LABEL, target]
    definition = _definition(numerical, categorical, target=target)
    cur, ref = _datasets(reference, current, cols, definition)
    report = Report([DataDriftPreset(columns=numerical + categorical, drift_share=0.3),
                     ClassificationPreset()])
    snapshot = report.run(current_data=cur, reference_data=ref)
    snapshot.save_html(str(out_path))
    return _parse(snapshot)
