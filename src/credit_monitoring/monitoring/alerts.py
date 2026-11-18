"""Regras de alerta do monitoramento (thresholds na seção monitoring de config/config.yaml)."""

from __future__ import annotations

import pandas as pd

from credit_monitoring.monitoring.telemetry import raise_alert


def evaluate_batch_alerts(batch: str, drift: pd.DataFrame, dataset_drift_share: float,
                          prediction_psi: float, perf: dict | None, reference_perf: dict,
                          fairness: dict | None, cfg: dict) -> list[dict]:
    alerts: list[dict] = []

    # 1) Drift por feature (PSI)
    crit = drift[drift["psi"] >= cfg["psi_critical"]]
    warn = drift[(drift["psi"] >= cfg["psi_warning"]) & (drift["psi"] < cfg["psi_critical"])]
    if not crit.empty:
        alerts.append(raise_alert(
            "critical", "feature_drift",
            f"{len(crit)} feature(s) com PSI >= {cfg['psi_critical']}: "
            + ", ".join(f"{r.feature} ({r.psi:.2f})" for r in crit.itertuples()),
            batch=batch, features=crit["feature"].tolist()))
    if not warn.empty:
        alerts.append(raise_alert(
            "warning", "feature_drift",
            f"{len(warn)} feature(s) com PSI entre {cfg['psi_warning']} e {cfg['psi_critical']}: "
            + ", ".join(warn["feature"]), batch=batch, features=warn["feature"].tolist()))

    # 2) Drift do dataset como um todo
    if dataset_drift_share >= cfg["dataset_drift_share"]:
        alerts.append(raise_alert(
            "critical", "dataset_drift",
            f"{dataset_drift_share:.0%} das features derivaram (limite {cfg['dataset_drift_share']:.0%}).",
            batch=batch, share=dataset_drift_share))

    # 3) Drift do score (prediction drift: sinal antecipado, não depende de rótulo)
    if prediction_psi >= cfg["prediction_psi_critical"]:
        alerts.append(raise_alert(
            "critical", "prediction_drift",
            f"Distribuição do score mudou (PSI={prediction_psi:.2f}).", batch=batch, psi=prediction_psi))

    # 4) Performance (quando os rótulos chegam)
    if perf:
        drop = reference_perf["auc"] - perf["auc"]
        if drop >= cfg["auc_drop_critical"]:
            sev = "critical"
        elif drop >= cfg["auc_drop_warning"]:
            sev = "warning"
        else:
            sev = None
        if sev:
            alerts.append(raise_alert(
                sev, "performance_degradation",
                f"AUC caiu {drop:.3f} (de {reference_perf['auc']:.3f} para {perf['auc']:.3f}).",
                batch=batch, auc=perf["auc"], auc_drop=drop))
        dr_change = perf["default_rate"] - reference_perf["default_rate"]
        if dr_change >= cfg["default_rate_change_critical"]:
            alerts.append(raise_alert(
                "critical", "default_rate_increase",
                f"Taxa de inadimplência subiu {dr_change * 100:.1f} p.p. "
                f"(de {reference_perf['default_rate']:.1%} para {perf['default_rate']:.1%}).",
                batch=batch, default_rate=perf["default_rate"]))

    # 5) Fairness (regra dos 4/5)
    if fairness:
        for attr in ("sex", "age_band"):
            di = fairness.get(f"di_{attr}")
            if di is not None and di < cfg["fairness_di_min"]:
                alerts.append(raise_alert(
                    "warning", "fairness",
                    f"Disparate Impact em '{attr}' = {di:.2f} (< {cfg['fairness_di_min']}).",
                    batch=batch, attribute=attr, disparate_impact=di))
    return alerts
