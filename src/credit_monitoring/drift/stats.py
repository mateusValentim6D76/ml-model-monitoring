"""Testes estatísticos de drift implementados do zero (auditáveis e independentes do Evidently).

Servem como *segunda opinião* (cross-check) dos resultados do Evidently AI:

* PSI (Population Stability Index): padrão de mercado em crédito / Basileia.
      PSI = Σ (p_cur - p_ref) · ln(p_cur / p_ref), bins definidos por quantis da referência.
      < 0,10 estável | 0,10-0,25 mudança moderada | > 0,25 mudança significativa.
* KS: Kolmogorov-Smirnov de 2 amostras (features numéricas contínuas).
* χ²: qui-quadrado de homogeneidade (features categóricas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

EPS = 1e-4


def psi(reference: pd.Series, current: pd.Series, bins: int = 10, categorical: bool = False) -> float:
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()
    if categorical:
        cats = sorted(set(ref.unique()) | set(cur.unique()))
        p_ref = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        p_cur = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:  # variável quase constante: trata como categórica
            return psi(ref, cur, categorical=True)
        edges[0], edges[-1] = -np.inf, np.inf
        p_ref = np.histogram(ref, bins=edges)[0] / len(ref)
        p_cur = np.histogram(cur, bins=edges)[0] / len(cur)
    p_ref = np.clip(p_ref, EPS, None)
    p_cur = np.clip(p_cur, EPS, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi_level(value: float, warning: float = 0.10, critical: float = 0.25) -> str:
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return "stable"


def drift_table(reference: pd.DataFrame, current: pd.DataFrame, numerical: list[str],
                categorical: list[str], psi_warning: float = 0.10, psi_critical: float = 0.25,
                alpha: float = 0.05) -> pd.DataFrame:
    """Tabela feature a feature com PSI + KS (numéricas) ou χ² (categóricas)."""
    rows = []
    for col in numerical + categorical:
        is_cat = col in categorical
        value = psi(reference[col], current[col], categorical=is_cat)
        if is_cat:
            cats = sorted(set(reference[col]) | set(current[col]))
            table = np.array([reference[col].value_counts().reindex(cats, fill_value=0),
                              current[col].value_counts().reindex(cats, fill_value=0)])
            table = table[:, table.sum(axis=0) > 0]
            stat, pvalue, *_ = chi2_contingency(table)
            test = "chi2"
        else:
            stat, pvalue = ks_2samp(reference[col], current[col])
            test = "ks"
        rows.append({
            "feature": col, "type": "categorical" if is_cat else "numerical",
            "psi": value, "psi_level": psi_level(value, psi_warning, psi_critical),
            "test": test, "statistic": float(stat), "p_value": float(pvalue),
            "stat_drift": bool(pvalue < alpha),
            "ref_mean": float(reference[col].mean()), "cur_mean": float(current[col].mean()),
        })
    out = pd.DataFrame(rows)
    out["mean_change_pct"] = (out["cur_mean"] - out["ref_mean"]) / out["ref_mean"].abs().replace(0, np.nan)
    # Decisão final: PSI é o critério de negócio; o teste de hipótese confirma
    out["drift_detected"] = out["psi"] >= psi_warning
    return out.sort_values("psi", ascending=False).reset_index(drop=True)
