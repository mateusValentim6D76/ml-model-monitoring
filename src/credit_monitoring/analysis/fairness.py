"""Auditoria de vieses (LGPD art. 6º, IX: não discriminação; art. 20: revisão de decisões).

Os atributos protegidos NÃO são features do modelo, mas um modelo pode discriminar
indiretamente via proxies. Por isso medimos, por grupo:

* approval_rate        taxa de aprovação (score abaixo do threshold)
* disparate impact     menor taxa de aprovação / maior taxa de aprovação (pela regra dos 4/5, deve ficar >= 0,8)
* TPR / FPR            igualdade de oportunidade (equalized odds), quando há rótulo
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_monitoring.data.privacy import age_band

MIN_GROUP_SIZE = 100

GROUP_LABELS = {
    "sex": {1: "masculino", 2: "feminino"},
}


def group_metrics(df: pd.DataFrame, proba: np.ndarray, threshold: float, group_col: str,
                  target: str | None = "default") -> pd.DataFrame:
    work = pd.DataFrame({"group": df[group_col].values, "proba": proba})
    work["approved"] = (work["proba"] < threshold).astype(int)
    has_target = target is not None and target in df.columns
    if has_target:
        work["y"] = df[target].values

    rows = []
    for g, part in work.groupby("group", observed=True):
        row = {"attribute": group_col, "group": GROUP_LABELS.get(group_col, {}).get(g, str(g)),
               "n": len(part), "approval_rate": part["approved"].mean(),
               "mean_score": part["proba"].mean()}
        if has_target:
            bad, good = part[part["y"] == 1], part[part["y"] == 0]
            row["default_rate"] = part["y"].mean()
            row["tpr"] = (1 - bad["approved"]).mean() if len(bad) else np.nan   # maus barrados
            row["fpr"] = (1 - good["approved"]).mean() if len(good) else np.nan  # bons negados
        rows.append(row)
    out = pd.DataFrame(rows)
    out["disparate_impact"] = out["approval_rate"] / out["approval_rate"].max()
    return out


def fairness_report(df: pd.DataFrame, proba: np.ndarray, threshold: float,
                    target: str | None = "default") -> tuple[pd.DataFrame, dict[str, float]]:
    work = df.copy()
    work["age_band"] = age_band(work["age"])
    tables = [group_metrics(work, proba, threshold, col, target) for col in ("sex", "age_band")]
    table = pd.concat(tables, ignore_index=True)
    summary = {}
    # Grupos pequenos geram DI instável (ruído amostral), então só entram no indicador
    # grupos com pelo menos MIN_GROUP_SIZE registros (continuam visíveis na tabela).
    table["in_di"] = table["n"] >= MIN_GROUP_SIZE
    for attr, part in table[table["in_di"]].groupby("attribute"):
        summary[f"di_{attr}"] = float(part["approval_rate"].min() / part["approval_rate"].max())
        if "tpr" in part and part["tpr"].notna().any():
            summary[f"eo_gap_{attr}"] = float(part["tpr"].max() - part["tpr"].min())
    return table, summary
