"""Análise causal do drift: atribuição de causa raiz por intervenção.

Pergunta causal: "Quanto da mudança no comportamento do modelo é CAUSADA pela
mudança na distribuição de cada variável?"

Método (intervenção na feature X_f, trocando sua distribuição pela de referência via quantile mapping):
    para cada feature f com drift:
        1. "reparamos" f no lote de produção, levando cada valor para o mesmo
           percentil da distribuição de referência (preserva o ranking do cliente);
        2. re-pontuamos o lote com o modelo, mantendo todo o resto fixo;
        3. o quanto o score médio volta para o patamar de referência é o efeito
           causal (no modelo) atribuído a f.

Decomposição da perda de performance:
    ΔAUC_total     = AUC_ref - AUC_prod
    ΔAUC_covariate = AUC_reparado_total - AUC_prod   (parcela recuperável ao desfazer o data drift)
    ΔAUC_concept   = AUC_ref - AUC_reparado_total     (resíduo: mudança em P(y|X), ou seja, concept drift)

Limitação (documentada): é uma intervenção sobre o *modelo*, não sobre o mundo.
Ela identifica quais entradas movem o score, não prova a causa econômica, que
vem do DAG de domínio em docs/causal_analysis.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def quantile_map(current: pd.Series, reference: pd.Series) -> np.ndarray:
    """Leva cada valor de `current` ao mesmo percentil na distribuição de `reference`."""
    ranks = current.rank(method="average", pct=True).to_numpy()
    return np.quantile(reference.to_numpy(), np.clip(ranks, 0, 1))


def root_cause_analysis(model_bundle: dict, reference: pd.DataFrame, current: pd.DataFrame,
                        drifted_features: list[str], target: str = "default") -> tuple[pd.DataFrame, dict]:
    model, feats = model_bundle["model"], model_bundle["features"]

    def score(frame: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(frame[feats])[:, 1]

    p_ref = score(reference)
    p_cur = score(current)
    gap = p_cur.mean() - p_ref.mean()
    has_y = target in current.columns and target in reference.columns
    auc_ref = roc_auc_score(reference[target], p_ref) if has_y else np.nan
    auc_cur = roc_auc_score(current[target], p_cur) if has_y else np.nan

    rows = []
    for f in drifted_features:
        repaired = current.copy()
        mapped = quantile_map(current[f], reference[f])
        repaired[f] = np.round(mapped).astype(current[f].dtype)
        p_rep = score(repaired)
        rows.append({
            "feature": f,
            "score_shift_caused": float(p_cur.mean() - p_rep.mean()),
            "share_of_score_drift": float((p_cur.mean() - p_rep.mean()) / gap) if abs(gap) > 1e-9 else np.nan,
            "auc_after_repair": float(roc_auc_score(current[target], p_rep)) if has_y else np.nan,
        })
    table = pd.DataFrame(rows, columns=["feature", "score_shift_caused", "share_of_score_drift",
                                        "auc_after_repair"])
    table = table.sort_values("score_shift_caused", key=np.abs, ascending=False)

    # Reparo conjunto de todas as features com drift
    repaired_all = current.copy()
    for f in drifted_features:
        repaired_all[f] = np.round(quantile_map(current[f], reference[f])).astype(current[f].dtype)
    p_all = score(repaired_all)
    auc_all = roc_auc_score(current[target], p_all) if has_y else np.nan

    summary = {
        "mean_score_reference": float(p_ref.mean()),
        "mean_score_production": float(p_cur.mean()),
        "mean_score_all_repaired": float(p_all.mean()),
        "auc_reference": float(auc_ref),
        "auc_production": float(auc_cur),
        "auc_all_repaired": float(auc_all),
        "auc_drop_total": float(auc_ref - auc_cur),
        "auc_drop_covariate": float(auc_all - auc_cur),
        "auc_drop_concept": float(auc_ref - auc_all),
    }
    return table.reset_index(drop=True), summary
