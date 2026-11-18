"""credit_monitoring: camada de sustentação e confiabilidade de um modelo de Credit Scoring.

Módulos:
    data          ingestão, limpeza, pseudonimização e separação referência/produção
    contracts     data contracts (Pandera) e quality gate de ingestão
    models        treino do baseline (XGBoost) e inferência
    drift         simulação de drift macroeconômico e detecção estatística (Evidently + PSI/KS)
    analysis      análise causal (root cause por intervenção) e auditoria de fairness
    monitoring    logs estruturados, métricas, alertas, MLflow e dashboard HTML
"""

__version__ = "1.0.0"
