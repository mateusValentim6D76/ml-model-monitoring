"""Etapa 2: simulação da passagem do tempo em produção (6 meses).

Cenário macroeconômico (narrativa causal detalhada em docs/causal_analysis.md):

    * Inflação em alta: o valor das faturas (bill_amt) sobe e, com ele, a utilização do limite.
    * Juros em alta: aperto de crédito, com novos limites (limit_bal) menores.
    * Desemprego em alta: mais atrasos (pay_status) e pagamentos (pay_amt) menores.
    * Novo público: entrada de clientes jovens (age menor), sem histórico.
    Todos esses fatores aumentam a inadimplência (default).

Dois tipos de degradação são injetados, com intensidade crescente mês a mês:

* DATA DRIFT (covariate shift): P(X) muda: faturas, limites, atrasos, pagamentos, idade.
* CONCEPT DRIFT: P(y | X) muda: clientes com perfil "aparentemente bom" (sem atraso)
  mas com alta utilização passam a inadimplir, pois a renda real foi corroída pela
  inflação. Essa variável (renda real) NÃO é observada pelo modelo, então ele não
  consegue reagir, e a performance cai silenciosamente.

Saída (por mês):
    data/production/batches/<mês>.parquet          features recebidas para inferência (sem rótulo)
    data/production/batches/<mês>_labels.parquet   rótulos que chegam com atraso (label delay)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from credit_monitoring.config import load_settings
from credit_monitoring.monitoring.telemetry import log, record_metric, track_stage

BILL = [f"bill_amt_{i}" for i in range(1, 7)]
PAY_AMT = [f"pay_amt_{i}" for i in range(1, 7)]
RECENT_STATUS = ["pay_status_1", "pay_status_2", "pay_status_3"]


def month_names(start: str, n: int) -> list[str]:
    return [str(p) for p in pd.period_range(start=start, periods=n, freq="M")]


def apply_drift(batch: pd.DataFrame, intensity: float, shocks: dict, concept: dict,
                rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Aplica os choques ao lote. Retorna (lote alterado, log de choques aplicados)."""
    df = batch.copy().reset_index(drop=True)
    n = len(df)
    applied: dict[str, float] = {"intensity": intensity}
    if intensity <= 0:
        return df, applied

    # 1) Inflação: faturas mais altas (ruído individual em torno do choque médio)
    infl = shocks["inflation_bill_increase"] * intensity
    factor = 1 + infl * rng.normal(1.0, 0.25, size=n).clip(0.2, 1.8)
    for c in BILL:
        df[c] = (df[c] * factor).round().astype("int64")
    applied["bill_increase_mean"] = float(factor.mean() - 1)

    # 2) Aperto de crédito: limites menores (múltiplos de 10 mil, piso da política)
    tight = shocks["limit_tightening"] * intensity
    new_limit = df["limit_bal"] * (1 - tight * rng.uniform(0.6, 1.4, size=n))
    df["limit_bal"] = (np.round(new_limit / 10_000) * 10_000).clip(10_000, 1_000_000).astype("int64")
    applied["limit_tightening_mean"] = float(tight)

    # 3) Desemprego: parcela da base passa a atrasar +1/+2 meses nos meses recentes
    share = shocks["payment_delay_share"] * intensity
    hit = rng.random(n) < share
    bump = rng.integers(1, 3, size=n)
    for c in RECENT_STATUS:
        base = df[c].clip(lower=0)
        df.loc[hit, c] = (base[hit] + bump[hit]).clip(upper=8)
    applied["payment_delay_share"] = float(hit.mean())

    # 4) Capacidade de pagamento menor
    drop = shocks["payment_capacity_drop"] * intensity
    for c in PAY_AMT:
        df[c] = (df[c] * (1 - drop)).round().astype("int64")
    applied["payment_capacity_drop"] = float(drop)

    # 5) Novo público: clientes jovens (21-30), com limites iniciais baixos
    young_share = shocks["young_customers_share"] * intensity
    young = rng.random(n) < young_share
    df.loc[young, "age"] = rng.integers(21, 31, size=int(young.sum()))
    df.loc[young, "limit_bal"] = np.minimum(df.loc[young, "limit_bal"], 80_000)
    applied["young_customers_share"] = float(young.mean())

    # --- Rótulos -----------------------------------------------------------
    # Coerência causal: quem passou a atrasar (desemprego) tende a inadimplir.
    y = df["default"].to_numpy().copy()
    became_default = hit & (rng.random(n) < 0.45)
    y[became_default] = 1

    # CONCEPT DRIFT: adimplentes "limpos" (sem atraso) com alta utilização passam a
    # inadimplir, uma mudança em P(y|X) que o modelo não aprendeu.
    utilization = df["bill_amt_1"] / df["limit_bal"]
    clean_history = (df[RECENT_STATUS] <= 0).all(axis=1)
    candidates = (y == 0) & clean_history.to_numpy() & (utilization > concept["utilization_threshold"]).to_numpy()
    flip = candidates & (rng.random(n) < concept["flip_share"] * intensity * 2)
    y[flip] = 1
    df["default"] = y.astype("int64")
    applied["concept_flips"] = int(flip.sum())
    applied["delay_induced_defaults"] = int(became_default.sum())
    return df, applied


def simulate() -> list[str]:
    s = load_settings()
    sim = s.section("simulation")
    rng = np.random.default_rng(s.seed)
    pool = pd.read_parquet(s.path("production_pool")).sample(frac=1.0, random_state=s.seed)

    months = month_names(sim["start_month"], sim["n_batches"])
    out_dir = s.path("production_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    size = min(sim["batch_size"], len(pool) // sim["n_batches"])
    manifest = []

    for i, (month, intensity) in enumerate(zip(months, sim["intensity_schedule"], strict=True)):
        with track_stage("drift_simulation", batch=month) as st:
            raw = pool.iloc[i * size:(i + 1) * size]
            st.rows_in = len(raw)
            drifted, applied = apply_drift(raw, intensity, sim["shocks"], sim["concept_drift"], rng)
            features = drifted.drop(columns=[s.target])
            labels = drifted[["customer_id", s.target]]
            features.to_parquet(out_dir / f"{month}.parquet", index=False)
            labels.to_parquet(out_dir / f"{month}_labels.parquet", index=False)
            st.rows_out = len(drifted)
            st.detail.update(applied)
            record_metric("simulated_intensity", intensity, batch=month)
            record_metric("observed_default_rate", drifted[s.target].mean(), batch=month)
            manifest.append({"batch": month, "rows": len(drifted),
                             "default_rate": float(drifted[s.target].mean()), **applied})
            log.info(f"Lote {month}: intensidade={intensity:.2f} | default="
                     f"{drifted[s.target].mean():.2%} | flips de conceito={applied.get('concept_flips', 0)}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return months
