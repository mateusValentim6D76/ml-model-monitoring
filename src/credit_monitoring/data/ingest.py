"""Etapa 0 — ingestão do dataset público, limpeza e separação Referência × Produção.

Dataset: *Default of Credit Card Clients* (Yeh & Lien, 2009) — 30.000 clientes de
um emissor de cartão de crédito, alvo binário = inadimplência no mês seguinte.
Disponível no UCI ML Repository e espelhado no OpenML (id 42477).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_monitoring.config import load_settings
from credit_monitoring.data.privacy import pseudonymize_column
from credit_monitoring.monitoring.telemetry import log, record_metric, track_stage

# Ordem oficial das colunas no arquivo da UCI (X1..X23, Y)
UCI_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

RENAME = {
    "LIMIT_BAL": "limit_bal", "SEX": "sex", "EDUCATION": "education",
    "MARRIAGE": "marriage", "AGE": "age",
    "PAY_0": "pay_status_1", "PAY_2": "pay_status_2", "PAY_3": "pay_status_3",
    "PAY_4": "pay_status_4", "PAY_5": "pay_status_5", "PAY_6": "pay_status_6",
    **{f"BILL_AMT{i}": f"bill_amt_{i}" for i in range(1, 7)},
    **{f"PAY_AMT{i}": f"pay_amt_{i}" for i in range(1, 7)},
}


def download() -> pd.DataFrame:
    """Baixa o dataset do OpenML (cache local em data/raw/openml) e salva o CSV bruto."""
    from sklearn.datasets import fetch_openml

    s = load_settings()
    raw_path = s.path("raw_data")
    if raw_path.exists():
        log.info(f"Dataset bruto já existe em {raw_path.relative_to(raw_path.parents[2])}")
        return pd.read_csv(raw_path)

    with track_stage("ingest_download") as st:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        bunch = fetch_openml(data_id=s.section("dataset")["openml_id"], as_frame=True,
                             parser="auto", data_home=str(raw_path.parent / "openml"))
        X = bunch.data.copy()
        if list(X.columns) != UCI_COLUMNS:          # OpenML publica como x1..x23
            X.columns = UCI_COLUMNS
        df = X.apply(pd.to_numeric)
        df.insert(0, "ID", np.arange(1, len(df) + 1))
        df["default_payment_next_month"] = pd.to_numeric(bunch.target).astype(int)
        df.to_csv(raw_path, index=False)
        st.rows_out = len(df)
    return df


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes, corrige códigos não documentados e pseudonimiza o ID."""
    df = raw.rename(columns=RENAME).rename(columns={"default_payment_next_month": "default"})

    # Códigos não documentados no dicionário oficial viram a categoria "outros"
    df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})
    df["marriage"] = df["marriage"].replace({0: 3})

    # Privacy by Design: o ID original nunca sai da camada de ingestão
    df = pseudonymize_column(df, "ID", "customer_id")

    int_cols = [c for c in df.columns if c != "customer_id"]
    df[int_cols] = df[int_cols].astype("int64")
    cols = ["customer_id"] + [c for c in df.columns if c != "customer_id"]
    return df[cols].drop_duplicates(subset=[c for c in cols if c != "customer_id"])


def prepare() -> None:
    """Gera os datasets de Referência (treino) e o Pool de Produção, fisicamente separados."""
    s = load_settings()
    ds = s.section("dataset")
    raw = download()

    with track_stage("ingest_prepare") as st:
        st.rows_in = len(raw)
        df = clean(raw)
        st.rows_out = len(df)
        dropped = st.rows_in - st.rows_out
        record_metric("ingest_duplicates_removed", dropped)
        s.path("processed_data").parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(s.path("processed_data"), index=False)

    with track_stage("split_reference_production") as st:
        st.rows_in = len(df)
        reference, production_pool = train_test_split(
            df, train_size=ds["reference_fraction"], stratify=df[s.target], random_state=s.seed)
        ref_train, ref_test = train_test_split(
            reference, test_size=ds["test_size"], stratify=reference[s.target], random_state=s.seed)

        for key, frame in [("reference_data", ref_train), ("reference_test", ref_test),
                           ("production_pool", production_pool)]:
            path = s.path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.reset_index(drop=True).to_parquet(path, index=False)

        st.rows_out = len(ref_train) + len(ref_test) + len(production_pool)
        st.detail.update(reference_train=len(ref_train), reference_test=len(ref_test),
                         production_pool=len(production_pool))
        record_metric("reference_rows", len(ref_train))
        record_metric("reference_default_rate", ref_train[s.target].mean())
        log.info(f"Referência (treino)={len(ref_train)} | Referência (teste)={len(ref_test)} | "
                 f"Pool de produção={len(production_pool)} | "
                 f"taxa de default={ref_train[s.target].mean():.2%}")
