"""Fixtures: lote sintético VÁLIDO segundo o contrato (não depende do download do dataset)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_monitoring.data.privacy import pseudonymize


def make_valid_batch(n: int = 400, seed: int = 0, with_target: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": [pseudonymize(i) for i in range(n)],
        "limit_bal": (rng.integers(1, 50, n) * 10_000).astype("int64"),
        "sex": rng.integers(1, 3, n).astype("int64"),
        "education": rng.integers(1, 5, n).astype("int64"),
        "marriage": rng.integers(1, 4, n).astype("int64"),
        "age": rng.integers(21, 70, n).astype("int64"),
    })
    for i in range(1, 7):
        df[f"pay_status_{i}"] = rng.integers(-2, 4, n).astype("int64")
    for i in range(1, 7):
        df[f"bill_amt_{i}"] = rng.integers(-1_000, 200_000, n).astype("int64")
    for i in range(1, 7):
        df[f"pay_amt_{i}"] = rng.integers(0, 20_000, n).astype("int64")
    if with_target:
        df["default"] = rng.integers(0, 2, n).astype("int64")
    return df


@pytest.fixture
def valid_batch() -> pd.DataFrame:
    return make_valid_batch()


@pytest.fixture
def inference_batch() -> pd.DataFrame:
    return make_valid_batch(with_target=False)


@pytest.fixture(autouse=True)
def _isolate_outputs(tmp_path, monkeypatch):
    """Redireciona logs/relatórios/quarentena para um diretório temporário."""
    from credit_monitoring.config import load_settings

    s = load_settings()
    original = s.path

    def fake_path(key: str):
        if key in {"logs_dir", "reports_dir", "quarantine_dir"}:
            return tmp_path / key
        return original(key)

    monkeypatch.setattr(type(s), "path", lambda self, key: fake_path(key))
    yield
