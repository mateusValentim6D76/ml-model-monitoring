"""Privacy by Design (pseudonimização) e auditoria de fairness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_monitoring.analysis.fairness import fairness_report
from credit_monitoring.config import load_settings
from credit_monitoring.data.privacy import age_band, pseudonymize

from .conftest import make_valid_batch


def test_pseudonym_is_deterministic_and_not_the_original():
    assert pseudonymize(123) == pseudonymize(123)
    assert pseudonymize(123) != pseudonymize(124)
    assert "123" not in pseudonymize(123)
    assert pseudonymize(1).startswith("cus_") and len(pseudonymize(1)) == 28


def test_pseudonym_depends_on_secret_key(monkeypatch):
    a = pseudonymize(42)
    monkeypatch.setenv("PSEUDONYMIZATION_KEY", "another-secret")
    assert pseudonymize(42) != a


def test_protected_attributes_are_not_model_features():
    s = load_settings()
    assert not set(s.protected) & set(s.features)
    assert "customer_id" not in s.features


def test_age_band_generalization():
    bands = age_band(pd.Series([18, 25, 30, 50, 70]))
    assert list(bands) == ["18-25", "18-25", "26-35", "46-60", "60+"]


def test_fairness_report_disparate_impact():
    df = make_valid_batch(1000, seed=9)
    proba = np.where(df["sex"] == 1, 0.1, 0.9)    # modelo enviesado de propósito
    _, summary = fairness_report(df, proba, threshold=0.5)
    assert summary["di_sex"] < 0.8
    neutral = np.full(len(df), 0.1)
    _, summary = fairness_report(df, neutral, threshold=0.5)
    assert summary["di_sex"] == 1.0
