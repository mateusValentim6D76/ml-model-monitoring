"""Testes estatísticos de drift e simulação."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_monitoring.config import load_settings
from credit_monitoring.drift.simulate import apply_drift
from credit_monitoring.drift.stats import drift_table, psi, psi_level

from .conftest import make_valid_batch


def test_psi_is_zero_for_identical_distributions():
    x = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert psi(x, x) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(0, 1, 5000))
    small = pd.Series(rng.normal(0.05, 1, 5000))
    big = pd.Series(rng.normal(1.0, 1, 5000))
    assert psi(ref, small) < 0.10
    assert psi(ref, big) > 0.25


def test_psi_categorical():
    ref = pd.Series([1] * 500 + [2] * 500)
    cur = pd.Series([1] * 900 + [2] * 100)
    assert psi(ref, cur, categorical=True) > 0.25


def test_psi_level_thresholds():
    assert psi_level(0.05) == "stable"
    assert psi_level(0.15) == "warning"
    assert psi_level(0.30) == "critical"


def test_no_shock_means_no_drift():
    s = load_settings()
    sim = s.section("simulation")
    batch = make_valid_batch(2000, seed=3)
    out, applied = apply_drift(batch, 0.0, sim["shocks"], sim["concept_drift"], np.random.default_rng(0))
    pd.testing.assert_frame_equal(out, batch)
    assert applied == {"intensity": 0.0}


def test_full_shock_moves_expected_features():
    s = load_settings()
    sim = s.section("simulation")
    ref = make_valid_batch(3000, seed=4)
    cur, applied = apply_drift(make_valid_batch(3000, seed=5), 1.0, sim["shocks"],
                               sim["concept_drift"], np.random.default_rng(0))
    table = drift_table(ref, cur, s.numerical, s.categorical).set_index("feature")
    for feature in ["bill_amt_1", "limit_bal", "age", "pay_status_1", "pay_amt_1"]:
        assert table.loc[feature, "drift_detected"], feature
    assert cur["bill_amt_1"].mean() > ref["bill_amt_1"].mean()      # inflação
    assert cur["limit_bal"].mean() < ref["limit_bal"].mean()        # aperto de crédito
    assert cur["age"].mean() < ref["age"].mean()                    # público mais jovem
    assert cur["default"].mean() > ref["default"].mean()            # mais inadimplência
    assert applied["concept_flips"] > 0                             # concept drift injetado


def test_drifted_batch_still_respects_contract():
    """Drift é mudança LEGÍTIMA de distribuição: o contrato não deve bloquear."""
    from credit_monitoring.contracts.validate import validate_batch

    s = load_settings()
    sim = s.section("simulation")
    cur, _ = apply_drift(make_valid_batch(2000, seed=6), 1.0, sim["shocks"], sim["concept_drift"],
                         np.random.default_rng(1))
    assert validate_batch(cur, "t_drifted", require_target=True, write_report=False).passed
