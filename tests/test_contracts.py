"""O contrato precisa APROVAR dados limpos e BLOQUEAR cada tipo de anomalia."""

from __future__ import annotations

import numpy as np
import pytest

from credit_monitoring.contracts.corrupt import make_corrupted_batch
from credit_monitoring.contracts.validate import DataContractViolation, validate_batch


def test_clean_batch_passes(inference_batch):
    result = validate_batch(inference_batch, batch="t_clean", write_report=False)
    assert result.passed
    assert result.rules_violated == []


def test_training_batch_requires_valid_target(valid_batch):
    assert validate_batch(valid_batch, "t_train", require_target=True, write_report=False).passed
    bad = valid_batch.copy()
    bad.loc[0, "default"] = 3
    res = validate_batch(bad, "t_train_bad", require_target=True, write_report=False)
    assert not res.passed and "R6" in res.rules_violated


@pytest.mark.parametrize("mutation, rule", [
    (lambda d: d.assign(limit_bal=d["limit_bal"].astype(float).where(d.index != 0, np.nan)), "R1"),
    (lambda d: d.assign(age=d["age"].where(d.index != 0, 16)), "R2"),
    (lambda d: d.assign(age=d["age"].where(d.index != 0, 150)), "R2"),
    (lambda d: d.assign(limit_bal=d["limit_bal"].where(d.index != 0, -10)), "R3"),
    (lambda d: d.assign(limit_bal=d["limit_bal"].where(d.index != 0, 5_000_000)), "R3"),
    (lambda d: d.assign(education=d["education"].where(d.index != 0, 9)), "R6"),
    (lambda d: d.assign(pay_status_1=d["pay_status_1"].where(d.index != 0, 12)), "R6"),
    (lambda d: d.assign(pay_amt_1=d["pay_amt_1"].where(d.index != 0, -1)), "R7"),
    (lambda d: d.assign(cpf="123.456.789-00"), "R8"),
    (lambda d: d.assign(customer_id=d["customer_id"].where(d.index != 0, "12345")), "R5"),
    (lambda d: d.head(50), "R10"),
])
def test_each_rule_blocks(inference_batch, mutation, rule):
    result = validate_batch(mutation(inference_batch), batch="t_rule", write_report=False)
    assert not result.passed
    assert rule in result.rules_violated


def test_duplicates_are_blocked(inference_batch):
    import pandas as pd

    dup = pd.concat([inference_batch, inference_batch.head(3)], ignore_index=True)
    result = validate_batch(dup, batch="t_dup", write_report=False)
    assert {"R4", "R5"} <= set(result.rules_violated)


def test_corrupted_batch_is_quarantined_and_raises(valid_batch, tmp_path):
    corrupted = make_corrupted_batch(valid_batch)
    with pytest.raises(DataContractViolation):
        validate_batch(corrupted, batch="t_corrupted", raise_on_failure=True)
    assert (tmp_path / "quarantine_dir" / "t_corrupted.csv").exists()
    assert (tmp_path / "reports_dir" / "data_contracts" / "contract_t_corrupted.html").exists()


def test_corrupted_batch_violates_at_least_seven_rules(valid_batch):
    result = validate_batch(make_corrupted_batch(valid_batch), batch="t_many", write_report=False)
    assert len(result.rules_violated) >= 7
    assert {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"} <= set(result.rules_violated)
