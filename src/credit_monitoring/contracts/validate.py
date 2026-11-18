"""Quality Gate de ingestão: valida um lote contra o Data Contract.

Fluxo:
    1. O lote é validado contra o contrato (Pandera com lazy=True, que coleta todas as violações).
    2. Se estiver OK, segue para a inferência.
    3. Se violar alguma regra, o lote é bloqueado e salvo em data/quarantine/, um relatório
       HTML é gerado, um alerta crítico é disparado e a métrica é registrada.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pandera.errors as pa_errors

from credit_monitoring.config import load_settings
from credit_monitoring.contracts.schemas import (
    BILL_COLS,
    PAY_AMT_COLS,
    PAY_STATUS_COLS,
    RULES,
    build_contract,
)
from credit_monitoring.monitoring.telemetry import log, raise_alert, record_metric, track_stage


class DataContractViolation(Exception):
    """Lançada quando `raise_on_failure=True` e o lote viola o contrato."""


@dataclass
class ValidationResult:
    batch: str
    passed: bool
    n_rows: int
    failure_cases: pd.DataFrame = field(default_factory=pd.DataFrame)
    rules_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    report_path: Path | None = None

    @property
    def rules_violated(self) -> list[str]:
        return [] if self.rules_summary.empty else sorted(self.rules_summary["rule"].unique())


# --------------------------------------------------------------------------- #
def _rule_for(row: pd.Series) -> str:
    """Mapeia uma falha do Pandera para a regra de negócio do contrato (R1..R10)."""
    check, col = str(row["check"]), str(row["column"])
    if "not_nullable" in check:
        return "R1"
    if check.startswith("dtype"):
        return "R9"
    if "column_in_schema" in check or "column_in_dataframe" in check:
        return "R8"
    if "R4" in check:
        return "R4"
    if "R10" in check:
        return "R10"
    if col == "customer_id":
        return "R5"
    if col == "age":
        return "R2"
    if col == "limit_bal":
        return "R3"
    if col in PAY_AMT_COLS or col in BILL_COLS:
        return "R7"
    if col in PAY_STATUS_COLS or col in {"education", "sex", "marriage", "default"}:
        return "R6"
    return "OUTRA"


def _summarize(failures: pd.DataFrame) -> pd.DataFrame:
    f = failures.copy()
    f["rule"] = f.apply(_rule_for, axis=1)
    f["failure_case"] = f["failure_case"].astype(str)
    summary = (f.groupby(["rule", "column", "check"], dropna=False)
                 .agg(violations=("failure_case", "size"),
                      examples=("failure_case", lambda s: ", ".join(pd.unique(s)[:5])))
                 .reset_index()
                 .sort_values(["rule", "violations"], ascending=[True, False]))
    summary["description"] = summary["rule"].map(RULES).fillna("-")
    return summary


def validate_batch(df: pd.DataFrame, batch: str, require_target: bool = False,
                   raise_on_failure: bool = False, write_report: bool = True) -> ValidationResult:
    """Valida `df` contra o contrato e aplica a política de bloqueio."""
    s = load_settings()
    contract = build_contract(require_target=require_target)

    with track_stage("data_contract_validation", batch=batch) as st:
        st.rows_in = len(df)
        try:
            contract.validate(df, lazy=True)
            result = ValidationResult(batch=batch, passed=True, n_rows=len(df))
            st.rows_out = len(df)
        except pa_errors.SchemaErrors as err:
            failures = err.failure_cases.copy()
            result = ValidationResult(batch=batch, passed=False, n_rows=len(df),
                                      failure_cases=failures, rules_summary=_summarize(failures))
            st.rows_out = 0
            st.block(f"contrato violado: regras {', '.join(result.rules_violated)}")

        record_metric("contract_passed", int(result.passed), batch=batch)
        record_metric("contract_violations", len(result.failure_cases), batch=batch)
        record_metric("contract_rules_violated", len(result.rules_violated), batch=batch)

        if write_report:
            result.report_path = write_validation_report(result, df)

        if not result.passed:
            qdir = s.path("quarantine_dir")
            qdir.mkdir(parents=True, exist_ok=True)
            df.to_csv(qdir / f"{batch}.csv", index=False)
            result.failure_cases.to_csv(qdir / f"{batch}_failures.csv", index=False)
            raise_alert("critical", "data_contract_violation",
                        f"Lote '{batch}' BLOQUEADO: {len(result.failure_cases)} violações nas regras "
                        f"{', '.join(result.rules_violated)}. Enviado para quarentena.",
                        batch=batch, rules=result.rules_violated)
        else:
            log.info(f"Lote '{batch}' aprovado no contrato ({len(df)} registros).")

    if not result.passed and raise_on_failure:
        raise DataContractViolation(f"Lote '{batch}' violou o contrato: {result.rules_violated}")
    return result


# --------------------------------------------------------------------------- #
# Relatório HTML do contrato
# --------------------------------------------------------------------------- #
def write_validation_report(result: ValidationResult, df: pd.DataFrame) -> Path:
    s = load_settings()
    out_dir = s.path("reports_dir") / "data_contracts"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"contract_{result.batch}.html"

    status = "APROVADO" if result.passed else "BLOQUEADO"
    color = "#1a7f37" if result.passed else "#cf222e"

    rule_rows = []
    violated = set(result.rules_violated)
    for code, desc in RULES.items():
        ok = code not in violated
        n = 0 if ok else int(result.rules_summary.loc[result.rules_summary["rule"] == code,
                                                     "violations"].sum())
        rule_rows.append(
            f"<tr><td><b>{code}</b></td><td>{html.escape(desc)}</td>"
            f"<td class='{'ok' if ok else 'ko'}'>{'✔ OK' if ok else '✘ VIOLADA'}</td>"
            f"<td>{n}</td></tr>")

    detail = ""
    if not result.passed:
        detail = (result.rules_summary[["rule", "column", "check", "violations", "examples"]]
                  .to_html(index=False, classes="tbl", escape=True, border=0))

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    page = f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Contract: {html.escape(result.batch)}</title>
<style>
 body{{font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f6f8fa;color:#1f2328}}
 main{{max-width:1100px;margin:auto;padding:24px 16px}}
 .badge{{display:inline-block;padding:6px 14px;border-radius:999px;color:#fff;font-weight:600;background:{color}}}
 .card{{background:#fff;border:1px solid #d0d7de;border-radius:10px;padding:16px 20px;margin:16px 0;overflow-x:auto}}
 table{{border-collapse:collapse;width:100%;font-size:14px}}
 th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #eaeef2;vertical-align:top}}
 th{{background:#f6f8fa}} .ok{{color:#1a7f37;font-weight:600}} .ko{{color:#cf222e;font-weight:600}}
 .kpis{{display:flex;gap:16px;flex-wrap:wrap}} .kpi{{flex:1;min-width:160px}}
 .kpi b{{font-size:26px;display:block}}
</style></head><body><main>
<h1>Data Contract <code>credit_scoring_contract_v1</code></h1>
<p>Lote: <b>{html.escape(result.batch)}</b> &nbsp; <span class="badge">{status}</span></p>
<div class="card kpis">
 <div class="kpi">Registros no lote<b>{result.n_rows:,}</b></div>
 <div class="kpi">Violações encontradas<b>{len(result.failure_cases):,}</b></div>
 <div class="kpi">Regras violadas<b>{len(violated)} / {len(RULES)}</b></div>
 <div class="kpi">Colunas recebidas<b>{df.shape[1]}</b></div>
</div>
<div class="card"><h2>Regras do contrato</h2>
<table><tr><th>Regra</th><th>Descrição</th><th>Status</th><th>Violações</th></tr>{''.join(rule_rows)}</table></div>
{'<div class="card"><h2>Detalhe das violações (Pandera)</h2>' + detail + '</div>' if detail else ''}
<div class="card"><h2>Política aplicada</h2>
<p>{'O lote seguiu para inferência.' if result.passed else
   'O lote foi <b>bloqueado</b>: nenhuma predição foi gerada, os dados foram movidos para '
   '<code>data/quarantine/</code> e um alerta crítico foi registrado em <code>logs/alerts.jsonl</code>.'}</p>
</div>
<p style="color:#59636e;font-size:12px">Gerado em {generated}</p>
</main></body></html>"""
    path.write_text(page, encoding="utf-8")
    (out_dir / f"contract_{result.batch}.json").write_text(json.dumps({
        "batch": result.batch, "passed": result.passed, "n_rows": result.n_rows,
        "violations": len(result.failure_cases), "rules_violated": result.rules_violated,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
