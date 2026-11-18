"""CLI do projeto.

    credit-monitor all              # pipeline completo (ingestão, treino, contrato, drift e dashboard)
    credit-monitor prepare          # baixa o dataset e separa Referência × Produção
    credit-monitor train            # treina o baseline (XGBoost) e registra no MLflow
    credit-monitor contract-demo    # força um lote corrompido e mostra o contrato bloqueando
    credit-monitor simulate         # gera 6 meses de produção com data/concept drift
    credit-monitor monitor          # executa o monitoramento lote a lote (Evidently, alertas, MLflow)
    credit-monitor dashboard        # gera reports/monitoring_dashboard.html a partir dos logs

Também pode ser chamado como `python -m credit_monitoring <comando>`.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time


def _prepare() -> None:
    from credit_monitoring.data.ingest import prepare

    prepare()


def _train() -> None:
    from credit_monitoring.models.train import train

    train()


def _contract_demo() -> None:
    import pandas as pd

    from credit_monitoring.config import load_settings
    from credit_monitoring.contracts.corrupt import make_corrupted_batch
    from credit_monitoring.contracts.validate import DataContractViolation, validate_batch
    from credit_monitoring.monitoring.telemetry import log

    s = load_settings()
    pool = pd.read_parquet(s.path("production_pool"))
    clean = pool.sample(1000, random_state=s.seed)

    log.info("=== DEMO 1: lote íntegro ===")
    validate_batch(clean.drop(columns=[s.target]), batch="demo_clean_batch")

    log.info("=== DEMO 2: lote corrompido (nulos, menores de idade, duplicatas, PII...) ===")
    try:
        validate_batch(make_corrupted_batch(clean, seed=s.seed), batch="demo_corrupted_batch",
                       raise_on_failure=True)
    except DataContractViolation as exc:
        log.info(f"Ingestão interrompida como esperado: {exc}")


def _simulate() -> None:
    from credit_monitoring.drift.simulate import simulate

    simulate()


def _monitor() -> None:
    from credit_monitoring.monitoring.monitor import run_monitoring

    run_monitoring()


def _dashboard() -> None:
    from credit_monitoring.monitoring.dashboard import build_dashboard

    build_dashboard()


def _all() -> None:
    from credit_monitoring.monitoring.telemetry import log, reset_logs

    reset_logs()
    t0 = time.perf_counter()
    for name, fn in [("prepare", _prepare), ("train", _train), ("contract-demo", _contract_demo),
                     ("simulate", _simulate), ("monitor", _monitor), ("dashboard", _dashboard)]:
        log.info(f"Etapa: {name}")
        fn()
    log.info(f"Pipeline completo em {time.perf_counter() - t0:.1f}s. "
             f"Abra reports/monitoring_dashboard.html")


COMMANDS = {
    "prepare": _prepare, "train": _train, "contract-demo": _contract_demo,
    "simulate": _simulate, "monitor": _monitor, "dashboard": _dashboard, "all": _all,
}


def _force_utf8_console() -> None:
    """O terminal do Windows usa cp1252 por padrão e falha ao imprimir alguns caracteres Unicode."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    parser = argparse.ArgumentParser(prog="credit-monitor", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS.keys())
    args = parser.parse_args(argv)
    COMMANDS[args.command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
