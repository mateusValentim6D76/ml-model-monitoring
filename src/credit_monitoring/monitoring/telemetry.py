"""Telemetria centralizada do pipeline.

Tudo o que acontece no pipeline vira um *evento estruturado* (JSON Lines) em `logs/`:

    logs/pipeline.jsonl   log de aplicação (nível, mensagem, run_id, stage, contexto)
    logs/stages.jsonl     uma linha por execução de etapa: status, latência, volume de dados
    logs/metrics.jsonl    métricas numéricas (drift, performance, qualidade, fairness)
    logs/alerts.jsonl     alertas disparados pelas regras de monitoramento

Esse formato é o mesmo consumido por stacks de observabilidade (Datadog, Loki,
CloudWatch, ELK): basta apontar um agente/forwarder para a pasta `logs/`.
O dashboard HTML local (`monitoring/dashboard.py`) e o MLflow leem estes arquivos.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from credit_monitoring.config import load_settings

RUN_ID = os.environ.get("CREDIT_MONITORING_RUN_ID") or (
    datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
)

LOG_FILES = ("pipeline.jsonl", "stages.jsonl", "metrics.jsonl", "alerts.jsonl")


def _logs_dir() -> Path:
    d = load_settings().path("logs_dir")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _append(filename: str, record: dict[str, Any]) -> None:
    with open(_logs_dir() / filename, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1024**2, 1)
    except Exception:  # pragma: no cover - psutil opcional
        return None


def reset_logs() -> None:
    """Zera os arquivos de telemetria (usado no início de uma execução completa)."""
    for name in LOG_FILES:
        f = _logs_dir() / name
        if f.exists():
            f.unlink()


# --------------------------------------------------------------------------- #
# Logger de aplicação: console legível + arquivo JSON Lines
# --------------------------------------------------------------------------- #
class _JsonLinesHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - IO trivial
        payload = {
            "timestamp": _now(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": RUN_ID,
            "message": record.getMessage(),
        }
        ctx = getattr(record, "ctx", None)
        if ctx:
            payload.update(ctx)
        if record.exc_info:
            payload["exception"] = "".join(traceback.format_exception(*record.exc_info))
        try:
            _append("pipeline.jsonl", payload)
        except Exception:
            self.handleError(record)


def get_logger(name: str = "credit_monitoring") -> logging.Logger:
    logger = logging.getLogger(name)
    if getattr(logger, "_configured", False):
        return logger
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                                          datefmt="%H:%M:%S"))
    logger.addHandler(console)
    logger.addHandler(_JsonLinesHandler())
    logger.propagate = False
    logger._configured = True  # type: ignore[attr-defined]
    return logger


log = get_logger()


# --------------------------------------------------------------------------- #
# Rastreamento de etapas (status, latência, volume)
# --------------------------------------------------------------------------- #
@dataclass
class StageRun:
    stage: str
    batch: str | None = None
    rows_in: int | None = None
    rows_out: int | None = None
    status: str = "running"
    detail: dict[str, Any] = field(default_factory=dict)

    def block(self, reason: str) -> None:
        """Marca a etapa como BLOQUEADA (falha controlada, por exemplo contrato violado)."""
        self.status = "blocked"
        self.detail["reason"] = reason


@contextmanager
def track_stage(stage: str, batch: str | None = None, **detail: Any) -> Iterator[StageRun]:
    """Context manager que mede latência e registra o resultado de uma etapa.

    status final:
        success  etapa concluída
        blocked  etapa interrompida de forma controlada (ex.: data contract violado)
        failed   exceção não tratada (a exceção é propagada)
    """
    run = StageRun(stage=stage, batch=batch, detail=dict(detail))
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    log.info(f"[{stage}] início" + (f" | lote={batch}" if batch else ""),
             extra={"ctx": {"stage": stage, "batch": batch, "event": "stage_start"}})
    try:
        yield run
        if run.status == "running":
            run.status = "success"
    except Exception as exc:
        run.status = "failed"
        run.detail["error"] = f"{type(exc).__name__}: {exc}"
        log.exception(f"[{stage}] falhou: {exc}",
                      extra={"ctx": {"stage": stage, "batch": batch, "event": "stage_failed"}})
        raise
    finally:
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        record = {
            "timestamp": _now(),
            "run_id": RUN_ID,
            "stage": stage,
            "batch": batch,
            "status": run.status,
            "duration_ms": duration_ms,
            "rows_in": run.rows_in,
            "rows_out": run.rows_out,
            # Infraestrutura: consumo de CPU e memória do processo na etapa
            "cpu_time_s": round(time.process_time() - cpu0, 3),
            "memory_rss_mb": _rss_mb(),
            **({"detail": run.detail} if run.detail else {}),
        }
        _append("stages.jsonl", record)
        level = logging.INFO if run.status == "success" else logging.WARNING
        log.log(level, f"[{stage}] {run.status.upper()} em {duration_ms} ms"
                       + (f" | rows_in={run.rows_in} rows_out={run.rows_out}" if run.rows_in else ""),
                extra={"ctx": {"stage": stage, "batch": batch, "event": "stage_end",
                               "status": run.status, "duration_ms": duration_ms}})


# --------------------------------------------------------------------------- #
# Métricas e alertas
# --------------------------------------------------------------------------- #
def record_metric(name: str, value: float, batch: str | None = None, **tags: Any) -> None:
    _append("metrics.jsonl", {
        "timestamp": _now(), "run_id": RUN_ID, "metric": name,
        "value": None if value is None else float(value), "batch": batch, **tags,
    })


def raise_alert(severity: str, kind: str, message: str, batch: str | None = None,
                **context: Any) -> dict[str, Any]:
    """Registra um alerta. severity ∈ {info, warning, critical}."""
    alert = {"timestamp": _now(), "run_id": RUN_ID, "severity": severity, "kind": kind,
             "batch": batch, "message": message, **context}
    _append("alerts.jsonl", alert)
    level = logging.CRITICAL if severity == "critical" else logging.WARNING
    log.log(level, f"ALERTA [{severity.upper()}] {kind}: {message}",
            extra={"ctx": {"event": "alert", "severity": severity, "kind": kind, "batch": batch}})
    return alert


def read_jsonl(filename: str) -> list[dict[str, Any]]:
    f = _logs_dir() / filename
    if not f.exists():
        return []
    with open(f, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
