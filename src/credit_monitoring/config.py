"""Carregamento da configuração central (config/config.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]

    # ---- atalhos -----------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve um caminho de `paths.<key>` relativo à raiz do projeto."""
        p = PROJECT_ROOT / self.raw["paths"][key]
        return p

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    @property
    def target(self) -> str:
        return self.raw["dataset"]["target"]

    @property
    def numerical(self) -> list[str]:
        return list(self.raw["features"]["numerical"])

    @property
    def categorical(self) -> list[str]:
        return list(self.raw["features"]["categorical"])

    @property
    def features(self) -> list[str]:
        return self.numerical + self.categorical

    @property
    def protected(self) -> list[str]:
        return list(self.raw["features"]["protected"])

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]


@lru_cache(maxsize=4)
def load_settings(config_path: str | Path | None = None) -> Settings:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        return Settings(raw=yaml.safe_load(fh))
