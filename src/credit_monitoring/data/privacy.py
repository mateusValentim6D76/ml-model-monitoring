"""Privacy by Design — utilitários de proteção de dados pessoais (LGPD).

* Pseudonimização (LGPD art. 13, §4º): o identificador original do cliente é
  substituído por um HMAC-SHA256 com chave secreta. Sem a chave não é possível
  reverter nem re-identificar por força bruta (diferente de um hash simples).
* Minimização (LGPD art. 6º, III): somente as colunas estritamente necessárias
  à finalidade (análise de crédito) seguem no pipeline.
* Atributos protegidos (sexo, estado civil) NUNCA são features do modelo; são
  retidos de forma segregada apenas para auditoria de discriminação (art. 6º, IX).
"""

from __future__ import annotations

import hashlib
import hmac
import os

import pandas as pd

# Em produção a chave vem de um secret manager (Vault / AWS Secrets Manager).
# O valor default existe apenas para tornar o projeto reprodutível localmente.
_DEFAULT_DEV_KEY = "dev-only-key-change-me"


def _key() -> bytes:
    return os.environ.get("PSEUDONYMIZATION_KEY", _DEFAULT_DEV_KEY).encode("utf-8")


def pseudonymize(value: str | int) -> str:
    """Retorna um pseudônimo determinístico e irreversível (sem a chave) para `value`."""
    digest = hmac.new(_key(), str(value).encode("utf-8"), hashlib.sha256).hexdigest()
    return f"cus_{digest[:24]}"


def pseudonymize_column(df: pd.DataFrame, column: str, new_column: str = "customer_id") -> pd.DataFrame:
    out = df.copy()
    out[new_column] = out[column].map(pseudonymize)
    return out.drop(columns=[column])


def age_band(age: pd.Series) -> pd.Series:
    """Generalização (k-anonimato): troca a idade exata pela faixa etária nos relatórios."""
    bins = [17, 25, 35, 45, 60, 200]
    labels = ["18-25", "26-35", "36-45", "46-60", "60+"]
    return pd.cut(age, bins=bins, labels=labels).astype(str)
