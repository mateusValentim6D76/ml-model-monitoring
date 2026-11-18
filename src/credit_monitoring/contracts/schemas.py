"""Data Contract do modelo de Credit Scoring (Pandera).

O contrato é a "API" entre quem produz os dados (originação de crédito / bureau)
e quem os consome (modelo). Qualquer lote que viole uma regra é BLOQUEADO e vai
para quarentena: "lixo não entra no modelo".

Regras rígidas (todas obrigatórias, sem tolerância):

| #  | Regra                                   | Motivação de negócio                                    |
|----|-----------------------------------------|---------------------------------------------------------|
| R1 | Nenhum valor nulo em nenhuma coluna     | XGBoost aceitaria NaN sem avisar e decidiria errado      |
| R2 | Idade entre 18 e 100 anos               | Crédito só para maiores de idade (Código Civil art. 5º)   |
| R3 | Limite de crédito entre 10 mil e 1 mi   | Faixa da política de crédito; fora disso = erro de carga |
| R4 | Sem registros duplicados                | Duplicatas inflam volume e enviesam métricas             |
| R5 | `customer_id` único e pseudonimizado    | Garante que nenhum ID real (PII) vazou da ingestão       |
| R6 | Domínios categóricos válidos            | education ∈ {1..4}, pay_status ∈ [-2, 8], sex, marriage  |
| R7 | Valores monetários dentro de limites    | Pagamento negativo é impossível; faturas com teto        |
| R8 | Schema estrito (sem colunas extras)     | Bloqueia vazamento de PII (ex.: CPF, nome) para o modelo |
| R9 | Tipos inteiros (sem coerção silenciosa) | Mudança de tipo indica quebra no produtor                |
| R10| Lote com no mínimo 100 registros        | Lotes truncados indicam falha de extração                |
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

PAY_STATUS_COLS = [f"pay_status_{i}" for i in range(1, 7)]
BILL_COLS = [f"bill_amt_{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"pay_amt_{i}" for i in range(1, 7)]

MIN_AGE, MAX_AGE = 18, 100
MIN_LIMIT, MAX_LIMIT = 10_000, 1_000_000
MIN_BATCH_ROWS = 100

_INT = "int64"


def _no_duplicate_records(df):
    """R4: linha a linha: False para registros integralmente repetidos (arquivo reprocessado).

    Clientes distintos podem legitimamente ter o mesmo perfil (ex.: sem movimentação),
    por isso a duplicidade considera o registro completo, incluindo o `customer_id`.
    """
    return df.duplicated(keep=False).eq(False)


def build_contract(require_target: bool = True) -> DataFrameSchema:
    """Constrói o contrato.

    `require_target=False` é usado em lotes de inferência, em que o rótulo
    (inadimplência) ainda não é conhecido, porque chega com atraso (label delay).
    """
    columns: dict[str, Column] = {
        # R5: identificador pseudonimizado (HMAC-SHA256), nunca o ID real
        "customer_id": Column(str, Check.str_matches(r"^cus_[0-9a-f]{24}$"),
                              unique=True, nullable=False,
                              description="Pseudônimo HMAC-SHA256 do cliente"),
        # R3: limite de crédito (proxy de renda/capacidade)
        "limit_bal": Column(_INT, Check.in_range(MIN_LIMIT, MAX_LIMIT), nullable=False,
                            description="Limite de crédito concedido (NT$)"),
        # R2: idade mínima legal
        "age": Column(_INT, Check.in_range(MIN_AGE, MAX_AGE), nullable=False,
                      description="Idade em anos"),
        # R6: domínios categóricos
        "education": Column(_INT, Check.isin([1, 2, 3, 4]), nullable=False),
        "sex": Column(_INT, Check.isin([1, 2]), nullable=False,
                      description="Atributo protegido, usado só na auditoria de viés"),
        "marriage": Column(_INT, Check.isin([1, 2, 3]), nullable=False,
                           description="Atributo protegido, usado só na auditoria de viés"),
        **{c: Column(_INT, Check.in_range(-2, 8), nullable=False) for c in PAY_STATUS_COLS},
        **{c: Column(_INT, Check.in_range(-1_000_000, 5_000_000), nullable=False)
           for c in BILL_COLS},
        # R7: pagamentos
        **{c: Column(_INT, Check.in_range(0, 2_000_000), nullable=False) for c in PAY_AMT_COLS},
    }
    if require_target:
        columns["default"] = Column(_INT, Check.isin([0, 1]), nullable=False,
                                    description="1 = inadimplente no mês seguinte")

    return DataFrameSchema(
        columns=columns,
        checks=[
            Check(_no_duplicate_records, error="R4: registros duplicados"),
            Check(lambda df: len(df) >= MIN_BATCH_ROWS, element_wise=False,
                  error=f"R10: lote com menos de {MIN_BATCH_ROWS} registros"),
        ],
        strict=True,        # R8: colunas não previstas no contrato são rejeitadas
        coerce=False,       # R9: nada de conversão silenciosa de tipos
        name="credit_scoring_contract_v1",
        description="Contrato de dados do modelo de Credit Scoring (v1)",
    )


# Versão "documentada" das regras para relatórios e README
RULES = {
    "R1": "Nenhum valor nulo",
    "R2": f"Idade entre {MIN_AGE} e {MAX_AGE} anos",
    "R3": f"Limite de crédito entre {MIN_LIMIT:,} e {MAX_LIMIT:,}",
    "R4": "Sem registros duplicados",
    "R5": "customer_id único e pseudonimizado (cus_<hmac>)",
    "R6": "Domínios categóricos válidos",
    "R7": "Valores monetários dentro de limites (pagamentos >= 0)",
    "R8": "Schema estrito: sem colunas extras (ex.: PII)",
    "R9": "Tipos inteiros sem coerção",
    "R10": f"Lote com >= {MIN_BATCH_ROWS} registros",
}

__all__ = ["build_contract", "RULES", "pa"]
