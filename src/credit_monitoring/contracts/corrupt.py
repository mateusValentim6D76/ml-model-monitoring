"""Gera um lote de produção propositalmente CORROMPIDO para provar que o contrato bloqueia lixo.

Cada defeito simula um incidente real de engenharia de dados:
    * nulos em limite de crédito ............ falha de join com o bureau
    * menores de idade e idade 150 .......... erro de digitação / data de nascimento
    * limite de crédito negativo/absurdo .... erro de unidade (centavos x reais)
    * registros duplicados .................. reprocessamento de arquivo
    * códigos categóricos inválidos ......... nova versão do sistema de origem
    * pagamento negativo .................... estorno lançado no campo errado
    * coluna extra "cpf" .................... VAZAMENTO DE PII para o pipeline do modelo
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_corrupted_batch(clean_batch: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = clean_batch.drop(columns=["default"], errors="ignore").copy().reset_index(drop=True)
    n = len(df)

    def pick(k: int) -> np.ndarray:
        return rng.choice(n, size=k, replace=False)

    df["limit_bal"] = df["limit_bal"].astype("float64")
    df.loc[pick(25), "limit_bal"] = np.nan                       # R1 / R9
    df.loc[pick(5), "limit_bal"] = -50_000                        # R3
    df.loc[pick(3), "limit_bal"] = 9_999_999                      # R3
    df.loc[pick(12), "age"] = rng.integers(14, 18, 12)            # R2 menores de idade
    df.loc[pick(2), "age"] = 150                                  # R2
    df.loc[pick(6), "education"] = 9                              # R6
    df.loc[pick(4), "pay_status_1"] = 12                          # R6
    df.loc[pick(5), "pay_amt_1"] = -1_500                         # R7
    # R8: vazamento de PII: CPF fictício (formato válido, números aleatórios)
    df["cpf"] = [f"{rng.integers(100, 999)}.{rng.integers(100, 999)}."
                 f"{rng.integers(100, 999)}-{rng.integers(10, 99)}" for _ in range(n)]
    # Arquivo reprocessado: registros integralmente repetidos
    dup = df.sample(15, random_state=seed)                        # R4 / R5
    df = pd.concat([df, dup], ignore_index=True)
    return df
