from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Literal

import pandas as pd

Pol = Literal["TE", "TM", "UNPOL"]


@dataclass(frozen=True)
class ValidationMeta:
    name: str
    theta_deg: float
    pol: Pol


def read_csv_text(text: str) -> pd.DataFrame:
    """Read CSV text into a pandas DataFrame with minimal assumptions.

    The caller provides column name mapping with `map_columns`.
    """
    return pd.read_csv(StringIO(text))


def map_columns(df: pd.DataFrame, *, lam_col: str, eps_col: str) -> pd.DataFrame:
    """Return a normalized DataFrame with columns: `lambda_um`, `eps`.

    Drops NA rows, sorts by `lambda_um`, clips `eps` to [0, 1].
    """
    if lam_col not in df.columns or eps_col not in df.columns:
        raise KeyError("Provided column names not present in CSV")
    out = df[[lam_col, eps_col]].rename(columns={lam_col: "lambda_um", eps_col: "eps"}).copy()
    out = out.dropna(subset=["lambda_um", "eps"]).reset_index(drop=True)
    out["lambda_um"] = out["lambda_um"].astype(float)
    out["eps"] = out["eps"].astype(float).clip(lower=0.0, upper=1.0)
    out = out.sort_values("lambda_um").reset_index(drop=True)
    return out
