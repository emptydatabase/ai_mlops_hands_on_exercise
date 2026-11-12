from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def pearson_correlation(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Return the Pearson correlation matrix for the given columns."""
    return df.loc[:, list(columns)].corr(method="pearson")


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Compute Cramér's V association between two categorical series.

    The value is derived from a contingency table, so the categorical values
    are never one-hot encoded. The result is clamped to ``[0, 1]`` to absorb
    floating-point noise.
    """
    table = pd.crosstab(x, y)
    if table.empty:
        return 0.0

    chi2 = chi2_contingency(table, correction=False)[0]
    sample_size = float(table.to_numpy().sum())
    denominator = sample_size * min(
        table.shape[0] - 1,
        table.shape[1] - 1,
    )
    if denominator <= 0:
        return 0.0

    value = float(np.sqrt(chi2 / denominator))
    return min(max(value, 0.0), 1.0)


def cramers_v_matrix(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Return the symmetric Cramér's V matrix for the given columns."""
    columns = list(columns)
    if not columns:
        return pd.DataFrame()

    matrix = pd.DataFrame(
        np.eye(len(columns)),
        index=columns,
        columns=columns,
    )

    for index, first in enumerate(columns):
        for second in columns[index + 1 :]:
            value = cramers_v(df[first], df[second])
            matrix.loc[first, second] = value
            matrix.loc[second, first] = value

    return matrix


__all__ = [
    "cramers_v",
    "cramers_v_matrix",
    "pearson_correlation",
]
