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


def confusion_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, int | float]:
    """Return thresholded confusion-matrix counts and summary metrics.

    A row is predicted positive when ``y_score > threshold``. The strict
    comparison matches the Task 1 evaluation code, so a score exactly equal
    to the threshold is classified as negative. This keeps the dashboard
    consistent with the saved Task 1 artefacts.

    Zero denominators (for example, no predicted positives) produce ``0.0``
    instead of raising an error.
    """
    truth = np.asarray(y_true).astype(bool)
    scores = np.asarray(y_score, dtype=float)

    if truth.shape != scores.shape:
        raise ValueError("y_true and y_score must have the same shape.")

    predicted = scores > threshold

    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive": tp + fp,
    }


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
    "confusion_metrics",
    "cramers_v",
    "cramers_v_matrix",
    "pearson_correlation",
]
