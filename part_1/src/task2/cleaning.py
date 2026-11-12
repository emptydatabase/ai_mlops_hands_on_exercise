from __future__ import annotations

import pandas as pd

SENTINEL_VALUE = 99999
SENTINEL_COLUMNS = ("capital-gain", "capital-loss")

DROP_COLUMNS = ("fnlwgt", "education-num")

CATEGORICAL_FEATURES = (
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
)

MISSING_COLUMNS = ("workclass", "occupation", "native-country")
MISSING_MARKER = "?"
UNKNOWN_VALUE = "Unknown"

TARGET_SOURCE_COLUMN = "income"
TARGET_COLUMN = "target"
TARGET_COLUMNS = (TARGET_SOURCE_COLUMN, TARGET_COLUMN)
ALLOWED_INCOME_VALUES = ("<=50K", ">50K")

RAW_COLUMN_ORDER = (
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
)

CLEAN_COLUMN_ORDER = (
    "age",
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
    "target",
)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stripped column names and no duplicate names."""
    normalised = df.copy()
    normalised.columns = [str(column).strip() for column in normalised.columns]

    duplicated = normalised.columns[normalised.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            "Duplicate column names after stripping whitespace: "
            f"{sorted(set(duplicated))}"
        )

    return normalised


def _require_raw_columns(df: pd.DataFrame) -> None:
    missing = [column for column in RAW_COLUMN_ORDER if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required raw columns: {missing}")


def _strip_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from categorical values; the caller owns the copy."""
    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].astype("string").str.strip()
    return df


def _sentinel_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index, dtype=bool)
    for column in SENTINEL_COLUMNS:
        mask |= df[column].eq(SENTINEL_VALUE)
    return mask


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns, excluding the target representations."""
    return [
        column
        for column in df.columns
        if column not in TARGET_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return categorical feature columns, excluding the target representations."""
    return [
        column
        for column in df.columns
        if column not in TARGET_COLUMNS
        and not pd.api.types.is_numeric_dtype(df[column])
    ]


def clean_census(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Adult Census Income data without one-hot encoding.

    The returned frame keeps categorical features as strings. Sentinel value
    rows are removed, ``fnlwgt`` and ``education-num`` are dropped, ``?`` is
    replaced with ``Unknown``, and ``income`` is normalised into a string plus
    a boolean ``target`` column.

    Parameters
    ----------
    df:
        Raw or already column-stripped census data. The input is not mutated.

    Returns
    -------
    pandas.DataFrame
        Cleaned data with ``CLEAN_COLUMN_ORDER`` columns.
    """
    prepared = _normalise_columns(df)
    _require_raw_columns(prepared)
    prepared = _strip_categorical_values(prepared)

    cleaned = prepared.loc[~_sentinel_mask(prepared)].copy()
    cleaned = cleaned.drop(columns=list(DROP_COLUMNS))

    for column in MISSING_COLUMNS:
        cleaned[column] = cleaned[column].mask(
            cleaned[column].eq(MISSING_MARKER),
            UNKNOWN_VALUE,
        )

    income = (
        cleaned[TARGET_SOURCE_COLUMN]
        .astype("string")
        .str.strip()
        .str.rstrip(".")
    )

    if income.isna().any():
        raise ValueError("The income column contains missing values.")

    unexpected = sorted(set(income.unique()) - set(ALLOWED_INCOME_VALUES))
    if unexpected:
        raise ValueError(
            "Unexpected income values after normalisation: "
            f"{unexpected}. Expected only {list(ALLOWED_INCOME_VALUES)}."
        )

    cleaned[TARGET_SOURCE_COLUMN] = income
    cleaned[TARGET_COLUMN] = income.eq(">50K").fillna(False).astype(bool)

    return cleaned.loc[:, list(CLEAN_COLUMN_ORDER)].reset_index(drop=True).copy()


def cleaning_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """Describe the effect of :func:`clean_census` on the raw data."""
    raw = _normalise_columns(raw_df)
    _require_raw_columns(raw)
    raw = _strip_categorical_values(raw)

    rows_before = len(raw)
    rows_after = len(clean_df)
    sentinel_mask = _sentinel_mask(raw)

    sentinel_counts = {
        column: int(raw[column].eq(SENTINEL_VALUE).sum())
        for column in SENTINEL_COLUMNS
    }
    missing_markers_raw = {
        column: int(raw[column].eq(MISSING_MARKER).sum())
        for column in MISSING_COLUMNS
    }
    missing_markers_cleaned = {
        column: int(clean_df[column].astype("string").eq(UNKNOWN_VALUE).sum())
        for column in MISSING_COLUMNS
    }

    positive_count = int(clean_df[TARGET_COLUMN].sum())
    negative_count = rows_after - positive_count

    return {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "rows_removed_sentinel": int(sentinel_mask.sum()),
        "sentinel_counts": sentinel_counts,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_count / rows_after if rows_after else 0.0,
        "columns": list(clean_df.columns),
        "numeric_columns": numeric_columns(clean_df),
        "categorical_columns": categorical_columns(clean_df),
        "target_columns": list(TARGET_COLUMNS),
        "missing_markers_raw": missing_markers_raw,
        "missing_markers_cleaned": missing_markers_cleaned,
    }


__all__ = [
    "ALLOWED_INCOME_VALUES",
    "CATEGORICAL_FEATURES",
    "CLEAN_COLUMN_ORDER",
    "DROP_COLUMNS",
    "MISSING_COLUMNS",
    "MISSING_MARKER",
    "RAW_COLUMN_ORDER",
    "SENTINEL_COLUMNS",
    "SENTINEL_VALUE",
    "TARGET_COLUMN",
    "TARGET_COLUMNS",
    "TARGET_SOURCE_COLUMN",
    "UNKNOWN_VALUE",
    "categorical_columns",
    "clean_census",
    "cleaning_summary",
    "numeric_columns",
]
