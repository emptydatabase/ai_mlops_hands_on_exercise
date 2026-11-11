import pandas as pd


def check_one_to_one_correspondence(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> pd.DataFrame:
    """Check whether two columns have a one-to-one correspondence.

    A one-to-one correspondence means each distinct value in ``col_a``
    maps to exactly one distinct value in ``col_b`` (and vice versa).
    This function returns the subset of rows that violate that rule.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    col_a : str
        Name of the first column.
    col_b : str
        Name of the second column.

    Returns
    -------
    pandas.DataFrame
        Rows from ``df`` where the value in ``col_a`` is associated with
        more than one value in ``col_b`` or where the value in ``col_b``
        is associated with more than one value in ``col_a``.
        Returns an empty dataframe if the correspondence is one-to-one.
    """

    if col_a not in df.columns or col_b not in df.columns:
        raise KeyError(
            f"Columns '{col_a}' and '{col_b}' must exist in the dataframe."
        )

    # Group by col_a -> set of col_b values
    a_to_b = df.groupby(col_a)[col_b].nunique()
    bad_a = a_to_b[a_to_b > 1].index

    # Group by col_b -> set of col_a values
    b_to_a = df.groupby(col_b)[col_a].nunique()
    bad_b = b_to_a[b_to_a > 1].index

    # Rows that break the correspondence in either direction
    mask = df[col_a].isin(bad_a) | df[col_b].isin(bad_b)
    return df[mask].copy()
