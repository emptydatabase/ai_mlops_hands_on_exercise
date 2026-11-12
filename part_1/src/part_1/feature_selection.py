from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier


def feature_selection(df: DataFrame, random_state: int, n_features: int | None = None):
    if n_features is not None:
        return df

    df_copy = df.copy()

    y = df_copy.pop("target")
    X = df_copy

    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X, y)

    out = DataFrame({
        "features": df.columns,
        "importance": rf.feature_importances_
    })
    out.sort_values(["importance"])
    cols = out.iloc[:n_features]["features"].tolist()
    return df[["target", *cols]]
