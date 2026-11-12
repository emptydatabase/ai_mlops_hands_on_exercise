from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier


def feature_selection(X: DataFrame, y, random_state: int, n_features):
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X, y)

    importance = DataFrame({
        "features": X.columns,
        "importance": rf.feature_importances_
    })
    importance = importance.sort_values(["importance"], ascending=False)
    cols = importance.iloc[:n_features]["features"].tolist()
    return X[cols]
