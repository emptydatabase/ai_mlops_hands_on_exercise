import numpy as np
from pandas import DataFrame
from sklearn.metrics import roc_curve, precision_recall_curve


def roc_prc(y_test, prediction):
    fpr, tpr, thresholds = roc_curve(y_test, prediction)

    roc = DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    })

    precision, recall, thresholds = precision_recall_curve(y_test, prediction)

    prc = DataFrame({
        "precision": precision[:-1],
        "recall": recall[:-1],
        "thresholds": thresholds
    })

    return roc, prc


def best_point(df: DataFrame, x: str, y: str, target_x: float, target_y: float):
    df = df.copy()
    df["dist"] = np.sqrt((df[x] - target_x) ** 2 + (df[y] - target_y) ** 2)
    return df[df["dist"] == df["dist"].min()]
