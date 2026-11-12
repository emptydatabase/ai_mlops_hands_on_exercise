import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
PATH_DF = ROOT / "dataset" / "census.csv"
PATH_IMPORTANCE = ROOT / "dataset" / "feature_importance.json"
PATH_ROC = ROOT / "dataset" / "mlp_roc_curve.csv"
PATH_PRC = ROOT / "dataset" / "mlp_prc_curve.csv"


with open(PATH_IMPORTANCE) as fp:
    importance = json.load(fp)


features = [
    col
    for col, _ in zip(importance, range(64))
]


df: DataFrame = pd.read_csv(PATH_DF)

X = df[features]
y = df["target"].astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42))
])

pipeline.fit(X_train, y_train)

prediction = pipeline.predict(X_test).astype(np.float32)

fpr, tpr, thresholds = roc_curve(y_test, prediction)

roc = DataFrame({
    "fpr": fpr,
    "tpr": tpr,
    "threadsholds": thresholds
})
roc.to_csv(PATH_ROC)

precision, recall, thresholds = precision_recall_curve(y_test, prediction)

prc = DataFrame({
    "precision": precision[:-1],
    "recall": recall[:-1],
    "threadsholds": thresholds
})
roc.to_csv(PATH_PRC)
