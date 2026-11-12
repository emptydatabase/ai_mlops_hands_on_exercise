import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
PATH_DF = ROOT / "dataset" / "census.csv"
PATH_IMPORTANCE = ROOT / "dataset" / "feature_importance.json"
PATH_OUT = ROOT / "dataset" / "rf_confusion_matrix.json"


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

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

prediction = rf.predict(X_test)

cf = confusion_matrix(y_test, prediction)

with open(PATH_OUT, "w") as fp:
    json.dump([
        row.tolist()
        for row in cf
    ], indent=4, fp=fp)
