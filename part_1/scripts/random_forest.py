import json
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).parent.parent
PATH_IN = ROOT / "dataset" / "census.csv"
PATH_OUT = ROOT / "dataset" / "feature_importance.json"

df: DataFrame = pd.read_csv(PATH_IN)

random_state = 42 # The only allowed one


y = df.pop("target")
X = df

rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
rf.fit(X, y)

importance = list(zip(
    df.columns,
    rf.feature_importances_
))
importance.sort(key=lambda item: item[1], reverse=True)

with open(PATH_OUT, "w") as fp:
    json.dump(dict(importance), indent=4, fp=fp)
