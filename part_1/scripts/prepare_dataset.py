from pathlib import Path

import pandas as pd
from pandas import DataFrame

# from part_1.utils import check_one_to_one_correspondence

ROOT = Path(__file__).parent.parent
PATH_IN = ROOT.parent / "datasets" / "census_income.csv"
PATH_OUT = ROOT / "dataset" / "census.csv"

df: DataFrame = pd.read_csv(PATH_IN)
initial_count = len(df.index)


# Drop invalid rows
df = df[df[" capital-gain"] != 99999]

drop_count = len(df.index)
print(f"{(initial_count - drop_count)*100/initial_count:.2f}% rows deleted")


# Drop features
df = df.drop([
    " fnlwgt",
    " education-num",
], axis=1)


# Categorical features
one_hot = []

for col in [
    " workclass",
    " education",
    " marital-status",
    " occupation",
    " relationship",
    " race",
    " sex",
    " native-country"
]:
    one_hot.append(pd.get_dummies(df.pop(col), prefix=col.strip()))

df = pd.concat([df, *one_hot], axis=1)


# Boolean target
df["target"] = df.pop(" income").isin([" >50K", " >50K."])


# Save
df.to_csv(PATH_OUT, index=False)
print("Dataset saved")
#print(json.dumps(list(df.columns), indent=4))
