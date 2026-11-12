from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from .feature_selection import feature_selection
from .models import neural_network_predict, random_forest_predict
from .plot import plot_curve
from .roc_prc import roc_prc, best_point


def part_1_main(
        dataset: Path,
        out_data: Path,
        n_features: int | None = 64,
        random_state: int = 42,
):
    df: DataFrame = pd.read_csv(dataset)

    print("- Feature selection")
    df = feature_selection(
        df=df,
        random_state=random_state,
        n_features=n_features
    )

    y = df.pop("target")
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    print("- RF predict")
    rf_pred = random_forest_predict(X_train, X_test, y_train, random_state)
    print("- NN predict")
    nn_pred = neural_network_predict(X_train, X_test, y_train.astype(np.float32), random_state)

    print("- Save results")


    # Save pred
    DataFrame({
        "real": y_test,
        "pred": rf_pred
    }).to_csv(out_data / "rf_pred.csv")

    DataFrame({
        "real": y_test.astype(np.float32),
        "pred": nn_pred
    }).to_csv(out_data / "nn_pred.csv")


def part_1_plot(
        data: Path,
        out_plots: Path,
):
    rf_df = pd.read_csv(data / "rf_pred.csv")
    nn_df = pd.read_csv(data / "nn_pred.csv")

    # Save curves
    roc, prc = roc_prc(nn_df["real"], nn_df["pred"])
    roc.to_csv(data / "nn_roc.csv")
    prc.to_csv(data / "nn_prc.csv")

    # Best
    best_roc = best_point(roc, "fpr", "tpr", 0, 1)
    best_prc = best_point(prc, "precision", "recall", 1, 1)

    # Plot curves
    plot_curve(
        roc,
        "fpr",
        "tpr",
        best_roc,
        "ROC curve",
        "False Positive Rate",
        "True Positive Rate",
        "NN ROC curve",
        out_plots / "nn_roc.png"
    )
    plot_curve(
        prc,
        "precision",
        "recall",
        best_prc,
        "PRC curve",
        "Precision",
        "Recall",
        "NN PRC curve",
        out_plots / "nn_prc.png"
    )
