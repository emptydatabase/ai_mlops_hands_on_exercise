from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import (
    CLEAN_COLUMN_ORDER,
    categorical_columns,
    numeric_columns,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "dataset" / "census_clean.csv"
DEFAULT_PREDICTIONS_DIR = PROJECT_ROOT / "data"

DATA_PATH_ENV = "CENSUS_DATA_PATH"
PREDICTIONS_DIR_ENV = "PREDICTIONS_DIR"

MISSING_DATA_MESSAGE = (
    "Cleaned dataset not found. Run: cd part_1 && "
    "uv run python scripts/clean_dataset.py"
)

NN_PREDICTIONS_FILE = "nn_pred.csv"
RF_PREDICTIONS_FILE = "rf_pred.csv"

DEFAULT_THRESHOLD = 0.362
FALLBACK_THRESHOLD = 0.5


@dataclass(frozen=True)
class PredictionData:
    """Task 1 predictions for the held-out test set."""

    y_true: np.ndarray
    nn_score: np.ndarray
    rf_score: np.ndarray
    rf_has_proba: bool
    threshold_default: float

    @property
    def size(self) -> int:
        return int(len(self.y_true))

    @property
    def score_min(self) -> float:
        return float(
            min(self.nn_score.min(), self.rf_score.min())
        )

    @property
    def score_max(self) -> float:
        return float(
            max(self.nn_score.max(), self.rf_score.max())
        )


def _resolve_data_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get(DATA_PATH_ENV, DEFAULT_DATA_PATH))


def _resolve_predictions_dir(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get(PREDICTIONS_DIR_ENV, DEFAULT_PREDICTIONS_DIR))


def load_clean_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Task 2 cleaned dataset and validate its schema."""
    data_path = _resolve_data_path(path)

    if not data_path.exists():
        raise FileNotFoundError(MISSING_DATA_MESSAGE)

    df = pd.read_csv(data_path)

    missing = [column for column in CLEAN_COLUMN_ORDER if column not in df.columns]
    if missing:
        raise ValueError(
            f"Cleaned dataset at {data_path} is missing required columns: {missing}"
        )

    df = df.copy()
    df["target"] = df["target"].astype(bool)
    df["income"] = df["income"].astype("string")

    return df


def _read_prediction_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}. "
            "Run: cd part_1 && uv run python run.py"
        )
    return pd.read_csv(path, index_col=0)


def load_predictions(pred_dir: str | Path | None = None) -> PredictionData:
    """Load the Task 1 predictions and align the two model outputs.

    The NN file contains continuous regressor scores. The RF file contains
    hard labels in the current artefacts; if a ``proba`` column is added
    later, it is used automatically.
    """
    predictions_dir = _resolve_predictions_dir(pred_dir)

    nn = _read_prediction_file(predictions_dir / NN_PREDICTIONS_FILE)
    rf = _read_prediction_file(predictions_dir / RF_PREDICTIONS_FILE)

    required_columns = {"real", "pred"}
    for name, frame in (("nn_pred.csv", nn), ("rf_pred.csv", rf)):
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )

    if not nn.index.equals(rf.index):
        raise ValueError(
            "nn_pred.csv and rf_pred.csv are not in the same row order."
        )

    y_true = nn["real"].astype(int).to_numpy()
    rf_y_true = rf["real"].astype(int).to_numpy()

    if not np.array_equal(y_true, rf_y_true):
        raise ValueError(
            "nn_pred.csv and rf_pred.csv do not contain the same real labels."
        )

    nn_score = nn["pred"].astype(float).to_numpy()

    rf_has_proba = "proba" in rf.columns
    if rf_has_proba:
        rf_score = rf["proba"].astype(float).to_numpy()
    else:
        rf_score = rf["pred"].astype(int).to_numpy()

    combined = np.concatenate([nn_score, rf_score])
    if combined.min() < DEFAULT_THRESHOLD < combined.max():
        threshold_default = DEFAULT_THRESHOLD
    else:
        threshold_default = FALLBACK_THRESHOLD

    return PredictionData(
        y_true=y_true,
        nn_score=nn_score,
        rf_score=rf_score,
        rf_has_proba=rf_has_proba,
        threshold_default=threshold_default,
    )


__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_PREDICTIONS_DIR",
    "PredictionData",
    "categorical_columns",
    "load_clean_dataset",
    "load_predictions",
    "numeric_columns",
]
