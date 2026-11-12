from __future__ import annotations

import os
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, html

from .data import PredictionData, load_clean_dataset, load_predictions

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8050
DEFAULT_DEBUG = False

DATA_SOURCE_NOTE = (
    "Data source: Adult Census Income (datasets/census_income.csv), cleaned to "
    "part_1/dataset/census_clean.csv. Model metrics use the Task 1 held-out "
    "test set."
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_layout(dataset: pd.DataFrame, predictions: PredictionData) -> dbc.Container:
    positive_rate = float(dataset["target"].mean())

    header = dbc.Container(
        [
            html.H1(
                "Adult Census Income Explorer",
                className="display-6 mb-1",
            ),
            html.P(
                "Explore the cleaned census data and the Task 1 models on "
                "held-out data.",
                className="lead text-muted mb-0",
            ),
        ],
        fluid=True,
        className="py-3",
    )

    data_tab = dbc.Card(
        [
            dbc.CardHeader("Data explorer"),
            dbc.CardBody(
                [
                    html.H5("Interactive exploration", className="card-title"),
                    html.P(
                        "The table view, feature histograms, and correlation "
                        "views will be added in the next prompts.",
                        className="card-text",
                    ),
                    html.P(
                        [
                            html.Strong(f"{len(dataset):,}"),
                            " rows · ",
                            html.Strong(f"{len(dataset.columns)}"),
                            " columns · ",
                            html.Strong(f"{positive_rate:.2%}"),
                            " positive class",
                        ],
                        className="text-muted mb-0",
                    ),
                ]
            ),
        ],
        className="shadow-sm",
    )

    model_tab = dbc.Card(
        [
            dbc.CardHeader("Model evaluation"),
            dbc.CardBody(
                [
                    html.H5("Thresholded model comparison", className="card-title"),
                    html.P(
                        "The score threshold, confusion matrices, and random "
                        "forest comparison will be added in the next prompts.",
                        className="card-text",
                    ),
                    html.P(
                        [
                            "Evaluation set: ",
                            html.Strong(f"{predictions.size:,}"),
                            " held-out rows",
                        ],
                        className="text-muted mb-0",
                    ),
                ]
            ),
        ],
        className="shadow-sm",
    )

    tabs = dbc.Tabs(
        [
            dbc.Tab(data_tab, label="Data explorer", tab_id="data-explorer"),
            dbc.Tab(model_tab, label="Model evaluation", tab_id="model-evaluation"),
        ],
        id="main-tabs",
        active_tab="data-explorer",
        className="mb-3",
    )

    footer = html.Footer(
        html.P(DATA_SOURCE_NOTE, className="text-muted small mb-0"),
        className="mt-4 pt-3 border-top",
    )

    return dbc.Container(
        [header, tabs, footer],
        fluid=True,
        className="px-3 px-md-4 pb-4",
    )


def create_app(
    data_path: str | Path | None = None,
    pred_dir: str | Path | None = None,
) -> Dash:
    """Create the Dash application and load its data exactly once."""
    dataset = load_clean_dataset(data_path)
    predictions = load_predictions(pred_dir)

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="Adult Census Income Explorer",
    )
    app.layout = _build_layout(dataset, predictions)

    @app.server.route("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
server = app.server


def main() -> None:
    """Run the development server using environment-based configuration."""
    host = os.environ.get("HOST", DEFAULT_HOST)
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    debug = _env_flag("DASH_DEBUG", DEFAULT_DEBUG)

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
