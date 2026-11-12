from __future__ import annotations

import os
from math import ceil, floor
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import (
    Dash,
    Input,
    Output,
    ctx,
    dash_table,
    dcc,
    html,
    no_update,
)
from plotly.subplots import make_subplots

from .data import (
    PredictionData,
    categorical_columns,
    load_clean_dataset,
    load_predictions,
    numeric_columns,
)
from .metrics import confusion_metrics, cramers_v_matrix, pearson_correlation

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8050
DEFAULT_DEBUG = False

DATA_SOURCE_NOTE = (
    "Data source: Adult Census Income (datasets/census_income.csv), cleaned to "
    "part_1/dataset/census_clean.csv. Model metrics use the Task 1 held-out "
    "test set."
)

VIZ_TYPE_OPTIONS = [
    {"label": "Table", "value": "Table"},
    {"label": "Histogram", "value": "Histogram"},
    {"label": "Correlation", "value": "Correlation"},
]

TARGET_LABELS = {True: ">50K", False: "<=50K"}
TARGET_COLORS = {False: "#8da0cb", True: "#e78ac3"}

MAX_HISTOGRAM_FEATURES = 4
HISTOGRAM_BINS = 30

CONFUSION_LABELS = ["<=50K", ">50K"]
RF_HARD_LABEL_THRESHOLD = 0.5


def pretty_column_name(column: str) -> str:
    """Convert a dataset column name into a readable table header."""
    return column.replace("-", " ").replace("_", " ").title()


def _table_columns(
    dataset: pd.DataFrame,
    selected_features: list[str] | None,
) -> list[str]:
    if not selected_features:
        return list(dataset.columns)
    return [column for column in selected_features if column in dataset.columns]


def _display_frame(dataset: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Build the display frame without mutating the loaded dataset."""
    display = dataset.loc[:, columns].copy()
    if "target" in display.columns:
        display["target"] = display["target"].map(TARGET_LABELS)
    return display


def _valid_features(
    dataset: pd.DataFrame,
    selected_features: list[str] | None,
) -> list[str]:
    return [
        column
        for column in (selected_features or [])
        if column in dataset.columns
    ]


def _is_numeric_feature(dataset: pd.DataFrame, column: str) -> bool:
    series = dataset[column]
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def _category_labels(values: pd.Index, column: str) -> list[str]:
    if column == "target":
        return [TARGET_LABELS[bool(value)] for value in values]
    return [str(value) for value in values]


def _build_table(
    dataset: pd.DataFrame,
    selected_features: list[str] | None,
) -> dash_table.DataTable | dbc.Alert:
    columns = _table_columns(dataset, selected_features)
    if not columns:
        return dbc.Alert(
            "No valid feature is selected. Choose at least one column.",
            color="warning",
            className="mb-0",
        )

    display = _display_frame(dataset, columns)
    table_columns = [
        {"name": pretty_column_name(column), "id": column}
        for column in columns
    ]

    return dash_table.DataTable(
        id="explorer-table",
        columns=table_columns,
        data=display.to_dict("records"),
        page_size=15,
        page_action="native",
        sort_action="native",
        filter_action="native",
        fixed_rows={"headers": True},
        export_format="csv",
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left",
            "padding": "0.45rem",
            "fontSize": "0.85rem",
            "minWidth": "90px",
            "maxWidth": "240px",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={
            "backgroundColor": "#f8f9fa",
            "fontWeight": "600",
            "borderBottom": "1px solid #dee2e6",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"}
        ],
    )


def _build_histograms(
    dataset: pd.DataFrame,
    selected_features: list[str] | None,
    split_by_target: bool = False,
) -> html.Div | dbc.Alert:
    features = _valid_features(dataset, selected_features)

    if not features:
        return dbc.Alert(
            "Select at least one feature to plot a distribution.",
            color="warning",
            className="mb-0",
        )

    if len(features) > MAX_HISTOGRAM_FEATURES:
        return dbc.Alert(
            "Select at most four features to compare distributions",
            color="warning",
            className="mb-0",
        )

    n_cols = min(2, len(features))
    n_rows = ceil(len(features) / n_cols)
    subplot_titles = [pretty_column_name(feature) for feature in features]

    subplot_kwargs: dict[str, float] = {}
    if n_rows > 1:
        subplot_kwargs["vertical_spacing"] = 0.16
    if n_cols > 1:
        subplot_kwargs["horizontal_spacing"] = 0.12

    figure = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        **subplot_kwargs,
    )

    for index, feature in enumerate(features):
        row = index // n_cols + 1
        column = index % n_cols + 1
        title = pretty_column_name(feature)
        show_legend = split_by_target and index == 0

        if _is_numeric_feature(dataset, feature):
            if split_by_target:
                for target_value in (False, True):
                    figure.add_trace(
                        go.Histogram(
                            x=dataset.loc[
                                dataset["target"].eq(target_value),
                                feature,
                            ],
                            nbinsx=HISTOGRAM_BINS,
                            name=TARGET_LABELS[target_value],
                            marker_color=TARGET_COLORS[target_value],
                            opacity=0.6,
                            showlegend=show_legend,
                        ),
                        row=row,
                        col=column,
                    )
            else:
                figure.add_trace(
                    go.Histogram(
                        x=dataset[feature],
                        nbinsx=HISTOGRAM_BINS,
                        name=title,
                        marker_color="#4c78a8",
                        showlegend=False,
                    ),
                    row=row,
                    col=column,
                )
            figure.update_xaxes(title_text=title, row=row, col=column)
        else:
            if split_by_target:
                counts = pd.crosstab(dataset[feature], dataset["target"])
                counts = counts.reindex(
                    columns=[False, True],
                    fill_value=0,
                )
                counts = counts.loc[
                    counts.sum(axis=1).sort_values(ascending=False).index
                ]
                positions = list(range(len(counts.index)))
                labels = _category_labels(counts.index, feature)
                bar_width = 0.38
                bar_offset = 0.21

                for target_value, direction in ((False, -1), (True, 1)):
                    figure.add_trace(
                        go.Bar(
                            x=[
                                position + direction * bar_offset
                                for position in positions
                            ],
                            y=counts[target_value],
                            width=bar_width,
                            name=TARGET_LABELS[target_value],
                            marker_color=TARGET_COLORS[target_value],
                            showlegend=show_legend,
                        ),
                        row=row,
                        col=column,
                    )

                figure.update_xaxes(
                    title_text=title,
                    tickmode="array",
                    tickvals=positions,
                    ticktext=labels,
                    tickangle=-45,
                    row=row,
                    col=column,
                )
            else:
                counts = dataset[feature].value_counts()
                figure.add_trace(
                    go.Bar(
                        x=_category_labels(counts.index, feature),
                        y=counts.to_numpy(),
                        name=title,
                        marker_color="#4c78a8",
                        showlegend=False,
                    ),
                    row=row,
                    col=column,
                )
                figure.update_xaxes(
                    title_text=title,
                    tickangle=-45,
                    row=row,
                    col=column,
                )

        figure.update_yaxes(
            title_text="Count",
            row=row,
            col=column,
        )

    if len(features) < n_rows * n_cols:
        figure.update_xaxes(visible=False, row=n_rows, col=n_cols)
        figure.update_yaxes(visible=False, row=n_rows, col=n_cols)

    figure.update_layout(
        title_text=(
            "Feature distributions split by income"
            if split_by_target
            else "Feature distributions"
        ),
        height=320 * n_rows,
        margin={"l": 60, "r": 20, "t": 70, "b": 60},
        bargap=0.05,
        barmode="overlay",
        showlegend=split_by_target,
    )
    if split_by_target:
        figure.update_layout(legend_title_text="Income")

    note = (
        "Numeric features use 30-bin histograms; categorical features use "
        "value-count bars sorted from most to least frequent."
    )
    if split_by_target:
        note = (
            "Distributions are split by observed income class; numeric "
            "histograms are overlaid at 60% opacity and categorical bars are "
            "grouped."
        )

    return html.Div(
        [
            dcc.Graph(
                figure=figure,
                config={"displaylogo": False, "responsive": True},
            ),
            html.P(note, className="text-muted small mt-2 mb-0"),
        ]
    )


def _correlation_heatmap(
    matrix: pd.DataFrame,
    title: str,
    colorscale: str,
    colorbar_title: str,
    note: str,
) -> html.Div:
    labels = [pretty_column_name(column) for column in matrix.columns]
    values = matrix.to_numpy()
    text = matrix.round(2).to_numpy()
    is_diverging = colorscale == "RdBu"

    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=labels,
            y=labels,
            zmin=-1 if is_diverging else 0,
            zmax=1,
            colorscale=colorscale,
            reversescale=is_diverging,
            colorbar={"title": colorbar_title},
            text=text,
            texttemplate="%{text:.2f}",
            hovertemplate="%{y} vs %{x}<br>%{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Feature",
        yaxis_title="Feature",
        height=max(420, 60 * len(labels) + 220),
        margin={"l": 70, "r": 20, "t": 70, "b": 90},
    )
    figure.update_xaxes(tickangle=-45)

    return html.Div(
        [
            dcc.Graph(
                figure=figure,
                config={"displaylogo": False, "responsive": True},
            ),
            html.P(note, className="text-muted small mt-2 mb-0"),
        ]
    )


def _build_correlation(
    dataset: pd.DataFrame,
    selected_features: list[str] | None,
    corr_method: str | None,
) -> html.Div | dbc.Alert:
    selected = _valid_features(dataset, selected_features)

    if corr_method == "categorical":
        columns = [
            column
            for column in categorical_columns(dataset)
            if column in selected
        ]
        if len(columns) < 2:
            return dbc.Alert(
                "Select at least two categorical features to compute "
                "Cramér's V.",
                color="warning",
                className="mb-0",
            )

        matrix = cramers_v_matrix(dataset, columns)
        return _correlation_heatmap(
            matrix,
            title="Cramér's V association between categorical features",
            colorscale="Blues",
            colorbar_title="Cramér's V",
            note=(
                "Cramér's V ranges from 0 (no association) to 1 (strong "
                "association); it is computed from contingency tables without "
                "one-hot encoding."
            ),
        )

    columns = [
        column
        for column in numeric_columns(dataset)
        if column in selected
    ]
    if len(columns) < 2:
        return dbc.Alert(
            "Select at least two numeric features to compute Pearson "
            "correlation.",
            color="warning",
            className="mb-0",
        )

    matrix = pearson_correlation(dataset, columns)
    return _correlation_heatmap(
        matrix,
        title="Pearson correlation between numeric features",
        colorscale="RdBu",
        colorbar_title="Pearson r",
        note=(
            "Pearson r ranges from -1 to 1; values near ±1 indicate a strong "
            "linear relationship."
        ),
    )


def _render_explorer(
    dataset: pd.DataFrame,
    viz_type: str | None,
    selected_features: list[str] | None,
    split_by_target: bool = False,
    corr_method: str | None = None,
) -> dash_table.DataTable | html.Div | dbc.Alert:
    if viz_type == "Table":
        return _build_table(dataset, selected_features)

    if viz_type == "Histogram":
        return _build_histograms(dataset, selected_features, split_by_target)

    if viz_type == "Correlation":
        return _build_correlation(dataset, selected_features, corr_method)

    return dbc.Alert(
        "Choose a visualisation type to continue.",
        color="warning",
        className="mb-0",
    )


def _score_bounds(predictions: PredictionData) -> tuple[float, float]:
    """Round the combined score range outward to two decimal places."""
    minimum = floor(predictions.score_min * 100) / 100
    maximum = ceil(predictions.score_max * 100) / 100
    return float(minimum), float(maximum)


def _threshold_marks(minimum: float, maximum: float) -> dict[float, str]:
    candidates = {minimum, maximum, 0.0, 0.25, 0.5, 0.75, 1.0}
    values = sorted(
        value for value in candidates if minimum <= value <= maximum
    )
    return {value: f"{value:g}" for value in values}


def _metric_card(label: str, value: float) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(label, className="text-muted small text-uppercase"),
                html.Div(f"{value:.2%}", className="fs-4 fw-semibold"),
            ],
            className="py-2",
        ),
        className="h-100 shadow-sm",
    )


def _confusion_figure(
    metrics: dict[str, int | float],
    title: str,
) -> dcc.Graph:
    values = [
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ]
    text = [[f"{int(value):,}" for value in row] for row in values]

    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=CONFUSION_LABELS,
            y=CONFUSION_LABELS,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
            hovertemplate=(
                "Actual %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Predicted label",
        yaxis_title="Actual label",
        height=360,
        margin={"l": 70, "r": 20, "t": 60, "b": 60},
    )

    return dcc.Graph(
        figure=figure,
        config={"displaylogo": False, "responsive": True},
    )


def _counts_line(metrics: dict[str, int | float]) -> html.P:
    return html.P(
        [
            html.Strong("TP"), f" {int(metrics['tp']):,}",
            " · ",
            html.Strong("FP"), f" {int(metrics['fp']):,}",
            " · ",
            html.Strong("FN"), f" {int(metrics['fn']):,}",
            " · ",
            html.Strong("TN"), f" {int(metrics['tn']):,}",
            " · ",
            html.Strong("Predicted positives"),
            f" {int(metrics['predicted_positive']):,}",
        ],
        className="text-muted small mb-3",
    )


def _comparison_table(
    nn_metrics: dict[str, int | float],
    rf_metrics: dict[str, int | float],
) -> dash_table.DataTable:
    rows = [
        {
            "Model": "Neural network",
            "Accuracy": f"{float(nn_metrics['accuracy']):.2%}",
            "Precision": f"{float(nn_metrics['precision']):.2%}",
            "Recall": f"{float(nn_metrics['recall']):.2%}",
            "F1": f"{float(nn_metrics['f1']):.2%}",
            "TP": int(nn_metrics["tp"]),
            "FP": int(nn_metrics["fp"]),
            "FN": int(nn_metrics["fn"]),
            "TN": int(nn_metrics["tn"]),
        },
        {
            "Model": "Random forest",
            "Accuracy": f"{float(rf_metrics['accuracy']):.2%}",
            "Precision": f"{float(rf_metrics['precision']):.2%}",
            "Recall": f"{float(rf_metrics['recall']):.2%}",
            "F1": f"{float(rf_metrics['f1']):.2%}",
            "TP": int(rf_metrics["tp"]),
            "FP": int(rf_metrics["fp"]),
            "FN": int(rf_metrics["fn"]),
            "TN": int(rf_metrics["tn"]),
        },
    ]

    return dash_table.DataTable(
        data=rows,
        columns=[
            {"name": column, "id": column}
            for column in (
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "TP",
                "FP",
                "FN",
                "TN",
            )
        ],
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "inherit",
            "fontSize": "0.875rem",
            "padding": "0.4rem 0.6rem",
            "textAlign": "center",
            "whiteSpace": "normal",
        },
        style_cell_conditional=[{"if": {"column_id": "Model"}, "textAlign": "left"}],
        style_header={"fontWeight": "600"},
        style_as_list_view=True,
    )


def _render_model_evaluation(
    predictions: PredictionData,
    threshold: float | None,
    compare_rf: bool,
) -> html.Div:
    current_threshold = (
        predictions.threshold_default if threshold is None else float(threshold)
    )
    nn_metrics = confusion_metrics(
        predictions.y_true,
        predictions.nn_score,
        current_threshold,
    )

    children: list = [
        html.H5("Thresholded model comparison", className="card-title mb-1"),
        html.P(
            [
                "Evaluation set: ",
                html.Strong(f"{predictions.size:,}"),
                " held-out rows. A row is predicted positive when its score "
                "is strictly greater than the threshold.",
            ],
            className="text-muted small mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    _metric_card("Accuracy", float(nn_metrics["accuracy"])),
                    xs=6,
                    md=3,
                ),
                dbc.Col(
                    _metric_card("Precision", float(nn_metrics["precision"])),
                    xs=6,
                    md=3,
                ),
                dbc.Col(
                    _metric_card("Recall", float(nn_metrics["recall"])),
                    xs=6,
                    md=3,
                ),
                dbc.Col(
                    _metric_card("F1", float(nn_metrics["f1"])),
                    xs=6,
                    md=3,
                ),
            ],
            className="g-2 mb-3",
        ),
        _counts_line(nn_metrics),
        html.P(
            "The neural network is a regressor; its scores are not calibrated "
            "probabilities.",
            className="text-muted small mb-3",
        ),
        html.P(
            "Task 1's report quotes a 0.362 cut-off, but its published metrics "
            "match a slightly higher effective threshold. The figures here are "
            "recomputed from the saved scores with the same strict "
            "score > threshold rule.",
            className="text-muted small fst-italic mb-4",
        ),
    ]

    nn_column = dbc.Col(
        _confusion_figure(
            nn_metrics,
            f"Neural network confusion matrix (threshold "
            f"{current_threshold:.3f})",
        ),
        xs=12,
        lg=6 if compare_rf else 12,
    )

    if compare_rf:
        if predictions.rf_has_proba:
            rf_threshold = current_threshold
            rf_metrics = confusion_metrics(
                predictions.y_true,
                predictions.rf_score,
                rf_threshold,
            )
            rf_title = (
                "Random forest confusion matrix (threshold "
                f"{rf_threshold:.3f})"
            )
        else:
            rf_threshold = RF_HARD_LABEL_THRESHOLD
            rf_metrics = confusion_metrics(
                predictions.y_true,
                predictions.rf_score,
                rf_threshold,
            )
            rf_title = (
                "Random forest confusion matrix "
                "(hard labels at the 0.5 cut-off)"
            )

        children.append(
            dbc.Row(
                [
                    nn_column,
                    dbc.Col(
                        _confusion_figure(rf_metrics, rf_title),
                        xs=12,
                        lg=6,
                    ),
                ],
                className="g-3",
            )
        )
        children.append(html.Div(className="mt-3"))
        children.append(
            html.H6("Model comparison", className="text-uppercase small")
        )
        children.append(_comparison_table(nn_metrics, rf_metrics))
        if not predictions.rf_has_proba:
            children.append(
                html.P(
                    "RF has no saved probabilities; this matrix is fixed at "
                    "the 0.5 hard-label cut-off.",
                    className="text-warning-emphasis small mt-2 mb-0",
                )
            )
    else:
        children.append(dbc.Row([nn_column], className="g-3"))

    return html.Div(children)


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
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Visualisation",
                                        html_for="viz-type",
                                    ),
                                    dcc.Dropdown(
                                        id="viz-type",
                                        options=VIZ_TYPE_OPTIONS,
                                        value="Table",
                                        clearable=False,
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Features",
                                        html_for="feature-select",
                                    ),
                                    dcc.Dropdown(
                                        id="feature-select",
                                        options=[
                                            {
                                                "label": pretty_column_name(
                                                    column
                                                ),
                                                "value": column,
                                            }
                                            for column in dataset.columns
                                        ],
                                        value=list(dataset.columns),
                                        multi=True,
                                        clearable=False,
                                    ),
                                ],
                                md=8,
                            ),
                        ],
                        className="g-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    "Select all",
                                    id="select-all",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Clear features",
                                    id="clear-features",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                ),
                                width="auto",
                            ),
                        ],
                        className="g-2 mt-1",
                    ),
                    dbc.Row(
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Correlation method",
                                    html_for="corr-method",
                                ),
                                dbc.RadioItems(
                                    id="corr-method",
                                    options=[
                                        {
                                            "label": "Numeric (Pearson)",
                                            "value": "numeric",
                                        },
                                        {
                                            "label": "Categorical (Cramér's V)",
                                            "value": "categorical",
                                        },
                                    ],
                                    value="numeric",
                                    inline=True,
                                    className="small",
                                ),
                            ],
                            md=8,
                        ),
                        className="g-3 mt-1",
                    ),
                    dbc.Checklist(
                        id="split-by-target",
                        options=[
                            {
                                "label": "Split distributions by income",
                                "value": "split",
                            }
                        ],
                        value=[],
                        switch=True,
                        inline=True,
                        className="small mt-2 mb-0",
                    ),
                    html.Hr(className="my-3"),
                    html.Div(
                        [
                            html.Strong(f"{len(dataset):,}"),
                            " rows · ",
                            html.Strong(f"{len(dataset.columns)}"),
                            " columns · ",
                            html.Strong(f"{positive_rate:.2%}"),
                            " positive class",
                        ],
                        id="explorer-summary",
                        className="text-muted small mb-2",
                    ),
                    dcc.Loading(
                        html.Div(id="explorer-output"),
                        type="default",
                    ),
                ]
            ),
        ],
        className="shadow-sm",
    )

    threshold_min, threshold_max = _score_bounds(predictions)
    model_tab = dbc.Card(
        [
            dbc.CardHeader("Model evaluation"),
            dbc.CardBody(
                [
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5(
                                    "Decision threshold",
                                    className="card-title",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Label(
                                                    "Score threshold",
                                                    html_for="threshold-slider",
                                                ),
                                                dcc.Slider(
                                                    id="threshold-slider",
                                                    min=threshold_min,
                                                    max=threshold_max,
                                                    step=0.001,
                                                    value=predictions.threshold_default,
                                                    marks=_threshold_marks(
                                                        threshold_min,
                                                        threshold_max,
                                                    ),
                                                    tooltip={
                                                        "placement": "bottom",
                                                        "always_visible": True,
                                                    },
                                                    className="mb-4",
                                                ),
                                            ],
                                            md=8,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Switch(
                                                    id="compare-rf",
                                                    label=(
                                                        "Compare with Random "
                                                        "Forest"
                                                    ),
                                                    value=False,
                                                ),
                                                html.P(
                                                    "The neural network score "
                                                    "is thresholded on every "
                                                    "change; the random forest "
                                                    "comparison uses saved "
                                                    "artefacts only.",
                                                    className="text-muted "
                                                    "small mb-0",
                                                ),
                                            ],
                                            md=4,
                                            className="pt-md-4",
                                        ),
                                    ],
                                    className="g-3",
                                ),
                            ]
                        ),
                        className="bg-light border-0",
                    ),
                    dcc.Loading(
                        html.Div(id="model-output"),
                        type="default",
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

    @app.callback(
        Output("explorer-output", "children"),
        Input("viz-type", "value"),
        Input("feature-select", "value"),
        Input("split-by-target", "value"),
        Input("corr-method", "value"),
    )
    def update_explorer(
        viz_type: str | None,
        selected_features: list[str] | None,
        split_value: list[str] | None,
        corr_method: str | None,
    ) -> dash_table.DataTable | html.Div | dbc.Alert:
        split_by_target = "split" in (split_value or [])
        return _render_explorer(
            dataset,
            viz_type,
            selected_features,
            split_by_target,
            corr_method,
        )

    @app.callback(
        Output("model-output", "children"),
        Input("threshold-slider", "value"),
        Input("compare-rf", "value"),
    )
    def update_model_evaluation(
        threshold: float | None,
        compare_rf: bool | None,
    ) -> html.Div:
        return _render_model_evaluation(
            predictions,
            threshold,
            bool(compare_rf),
        )

    @app.callback(
        Output("feature-select", "value"),
        Input("select-all", "n_clicks"),
        Input("clear-features", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_feature_selection(
        _select_all_clicks: int | None,
        _clear_clicks: int | None,
    ):
        if ctx.triggered_id == "select-all":
            return list(dataset.columns)
        if ctx.triggered_id == "clear-features":
            return []
        return no_update

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
