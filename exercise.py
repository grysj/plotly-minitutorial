"""Exercise: Business time-series dashboard (Darts + Plotly + Dash).

Goal
- Practice building an interactive dashboard around a real time-series dataset.
- Use Darts to load data and generate a simple forecast.

Exercise focus
- The dashboard runs as-is, but the plots are intentionally barebones.
- Your task is to improve the figures by implementing the TODOs inside the code
    (legend, colors, templates, hover mode, axis formatting, annotations, etc.).

Run
- `uv run python exercise.py`
"""

from __future__ import annotations

import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from darts.datasets import AirPassengersDataset
from darts.metrics import mape
from darts.models import NaiveDrift, NaiveSeasonal

warnings.filterwarnings("ignore")

APP_TITLE = "Exercise: Time-series dashboard"
DEFAULT_SEASONAL_PERIOD = 12  # monthly seasonality for AirPassengers


def load_series():
    """Load a public business-ish time series via Darts.

    AirPassengers: monthly international airline passengers (1949-1960).
    """

    # Darts will download/cache the dataset automatically.
    return AirPassengersDataset().load()


SERIES = load_series()


def series_to_frame(series) -> pd.DataFrame:

    if hasattr(series, "pd_dataframe"):
        df = series.pd_dataframe()
    else:
        df = series.to_dataframe()

    df = df.reset_index()
    date_col = df.columns[0]

    value_cols = [c for c in df.columns if c != date_col]
    if len(value_cols) != 1:
        raise ValueError(
            "Expected a univariate time series; got columns: "
            + ", ".join(map(str, value_cols))
        )

    df = df.rename(columns={date_col: "date", value_cols[0]: "value"})
    return df[["date", "value"]]


def make_raw_series_figure(df: pd.DataFrame) -> go.Figure:
    fig = px.line(df, x="date", y="value")

    # TODO: Add a title.

    # TODO: Add y-axis title and tick formatting (thousands separators).

    # TODO: Set a template (e.g. "plotly_white" / "plotly_dark").

    # TODO: Pick a consistent color and line width.

    # TODO: Turn legend on and move it (top/bottom/right).
    fig.update_layout(showlegend=False)

    # TODO: Improve hover (e.g. unified hover on x).

    return fig


def make_rolling_figure(df: pd.DataFrame, window: int) -> go.Figure:
    data = df.copy()
    data["rolling_mean"] = data["value"].rolling(window=window).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["value"],
            name="Raw",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["rolling_mean"],
            name=f"Rolling mean ({window})",
            mode="lines",
        )
    )

    # Hide legend by default so students add it.
    # TODO: Enable legend and position it neatly.
    fig.update_layout(showlegend=False)

    # TODO: Add a title and axis labels.


    # TODO: Style the rolling mean differently (color, width) and make raw line lighter.

    # TODO: Improve hover mode (e.g. unified hover on x) and add annotations.

    return fig


def fit_and_forecast(model_name: str, horizon: int, season_length: int):
    # Cutoff is chosen so that we always have `horizon` ground-truth points
    # to evaluate on (unless horizon is too large).
    horizon = int(max(1, horizon))
    if horizon >= len(SERIES):
        horizon = len(SERIES) - 1

    train, test = SERIES[:-horizon], SERIES[-horizon:]

    if model_name == "naive_seasonal":
        model = NaiveSeasonal(K=season_length)
    elif model_name == "naive_drift":
        model = NaiveDrift()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(train)
    forecast = model.predict(horizon)

    error = float(mape(test, forecast))
    return train, test, forecast, error


def make_forecast_figure(train, test, forecast) -> go.Figure:
    df_train = series_to_frame(train)
    df_test = series_to_frame(test)
    df_fc = series_to_frame(forecast)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_train["date"],
            y=df_train["value"],
            name="Train",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_test["date"],
            y=df_test["value"],
            name="Test",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_fc["date"],
            y=df_fc["value"],
            name="Forecast",
            mode="lines",
        )
    )

    # Hide legend by default so students add it.
    # TODO: Enable legend and position it.
    fig.update_layout(showlegend=False)

    # TODO: Add a title, axis titles, template, hover mode.

    # TODO: Make each series visually distinct (color, width, dashes, markers).

    # TODO: Add a vertical cutoff line at the train/test boundary.

    # TODO: Shade the forecast region (after cutoff) with `add_vrect`.

    return fig


app = Dash(__name__)
app.title = APP_TITLE

raw_df = series_to_frame(SERIES)

app.layout = html.Div(
    [
        html.H1(APP_TITLE),
        html.P(
            "Use the controls to produce a simple baseline forecast."
        ),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Overview",
                    children=[
                        dcc.Graph(figure=make_raw_series_figure(raw_df)),
                        html.Div(
                            [
                                html.Label("Rolling window"),
                                dcc.Slider(
                                    id="roll-window",
                                    min=3,
                                    max=24,
                                    step=1,
                                    value=12,
                                    marks={i: str(i) for i in [3, 6, 12, 18, 24]},
                                ),
                            ],
                            style={"padding": "8px 0"},
                        ),
                        dcc.Graph(id="rolling-graph"),
                    ],
                ),
                dcc.Tab(
                    label="Forecast",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Model"),
                                        dcc.Dropdown(
                                            id="model-name",
                                            options=[
                                                {
                                                    "label": "Naive seasonal (repeat last season)",
                                                    "value": "naive_seasonal",
                                                },
                                                {
                                                    "label": "Naive drift (linear extrapolation)",
                                                    "value": "naive_drift",
                                                },
                                            ],
                                            value="naive_seasonal",
                                            clearable=False,
                                        ),
                                    ],
                                    style={"width": "48%", "display": "inline-block"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Forecast horizon (months)"),
                                        dcc.Slider(
                                            id="horizon",
                                            min=6,
                                            max=48,
                                            step=1,
                                            value=24,
                                            marks={i: str(i) for i in [6, 12, 24, 36, 48]},
                                        ),
                                    ],
                                    style={"width": "48%", "display": "inline-block", "float": "right"},
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Label("Season length (only for Naive seasonal)"),
                                dcc.Slider(
                                    id="season-length",
                                    min=2,
                                    max=24,
                                    step=1,
                                    value=DEFAULT_SEASONAL_PERIOD,
                                    marks={i: str(i) for i in [4, 6, 12, 18, 24]},
                                ),
                            ],
                            style={"padding": "10px 0"},
                        ),
                        html.Div(id="kpi-row"),
                        dcc.Graph(id="forecast-graph"),
                    ],
                ),
            ]
        ),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"},
)


@app.callback(Output("rolling-graph", "figure"), Input("roll-window", "value"))
def update_rolling(window: int):
    window = int(max(2, window))
    return make_rolling_figure(raw_df, window=window)


@app.callback(
    Output("forecast-graph", "figure"),
    Output("kpi-row", "children"),
    Input("model-name", "value"),
    Input("horizon", "value"),
    Input("season-length", "value"),
)
def update_forecast(model_name: str, horizon: int, season_length: int):
    train, test, forecast, error = fit_and_forecast(
        model_name=model_name,
        horizon=int(horizon),
        season_length=int(season_length),
    )

    kpis = html.Div(
        [
            html.B(f"MAPE: {error:.2f}%"),
            html.Span("  "),
            html.Span(f"Train points: {len(train)} | Test points: {len(test)}"),
        ],
        style={"padding": "6px 0"},
    )

    # TODO: Style KPIs as a compact row with 2-3 cards (MAPE, last actual, last forecast).

    return make_forecast_figure(train, test, forecast), kpis


if __name__ == "__main__":
    app.run(debug=True)
