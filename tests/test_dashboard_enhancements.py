"""Cobertura ligera de los controles multiagente del dashboard."""

import pandas as pd
import plotly.graph_objects as go
from dash import dcc

from dashboard import app as dashboard_app
from dashboard.components.figures import (
    create_agent_metric_bar,
    create_cross_ticker_ivl_chart,
    create_ivl_delta_bar,
    create_seed_metric_boxplot,
)


def _ivl_records():
    return [
        {
            "ticker": "SPY",
            "latent_agent": agent,
            "ivl": ivl,
            "interpretation": "latent_advantage" if ivl > 0 else "direct_advantage",
            "delta_sharpe": ivl,
            "delta_mdd": -ivl,
            "delta_seed_std": -ivl / 2,
            "delta_is_oos_gap": -ivl / 3,
        }
        for agent, ivl in (("B", 0.2), ("C", 0.5), ("D", -0.1))
    ]


def test_dashboard_ivl_callback_handles_all_latent_agents():
    output = dashboard_app.update_ivl_section({"ivl_results": _ivl_records()}, "ALL")

    assert len(output) == 5
    assert output[0].children[1].children == "C · LatentDQN-IS-Frozen"
    assert isinstance(output[4], dcc.Graph)
    assert len(output[4].figure.data) == 3


def test_dashboard_ivl_component_figure_can_filter_one_agent():
    frame = pd.DataFrame(_ivl_records())
    figure = create_ivl_delta_bar(frame[frame["latent_agent"] == "B"])

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1
    assert figure.data[0].name == "B · LatentDQN-FT"


def test_dashboard_metric_and_seed_figures_use_requested_metric():
    summary = pd.DataFrame([
        {
            "agent_name": "A",
            "mean_sharpe_oos": 0.4,
            "mean_return_oos": 0.1,
            "mean_mdd_oos": -0.2,
            "mean_sharpe_is": 1.0,
            "mean_n_trades": 4,
            "mean_equity_oos": 11_000,
        },
        {
            "agent_name": "C",
            "mean_sharpe_oos": 0.7,
            "mean_return_oos": 0.2,
            "mean_mdd_oos": -0.25,
            "mean_sharpe_is": 0.9,
            "mean_n_trades": 2,
            "mean_equity_oos": 12_000,
        },
    ])
    seeds = pd.DataFrame([
        {"agent_name": agent, "seed": seed, "split": "oos", "sharpe": value}
        for agent, value in (("A", 0.4), ("C", 0.7))
        for seed in (0, 1)
    ])

    metric_figure = create_agent_metric_bar(summary, "sharpe_gap")
    seed_figure = create_seed_metric_boxplot(seeds, "sharpe")

    assert isinstance(metric_figure, go.Figure)
    assert "Gap Sharpe" in metric_figure.layout.title.text
    assert isinstance(seed_figure, go.Figure)
    assert "Sharpe OOS" in seed_figure.layout.title.text


def test_cross_ticker_heatmap_adds_mean_and_excludes_std():
    frame = pd.DataFrame(
        _ivl_records()
        + [{**row, "ticker": "TSLA", "ivl": row["ivl"] + 0.1}
           for row in _ivl_records()]
        + [{**_ivl_records()[0], "ticker": "STD", "ivl": 99.0}]
    )

    figure = create_cross_ticker_ivl_chart(frame, view="heatmap")

    assert isinstance(figure, go.Figure)
    assert list(figure.data[0].x) == ["SPY", "TSLA", "MEAN"]


def test_dashboard_has_no_memory_view():
    assert "Memoria" not in str(dashboard_app.app.layout)
