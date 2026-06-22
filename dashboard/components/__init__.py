"""Modulo de componentes del dashboard."""

from .cards import create_kpi_card, create_ivl_card, create_ivl_kpi_card, create_delta_card
from .figures import (
    create_agent_return_bar,
    create_agent_metric_bar,
    create_seed_return_boxplot,
    create_seed_return_scatter,
    create_seed_metric_boxplot,
    create_seed_metric_scatter,
    create_ivl_delta_bar,
    create_cross_ticker_ivl_chart,
)

__all__ = [
    "create_kpi_card",
    "create_ivl_card",
    "create_ivl_kpi_card",
    "create_delta_card",
    "create_agent_return_bar",
    "create_agent_metric_bar",
    "create_seed_return_boxplot",
    "create_seed_return_scatter",
    "create_seed_metric_boxplot",
    "create_seed_metric_scatter",
    "create_ivl_delta_bar",
    "create_cross_ticker_ivl_chart",
]
