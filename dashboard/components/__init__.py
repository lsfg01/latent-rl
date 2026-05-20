"""Módulo de componentes del dashboard."""

from .cards import create_kpi_card, create_ivl_card, create_ivl_kpi_card, create_delta_card
from .figures import (
    create_agent_return_bar,
    create_seed_return_boxplot,
    create_seed_return_scatter,
    create_ivl_delta_bar,
    create_price_vs_latent_plot,
    create_latent_indices_plot,
    create_pca_vs_next_return_scatter
)

__all__ = [
    "create_kpi_card",
    "create_ivl_card",
    "create_ivl_kpi_card",
    "create_delta_card",
    "create_agent_return_bar",
    "create_seed_return_boxplot",
    "create_seed_return_scatter",
    "create_ivl_delta_bar",
    "create_price_vs_latent_plot",
    "create_latent_indices_plot",
    "create_pca_vs_next_return_scatter"
]