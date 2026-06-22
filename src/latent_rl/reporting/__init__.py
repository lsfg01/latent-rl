"""Carga y visualizacion reproducible de resultados experimentales."""

from .agent_metadata import (
    AGENT_COLORS,
    AGENT_DESCRIPTIONS,
    AGENT_DISPLAY,
    AGENT_ORDER,
    agent_display_name,
)
from .results_loader import AggregatedResults, load_aggregated_results

__all__ = [
    "AGENT_COLORS",
    "AGENT_DESCRIPTIONS",
    "AGENT_DISPLAY",
    "AGENT_ORDER",
    "AggregatedResults",
    "agent_display_name",
    "load_aggregated_results",
]
