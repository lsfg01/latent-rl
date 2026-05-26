"""Módulo de experimentación para comparación de agentes."""

from .utils import (
    load_yfinance_data,
    split_data,
    select_action_for_evaluation,
    evaluate_agent,
    aggregate_results,
    print_summary_table,
    print_ranking,
    export_dashboard_results,
)
from .runner import (
    train_dqn_agent,
    train_latent_dqn_agent,
    run_single_seed,
)

__all__ = [
    # utils
    "load_yfinance_data",
    "split_data",
    "select_action_for_evaluation",
    "evaluate_agent",
    "aggregate_results",
    "print_summary_table",
    "print_ranking",
    "export_dashboard_results",
    # runner
    "train_dqn_agent",
    "train_latent_dqn_agent",
    "run_single_seed",
]
