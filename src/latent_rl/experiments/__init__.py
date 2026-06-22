"""Modulo de experimentacion para comparacion de agentes."""

from .config import ExperimentConfig, TickerConfig
from .utils import (
    load_yfinance_data,
    load_tickers_data,
    load_ticker_with_config,
    split_data,
    normalize_is_oos,
    normalize_train_val_oos,
    split_internal_validation,
    walk_forward_splits,
    load_context_features,
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
    compute_ivl,
    export_ivl_results,
    aggregate_ticker_results,
    export_cross_ticker_results,
    run_experiment,
)

__all__ = [
    # config
    "ExperimentConfig",
    # utils
    "load_yfinance_data",
    "load_tickers_data",
    "split_data",
    "normalize_is_oos",
    "normalize_train_val_oos",
    "split_internal_validation",
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
    "compute_ivl",
    "export_ivl_results",
    "aggregate_ticker_results",
    "export_cross_ticker_results",
    "run_experiment",
]
