"""Sweep robusto del brazo A sobre hidden_dim=64/128.

Uso:
    python -m scripts.experiments.a_hidden_sweep --smoke
    python -m scripts.experiments.a_hidden_sweep
    python -m scripts.experiments.a_hidden_sweep --results-dir results/custom_sweep
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from scripts.experiments.config import (
    EVAL_TICKERS,
    FEATURES,
    LOOKBACK,
    default_results_dir,
)
from latent_rl.experiments import ExperimentConfig, TickerConfig, run_experiment


def build_config(
    hidden_dim: int,
    smoke: bool = False,
    results_dir: str | None = None,
) -> ExperimentConfig:
    """Construye una configuracion A-only para un tamano de cabeza Q."""
    results_dir = results_dir or default_results_dir("a_hidden_sweep", smoke=smoke)
    start_date = "2020-01-01" if smoke else "2015-01-01"
    return ExperimentConfig(
        tickers=EVAL_TICKERS,
        ticker_configs=[TickerConfig("BTC-USD", start_date=start_date)],
        start_date=start_date,
        end_date="2024-01-01",
        train_ratio=0.7,
        interval="1d",
        cache_dir=".data_cache",
        features=FEATURES,
        normalize_features=True,
        lookback_window=LOOKBACK,
        initial_balance=10_000.0,
        transaction_cost=0.001,
        run_arms=["A"],
        direct_agent="A",
        latent_agents=[],
        seeds=[0] if smoke else [0, 1, 2, 3, 4],
        n_training_episodes=3 if smoke else 30,
        n_eval_episodes=1 if smoke else 3,
        max_steps_per_episode=100 if smoke else 500,
        random_start_train=True,
        reward_mode="log_return",
        reward_clip=None,
        trade_penalty=0.0,
        use_internal_validation=True,
        internal_val_ratio=0.2,
        validation_eval_freq=5,
        validation_patience=None,
        validation_score_mdd_weight=0.25,
        validation_score_trade_weight=0.05,
        dqn_lr=5e-4,
        dqn_gamma=0.99,
        dqn_hidden_dim=hidden_dim,
        dqn_batch_size=64,
        dqn_buffer_capacity=5_000,
        dqn_target_update=100,
        dqn_epsilon_start=1.0,
        dqn_epsilon_end=0.1,
        dqn_epsilon_decay=0.998,
        dqn_weight_decay=1e-4,
        dqn_grad_clip_norm=1.0,
        dqn_dropout=0.0,
        device="cpu",
        results_dir=str(Path(results_dir) / f"h{hidden_dim}"),
    )


def _collect_outputs(
    root: Path,
    hidden_sizes: List[int],
    tickers: List[str],
) -> None:
    summary_rows = []
    seed_frames = []

    for hidden_dim in hidden_sizes:
        for ticker in tickers:
            ticker_dir = root / f"h{hidden_dim}" / ticker
            summary = pd.read_csv(ticker_dir / "agent_summary.csv")
            summary = summary[summary["agent_name"] == "A"].copy()
            if summary.empty:
                continue
            row = summary.iloc[0].to_dict()
            row.update({"hidden_dim": hidden_dim, "ticker": ticker})

            validation_path = ticker_dir / "validation_metrics.csv"
            if validation_path.exists():
                validation = pd.read_csv(validation_path)
                validation = validation[validation["agent_name"] == "A"]
                for col in (
                    "best_val_score",
                    "best_val_sharpe",
                    "best_val_mdd",
                    "best_val_episode",
                ):
                    row[f"mean_{col}"] = float(validation[col].mean())
            summary_rows.append(row)

            seed_metrics = pd.read_csv(ticker_dir / "agent_seed_metrics.csv")
            seed_metrics = seed_metrics[seed_metrics["agent_name"] == "A"].copy()
            seed_metrics.insert(0, "ticker", ticker)
            seed_metrics.insert(0, "hidden_dim", hidden_dim)
            seed_frames.append(seed_metrics)

    required_summary_cols = [
        "hidden_dim", "ticker", "agent_name",
        "mean_return_is", "mean_sharpe_is", "mean_mdd_is",
        "mean_return_oos", "mean_sharpe_oos", "mean_mdd_oos",
        "std_return_oos", "seed_std_sharpe_oos", "mean_n_trades",
        "mean_best_val_score", "mean_best_val_sharpe", "mean_best_val_mdd",
        "mean_best_val_episode",
    ]
    summary_df = pd.DataFrame(summary_rows)
    for col in required_summary_cols:
        if col not in summary_df:
            summary_df[col] = pd.NA
    summary_df[required_summary_cols].to_csv(
        root / "hidden_sweep_summary.csv", index=False
    )

    if seed_frames:
        seed_df = pd.concat(seed_frames, ignore_index=True)
    else:
        seed_df = pd.DataFrame()
    required_seed_cols = [
        "hidden_dim", "ticker", "seed", "split", "total_reward",
        "total_return", "final_equity", "n_trades", "steps", "sharpe",
        "max_drawdown",
    ]
    for col in required_seed_cols:
        if col not in seed_df:
            seed_df[col] = pd.NA
    seed_df[required_seed_cols].to_csv(
        root / "hidden_sweep_seed_metrics.csv", index=False
    )


def run_hidden_sweep(
    smoke: bool = False,
    results_dir: str | None = None,
) -> Dict[int, ExperimentConfig]:
    results_dir = results_dir or default_results_dir("a_hidden_sweep", smoke=smoke)
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)
    hidden_sizes = [64] if smoke else [64, 128]
    configs: Dict[int, ExperimentConfig] = {}

    for hidden_dim in hidden_sizes:
        cfg = build_config(hidden_dim, smoke=smoke, results_dir=results_dir)
        configs[hidden_dim] = cfg
        print(f"\nEjecutando brazo A con hidden_dim={hidden_dim}")
        run_experiment(cfg)

    _collect_outputs(root, hidden_sizes, EVAL_TICKERS)
    config_payload = {
        "smoke": smoke,
        "hidden_sizes": hidden_sizes,
        "configs": {f"h{k}": asdict(v) for k, v in configs.items()},
    }
    with (root / "experiment_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config_payload, fh, indent=2, ensure_ascii=False)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep A-only de hidden_dim")
    parser.add_argument("--smoke", action="store_true", help="Ejecucion corta")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directorio de salida (default: results/<campaña>/)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_hidden_sweep(smoke=args.smoke, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
