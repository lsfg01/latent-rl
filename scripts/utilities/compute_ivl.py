"""Recalcula el IVL a partir de resultados existentes.

Ejemplo:
    python -m scripts.utilities.compute_ivl --results-dir results/a_vs_d \
        --tickers SPY TSLA BTC-USD --direct-agent A --latent-agents D
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from latent_rl.experiments import ExperimentConfig
from latent_rl.experiments.runner import compute_ivl, export_ivl_results


DEFAULT_LATENT_AGENTS = [
    "LatentDQNAgent (no pretrained)",
    "LatentDQNAgent (pretrained)",
]


def build_config(
    *,
    results_dir: str = "results",
    tickers: list[str] | None = None,
    direct_agent: str = "DQNAgent",
    latent_agents: list[str] | None = None,
) -> ExperimentConfig:
    """Crea la configuración mínima necesaria para recalcular el IVL."""
    return ExperimentConfig(
        tickers=tickers or ["SPY"],
        direct_agent=direct_agent,
        latent_agents=latent_agents or DEFAULT_LATENT_AGENTS,
        ivl_weights={
            "sharpe": 0.25,
            "mdd": 0.25,
            "seed_std": 0.25,
            "is_oos_gap": 0.25,
        },
        results_dir=results_dir,
    )


def _load_summary(
    config: ExperimentConfig,
    ticker: str,
) -> tuple[pd.DataFrame, Path]:
    """Carga ``agent_summary.csv`` en formato multi-ticker o plano."""
    results_base = Path(config.results_dir)
    ticker_dir = results_base / ticker
    ticker_path = ticker_dir / "agent_summary.csv"
    if ticker_path.exists():
        return pd.read_csv(ticker_path).set_index("agent_name"), ticker_dir

    flat_path = results_base / "agent_summary.csv"
    if flat_path.exists():
        return pd.read_csv(flat_path).set_index("agent_name"), results_base

    raise FileNotFoundError(
        f"No se encuentra agent_summary.csv para ticker='{ticker}'.\n"
        f"  Buscado en: {ticker_path}\n"
        f"  Buscado en: {flat_path}\n"
        "Ejecuta primero una campaña experimental."
    )


def _rebuild_aggregated(summary_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Reconstruye el formato agregado que consume ``compute_ivl``."""
    metric_columns = [
        "mean_return_is",
        "std_return_is",
        "mean_sharpe_is",
        "mean_mdd_is",
        "mean_equity_is",
        "mean_return_oos",
        "std_return_oos",
        "mean_sharpe_oos",
        "mean_mdd_oos",
        "mean_equity_oos",
        "mean_n_trades",
        "seed_std_return_is",
        "seed_std_sharpe_oos",
    ]
    return {
        name: {"name": name, **{column: row[column] for column in metric_columns}}
        for name, row in summary_df.iterrows()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalcula el IVL desde resultados existentes",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tickers", nargs="+", default=["SPY"])
    parser.add_argument("--direct-agent", default="DQNAgent")
    parser.add_argument("--latent-agents", nargs="+", default=DEFAULT_LATENT_AGENTS)
    args = parser.parse_args()

    config = build_config(
        results_dir=args.results_dir,
        tickers=args.tickers,
        direct_agent=args.direct_agent,
        latent_agents=args.latent_agents,
    )

    for ticker in config.tickers:
        print(f"\nProcesando ticker: {ticker}")
        print("-" * 60)
        summary_df, output_dir = _load_summary(config, ticker)

        if config.direct_agent not in summary_df.index:
            raise ValueError(
                f"No se encuentra '{config.direct_agent}' en los resultados de "
                f"{ticker}. Agentes disponibles: {list(summary_df.index)}"
            )

        aggregated = _rebuild_aggregated(summary_df)

        records = compute_ivl(aggregated, config)
        export_ivl_results(records, output_dir)


if __name__ == "__main__":
    main()
