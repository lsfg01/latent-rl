"""Recalcula el IVL a partir de los resultados ya generados.

Lee agent_summary.csv generado por compare_agents_multiseed_experiment.py
y recalcula el IVL. Util para explorar el efecto de cambiar los pesos del
IVL sin necesidad de reejecutar el experimento completo.

Soporta tanto el formato de un ticker como el multi-ticker:
- Un ticker:     lee results/{ticker}/agent_summary.csv
- Multi-ticker:  procesa cada ticker por separado

Uso::

    python examples/compute_ivl_from_results.py

Salida::

    results/{ticker}/ivl_results.csv   (uno por cada ticker en config.tickers)
"""

from pathlib import Path

import pandas as pd

from latent_rl.experiments import ExperimentConfig
from latent_rl.experiments.runner import compute_ivl, export_ivl_results

# ── Personaliza aqui los parametros del IVL ──────────────────────────────────
config = ExperimentConfig(
    # Tickers a procesar (deben coincidir con los usados al generar los resultados)
    tickers=["SPY"],

    # Agente directo de referencia
    direct_agent="DQNAgent",

    # Agentes latentes a comparar contra el directo
    latent_agents=[
        "LatentDQNAgent (no pretrained)",
        "LatentDQNAgent (pretrained)",
    ],

    # Pesos de los componentes del IVL (deben sumar 1.0)
    # Modifica estos valores para explorar distintas definiciones del IVL
    ivl_weights={
        "sharpe":     0.25,
        "mdd":        0.25,
        "seed_std":   0.25,
        "is_oos_gap": 0.25,
    },

    results_dir="results",
)
# ─────────────────────────────────────────────────────────────────────────────


def _load_summary(ticker: str) -> tuple[pd.DataFrame, Path]:
    """
    Carga agent_summary.csv para un ticker.

    Busca primero en results/{ticker}/ (formato multi-ticker) y luego en
    results/ (formato plano legacy). Devuelve (DataFrame, directorio).

    Raises:
        FileNotFoundError: Si no se encuentra el archivo en ninguna ruta.
    """
    results_base = Path(config.results_dir)

    # Formato multi-ticker: results/{ticker}/agent_summary.csv
    ticker_dir = results_base / ticker
    ticker_path = ticker_dir / "agent_summary.csv"
    if ticker_path.exists():
        return pd.read_csv(ticker_path).set_index("agent_name"), ticker_dir

    # Fallback: formato plano legacy: results/agent_summary.csv
    flat_path = results_base / "agent_summary.csv"
    if flat_path.exists():
        return pd.read_csv(flat_path).set_index("agent_name"), results_base

    raise FileNotFoundError(
        f"No se encuentra agent_summary.csv para ticker='{ticker}'.\n"
        f"  Buscado en: {ticker_path}\n"
        f"  Buscado en: {flat_path}\n"
        "Ejecuta primero compare_agents_multiseed_experiment.py"
    )


def main() -> None:
    for ticker in config.tickers:
        print(f"\nProcesando ticker: {ticker}")
        print("-" * 60)

        summary_df, output_dir = _load_summary(ticker)

        if config.direct_agent not in summary_df.index:
            raise ValueError(
                f"No se encuentra '{config.direct_agent}' en los resultados de {ticker}. "
                f"Agentes disponibles: {list(summary_df.index)}"
            )

        # Reconstruir aggregated desde el CSV para reutilizar compute_ivl()
        aggregated = {
            name: {
                "name":               name,
                "mean_return_is":     row["mean_return_is"],
                "std_return_is":      row["std_return_is"],
                "mean_sharpe_is":     row["mean_sharpe_is"],
                "mean_mdd_is":        row["mean_mdd_is"],
                "mean_equity_is":     row["mean_equity_is"],
                "mean_return_oos":    row["mean_return_oos"],
                "std_return_oos":     row["std_return_oos"],
                "mean_sharpe_oos":    row["mean_sharpe_oos"],
                "mean_mdd_oos":       row["mean_mdd_oos"],
                "mean_equity_oos":    row["mean_equity_oos"],
                "mean_n_trades":      row["mean_n_trades"],
                "seed_std_return_is": row["seed_std_return_is"],
            }
            for name, row in summary_df.iterrows()
        }

        records = compute_ivl(aggregated, config)
        export_ivl_results(records, output_dir)


if __name__ == "__main__":
    main()
