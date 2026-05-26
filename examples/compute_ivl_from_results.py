"""Recalcula el IVL a partir de los resultados ya generados.

Lee agent_summary.csv generado por compare_agents_multiseed_experiment.py
y recalcula el IVL. Util para explorar el efecto de cambiar los pesos del
IVL sin necesidad de reejecutar el experimento completo.

Uso::

    python examples/compute_ivl_from_results.py

Salida::

    results/ivl_results.csv
"""

from pathlib import Path

import pandas as pd

from latent_rl.experiments import ExperimentConfig
from latent_rl.experiments.runner import compute_ivl, export_ivl_results

# ── Personaliza aqui los parametros del IVL ──────────────────────────────────
config = ExperimentConfig(
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


def main() -> None:
    summary_path = Path(config.results_dir) / "agent_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"No se encuentra {summary_path}. "
            "Ejecuta primero compare_agents_multiseed_experiment.py"
        )

    summary_df = pd.read_csv(summary_path).set_index("agent_name")

    if config.direct_agent not in summary_df.index:
        raise ValueError(
            f"No se encuentra '{config.direct_agent}' en {summary_path}. "
            f"Agentes disponibles: {list(summary_df.index)}"
        )

    # Reconstruir aggregated desde el CSV para reutilizar compute_ivl()
    aggregated = {
        name: {
            "name":              name,
            "mean_return_is":    row["mean_return_is"],
            "std_return_is":     row["std_return_is"],
            "mean_sharpe_is":    row["mean_sharpe_is"],
            "mean_mdd_is":       row["mean_mdd_is"],
            "mean_equity_is":    row["mean_equity_is"],
            "mean_return_oos":   row["mean_return_oos"],
            "std_return_oos":    row["std_return_oos"],
            "mean_sharpe_oos":   row["mean_sharpe_oos"],
            "mean_mdd_oos":      row["mean_mdd_oos"],
            "mean_equity_oos":   row["mean_equity_oos"],
            "mean_n_trades":     row["mean_n_trades"],
            "seed_std_return_is": row["seed_std_return_is"],
        }
        for name, row in summary_df.iterrows()
    }

    records = compute_ivl(aggregated, config)
    export_ivl_results(records, config)


if __name__ == "__main__":
    main()
