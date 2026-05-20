"""Calcula el IVL (Índice de Ventaja Latente) a partir de los resultados del experimento.

Lee agent_summary.csv generado por compare_agents_multiseed_experiment.py y calcula
el IVL comparando cada variante LatentDQN contra DQNAgent como baseline directo.

Uso:
    python examples/compute_ivl_from_results.py

Salida:
    results/ivl_results.csv  — una fila por par (directo, latente)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from latent_rl.evaluation.latent_advantage import LatentAdvantageIndex


RESULTS_DIR = Path("results")
SUMMARY_CSV = RESULTS_DIR / "agent_summary.csv"
OUTPUT_CSV = RESULTS_DIR / "ivl_results.csv"

# Agente directo de referencia
DIRECT_AGENT = "DQNAgent"

# Agentes latentes a comparar contra el directo
LATENT_AGENTS = [
    "LatentDQNAgent (no pretrained)",
    "LatentDQNAgent (pretrained)",
]


def build_metrics_dict(row: pd.Series, split: str) -> dict:
    """
    Construye el diccionario de métricas que espera LatentAdvantageIndex.compute().

    Args:
        row: Fila de agent_summary.csv
        split: "is" o "oos"

    Returns:
        Dict con claves sharpe, max_drawdown, seed_std, perf_is, perf_oos
    """
    return {
        "sharpe": row[f"mean_sharpe_{split}"],
        "max_drawdown": row[f"mean_mdd_{split}"],
        "seed_std": row["seed_std_return_is"],
        "perf_is": row["mean_return_is"],
        "perf_oos": row["mean_return_oos"],
    }


def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"No se encuentra {SUMMARY_CSV}. "
            "Ejecuta primero compare_agents_multiseed_experiment.py"
        )

    summary_df = pd.read_csv(SUMMARY_CSV)
    summary_df = summary_df.set_index("agent_name")

    if DIRECT_AGENT not in summary_df.index:
        raise ValueError(f"No se encuentra '{DIRECT_AGENT}' en {SUMMARY_CSV}")

    ivl_calculator = LatentAdvantageIndex()
    direct_row = summary_df.loc[DIRECT_AGENT]

    # Métricas del agente directo (se usan las mismas para IS y OOS del IVL,
    # ya que build_metrics_dict incluye perf_is/perf_oos internamente)
    direct_metrics = build_metrics_dict(direct_row, split="is")

    records = []

    for latent_name in LATENT_AGENTS:
        if latent_name not in summary_df.index:
            print(f"  Advertencia: '{latent_name}' no encontrado en {SUMMARY_CSV}, omitiendo.")
            continue

        latent_row = summary_df.loc[latent_name]
        latent_metrics = build_metrics_dict(latent_row, split="is")

        result = ivl_calculator.compute(direct_metrics, latent_metrics)

        records.append({
            "direct_agent": DIRECT_AGENT,
            "latent_agent": latent_name,
            "ivl": result["ivl"],
            "delta_sharpe": result["delta_sharpe"],
            "delta_mdd": result["delta_mdd"],
            "delta_seed_std": result["delta_seed_std"],
            "delta_is_oos_gap": result["delta_is_oos_gap"],
            "interpretation": result["interpretation"],
            # Métricas base para trazabilidad
            "direct_sharpe_is": direct_row["mean_sharpe_is"],
            "direct_return_is": direct_row["mean_return_is"],
            "direct_return_oos": direct_row["mean_return_oos"],
            "latent_sharpe_is": latent_row["mean_sharpe_is"],
            "latent_return_is": latent_row["mean_return_is"],
            "latent_return_oos": latent_row["mean_return_oos"],
        })

        print(f"\nIVL: {DIRECT_AGENT} vs {latent_name}")
        print(f"  IVL          = {result['ivl']:.4f}")
        print(f"  d Sharpe     = {result['delta_sharpe']:.4f}")
        print(f"  d MDD        = {result['delta_mdd']:.4f}")
        print(f"  d Seed Std   = {result['delta_seed_std']:.4f}")
        print(f"  d IS/OOS Gap = {result['delta_is_oos_gap']:.4f}")
        print(f"  Interpretación: {result['interpretation']}")

    if not records:
        print("No se calculó ningún IVL.")
        return

    ivl_df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ivl_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResultados IVL guardados en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
