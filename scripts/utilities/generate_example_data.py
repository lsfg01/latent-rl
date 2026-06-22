"""
Genera datos de ejemplo realistas para el dashboard sin necesitar internet.

Crea resultados sinteticos que muestran:
  - 5 agentes con rendimiento diferenciado
  - 5 semillas con variabilidad realista
  - IS/OOS split temporal
  - IVL con ventaja del agente latente preentrenado

Salida:
    results/SPY/agent_summary.csv
    results/SPY/agent_seed_metrics.csv
    results/SPY/ivl_results.csv

Uso:
    python -m scripts.utilities.generate_example_data
    python dashboard/app.py
"""

import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from latent_rl.data.features import FeatureEngineer
from latent_rl.experiments.config import ExperimentConfig
from latent_rl.experiments.utils import split_data, normalize_is_oos
from latent_rl.experiments.runner import run_single_seed
from latent_rl.experiments.utils import aggregate_results, export_dashboard_results
from latent_rl.experiments.runner import compute_ivl, export_ivl_results


# ── Datos sinteticos ──────────────────────────────────────────────────────────

def make_ohlcv(n=700, seed=42):
    rng = np.random.default_rng(seed)
    price = 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.013, n))
    noise = lambda s: rng.normal(0, s, n)
    return pd.DataFrame({
        "open":   price * (1 + noise(0.003)),
        "high":   price * (1 + np.abs(noise(0.006))),
        "low":    price * (1 - np.abs(noise(0.006))),
        "close":  price,
        "volume": rng.integers(8_000, 80_000, n).astype(float),
    })


def main():
    print("=" * 60)
    print("Generando datos de ejemplo para el dashboard")
    print("=" * 60)

    # ── 1. Datos con features ─────────────────────────────────────────────────
    print("\n[1] Preparando datos sinteticos con features...")
    fe = FeatureEngineer()
    df_raw = make_ohlcv(700)
    data = fe.transform(df_raw, [
        "log_return", "high_low_range", "close_open_pct",
        "volume_ratio", "rsi_14", "atr_pct",
    ])
    print(f"     {len(data)} filas x {data.shape[1]} columnas")

    # ── 2. Split IS/OOS + normalizacion ───────────────────────────────────────
    print("\n[2] Split IS/OOS + normalizacion...")
    data_is, data_oos = split_data(data, train_ratio=0.7)
    data_is_n, data_oos_n, _ = normalize_is_oos(data_is, data_oos)
    print(f"     IS: {len(data_is_n)} filas  |  OOS: {len(data_oos_n)} filas")

    # ── 3. Configuracion del experimento ──────────────────────────────────────
    SEEDS = [0, 1, 2, 3, 4]
    cfg = ExperimentConfig(
        seeds=SEEDS,
        n_training_episodes=8,
        n_eval_episodes=2,
        max_steps_per_episode=120,
        lookback_window=10,
        dqn_hidden_dim=64,
        dqn_buffer_capacity=1_000,
        dqn_batch_size=32,
        latent_dim=16,
        encoder_hidden_dims=[16, 32],
        encoder_type="conv1d",
        latent_q_hidden_dim=64,
        latent_buffer_capacity=1_000,
        latent_batch_size=32,
        pretrain_n_samples=80,
        pretrain_n_epochs=5,
        features=["log_return", "high_low_range", "close_open_pct",
                  "volume_ratio", "rsi_14", "atr_pct"],
        normalize_features=True,
        results_dir="results",
    )

    # ── 4. Ejecutar experimento multi-semilla ─────────────────────────────────
    print(f"\n[3] Ejecutando experimento ({len(SEEDS)} semillas)...")
    print("     Esto puede tardar 1-2 minutos...\n")

    all_results = []
    for seed in SEEDS:
        print(f"     Semilla {seed}...")
        res = run_single_seed(seed=seed, data_is=data_is_n, data_oos=data_oos_n, cfg=cfg)
        all_results.append(res)
        for agent_name, splits in res.items():
            m_is  = splits["is"]
            m_oos = splits["oos"]
            print(
                f"       {agent_name:<40}: "
                f"IS={m_is['total_return']:+.4f}  OOS={m_oos['total_return']:+.4f}"
            )

    # ── 5. Agregar y guardar ──────────────────────────────────────────────────
    print("\n[4] Agregando resultados y guardando CSVs...")
    aggregated = aggregate_results(all_results)

    results_dir = PROJECT_ROOT / "results" / "SPY"
    export_dashboard_results(all_results, aggregated, results_dir=results_dir)

    ivl_records = compute_ivl(aggregated, cfg)
    export_ivl_results(ivl_records, results_dir=results_dir)

    # ── 6. Resumen ────────────────────────────────────────────────────────────
    print("\n[5] Resumen de resultados:")
    print(f"\n  {'Agente':<40} {'IS Return':>10} {'OOS Return':>11} {'Sharpe OOS':>11}")
    print(f"  {'-'*74}")
    for agent, metrics in aggregated.items():
        print(
            f"  {agent:<40} "
            f"{metrics['mean_return_is']:>+10.4f} "
            f"{metrics['mean_return_oos']:>+11.4f} "
            f"{metrics['mean_sharpe_oos']:>+11.3f}"
        )

    if ivl_records:
        print(f"\n  IVL calculado:")
        for rec in ivl_records:
            print(
                f"    {rec['direct_agent']} vs {rec['latent_agent']}: "
                f"IVL={rec['ivl']:+.4f}  ({rec['interpretation']})"
            )

    print(f"\n  Archivos generados:")
    for fname in ["agent_summary.csv", "agent_seed_metrics.csv", "ivl_results.csv"]:
        path = results_dir / fname
        print(f"    {path}")

    print("\n" + "=" * 60)
    print("Datos de ejemplo generados. Ejecuta ahora:")
    print("    python dashboard/app.py")
    print("    Abre: http://127.0.0.1:8050")
    print("=" * 60)


if __name__ == "__main__":
    main()
