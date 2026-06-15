"""
Caso de uso 3: Walk-Forward Analysis.

Demuestra:
  - walk_forward_splits(): genera N ventanas IS/OOS consecutivas
  - Normalizacion per-ventana (sin leakage cross-window)
  - Evaluacion del agente en multiples regimenes de mercado
  - Detecta si el agente es robusto o sobreajustado a un periodo

Ejecutar:
    python examples/uc3_walk_forward.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latent_rl.data.features import FeatureEngineer
from latent_rl.experiments.utils import walk_forward_splits, normalize_is_oos
from latent_rl.experiments.config import ExperimentConfig
from latent_rl.experiments.runner import run_single_seed


# ── Datos sinteticos con regimenes distintos ──────────────────────────────────

def make_ohlcv_multi_regime(n=600, seed=7):
    """Precio simulado con 3 regimenes: alcista -> lateral -> bajista."""
    rng = np.random.default_rng(seed)
    seg = n // 3
    rets = np.concatenate([
        rng.normal(+0.0008, 0.012, seg),   # regimen alcista
        rng.normal(+0.0001, 0.008, seg),   # regimen lateral
        rng.normal(-0.0005, 0.014, n - 2*seg),  # regimen bajista
    ])
    price = 100.0 * np.cumprod(1 + rets)
    noise = lambda s: rng.normal(0, s, n)
    return pd.DataFrame({
        "open":   price * (1 + noise(0.003)),
        "high":   price * (1 + np.abs(noise(0.007))),
        "low":    price * (1 - np.abs(noise(0.007))),
        "close":  price,
        "volume": rng.integers(5_000, 50_000, n).astype(float),
    })


def main():
    print("=" * 60)
    print("UC3: Walk-Forward Analysis")
    print("=" * 60)

    # ── 1. Datos con 3 regimenes de mercado ───────────────────────────────────
    print("\n[1] Generando datos con 3 regimenes (alcista / lateral / bajista)...")
    df_raw = make_ohlcv_multi_regime(600)
    fe = FeatureEngineer()
    data = fe.transform(df_raw, ["log_return", "ma_ratio", "market_regime"])
    print(f"     {len(data)} filas x {data.shape[1]} columnas")

    # ── 2. Comparacion: split unico vs walk-forward ───────────────────────────
    print("\n[2] Comparacion de enfoques:")
    print("     Split unico  → IS: 70%  OOS: 30% (solo 1 ventana)")
    print("     Walk-Forward → 5 ventanas de 20% cada una (is_ratio=0.6 dentro de c/u)")

    # ── 3. Generar ventanas walk-forward ──────────────────────────────────────
    N_WINDOWS = 5
    IS_RATIO  = 0.6
    splits = walk_forward_splits(data, n_windows=N_WINDOWS, is_ratio=IS_RATIO)

    print(f"\n[3] Ventanas walk-forward ({N_WINDOWS} x {IS_RATIO*100:.0f}/{(1-IS_RATIO)*100:.0f}):")
    print(f"{'Ventana':<10} {'IS filas':>10} {'OOS filas':>10} {'Periodo IS (aprox)':>22} {'Periodo OOS (aprox)':>22}")
    print("-" * 80)

    window_size = len(data) // N_WINDOWS
    for i, (wf_is, wf_oos) in enumerate(splits):
        start = i * window_size
        is_end = start + len(wf_is)
        oos_end = is_end + len(wf_oos)
        print(
            f"  {i+1:<8} {len(wf_is):>10} {len(wf_oos):>10} "
            f"  fila {start:>4}-{is_end:<4}      fila {is_end:>4}-{oos_end:<4}"
        )

    # ── 4. Ejecutar evaluacion mini por ventana ───────────────────────────────
    print("\n[4] Evaluacion de BuyAndHold por ventana (sin entrenamiento):")

    cfg = ExperimentConfig(
        seeds=[0],
        n_training_episodes=3,
        n_eval_episodes=1,
        max_steps_per_episode=50,
        lookback_window=10,
        dqn_hidden_dim=32,
        dqn_buffer_capacity=200,
        dqn_batch_size=16,
        latent_dim=8,
        encoder_hidden_dims=[8, 16],
        encoder_type="conv1d",
        latent_q_hidden_dim=32,
        latent_buffer_capacity=200,
        latent_batch_size=16,
        pretrain_n_samples=20,
        pretrain_n_epochs=2,
        features=["log_return", "ma_ratio", "market_regime"],
        normalize_features=True,
    )

    # Re-generar splits (generador agotado arriba)
    splits2 = walk_forward_splits(data, n_windows=N_WINDOWS, is_ratio=IS_RATIO)

    print(f"\n{'Ventana':<10} {'BuyHold IS':>12} {'BuyHold OOS':>12} {'DQN IS':>12} {'DQN OOS':>12}")
    print("-" * 60)

    wf_results = []
    for i, (wf_is, wf_oos) in enumerate(splits2):
        wf_is_n, wf_oos_n, _ = normalize_is_oos(wf_is, wf_oos)
        res = run_single_seed(seed=0, data_is=wf_is_n, data_oos=wf_oos_n, cfg=cfg)
        wf_results.append(res)

        bh_is  = res["BuyAndHoldAgent"]["is"]["total_return"]
        bh_oos = res["BuyAndHoldAgent"]["oos"]["total_return"]
        dq_is  = res["DQNAgent"]["is"]["total_return"]
        dq_oos = res["DQNAgent"]["oos"]["total_return"]
        print(f"  {i+1:<8} {bh_is:>+12.4f} {bh_oos:>+12.4f} {dq_is:>+12.4f} {dq_oos:>+12.4f}")

    # ── 5. Variabilidad entre ventanas ────────────────────────────────────────
    bh_oos_returns = [r["BuyAndHoldAgent"]["oos"]["total_return"] for r in wf_results]
    dqn_oos_returns = [r["DQNAgent"]["oos"]["total_return"] for r in wf_results]

    print(f"\n[5] Variabilidad OOS entre ventanas (std = estabilidad):")
    print(f"     BuyAndHold: mean={np.mean(bh_oos_returns):+.4f}  std={np.std(bh_oos_returns):.4f}")
    print(f"     DQNAgent:   mean={np.mean(dqn_oos_returns):+.4f}  std={np.std(dqn_oos_returns):.4f}")
    print("\n     Menor std -> agente mas robusto a cambios de regimen")

    # ── 6. Ventaja del walk-forward ───────────────────────────────────────────
    print("\n[6] Por que walk-forward vs split unico:")
    print("     Split unico: 1 punto de evaluacion -> puede ser suerte")
    print(f"     Walk-Forward: {N_WINDOWS} puntos de evaluacion -> detecta robustez real")
    print("     Si el agente es bueno solo en 1 de 5 ventanas -> sobreajuste")
    print("     Si es bueno en 4-5 -> ventaja real del encoder latente")

    print("\n" + "=" * 60)
    print("UC3 completado correctamente.")
    print("=" * 60)


if __name__ == "__main__":
    main()
