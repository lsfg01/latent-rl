"""
Caso de uso 5: Pipeline completo de datos con todas las mejoras.

Demuestra el flujo completo de datos mejorado:
  1. DataCache: descarga una vez, reutiliza
  2. FeatureEngineer: 8 indicadores tecnicos
  3. FeatureNormalizer: z-score IS/OOS sin leakage
  4. Walk-Forward: robustez en multiples regimenes
  5. ExperimentConfig: centralizacion de todos los parametros

Arquitectura del pipeline (este script):

  Yahoo Finance (o cache)
        |
        v
  DataCache (CSV.gz en disco)
        |
        v
  FeatureEngineer
  [OHLCV + log_return + rsi_14 + atr_pct + market_regime + ma_ratio]
        |
        v
  walk_forward_splits  [5 ventanas temporales]
        |
        v
  normalize_is_oos (por ventana, fit en IS)
        |
        v
  FinancialEnv (IS) -> DQNAgent / LatentDQNAgent training
        |
        v
  FinancialEnv (OOS) -> Evaluacion
        |
        v
  Metricas: IS/OOS Return, Sharpe, MDD -> IVL

Ejecutar:
    python examples/uc5_full_data_pipeline.py
"""

import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latent_rl.data.cache import DataCache
from latent_rl.data.features import FeatureEngineer
from latent_rl.data.normalizer import FeatureNormalizer
from latent_rl.experiments.config import ExperimentConfig, TickerConfig
from latent_rl.experiments.utils import (
    walk_forward_splits,
    normalize_is_oos,
    split_data,
)
from latent_rl.experiments.runner import run_single_seed
from latent_rl.evaluation.latent_advantage import LatentAdvantageIndex


# ── Datos sinteticos ──────────────────────────────────────────────────────────

def make_ohlcv(n=500, seed=42):
    rng = np.random.default_rng(seed)
    price = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    noise = lambda s: rng.normal(0, s, n)
    return pd.DataFrame({
        "Open":   price * (1 + noise(0.003)),
        "High":   price * (1 + np.abs(noise(0.007))),
        "Low":    price * (1 - np.abs(noise(0.007))),
        "Close":  price,
        "Volume": rng.integers(5_000, 50_000, n).astype(float),
    })


def mock_download(*a, **kw):
    return make_ohlcv()


def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    banner("UC5: Pipeline completo de datos con todas las mejoras")

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── PASO 1: DataCache ─────────────────────────────────────────────────
        banner("PASO 1/5 — DataCache")
        cache = DataCache(tmpdir)

        with patch("yfinance.download", side_effect=mock_download):
            df_raw = cache.get_or_download("SPY", "2020-01-01", "2024-01-01")

        print(f"  Cargado: {len(df_raw)} filas x {df_raw.shape[1]} cols")
        print(f"  Columnas: {list(df_raw.columns)}")

        # Cache hit (sin red)
        with patch("yfinance.download", side_effect=Exception("Cache miss!")):
            df_raw2 = cache.get_or_download("SPY", "2020-01-01", "2024-01-01")
        print(f"  Cache hit: {len(df_raw2)} filas (sin descarga)")

        # ── PASO 2: FeatureEngineer ───────────────────────────────────────────
        banner("PASO 2/5 — FeatureEngineer")

        FEATURES_SELECTED = [
            "log_return",      # retorno logaritmico diario
            "rsi_14",          # momentum (0-100)
            "atr_pct",         # volatilidad relativa
            "market_regime",   # tendencia -1/0/+1
            "ma_ratio",        # intensidad de la tendencia (continua)
        ]

        fe = FeatureEngineer()
        data = fe.transform(df_raw, FEATURES_SELECTED)

        print(f"  Features seleccionados: {FEATURES_SELECTED}")
        print(f"  Shape final: {data.shape}  ({data.shape[1]} columnas)")
        print(f"  Columna 3 = '{data.columns[3]}' (FinancialEnv price)")
        print(f"\n  Stats de features (descripcion):")
        print(data[FEATURES_SELECTED].describe().round(4).to_string())

        # ── PASO 3: Walk-Forward splits ────────────────────────────────────────
        banner("PASO 3/5 — Walk-Forward splits")

        N_WINDOWS = 4
        IS_RATIO  = 0.65
        splits = walk_forward_splits(data, n_windows=N_WINDOWS, is_ratio=IS_RATIO)

        print(f"  Configuracion: {N_WINDOWS} ventanas, IS={IS_RATIO*100:.0f}%")
        print(f"\n  {'Ventana':<10} {'IS filas':>10} {'OOS filas':>10} {'close IS [min,max]':>24}")
        for i, (wf_is, wf_oos) in enumerate(splits):
            print(
                f"    {i+1:<8} {len(wf_is):>10} {len(wf_oos):>10} "
                f"  [{wf_is['close'].min():.1f}, {wf_is['close'].max():.1f}]"
            )

        # ── PASO 4: FeatureNormalizer (por ventana) ────────────────────────────
        banner("PASO 4/5 — FeatureNormalizer (per-ventana, sin leakage)")

        splits2 = walk_forward_splits(data, n_windows=N_WINDOWS, is_ratio=IS_RATIO)
        normalized_splits = []

        for i, (wf_is, wf_oos) in enumerate(splits2):
            wf_is_n, wf_oos_n, norm = normalize_is_oos(wf_is, wf_oos)
            normalized_splits.append((wf_is_n, wf_oos_n))

            lr_is_mean = wf_is_n["log_return"].mean()
            lr_is_std  = wf_is_n["log_return"].std()
            lr_oos_mean = wf_oos_n["log_return"].mean()
            print(
                f"  Ventana {i+1}: log_return IS mean={lr_is_mean:.2e} std={lr_is_std:.4f}"
                f"  | OOS mean={lr_oos_mean:+.4f} (no cero, regimen distinto)"
            )

        print(f"\n  Clave: IS std=1.0 garantizado, OOS libre (sin leakage)")

        # ── PASO 5: Experimento mini ───────────────────────────────────────────
        banner("PASO 5/5 — Experimento multi-ventana (1 semilla)")

        cfg = ExperimentConfig(
            seeds=[42],
            n_training_episodes=3,
            n_eval_episodes=1,
            max_steps_per_episode=60,
            lookback_window=10,
            dqn_hidden_dim=64,
            dqn_buffer_capacity=500,
            dqn_batch_size=32,
            latent_dim=16,
            encoder_hidden_dims=[16, 32],
            encoder_type="conv1d",
            latent_q_hidden_dim=64,
            latent_buffer_capacity=500,
            latent_batch_size=32,
            pretrain_n_samples=40,
            pretrain_n_epochs=3,
            features=FEATURES_SELECTED,
            normalize_features=True,
        )

        print(f"\n  {'Ventana':<10} {'RandomAgent OOS':>18} {'DQN OOS':>18} {'LatentDQN OOS':>18}")
        print(f"  {'-'*68}")

        all_window_results = []
        for i, (wf_is_n, wf_oos_n) in enumerate(normalized_splits):
            res = run_single_seed(seed=42, data_is=wf_is_n, data_oos=wf_oos_n, cfg=cfg)
            all_window_results.append(res)

            rnd  = res["RandomAgent"]["oos"]["total_return"]
            dqn  = res["DQNAgent"]["oos"]["total_return"]
            lat  = res["LatentDQNAgent (pretrained)"]["oos"]["total_return"]
            print(f"    {i+1:<8} {rnd:>+18.4f} {dqn:>+18.4f} {lat:>+18.4f}")

        # Agregados
        print(f"\n  Agregados ({N_WINDOWS} ventanas):")
        for agent in ["RandomAgent", "DQNAgent", "LatentDQNAgent (pretrained)"]:
            returns = [r[agent]["oos"]["total_return"] for r in all_window_results]
            sharpes = [r[agent]["oos"]["sharpe"]        for r in all_window_results]
            print(
                f"    {agent:<35} "
                f"OOS return: mean={np.mean(returns):+.4f} std={np.std(returns):.4f} | "
                f"Sharpe mean={np.mean(sharpes):+.3f}"
            )

        print("\n  Pipeline completo ejecutado con exito.")
        print("  Para datos reales: elimina el mock y usa DataCache con internet.")

    banner("UC5 completado correctamente")


if __name__ == "__main__":
    main()
