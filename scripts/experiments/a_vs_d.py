"""
Experimento A vs D (5 semillas).

Compara DQN directo (brazo A) frente a LatentDQN con encoder gordo preentrenado
y congelado (brazo D). El encoder gordo se carga de `models/encoders/tcn_heavy.pt`
(Fase 1 ya ejecutada); este script no regenera el artefacto.

Uso:
    # Smoke-run (validar carga de artefacto y flujo completo, 1 semilla)
    python -m scripts.experiments.a_vs_d --smoke

    # Experimento real (5 semillas x 3 tickers)
    python -m scripts.experiments.a_vs_d

    # Walk-forward complementario
    python -m scripts.experiments.a_vs_d --wf
"""

import argparse
import logging
from pathlib import Path

from latent_rl.experiments import ExperimentConfig, TickerConfig, run_experiment
from scripts.experiments.config import (
    LOOKBACK, FEATURES, ENCODER_TYPE, LATENT_DIM,
    TCN_KERNEL, TCN_DILATIONS, TCN_CHANNELS,
    K_FORECAST, LAMBDA_FORECAST,
    EVAL_TICKERS, ENCODER_ARTIFACT, SMOKE_ENCODER_ARTIFACT,
    default_results_dir,
)


# ---------------------------------------------------------------------------
# Diagnóstico de normalización compartido con la campaña A/B/C/D.
# ---------------------------------------------------------------------------

def _check_norm_stats(cfg: ExperimentConfig) -> None:
    if not cfg.heavy_encoder_path or not Path(cfg.heavy_encoder_path).exists():
        print("  [norm-check] Artefacto gordo no disponible; diagnostico omitido.")
        return

    import torch
    import numpy as np
    from latent_rl.experiments.utils import load_ticker_with_config, split_data, normalize_is_oos

    ckpt = torch.load(cfg.heavy_encoder_path, map_location="cpu", weights_only=False)
    ns = ckpt.get("norm_stats", {})
    art_mean: dict = ns.get("mean", {})
    art_std:  dict = ns.get("std", {})
    art_features = ns.get("feature_names", [])

    if not art_features:
        print("  [norm-check] norm_stats vacio en el artefacto; diagnostico omitido.")
        return

    print("\n  [norm-check] Comparando norm_stats del artefacto vs IS de cada ticker:")

    for ticker in cfg.tickers:
        try:
            data = load_ticker_with_config(ticker, cfg)
            data_is, _ = split_data(data, cfg.train_ratio)
            warnings_found = False
            for feat in art_features:
                if feat not in data_is.columns:
                    continue
                col = data_is[feat].dropna()
                exp_mean = float(col.mean())
                exp_std  = float(col.std())
                a_mean   = art_mean.get(feat, 0.0)
                a_std    = art_std.get(feat, 1.0)
                z_mean   = abs(exp_mean - a_mean) / (a_std + 1e-8)
                ratio_std = exp_std / (a_std + 1e-8)
                if z_mean > 3.0 or not (1 / 3 < ratio_std < 3.0):
                    print(
                        f"    AVISO {ticker}/{feat}: "
                        f"IS mean={exp_mean:.4f} (art={a_mean:.4f}, z={z_mean:.1f}x)  "
                        f"IS std={exp_std:.4f} (art={a_std:.4f}, ratio={ratio_std:.2f}x)"
                    )
                    warnings_found = True
            if not warnings_found:
                print(f"    {ticker}: OK - todas las features dentro de rango.")
        except Exception as exc:
            print(f"    [norm-check] {ticker}: error al cargar datos ({exc})")

    print("  [norm-check] Diagnostico completado.\n")


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

def build_config(
    smoke: bool = False,
    wf: bool = False,
    results_dir: str | None = None,
) -> ExperimentConfig:
    """Config para el experimento A vs D."""

    experiment_name = "a_vs_d_wf" if wf else "a_vs_d"
    results_dir = results_dir or default_results_dir(experiment_name, smoke=smoke)

    if smoke:
        return ExperimentConfig(
            tickers=EVAL_TICKERS,
            ticker_configs=[
                TickerConfig("BTC-USD", start_date="2020-01-01"),
            ],
            start_date="2020-01-01",
            end_date="2024-01-01",
            train_ratio=0.7,
            interval="1d",
            cache_dir=".data_cache",
            features=FEATURES,
            normalize_features=True,
            seeds=[0],
            n_training_episodes=3,
            n_eval_episodes=2,
            max_steps_per_episode=100,
            lookback_window=LOOKBACK,
            initial_balance=10_000.0,
            transaction_cost=0.001,
            # DQN (brazo A)
            dqn_lr=5e-4,
            dqn_hidden_dim=64,
            dqn_batch_size=32,
            dqn_buffer_capacity=500,
            dqn_target_update=50,
            dqn_epsilon_decay=0.99,
            # LatentDQN (brazo D) — arquitectura debe coincidir con el artefacto
            latent_dim=LATENT_DIM,
            encoder_type=ENCODER_TYPE,
            tcn_kernel_size=TCN_KERNEL,
            tcn_dilations=TCN_DILATIONS,
            tcn_channels=TCN_CHANNELS,
            latent_q_hidden_dim=64,
            latent_buffer_capacity=500,
            latent_epsilon_decay=0.99,
            align_latent_q_with_dqn=True,
            # Brazo D
            heavy_encoder_path=SMOKE_ENCODER_ARTIFACT,
            run_arms=["A", "D"],
            # IVL: A vs D
            direct_agent="A",
            latent_agents=["D"],
            device="cpu",
            results_dir=results_dir,
        )

    return ExperimentConfig(
        tickers=EVAL_TICKERS,
        ticker_configs=[
            TickerConfig("BTC-USD", start_date="2015-01-01"),
        ],
        start_date="2015-01-01",
        end_date="2024-01-01",
        train_ratio=0.7,
        interval="1d",
        cache_dir=".data_cache",
        features=FEATURES,
        normalize_features=True,
        wf_enabled=wf,
        wf_n_windows=5,
        wf_is_ratio=0.6,
        seeds=[0, 1, 2, 3, 4],
        n_training_episodes=50,
        n_eval_episodes=3,
        max_steps_per_episode=500,
        lookback_window=LOOKBACK,
        initial_balance=10_000.0,
        transaction_cost=0.001,
        # DQN (brazo A)
        dqn_lr=5e-4,
        dqn_hidden_dim=128,
        dqn_batch_size=64,
        dqn_buffer_capacity=5_000,
        dqn_target_update=100,
        dqn_epsilon_decay=0.998,
        # LatentDQN (brazo D) — DEBE coincidir con Fase 1
        latent_dim=LATENT_DIM,
        encoder_type=ENCODER_TYPE,
        tcn_kernel_size=TCN_KERNEL,
        tcn_dilations=TCN_DILATIONS,
        tcn_channels=TCN_CHANNELS,
        latent_q_hidden_dim=128,
        latent_buffer_capacity=5_000,
        latent_epsilon_decay=0.998,
        align_latent_q_with_dqn=True,
        # Brazo D — artefacto gordo de Fase 1
        heavy_encoder_path=ENCODER_ARTIFACT,
        run_arms=["A", "D"],
        # IVL: A vs D
        direct_agent="A",
        latent_agents=["D"],
        device="cpu",
        results_dir=results_dir,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento A vs D (5 semillas)")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-run (1 semilla, 3 episodios) para validar artefacto y flujo",
    )
    parser.add_argument(
        "--wf", action="store_true",
        help="Walk-forward en lugar del split IS/OOS estandar",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Directorio de salida (default: results/<campaña>/)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = build_config(smoke=args.smoke, wf=args.wf, results_dir=args.results_dir)

    print("=" * 70)
    mode = "SMOKE RUN" if args.smoke else ("WALK-FORWARD" if args.wf else "EXPERIMENTO REAL")
    print(f"EXPERIMENTO A vs D  [{mode}]")
    print("=" * 70)
    print(f"  Eval tickers     : {cfg.tickers}")
    print(f"  Periodo          : {cfg.start_date} -> {cfg.end_date}")
    print(f"  Features ({len(cfg.features)})   : {cfg.features}")
    print(f"  Lookback (L)     : {cfg.lookback_window}")
    print(f"  Encoder (brazo D): {cfg.encoder_type}  latent_dim={cfg.latent_dim}")
    print(f"  TCN kernel={cfg.tcn_kernel_size} dilations={cfg.tcn_dilations} channels={cfg.tcn_channels}")
    print(f"  RF               : {1 + (cfg.tcn_kernel_size - 1) * sum(cfg.tcn_dilations)} (>= L={cfg.lookback_window}?)")
    print(f"  Semillas         : {cfg.seeds}")
    print(f"  Episodios train  : {cfg.n_training_episodes}")
    print(f"  Brazos           : {cfg.run_arms}")
    print(f"  Artefacto gordo  : {cfg.heavy_encoder_path}")
    print(f"  Resultados en    : {cfg.results_dir}/")
    print("=" * 70)

    _check_norm_stats(cfg)

    artifact = Path(cfg.heavy_encoder_path)
    if not artifact.exists():
        print(
            f"  ERROR: Artefacto gordo no encontrado en '{artifact}'. "
            "El brazo D no puede ejecutarse. Regenera la Fase 1 con "
            "python -m scripts.experiments.pretrain_encoder."
        )
        return

    run_experiment(cfg)


if __name__ == "__main__":
    main()
