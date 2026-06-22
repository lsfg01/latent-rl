"""
Fase 2 — Experimento de evaluación A/B/C/D sobre activos de evaluación.

Brazos:
  A  DQNAgent directo (baseline)
  B  LatentDQN encoder random, finetune completo
  C  LatentDQN encoder ligero pretrained en IS (30 épocas), frozen
  D  LatentDQN encoder gordo cargado del artefacto (Fase 1), frozen

Uso:
    # Smoke-run (validar flujo end-to-end sin coste de tiempo)
    python -m scripts.experiments.latent_abcd --smoke

    # Experimento real (5 semillas × 3 tickers × 4 brazos)
    python -m scripts.experiments.latent_abcd

    # Walk-forward complementario
    python -m scripts.experiments.latent_abcd --wf
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
# Diagnóstico de normalización  (comprobación de aceptación #3)
# ---------------------------------------------------------------------------

def _check_norm_stats(cfg: ExperimentConfig) -> None:
    """
    Compara los norm_stats del artefacto gordo con la estadística IS de cada
    ticker de evaluación. Emite un aviso si media o std divergen > 3×std o
    > factor 3, respectivamente.  No es bloqueante: solo trazabilidad.
    """
    if not cfg.heavy_encoder_path or not Path(cfg.heavy_encoder_path).exists():
        print("  [norm-check] Artefacto gordo no disponible; diagnóstico omitido.")
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
        print("  [norm-check] norm_stats vacío en el artefacto; diagnóstico omitido.")
        return

    print("\n  [norm-check] Comparando norm_stats del artefacto vs IS de cada ticker:")

    for ticker in cfg.tickers:
        try:
            data = load_ticker_with_config(ticker, cfg)
            data_is, _ = split_data(data, cfg.train_ratio)
            # Comparar datos IS en escala RAW (sin z-score) vs norm_stats del artefacto

            warnings_found = False
            for feat in art_features:
                if feat not in data_is.columns:
                    continue
                col = data_is[feat].dropna()
                exp_mean = float(col.mean())
                exp_std  = float(col.std())
                a_mean   = art_mean.get(feat, 0.0)
                a_std    = art_std.get(feat, 1.0)

                # Aviso si la media IS difiere en más de 3×std del artefacto
                z_mean = abs(exp_mean - a_mean) / (a_std + 1e-8)
                # Aviso si el std IS es >3× o <1/3 del artefacto
                ratio_std = exp_std / (a_std + 1e-8)
                if z_mean > 3.0 or not (1 / 3 < ratio_std < 3.0):
                    print(
                        f"    AVISO {ticker}/{feat}: "
                        f"IS mean={exp_mean:.4f} (art={a_mean:.4f}, z={z_mean:.1f}x)  "
                        f"IS std={exp_std:.4f} (art={a_std:.4f}, ratio={ratio_std:.2f}x)"
                    )
                    warnings_found = True
            if not warnings_found:
                print(f"    {ticker}: OK — todas las features dentro de rango.")
        except Exception as exc:
            print(f"    [norm-check] {ticker}: error al cargar datos ({exc})")

    print("  [norm-check] Diagnóstico completado.\n")


# ---------------------------------------------------------------------------
# Construcción de la configuración
# ---------------------------------------------------------------------------

def build_config(
    smoke: bool = False,
    wf: bool = False,
    results_dir: str | None = None,
) -> ExperimentConfig:
    """Devuelve ExperimentConfig usando las constantes compartidas."""

    experiment_name = "latent_abcd_wf" if wf else "latent_abcd"
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
            # Smoke: 1 semilla, episodios mínimos
            seeds=[0],
            n_training_episodes=3,
            n_eval_episodes=2,
            max_steps_per_episode=100,
            # Entorno
            lookback_window=LOOKBACK,
            initial_balance=10_000.0,
            transaction_cost=0.001,
            # DQN
            dqn_lr=5e-4,
            dqn_hidden_dim=64,
            dqn_batch_size=32,
            dqn_buffer_capacity=500,
            dqn_target_update=50,
            dqn_epsilon_decay=0.99,
            # LatentDQN — arquitectura igual a Fase 1
            latent_dim=LATENT_DIM,
            encoder_type=ENCODER_TYPE,
            tcn_kernel_size=TCN_KERNEL,
            tcn_dilations=TCN_DILATIONS,
            tcn_channels=TCN_CHANNELS,
            latent_q_hidden_dim=64,
            latent_buffer_capacity=500,
            latent_epsilon_decay=0.99,
            align_latent_q_with_dqn=True,
            # Preentrenamiento brazo C (smoke: épocas mínimas)
            pretrain_n_epochs=3,
            pretrain_lr=5e-4,
            pretrain_batch_size=32,
            pretrain_lambda_forecast=LAMBDA_FORECAST,
            pretrain_k_forecast=K_FORECAST,
            # Brazo D
            heavy_encoder_path=SMOKE_ENCODER_ARTIFACT,
            run_arms=["A", "B", "C", "D"],
            # IVL
            direct_agent="A",
            latent_agents=["B", "C", "D"],
            device="cpu",
            results_dir=results_dir,
        )

    return ExperimentConfig(
        tickers=EVAL_TICKERS,
        ticker_configs=[
            # BTC-USD tiene histórico desde ~2014; fijar start por seguridad
            TickerConfig("BTC-USD", start_date="2015-01-01"),
        ],
        start_date="2015-01-01",
        end_date="2024-01-01",
        train_ratio=0.7,
        interval="1d",
        cache_dir=".data_cache",
        features=FEATURES,
        normalize_features=True,
        # Walk-forward o split estándar
        wf_enabled=wf,
        wf_n_windows=5,
        wf_is_ratio=0.6,
        # Protocolo primario: 5 semillas
        seeds=[0, 1, 2, 3, 4],
        n_training_episodes=50,
        n_eval_episodes=3,
        max_steps_per_episode=500,
        # Entorno
        lookback_window=LOOKBACK,
        initial_balance=10_000.0,
        transaction_cost=0.001,
        # DQN
        dqn_lr=5e-4,
        dqn_hidden_dim=128,
        dqn_batch_size=64,
        dqn_buffer_capacity=5_000,
        dqn_target_update=100,
        dqn_epsilon_decay=0.998,
        # LatentDQN — arquitectura DEBE coincidir con Fase 1
        latent_dim=LATENT_DIM,
        encoder_type=ENCODER_TYPE,
        tcn_kernel_size=TCN_KERNEL,
        tcn_dilations=TCN_DILATIONS,
        tcn_channels=TCN_CHANNELS,
        latent_q_hidden_dim=128,
        latent_buffer_capacity=5_000,
        latent_epsilon_decay=0.998,
        align_latent_q_with_dqn=True,
        # Preentrenamiento brazo C (ligero, mismo objetivo que Fase 1)
        pretrain_n_epochs=30,
        pretrain_lr=5e-4,
        pretrain_batch_size=128,
        pretrain_lambda_forecast=LAMBDA_FORECAST,
        pretrain_k_forecast=K_FORECAST,
        # Brazo D — apunta al artefacto gordo de la Fase 1
        heavy_encoder_path=ENCODER_ARTIFACT,
        run_arms=["A", "B", "C", "D"],
        # IVL
        direct_agent="A",
        latent_agents=["B", "C", "D"],
        device="cpu",
        results_dir=results_dir,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 2: experimento A/B/C/D")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-run con parámetros mínimos (1 semilla, 3 episodios)",
    )
    parser.add_argument(
        "--wf", action="store_true",
        help="Walk-forward Analysis en lugar del split estándar IS/OOS",
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
    print(f"{mode}")
    print("=" * 70)
    print(f"  Eval tickers     : {cfg.tickers}")
    print(f"  Periodo          : {cfg.start_date} -> {cfg.end_date}")
    print(f"  Features ({len(cfg.features)})   : {cfg.features}")
    print(f"  Lookback (L)     : {cfg.lookback_window}")
    print(f"  Encoder          : {cfg.encoder_type}  latent_dim={cfg.latent_dim}")
    print(f"  TCN kernel={cfg.tcn_kernel_size} dilations={cfg.tcn_dilations} channels={cfg.tcn_channels}")
    print(f"  RF               : {1 + (cfg.tcn_kernel_size - 1) * sum(cfg.tcn_dilations)} (>= L={cfg.lookback_window}?)")
    print(f"  Semillas         : {cfg.seeds}")
    print(f"  Episodios train  : {cfg.n_training_episodes}")
    print(f"  Brazos           : {cfg.run_arms}")
    print(f"  Artefacto gordo  : {cfg.heavy_encoder_path}")
    print(f"  Resultados en    : {cfg.results_dir}/")
    print("=" * 70)

    # Comprobación de aceptación #3: diagnóstico de normalización
    _check_norm_stats(cfg)

    # Verificar disponibilidad del artefacto gordo
    if "D" in cfg.run_arms:
        artifact = Path(cfg.heavy_encoder_path)
        if not artifact.exists():
            print(
                f"  AVISO: Artefacto gordo no encontrado en '{artifact}'. "
                "El brazo D será omitido. Ejecuta primero "
                "python -m scripts.experiments.pretrain_encoder."
            )

    run_experiment(cfg)


if __name__ == "__main__":
    main()
