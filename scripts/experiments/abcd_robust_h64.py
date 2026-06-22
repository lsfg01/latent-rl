"""Run A/B/C/D con el protocolo robusto y cabeza Q de 64 unidades.

La eleccion de h=64 prioriza la menor variabilidad entre semillas observada
en el sweep A-only. No implica que h=64 dominase a h=128 en todos los tickers.

Uso:
    python -m scripts.experiments.abcd_robust_h64 --smoke
    python -m scripts.experiments.abcd_robust_h64
    python -m scripts.experiments.abcd_robust_h64 --results-dir results/custom_abcd
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from scripts.experiments.config import (
    ENCODER_ARTIFACT,
    SMOKE_ENCODER_ARTIFACT,
    ENCODER_TYPE,
    EVAL_TICKERS,
    FEATURES,
    K_FORECAST,
    LAMBDA_FORECAST,
    LATENT_DIM,
    LOOKBACK,
    TCN_CHANNELS,
    TCN_DILATIONS,
    TCN_KERNEL,
    default_results_dir,
)
from latent_rl.experiments import ExperimentConfig, TickerConfig, run_experiment
from latent_rl.representations.artifact import load_encoder_artifact


def _common_kwargs(results_dir: str, device: str) -> Dict[str, Any]:
    return {
        "tickers": EVAL_TICKERS,
        "ticker_configs": [TickerConfig("BTC-USD", start_date="2015-01-01")],
        "start_date": "2015-01-01",
        "end_date": "2024-01-01",
        "train_ratio": 0.7,
        "interval": "1d",
        "cache_dir": ".data_cache",
        "features": FEATURES,
        "normalize_features": True,
        "lookback_window": LOOKBACK,
        "initial_balance": 10_000.0,
        "transaction_cost": 0.001,
        # Protocolo anti-overfitting validado en el sweep A-only.
        "random_start_train": True,
        "reward_mode": "log_return",
        "reward_clip": None,
        "trade_penalty": 0.0,
        "use_internal_validation": True,
        "internal_val_ratio": 0.2,
        "validation_eval_freq": 5,
        "validation_patience": None,
        "validation_score_mdd_weight": 0.25,
        "validation_score_trade_weight": 0.05,
        # DQN directo.
        "dqn_lr": 5e-4,
        "dqn_gamma": 0.99,
        "dqn_hidden_dim": 64,
        "dqn_batch_size": 64,
        "dqn_buffer_capacity": 5_000,
        "dqn_target_update": 100,
        "dqn_epsilon_start": 1.0,
        "dqn_epsilon_end": 0.1,
        "dqn_epsilon_decay": 0.998,
        "dqn_weight_decay": 1e-4,
        "dqn_grad_clip_norm": 1.0,
        "dqn_dropout": 0.0,
        # Encoder compartido por C/D.
        "latent_dim": LATENT_DIM,
        "encoder_type": ENCODER_TYPE,
        "tcn_kernel_size": TCN_KERNEL,
        "tcn_dilations": TCN_DILATIONS,
        "tcn_channels": TCN_CHANNELS,
        # Valores explicitos para trazabilidad; el flag garantiza la alineacion.
        "latent_lr": 5e-4,
        "latent_gamma": 0.99,
        "latent_q_hidden_dim": 64,
        "latent_batch_size": 64,
        "latent_buffer_capacity": 5_000,
        "latent_target_update": 100,
        "latent_epsilon_start": 1.0,
        "latent_epsilon_end": 0.1,
        "latent_epsilon_decay": 0.998,
        "latent_weight_decay": 1e-4,
        "latent_grad_clip_norm": 1.0,
        "latent_q_dropout": 0.0,
        "align_latent_q_with_dqn": True,
        # Preentrenamiento ligero del brazo C, solo en IS_train.
        "pretrain_n_epochs": 20,
        "pretrain_lr": 5e-4,
        "pretrain_batch_size": 128,
        "pretrain_lambda_forecast": LAMBDA_FORECAST,
        "pretrain_k_forecast": K_FORECAST,
        # Brazos e IVL.
        "heavy_encoder_path": ENCODER_ARTIFACT,
        "run_arms": ["A", "B", "C", "D"],
        "direct_agent": "A",
        "latent_agents": ["B", "C", "D"],
        "device": device,
        "results_dir": results_dir,
    }


def build_config(
    smoke: bool = False,
    results_dir: str | None = None,
    device: str = "cpu",
) -> ExperimentConfig:
    results_dir = results_dir or default_results_dir("abcd_robust_h64", smoke=smoke)
    kwargs = _common_kwargs(results_dir=results_dir, device=device)
    if smoke:
        kwargs.update(
            heavy_encoder_path=SMOKE_ENCODER_ARTIFACT,
            start_date="2020-01-01",
            ticker_configs=[TickerConfig("BTC-USD", start_date="2020-01-01")],
            seeds=[0],
            n_training_episodes=3,
            n_eval_episodes=1,
            max_steps_per_episode=100,
            dqn_batch_size=32,
            dqn_buffer_capacity=500,
            dqn_target_update=50,
            dqn_epsilon_decay=0.99,
            latent_batch_size=32,
            latent_buffer_capacity=500,
            latent_target_update=50,
            latent_epsilon_decay=0.99,
            pretrain_n_epochs=2,
            pretrain_batch_size=32,
        )
    else:
        kwargs.update(
            seeds=[0, 1, 2, 3, 4],
            n_training_episodes=30,
            n_eval_episodes=3,
            max_steps_per_episode=500,
        )
    return ExperimentConfig(**kwargs)


def _preflight_heavy_encoder(
    cfg: ExperimentConfig,
    allow_missing: bool = False,
) -> bool:
    """Valida existencia y arquitectura de D antes de iniciar la run costosa."""
    if "D" not in cfg.run_arms:
        return False
    try:
        _, norm_stats, provenance = load_encoder_artifact(
            cfg.heavy_encoder_path, cfg=cfg
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        if not allow_missing:
            raise RuntimeError(
                f"El artefacto D no esta disponible o es incompatible: {exc}"
            ) from exc
        cfg.run_arms = [arm for arm in cfg.run_arms if arm != "D"]
        cfg.latent_agents = [arm for arm in cfg.latent_agents if arm != "D"]
        print(f"AVISO: brazo D desactivado por preflight fallido: {exc}")
        return False

    print(
        "Artefacto D validado: "
        f"norm_stats={'si' if norm_stats else 'no'}, "
        f"trained_at={provenance.get('trained_at', 'desconocido')}"
    )
    return True


def _save_effective_config(cfg: ExperimentConfig) -> Path:
    out = Path(cfg.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "experiment_config.json"
    path.write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run A/B/C/D robusta con hidden_dim=64"
    )
    parser.add_argument("--smoke", action="store_true", help="Prueba minima")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directorio de salida (default: results/<campaña>/)",
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, etc.")
    parser.add_argument(
        "--allow-missing-d",
        action="store_true",
        help="Continua como A/B/C si D falta o es incompatible",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = build_config(args.smoke, args.results_dir, args.device)

    print("=" * 78)
    print("RUN A/B/C/D ROBUSTA h=64" + (" [SMOKE]" if args.smoke else ""))
    print("=" * 78)
    print(f"Tickers          : {cfg.tickers}")
    print(f"Periodo          : {cfg.start_date} -> {cfg.end_date}")
    print(f"Semillas         : {cfg.seeds}")
    print(f"Brazos           : {cfg.run_arms}")
    print(f"Episodios/steps  : {cfg.n_training_episodes} / {cfg.max_steps_per_episode}")
    print(f"Q hidden A/latent: {cfg.dqn_hidden_dim} / {cfg.latent_q_hidden_dim}")
    print(f"Reward/random    : {cfg.reward_mode} / {cfg.random_start_train}")
    print(f"Validacion       : {cfg.internal_val_ratio:.0%} cada {cfg.validation_eval_freq} ep")
    print(f"Encoder          : {cfg.encoder_type}, latent_dim={cfg.latent_dim}")
    print(f"Resultados       : {cfg.results_dir}")
    print("=" * 78)

    _preflight_heavy_encoder(cfg, allow_missing=args.allow_missing_d)

    # El encoder C se entrena una vez antes del bucle de semillas. Fijamos su
    # inicializacion para que la run completa sea reproducible.
    shared_seed = cfg.seeds[0]
    np.random.seed(shared_seed)
    torch.manual_seed(shared_seed)

    config_path = _save_effective_config(cfg)
    print(f"Configuracion efectiva: {config_path}")
    run_experiment(cfg)


if __name__ == "__main__":
    main()
