"""Módulo de preentrenamiento offline del encoder robusto (brazo D)."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from latent_rl.data.cache import DataCache
from latent_rl.data.features import FeatureEngineer
from latent_rl.data.normalizer import FeatureNormalizer
from latent_rl.representations.factory import build_encoder
from latent_rl.pretraining.config import PretrainConfig
from latent_rl.pretraining.encoder_pretrainer import EncoderPretrainer

logger = logging.getLogger(__name__)


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _load_asset(
    ticker: str,
    cfg: PretrainConfig,
    cache: DataCache,
    engineer: FeatureEngineer,
) -> Optional[pd.DataFrame]:
    """Descarga (con caché) y calcula features para un activo."""
    try:
        raw = cache.get_or_download(ticker, cfg.start_date, cfg.end_date, cfg.interval)
        if raw is None or len(raw) < cfg.min_asset_length + cfg.lookback + cfg.k:
            logger.debug("Ticker %s descartado: longitud insuficiente.", ticker)
            return None
        df = engineer.transform(raw, cfg.features)
        df = df.dropna()
        if len(df) < cfg.min_asset_length + cfg.lookback + cfg.k:
            logger.debug("Ticker %s descartado tras dropna.", ticker)
            return None
        return df
    except Exception as exc:
        logger.warning("Error cargando %s: %s", ticker, exc)
        return None


def pretrain_offline(cfg: PretrainConfig) -> str:
    """
    Ejecuta el pipeline completo de preentrenamiento offline.

    Pasos:
    1. Descarga y calcula features para cada activo del universo.
    2. Descarta activos con longitud insuficiente.
    3. Ajusta FeatureNormalizer solo sobre el tramo train de cada activo.
    4. Genera ventanas (X, Y) por activo sin cruzar activos.
    5. Entrena el encoder con pérdida recon + forecast.
    6. Guarda el artefacto en cfg.output_path.

    Args:
        cfg: Configuración de preentrenamiento.

    Returns:
        Ruta del artefacto guardado.
    """
    logger.info("=== Preentrenamiento offline ===")
    logger.info(
        "Universo: %d activos | eval_tickers: %s", len(cfg.universe), cfg.eval_tickers
    )
    logger.info("Excluidos: %s", cfg.excluded_symbols())

    cache = DataCache(cfg.cache_dir)
    engineer = FeatureEngineer()

    # --- 1. Cargar activos -------------------------------------------------------
    loaded: Dict[str, pd.DataFrame] = {}
    for ticker in cfg.universe:
        df = _load_asset(ticker, cfg, cache, engineer)
        if df is not None:
            loaded[ticker] = df
            logger.info("  %s: %d filas", ticker, len(df))

    if not loaded:
        raise RuntimeError("No se pudo cargar ningún activo del universo.")
    logger.info("Activos efectivos: %d / %d", len(loaded), len(cfg.universe))

    # --- 2. Normalización: ajuste solo en tramo train de cada activo --------------
    # Recolectar DataFrames de train por activo para fitting del normalizer
    n_val_fraction = cfg.val_ratio
    all_train_feature_dfs: List[pd.DataFrame] = []
    asset_full_dfs: Dict[str, pd.DataFrame] = {}

    for ticker, df in loaded.items():
        n_train = max(1, int(len(df) * (1 - n_val_fraction)))
        df_train = df.iloc[:n_train]
        all_train_feature_dfs.append(df_train[cfg.features])
        asset_full_dfs[ticker] = df

    train_concat = pd.concat(all_train_feature_dfs, ignore_index=True)
    normalizer = FeatureNormalizer()
    normalizer.fit(train_concat)

    norm_stats: Dict[str, Any] = {
        "mean": dict(normalizer.mean_),
        "std": dict(normalizer.std_),
        "feature_names": cfg.features,
    }

    # --- 3. Construir encoder y pretrainer ---------------------------------------
    np.random.seed(cfg.seed)

    encoder = build_encoder(
        cfg.encoder_type,
        input_len=cfg.lookback,
        n_features=len(cfg.features),
        latent_dim=cfg.latent_dim,
        kernel_size=cfg.kernel_size,
        dilations=cfg.dilations,
        channels=cfg.channels,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    )

    pretrainer = EncoderPretrainer(
        encoder=encoder,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        lambda_forecast=cfg.lambda_forecast,
        k=cfg.k,
    )

    # --- 4. Generar ventanas por activo (sin cruzar activos) ----------------------
    all_X: List[np.ndarray] = []
    all_Y: List[np.ndarray] = []

    for ticker, df in asset_full_dfs.items():
        # Normalizar features (OHLCV se mantiene sin cambios por el normalizer)
        df_norm = normalizer.transform(df)
        # Para Y usamos log_return sin normalizar (train() lo estandariza)
        df_for_windows = df_norm.copy()
        df_for_windows["log_return"] = df["log_return"].values

        X, Y = pretrainer.make_windows(df_for_windows, cfg.features, cfg.lookback)
        if len(X) == 0:
            logger.debug("Ticker %s sin ventanas suficientes.", ticker)
            continue
        all_X.append(X)
        all_Y.append(Y)

    if not all_X:
        raise RuntimeError("No se generaron ventanas de entrenamiento.")

    X_all = np.concatenate(all_X, axis=0)
    Y_all = np.concatenate(all_Y, axis=0)
    logger.info("Total ventanas: %d", len(X_all))

    # --- 5. Entrenamiento --------------------------------------------------------
    history = pretrainer.train(
        X=X_all,
        Y=Y_all,
        n_epochs=cfg.n_epochs,
        val_ratio=cfg.val_ratio,
        early_stopping_patience=cfg.early_stopping_patience,
        seed=cfg.seed,
    )
    best_val = min(history["val_loss"]) if history["val_loss"] else float("inf")
    logger.info("Mejor val_loss: %.4f", best_val)

    # --- 6. Guardar artefacto ----------------------------------------------------
    arch_config = encoder.get_arch_config()
    arch_config["feature_names"] = list(cfg.features)
    arch_config["L"] = cfg.lookback
    arch_config["F"] = len(cfg.features)

    provenance: Dict[str, Any] = {
        "universe": list(loaded.keys()),
        "eval_tickers": cfg.eval_tickers,
        "excluded": cfg.excluded_symbols(),
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "interval": cfg.interval,
        "feature_names": list(cfg.features),
        "lookback": cfg.lookback,
        "seed": cfg.seed,
        "n_assets_requested": len(cfg.universe),
        "n_assets_effective": len(loaded),
        "n_total_windows": int(len(X_all)),
        "best_val_loss": float(best_val),
        "trained_at": datetime.utcnow().isoformat(),
        "git_commit": _git_commit(),
    }

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "arch_config": arch_config,
            "norm_stats": norm_stats,
            "provenance": provenance,
        },
        output_path,
    )

    logger.info("Artefacto guardado en: %s", cfg.output_path)
    return str(cfg.output_path)
