"""Configuración del módulo de preentrenamiento offline del encoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from latent_rl.data.features import AVAILABLE_FEATURES

# Features que se excluyen de la observación si están presentes en el corpus
_OHLCV = {"open", "high", "low", "close", "volume"}


@dataclass
class PretrainConfig:
    """Configuración del preentrenamiento offline del encoder robusto (brazo D).

    El universo de activos debe ser **disjunto** de cada eval_ticker y sus
    parientes; ``__post_init__`` lo valida y aborta si detecta solapamiento.

    Attributes:
        universe: Símbolos de activos del corpus de entrenamiento.
        eval_tickers: Activos que se usarán en el experimento (deben estar
            excluidos del universo).
        relatives: Mapa eval_ticker → lista de símbolos relacionados que
            también deben excluirse del universo.
        start_date: Inicio del periodo de descarga (YYYY-MM-DD).
        end_date: Fin del periodo de descarga (YYYY-MM-DD).
        interval: Intervalo de velas ("1d", "1wk", "1h").
        cache_dir: Directorio de cache de descargas.
        features: Features a calcular (deben pertenecer a AVAILABLE_FEATURES).
        lookback: Longitud de la ventana de entrada (L pasos).
        encoder_type: Tipo de encoder ("tcn", "mlp", "gru").
        latent_dim: Dimensión del espacio latente.
        kernel_size: (TCN) tamaño del kernel.
        dilations: (TCN) lista de dilataciones.
        channels: (TCN) número de canales de la TCN.
        hidden_dim: (GRU) dimensión oculta.
        num_layers: (GRU) número de capas.
        k: Horizonte de forecasting (pasos futuros).
        lambda_forecast: Peso de la pérdida de forecasting.
        n_epochs: Épocas máximas de entrenamiento.
        batch_size: Tamaño del batch.
        learning_rate: Tasa de aprendizaje.
        val_ratio: Fracción de validación.
        early_stopping_patience: Épocas sin mejora antes de parar.
        min_asset_length: Mínimo de filas para incluir un activo.
        seed: Semilla global de reproducibilidad.
        output_path: Ruta de salida del artefacto.
    """

    # Corpus
    universe: List[str] = field(default_factory=list)
    eval_tickers: List[str] = field(
        default_factory=lambda: ["SPY", "TSLA", "BTC-USD"]
    )
    relatives: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "SPY": ["IVV", "VOO"],
            "BTC-USD": ["BTC-EUR", "ETH-USD"],
            "TSLA": [],
        }
    )
    start_date: str = "2010-01-01"
    end_date: str = "2023-12-31"
    interval: str = "1d"
    cache_dir: str = ".data_cache"

    # Features y ventana
    features: List[str] = field(
        default_factory=lambda: [
            "log_return", "high_low_range", "close_open_pct",
            "volume_ratio", "rsi_14", "atr_pct", "market_regime", "ma_ratio",
        ]
    )
    lookback: int = 20

    # Arquitectura del encoder
    encoder_type: str = "tcn"
    latent_dim: int = 32
    kernel_size: int = 3
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4])
    channels: int = 64
    hidden_dim: int = 64
    num_layers: int = 1

    # Objetivo
    k: int = 5
    lambda_forecast: float = 0.5

    # Entrenamiento
    n_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 5e-4
    val_ratio: float = 0.15
    early_stopping_patience: int = 10
    min_asset_length: int = 200
    seed: int = 42

    # Salida
    output_path: str = "models/encoders/tcn_heavy.pt"

    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.universe:
            raise ValueError("PretrainConfig.universe no puede estar vacío.")

        # Construir conjunto de excluidos
        excluded: set = set(self.eval_tickers)
        for ticker, rels in self.relatives.items():
            excluded.update(rels)

        # Verificar disjunción
        overlaps = excluded & set(self.universe)
        if overlaps:
            raise ValueError(
                f"PretrainConfig: los siguientes símbolos están en universe "
                f"y en eval_tickers/relatives (violación anti-fuga): {sorted(overlaps)}"
            )

        # Validar ratio val
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError(
                f"val_ratio debe estar en (0, 1), got {self.val_ratio}"
            )

        if self.k < 1:
            raise ValueError(f"k debe ser >= 1, got {self.k}")

        if self.lambda_forecast < 0:
            raise ValueError(
                f"lambda_forecast debe ser >= 0, got {self.lambda_forecast}"
            )

        unknown_features = [f for f in self.features if f not in AVAILABLE_FEATURES]
        if unknown_features:
            raise ValueError(
                f"features contiene nombres no reconocidos: {unknown_features}. "
                f"Disponibles: {sorted(AVAILABLE_FEATURES)}"
            )

        # "log_return" es necesario para forecasting
        if "log_return" not in self.features:
            raise ValueError(
                "PretrainConfig.features debe incluir 'log_return' para "
                "construir targets de forecasting."
            )

    def excluded_symbols(self) -> List[str]:
        """Retorna la lista completa de símbolos excluidos del universo."""
        excluded: set = set(self.eval_tickers)
        for rels in self.relatives.values():
            excluded.update(rels)
        return sorted(excluded)
