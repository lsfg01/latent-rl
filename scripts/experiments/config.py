"""
Fuente única de verdad para las dos fases del experimento.

Importar desde los entrypoints de ``scripts.experiments`` para
garantizar que el encoder de la Fase 1 (artefacto gordo) y el de la Fase 2
(brazo C ligero + brazo D cargado) tengan exactamente la misma arquitectura.
Si L, F o latent_dim difieren entre fases el artefacto falla ruidosamente
al cargar; este módulo elimina esa posibilidad.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Ventana temporal y features  (deben coincidir en AMBAS fases)
# ---------------------------------------------------------------------------
LOOKBACK = 30

FEATURES = [
    "log_return",
    "high_low_range",
    "close_open_pct",
    "volume_ratio",
    "rsi_14",
    "atr_pct",
    "market_regime",
    "ma_ratio",
]

# ---------------------------------------------------------------------------
# Arquitectura del encoder  (debe coincidir en AMBAS fases)
# ---------------------------------------------------------------------------
ENCODER_TYPE = "tcn"
LATENT_DIM   = 32
TCN_KERNEL   = 3
# RF = 1 + (3-1)*(1+2+4+8) = 31 >= LOOKBACK=30  →  cobertura completa
TCN_DILATIONS = [1, 2, 4, 8]
TCN_CHANNELS  = 64

# ---------------------------------------------------------------------------
# Objetivo de preentrenamiento  (mismo en Fase 1 y brazo C de Fase 2)
# ---------------------------------------------------------------------------
K_FORECAST      = 5
LAMBDA_FORECAST = 0.5

# ---------------------------------------------------------------------------
# Evaluación y rutas
# ---------------------------------------------------------------------------
EVAL_TICKERS     = ["SPY", "TSLA", "BTC-USD"]
ENCODER_ARTIFACT = "models/encoders/tcn_heavy.pt"
SMOKE_ENCODER_ARTIFACT = "models/smoke/encoders/tcn_heavy.pt"


def default_results_dir(experiment: str, *, smoke: bool = False) -> str:
    """Construye la ruta de salida estándar de una campaña."""
    root = Path("results") / "smoke" if smoke else Path("results")
    return str(root / experiment)
