"""
Configuracion centralizada de experimentos de comparacion de agentes.

Todos los valores hardcodeados del pipeline experimental (hiperparametros
de agentes, entorno, preentrenamiento, IVL) se definen aqui con sus valores
por defecto. Para personalizar un experimento basta con sobrescribir los campos
deseados al construir ExperimentConfig.

Ejemplo::

    from latent_rl.experiments import ExperimentConfig, run_experiment

    # Un solo ticker (comportamiento previo)
    config = ExperimentConfig(
        tickers=["SPY"],
        seeds=[0, 1, 2],
        latent_dim=8,
        ivl_weights={"sharpe": 0.4, "mdd": 0.3, "seed_std": 0.15, "is_oos_gap": 0.15},
    )

    # Varios tickers con configuracion por activo
    config_multi = ExperimentConfig(
        tickers=["SPY", "AAPL", "BTC-USD"],
        ticker_configs=[
            TickerConfig("BTC-USD", start_date="2018-01-01"),
        ],
        seeds=[0, 1, 2],
        features=["log_return", "rsi_14"],
        normalize_features=True,
    )
    run_experiment(config_multi)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TickerConfig:
    """Configuracion especifica por ticker que sobreescribe los valores globales.

    Permite asignar rangos de fechas, intervalo y numero de observaciones
    diferentes para cada activo dentro del mismo experimento.

    Ejemplo::

        TickerConfig("SPY",     start_date="2010-01-01")
        TickerConfig("BTC-USD", start_date="2018-01-01", n_obs=1500)
        TickerConfig("EURUSD=X", interval="1h")

    Attributes:
        ticker: Simbolo del activo. Debe coincidir con uno de ExperimentConfig.tickers.
        start_date: Fecha de inicio. Si None, usa ExperimentConfig.start_date.
        end_date: Fecha de fin. Si None, usa ExperimentConfig.end_date.
        n_obs: Recorta las ultimas n_obs filas. Si None, usa ExperimentConfig.n_obs.
        interval: Intervalo de velas ("1d", "1wk", "1h"). Si None, usa global.
        context_tickers: Tickers adicionales cuyo log_return se une como features.
    """

    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    n_obs: Optional[int] = None
    interval: Optional[str] = None
    context_tickers: List[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """Configuracion completa de un experimento de comparacion de agentes.

    Agrupa todos los parametros ajustables del pipeline en un unico objeto
    con valores por defecto reproducibles. El objeto se valida al crearse.

    Brazos experimentales (``run_arms``):
      A = DQNAgent directo
      B = LatentDQNAgent  (encoder random, finetune)
      C = LatentDQNAgent  (encoder ligero IS, frozen)
      D = LatentDQNAgent  (encoder heavy offline, frozen) — requiere heavy_encoder_path

    Attributes:
        tickers: Lista de simbolos de activos a descargar de Yahoo Finance.
        ticker_configs: Configuraciones por ticker (sobrescriben valores globales).
        start_date: Fecha de inicio del historico (formato YYYY-MM-DD).
        end_date: Fecha de fin del historico (formato YYYY-MM-DD).
        n_obs: Si se especifica, recorta las ultimas n_obs filas del historico.
        train_ratio: Fraccion de datos para In-Sample (el resto es OOS).
        interval: Intervalo de velas para la descarga ("1d", "1wk", "1h").
        cache_dir: Directorio de cache local para datos descargados.
        features: Features tecnicos a anadir tras OHLCV.
        normalize_features: Si True, aplica z-score (fit en IS, sin leakage a OOS).

        wf_enabled: Activa Walk-Forward Analysis en lugar del split simple IS/OOS.
        wf_n_windows: Numero de ventanas WF consecutivas (>= 2).
        wf_is_ratio: Fracción inicial de anclaje. La primera ventana IS abarca
            ``floor(n * wf_is_ratio)`` filas desde el inicio; las ventanas
            siguientes heredan ese ancla y la amplían. Diferente a la semántica
            anterior (fracción IS dentro de cada bloque disjunto).
        wf_mode: Estrategia de ventana. "expanding" (anclada, por defecto) o
            "sliding" (reservado para implementación futura).

        seeds: Semillas aleatorias para la inicializacion de agentes.
        n_training_episodes: Episodios de entrenamiento RL por agente y semilla.
        n_eval_episodes: Episodios de evaluacion por agente, split y semilla.
        max_steps_per_episode: Limite de pasos por episodio de entrenamiento.

        lookback_window: Ventana de observacion historica de FinancialEnv.
        initial_balance: Capital inicial del entorno de trading (USD).
        transaction_cost: Coste de transaccion como fraccion del valor operado.

        run_arms: Brazos a ejecutar. Subconjunto de ["A", "B", "C", "D"].
        encoder_type: Tipo de encoder para brazos B/C/D ("tcn", "mlp", "gru").
        heavy_encoder_path: Ruta al artefacto del encoder gordo (brazo D).
        tcn_kernel_size: Tamaño del kernel TCN.
        tcn_dilations: Dilataciones TCN.
        tcn_channels: Canales TCN.
        gru_hidden_dim: Dimensión oculta GRU.
        gru_num_layers: Número de capas GRU.
        pretrain_lambda_forecast: Peso del objetivo de forecasting en el pretrainer.
        pretrain_k_forecast: Horizonte de forecasting en el pretrainer.

        device: Dispositivo PyTorch ("cpu" o "cuda").
        results_dir: Directorio donde se guardan los CSVs de resultados.
    """

    # -- Datos ----------------------------------------------------------------
    tickers:        List[str]          = field(default_factory=lambda: ["SPY"])
    ticker_configs: List[TickerConfig] = field(default_factory=list)
    start_date:     str                = "2020-01-01"
    end_date:       str                = "2023-12-31"
    n_obs:          Optional[int]      = None
    train_ratio:    float              = 0.7
    interval:       str                = "1d"
    cache_dir:      str                = ".data_cache"
    features:       List[str]          = field(default_factory=list)
    normalize_features: bool           = True

    # -- Walk-Forward Analysis ------------------------------------------------
    wf_enabled:   bool  = False
    wf_n_windows: int   = 5
    wf_is_ratio:  float = 0.6
    wf_mode:      str   = "expanding"  # "expanding" | "sliding" (futuro)

    # -- Experimento ----------------------------------------------------------
    seeds:                 List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_training_episodes:   int       = 10
    n_eval_episodes:       int       = 3
    max_steps_per_episode: int       = 200

    # -- Entorno --------------------------------------------------------------
    lookback_window:  int   = 10
    initial_balance:  float = 10_000.0
    transaction_cost: float = 0.001

    # -- DQN ------------------------------------------------------------------
    dqn_lr:              float = 1e-3
    dqn_gamma:           float = 0.99
    dqn_hidden_dim:      int   = 128
    dqn_batch_size:      int   = 32
    dqn_buffer_capacity: int   = 5_000
    dqn_target_update:   int   = 50
    dqn_epsilon_start:   float = 1.0
    dqn_epsilon_end:     float = 0.1
    dqn_epsilon_decay:   float = 0.995

    # -- LatentDQN ------------------------------------------------------------
    latent_dim:              int        = 16
    encoder_hidden_dims:     List[int]  = field(default_factory=lambda: [64, 32])
    encoder_dropout:         float      = 0.0
    encoder_activation:      str        = "relu"
    latent_lr:               float      = 1e-3
    latent_gamma:            float      = 0.99
    latent_q_hidden_dim:     int        = 128
    latent_batch_size:       int        = 32
    latent_buffer_capacity:  int        = 5_000
    latent_target_update:    int        = 50
    latent_epsilon_start:    float      = 1.0
    latent_epsilon_end:      float      = 0.1
    latent_epsilon_decay:    float      = 0.995

    # -- Brazos experimentales ------------------------------------------------
    run_arms:             List[str]      = field(default_factory=lambda: ["A", "B", "C", "D"])
    encoder_type:         str            = "tcn"
    heavy_encoder_path:   Optional[str]  = None
    tcn_kernel_size:      int            = 3
    tcn_dilations:        List[int]      = field(default_factory=lambda: [1, 2, 4])
    tcn_channels:         int            = 32
    gru_hidden_dim:       int            = 64
    gru_num_layers:       int            = 1

    # -- Preentrenamiento (brazo C y offline) ---------------------------------
    pretrain_n_samples:       int   = 100   # legado AutoencoderTrainer
    pretrain_n_epochs:        int   = 10
    pretrain_lr:              float = 1e-3
    pretrain_batch_size:      int   = 32
    pretrain_lambda_forecast: float = 0.5
    pretrain_k_forecast:      int   = 5

    # -- IVL ------------------------------------------------------------------
    direct_agent:  str       = "A"
    latent_agents: List[str] = field(default_factory=lambda: ["B", "C", "D"])
    ivl_weights: Dict[str, float] = field(default_factory=lambda: {
        "sharpe":     0.35,
        "mdd":        0.25,
        "seed_std":   0.20,
        "is_oos_gap": 0.20,
    })

    # -- Sistema --------------------------------------------------------------
    device:      str = "cpu"
    results_dir: str = "results"

    # -------------------------------------------------------------------------

    def get_ticker_config(self, ticker: str) -> TickerConfig:
        """Devuelve el TickerConfig para un ticker, o uno vacio si no existe."""
        for tc in self.ticker_configs:
            if tc.ticker == ticker:
                return tc
        return TickerConfig(ticker=ticker)

    def resolve_ticker_params(self, ticker: str) -> dict:
        """Resuelve los parametros efectivos (start, end, n_obs, interval) para un ticker."""
        tc = self.get_ticker_config(ticker)
        return {
            "start":    tc.start_date or self.start_date,
            "end":      tc.end_date   or self.end_date,
            "n_obs":    tc.n_obs      if tc.n_obs is not None else self.n_obs,
            "interval": tc.interval   or self.interval,
        }

    def encoder_kwargs(self) -> dict:
        """Devuelve kwargs de arquitectura para build_encoder según encoder_type."""
        base = {
            "latent_dim": self.latent_dim,
            "activation": self.encoder_activation,
            "dropout": self.encoder_dropout,
        }
        if self.encoder_type == "mlp":
            base["hidden_dims"] = self.encoder_hidden_dims
        elif self.encoder_type == "tcn":
            base["kernel_size"] = self.tcn_kernel_size
            base["dilations"] = self.tcn_dilations
            base["channels"] = self.tcn_channels
        elif self.encoder_type == "gru":
            base["hidden_dim"] = self.gru_hidden_dim
            base["num_layers"] = self.gru_num_layers
        return base

    def __post_init__(self) -> None:
        """Valida los valores de configuracion al construir el objeto."""
        if not self.tickers:
            raise ValueError("tickers no puede estar vacio")
        if any(not t or not t.strip() for t in self.tickers):
            raise ValueError(
                f"Todos los tickers deben ser strings no vacios, got {self.tickers}"
            )
        if len(self.tickers) != len(set(self.tickers)):
            dupes = [t for t in self.tickers if self.tickers.count(t) > 1]
            raise ValueError(
                f"tickers contiene duplicados: {list(set(dupes))}"
            )
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError(
                f"train_ratio debe estar en (0, 1), got {self.train_ratio}"
            )
        if self.lookback_window < 1:
            raise ValueError(
                f"lookback_window debe ser >= 1, got {self.lookback_window}"
            )
        if self.initial_balance <= 0:
            raise ValueError(
                f"initial_balance debe ser > 0, got {self.initial_balance}"
            )
        if not 0.0 <= self.transaction_cost < 1.0:
            raise ValueError(
                f"transaction_cost debe estar en [0, 1), got {self.transaction_cost}"
            )
        total_w = sum(self.ivl_weights.values())
        if abs(total_w - 1.0) > 1e-6:
            raise ValueError(
                f"ivl_weights deben sumar 1.0, pero suman {total_w:.6f}. "
                f"Valores actuales: {self.ivl_weights}"
            )
        required_ivl_keys = {"sharpe", "mdd", "seed_std", "is_oos_gap"}
        missing = required_ivl_keys - set(self.ivl_weights)
        if missing:
            raise ValueError(
                f"Faltan claves en ivl_weights: {missing}. "
                f"Claves requeridas: {required_ivl_keys}"
            )
        if not self.seeds:
            raise ValueError("seeds no puede estar vacio")

        if self.features:
            from latent_rl.data.features import AVAILABLE_FEATURES
            unknown = [f for f in self.features if f not in AVAILABLE_FEATURES]
            if unknown:
                raise ValueError(
                    f"features contiene nombres no reconocidos: {unknown}. "
                    f"Disponibles: {AVAILABLE_FEATURES}"
                )

        tc_tickers = [tc.ticker for tc in self.ticker_configs]
        unknown_tickers = [t for t in tc_tickers if t not in self.tickers]
        if unknown_tickers:
            raise ValueError(
                f"ticker_configs contiene tickers no presentes en tickers: {unknown_tickers}"
            )

        if self.wf_enabled:
            if self.wf_n_windows < 2:
                raise ValueError(
                    f"wf_n_windows debe ser >= 2, got {self.wf_n_windows}"
                )
            if not 0.0 < self.wf_is_ratio < 1.0:
                raise ValueError(
                    f"wf_is_ratio debe estar en (0, 1), got {self.wf_is_ratio}"
                )

        valid_arms = {"A", "B", "C", "D"}
        unknown_arms = set(self.run_arms) - valid_arms
        if unknown_arms:
            raise ValueError(
                f"run_arms contiene brazos desconocidos: {unknown_arms}. "
                f"Válidos: {valid_arms}"
            )
