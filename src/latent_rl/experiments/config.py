"""
Configuración centralizada de experimentos de comparación de agentes.

Todos los valores hardcodeados del pipeline experimental (hiperparámetros
de agentes, entorno, preentrenamiento, IVL) se definen aquí con sus valores
por defecto. Para personalizar un experimento basta con sobrescribir los campos
deseados al construir ExperimentConfig.

Ejemplo::

    from latent_rl.experiments import ExperimentConfig, run_experiment

    config = ExperimentConfig(
        ticker="AAPL",
        seeds=[0, 1, 2],
        latent_dim=8,
        ivl_weights={"sharpe": 0.4, "mdd": 0.3, "seed_std": 0.15, "is_oos_gap": 0.15},
    )
    run_experiment(config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Configuración completa de un experimento de comparación de agentes.

    Agrupa todos los parámetros ajustables del pipeline en un único objeto
    con valores por defecto reproducibles. El objeto se valida al crearse.

    Attributes:
        ticker: Símbolo del activo a descargar de Yahoo Finance.
        start_date: Fecha de inicio del histórico (formato YYYY-MM-DD).
        end_date: Fecha de fin del histórico (formato YYYY-MM-DD).
        n_obs: Si se especifica, recorta las últimas n_obs filas del histórico.
        train_ratio: Fracción de datos para In-Sample (el resto es OOS).

        seeds: Semillas aleatorias. Cada semilla ejecuta un experimento completo;
               solo afecta a la inicialización de los agentes (los datos son fijos).
        n_training_episodes: Episodios de entrenamiento RL por agente y semilla.
        n_eval_episodes: Episodios de evaluación por agente, split y semilla.
        max_steps_per_episode: Límite de pasos por episodio de entrenamiento.

        lookback_window: Ventana de observación histórica de FinancialEnv.
                         Controla también input_dim del encoder (lookback * 5 features).
        initial_balance: Capital inicial del entorno de trading (USD).
        transaction_cost: Coste de transacción como fracción del valor operado.

        dqn_lr: Learning rate del optimizador Adam del DQNAgent.
        dqn_gamma: Factor de descuento temporal del DQNAgent.
        dqn_hidden_dim: Dimensión de las capas ocultas de la QNetwork del DQNAgent.
        dqn_batch_size: Tamaño del minibatch de entrenamiento del DQNAgent.
        dqn_buffer_capacity: Capacidad del replay buffer del DQNAgent.
        dqn_target_update: Frecuencia (en pasos) de copia a la target network.
        dqn_epsilon_start: Epsilon inicial de la política epsilon-greedy.
        dqn_epsilon_end: Epsilon mínimo de la política epsilon-greedy.
        dqn_epsilon_decay: Factor multiplicativo de decaimiento de epsilon por paso.

        latent_dim: Dimensión del espacio latente del encoder.
        encoder_hidden_dims: Capas ocultas del MLPLatentEncoder.
        encoder_dropout: Tasa de dropout del encoder (0.0 = sin dropout).
        encoder_activation: Función de activación del encoder ("relu", "tanh", "gelu").
        latent_lr: Learning rate del LatentDQNAgent.
        latent_gamma: Factor de descuento temporal del LatentDQNAgent.
        latent_q_hidden_dim: Dimensión de las capas ocultas de la QNetwork latente.
        latent_batch_size: Tamaño del minibatch del LatentDQNAgent.
        latent_buffer_capacity: Capacidad del replay buffer del LatentDQNAgent.
        latent_target_update: Frecuencia de copia a la target network latente.
        latent_epsilon_start: Epsilon inicial del LatentDQNAgent.
        latent_epsilon_end: Epsilon mínimo del LatentDQNAgent.
        latent_epsilon_decay: Factor de decaimiento de epsilon del LatentDQNAgent.

        pretrain_n_samples: Observaciones a extraer para preentrenar el encoder.
        pretrain_n_epochs: Epochs de entrenamiento del autoencoder.
        pretrain_lr: Learning rate del AutoencoderTrainer.
        pretrain_batch_size: Batch size del AutoencoderTrainer.

        direct_agent: Nombre del agente directo de referencia para el IVL.
        latent_agents: Nombres de los agentes latentes a comparar contra el directo.
        ivl_weights: Pesos de los cuatro componentes del IVL (deben sumar 1.0).
                     Claves: "sharpe", "mdd", "seed_std", "is_oos_gap".

        device: Dispositivo PyTorch ("cpu" o "cuda").
        results_dir: Directorio donde se guardan los CSVs de resultados.
    """

    # ── Datos ────────────────────────────────────────────────────────────────
    ticker:      str           = "SPY"
    start_date:  str           = "2020-01-01"
    end_date:    str           = "2023-12-31"
    n_obs:       Optional[int] = None
    train_ratio: float         = 0.7

    # ── Experimento ──────────────────────────────────────────────────────────
    seeds:                 List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_training_episodes:   int       = 10
    n_eval_episodes:       int       = 3
    max_steps_per_episode: int       = 200

    # ── Entorno ──────────────────────────────────────────────────────────────
    lookback_window:  int   = 10
    initial_balance:  float = 10_000.0
    transaction_cost: float = 0.001

    # ── DQN ──────────────────────────────────────────────────────────────────
    dqn_lr:              float = 1e-3
    dqn_gamma:           float = 0.99
    dqn_hidden_dim:      int   = 128
    dqn_batch_size:      int   = 32
    dqn_buffer_capacity: int   = 5_000
    dqn_target_update:   int   = 50
    dqn_epsilon_start:   float = 1.0
    dqn_epsilon_end:     float = 0.1
    dqn_epsilon_decay:   float = 0.995

    # ── LatentDQN ────────────────────────────────────────────────────────────
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

    # ── Preentrenamiento ─────────────────────────────────────────────────────
    pretrain_n_samples:  int   = 100
    pretrain_n_epochs:   int   = 10
    pretrain_lr:         float = 1e-3
    pretrain_batch_size: int   = 32

    # ── IVL ──────────────────────────────────────────────────────────────────
    direct_agent:  str       = "DQNAgent"
    latent_agents: List[str] = field(default_factory=lambda: [
        "LatentDQNAgent (no pretrained)",
        "LatentDQNAgent (pretrained)",
    ])
    ivl_weights: Dict[str, float] = field(default_factory=lambda: {
        "sharpe":     0.25,
        "mdd":        0.25,
        "seed_std":   0.25,
        "is_oos_gap": 0.25,
    })

    # ── Sistema ──────────────────────────────────────────────────────────────
    device:      str = "cpu"
    results_dir: str = "results"

    # -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Valida los valores de configuración al construir el objeto."""
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
            raise ValueError("seeds no puede estar vacío")
