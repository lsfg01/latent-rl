"""
Experimento con datos REALES de Yahoo Finance.

Descarga SPY (2018-2024), aplica features tecnicos, normaliza IS/OOS
y compara los 5 agentes con 3 semillas.

Uso:
    python -m scripts.experiments.real_data_baseline
    python dashboard/app.py --results-dir results/real_data_baseline
"""

from latent_rl.experiments import ExperimentConfig, run_experiment

config = ExperimentConfig(
    # Datos reales — rango amplio para tener IS y OOS con historia suficiente
    tickers     = ["SPY"],
    start_date  = "2018-01-01",
    end_date    = "2024-01-01",
    train_ratio = 0.7,          # IS: 2018-2022  |  OOS: 2022-2024 aprox
    interval    = "1d",
    cache_dir   = ".data_cache",

    # Features tecnicos sobre OHLCV
    features = [
        "log_return",
        "high_low_range",
        "close_open_pct",
        "volume_ratio",
        "rsi_14",
        "atr_pct",
    ],

    # Normalizacion IS/OOS sin leakage
    normalize_features = True,

    # Experimento (3 semillas para ser rapido)
    seeds                = [0, 1, 2],
    n_training_episodes  = 20,
    n_eval_episodes      = 3,
    max_steps_per_episode = 300,

    # Entorno
    lookback_window  = 20,
    initial_balance  = 10_000.0,
    transaction_cost = 0.001,

    # DQN
    dqn_lr              = 5e-4,
    dqn_hidden_dim      = 128,
    dqn_batch_size      = 64,
    dqn_buffer_capacity = 5_000,
    dqn_target_update   = 100,
    dqn_epsilon_decay   = 0.998,

    # LatentDQN
    latent_dim          = 32,
    encoder_hidden_dims = [32, 64],
    encoder_dropout     = 0.0,
    encoder_activation  = "gelu",
    encoder_type        = "conv1d",
    latent_q_hidden_dim = 128,
    latent_buffer_capacity = 5_000,
    latent_epsilon_decay   = 0.998,
    align_latent_q_with_dqn = True,

    # Preentrenamiento del encoder
    pretrain_n_samples = 500,
    pretrain_n_epochs  = 30,
    pretrain_lr        = 1e-3,
    pretrain_batch_size = 64,

    # IVL
    direct_agent  = "DQNAgent",
    latent_agents = [
        "LatentDQNAgent (no pretrained)",
        "LatentDQNAgent (pretrained)",
    ],
    ivl_weights = {
        "sharpe":     0.25,
        "mdd":        0.25,
        "seed_std":   0.25,
        "is_oos_gap": 0.25,
    },

    device      = "cpu",
    results_dir = "results/real_data_baseline",
)

if __name__ == "__main__":
    run_experiment(config)
