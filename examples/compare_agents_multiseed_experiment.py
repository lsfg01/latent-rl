"""Experimento de comparación multi-semilla de agentes (IS/OOS).

Compara RandomAgent, BuyAndHoldAgent, DQNAgent y dos variantes de LatentDQNAgent
usando datos reales de Yahoo Finance con split temporal In-Sample / Out-of-Sample.

Para personalizar el experimento, modifica los campos de ExperimentConfig.
Todos los hiperparámetros tienen valores por defecto reproducibles.

Uso::

    python examples/compare_agents_multiseed_experiment.py

Salida (un ticker)::

    results/SPY/agent_summary.csv       -- metricas IS/OOS agregadas por agente
    results/SPY/agent_seed_metrics.csv  -- metricas por (agente, semilla, split)
    results/SPY/ivl_results.csv         -- IVL por par (directo, latente)

Salida (varios tickers)::

    results/{ticker}/...                -- resultados por ticker
    results/ticker_comparison.csv       -- comparacion cross-ticker del IVL
"""

from latent_rl.experiments import ExperimentConfig, run_experiment

# Personaliza aqui tu experimento.
# Solo cambia los campos que quieras modificar; el resto usa los valores por
# defecto definidos en ExperimentConfig.
# Para varios tickers: tickers=["SPY", "AAPL", "BTC-USD"]
config = ExperimentConfig(
    # Datos
    tickers=["SPY"],
    start_date="2020-01-01",
    end_date="2023-12-31",
    n_obs=None,           # None = todo el rango; int = ultimas N filas
    train_ratio=0.7,      # 70% IS, 30% OOS

    # Experimento
    seeds=[0, 1, 2, 3, 4],
    n_training_episodes=10,
    n_eval_episodes=3,
    max_steps_per_episode=200,

    # Entorno
    lookback_window=10,
    initial_balance=10_000.0,
    transaction_cost=0.001,

    # DQN
    dqn_lr=1e-3,
    dqn_hidden_dim=128,
    dqn_batch_size=32,
    dqn_buffer_capacity=5_000,
    dqn_target_update=50,

    # LatentDQN
    latent_dim=16,
    encoder_hidden_dims=[64, 32],
    encoder_dropout=0.0,
    encoder_activation="relu",
    latent_q_hidden_dim=128,

    # Preentrenamiento
    pretrain_n_samples=100,
    pretrain_n_epochs=10,

    # IVL — pesos de los 4 componentes (deben sumar 1.0)
    direct_agent="DQNAgent",
    latent_agents=[
        "LatentDQNAgent (no pretrained)",
        "LatentDQNAgent (pretrained)",
    ],
    ivl_weights={
        "sharpe":     0.25,
        "mdd":        0.25,
        "seed_std":   0.25,
        "is_oos_gap": 0.25,
    },

    # Sistema
    device="cpu",
    results_dir="results",
)

if __name__ == "__main__":
    run_experiment(config)
