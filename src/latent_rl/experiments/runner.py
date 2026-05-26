"""
Orquestación de experimentos: entrenamiento y ejecución por semilla.

Separa la lógica de entrenamiento (train_dqn_agent, train_latent_dqn_agent)
y la orquestación por semilla (run_single_seed) de las utilidades de datos
y evaluación en utils.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from latent_rl.envs import FinancialEnv
from latent_rl.agents import DQNAgent, LatentDQNAgent
from latent_rl.representations import MLPLatentEncoder
from latent_rl.pretraining import AutoencoderTrainer
from latent_rl.agents import RandomAgent, BuyAndHoldAgent

from .utils import evaluate_agent


# ---------------------------------------------------------------------------
# Hiperparámetros por defecto
# ---------------------------------------------------------------------------

_DQN_DEFAULTS = dict(
    learning_rate=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=0.995,
    batch_size=32,
    buffer_capacity=5000,
    target_update_freq=50,
    hidden_dim=128,
    device="cpu",
)

_LATENT_DQN_DEFAULTS = dict(
    latent_dim=16,
    encoder_hidden_dims=[64, 32],
    learning_rate=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=0.995,
    batch_size=32,
    buffer_capacity=5000,
    target_update_freq=50,
    q_hidden_dim=128,
    device="cpu",
)

_MAX_STEPS_PER_EPISODE = 200
_LOOKBACK_WINDOW = 10


# ---------------------------------------------------------------------------
# Entrenamiento de agentes
# ---------------------------------------------------------------------------

def train_dqn_agent(env: FinancialEnv, n_episodes: int = 10) -> DQNAgent:
    """
    Crea y entrena un DQNAgent sobre el entorno dado.

    El entorno debe corresponder al split IS para evitar data leakage.

    Args:
        env: Entorno financiero (IS).
        n_episodes: Número de episodios de entrenamiento.

    Returns:
        Agente DQN entrenado.
    """
    agent = DQNAgent(
        action_space=env.action_space,
        observation_shape=env.observation_space.shape,
        **_DQN_DEFAULTS,
    )

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done and step < _MAX_STEPS_PER_EPISODE:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)
            if len(agent.replay_buffer) >= agent.batch_size:
                agent.update()

            obs = next_obs
            step += 1

    return agent


def train_latent_dqn_agent(
    env: FinancialEnv,
    n_episodes: int = 10,
    pretrained: bool = False,
    pretraining_data: Optional[pd.DataFrame] = None,
) -> LatentDQNAgent:
    """
    Crea y entrena un LatentDQNAgent, opcionalmente con encoder preentrenado.

    El preentrenamiento del encoder usa SOLO datos IS (pretraining_data) para
    evitar data leakage: el encoder nunca ve datos OOS antes de la evaluación.

    Args:
        env: Entorno financiero (IS).
        n_episodes: Número de episodios de entrenamiento RL.
        pretrained: Si True, preentrenar el encoder antes del RL.
        pretraining_data: DataFrame IS para preentrenamiento (requerido si pretrained=True).

    Returns:
        Agente LatentDQN entrenado.

    Raises:
        ValueError: Si pretrained=True y pretraining_data es None.
    """
    agent = LatentDQNAgent(
        action_space=env.action_space,
        observation_shape=env.observation_space.shape,
        **_LATENT_DQN_DEFAULTS,
    )

    if pretrained:
        if pretraining_data is None:
            raise ValueError("pretraining_data es requerido cuando pretrained=True")

        input_dim = _LOOKBACK_WINDOW * 5  # 5 features OHLCV
        encoder = MLPLatentEncoder(
            input_dim=input_dim,
            latent_dim=_LATENT_DQN_DEFAULTS["latent_dim"],
            hidden_dims=_LATENT_DQN_DEFAULTS["encoder_hidden_dims"],
            activation="relu",
            dropout=0.0,
        )

        trainer = AutoencoderTrainer(
            encoder=encoder,
            learning_rate=1e-3,
            batch_size=32,
            device="cpu",
        )

        observations = trainer.collect_observations(
            data=pretraining_data,
            lookback_window=_LOOKBACK_WINDOW,
            n_samples=100,
        )
        observations_normalized = trainer.fit_transform_observations(observations)
        trainer.train(observations_normalized, n_epochs=10)

        with tempfile.TemporaryDirectory() as tmp_dir:
            encoder_path = Path(tmp_dir) / "pretrained_encoder.pth"
            trainer.save_encoder(str(encoder_path))
            agent.load_pretrained_encoder(str(encoder_path))

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done and step < _MAX_STEPS_PER_EPISODE:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)
            if len(agent.replay_buffer) >= agent.batch_size:
                agent.update()

            obs = next_obs
            step += 1

    return agent


# ---------------------------------------------------------------------------
# Ejecución por semilla
# ---------------------------------------------------------------------------

def run_single_seed(
    seed: int,
    data_is: pd.DataFrame,
    data_oos: pd.DataFrame,
    n_training_episodes: int = 10,
    n_eval_episodes: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Ejecuta el experimento completo para una semilla específica.

    Los datos IS y OOS deben estar ya divididos antes de llamar a esta función
    (split temporal idéntico para todas las semillas). La semilla controla
    únicamente la inicialización estocástica de los agentes.

    Flujo:
        1. Fija ``np.random.seed(seed)`` para reproducibilidad del agente.
        2. Entrena cada agente **solo en IS** (sin ver datos OOS).
        3. Evalúa cada agente por separado en IS y OOS.
        4. Devuelve métricas IS y OOS para cada agente.

    Args:
        seed: Semilla aleatoria (afecta solo a los agentes, no a los datos).
        data_is: DataFrame IS con columnas OHLCV en minúsculas.
        data_oos: DataFrame OOS con columnas OHLCV en minúsculas.
        n_training_episodes: Episodios de entrenamiento para agentes RL.
        n_eval_episodes: Episodios de evaluación por agente y split.

    Returns:
        ``results[agent_name]["is" | "oos"]`` con el dict de métricas de evaluate_agent.
    """
    print(f"\n--- Ejecutando semilla {seed} ---")
    np.random.seed(seed)

    env_cfg = dict(lookback_window=10, initial_balance=10_000, transaction_cost=0.001)
    env_is  = FinancialEnv(data=data_is,  **env_cfg)
    env_oos = FinancialEnv(data=data_oos, **env_cfg)

    results: Dict[str, Dict[str, Any]] = {}

    # 1. RandomAgent
    random_agent = RandomAgent(action_space=env_is.action_space, seed=seed)
    results["RandomAgent"] = {
        "is":  evaluate_agent(random_agent, env_is,  "RandomAgent", n_episodes=n_eval_episodes),
        "oos": evaluate_agent(random_agent, env_oos, "RandomAgent", n_episodes=n_eval_episodes),
    }

    # 2. BuyAndHoldAgent
    bah_agent = BuyAndHoldAgent(action_space=env_is.action_space)
    results["BuyAndHoldAgent"] = {
        "is":  evaluate_agent(bah_agent, env_is,  "BuyAndHoldAgent", n_episodes=n_eval_episodes),
        "oos": evaluate_agent(bah_agent, env_oos, "BuyAndHoldAgent", n_episodes=n_eval_episodes),
    }

    # 3. DQNAgent — entrenado solo en IS
    dqn_agent = train_dqn_agent(env_is, n_episodes=n_training_episodes)
    results["DQNAgent"] = {
        "is":  evaluate_agent(dqn_agent, env_is,  "DQNAgent", n_episodes=n_eval_episodes),
        "oos": evaluate_agent(dqn_agent, env_oos, "DQNAgent", n_episodes=n_eval_episodes),
    }

    # 4. LatentDQNAgent sin preentrenamiento — entrenado solo en IS
    latent_agent = train_latent_dqn_agent(env_is, n_episodes=n_training_episodes, pretrained=False)
    results["LatentDQNAgent (no pretrained)"] = {
        "is":  evaluate_agent(latent_agent, env_is,  "LatentDQNAgent (no pretrained)", n_episodes=n_eval_episodes),
        "oos": evaluate_agent(latent_agent, env_oos, "LatentDQNAgent (no pretrained)", n_episodes=n_eval_episodes),
    }

    # 5. LatentDQNAgent con encoder preentrenado — preentrenamiento y RL solo en IS
    latent_pretrained_agent = train_latent_dqn_agent(
        env_is,
        n_episodes=n_training_episodes,
        pretrained=True,
        pretraining_data=data_is,  # solo IS — sin data leakage
    )
    results["LatentDQNAgent (pretrained)"] = {
        "is":  evaluate_agent(latent_pretrained_agent, env_is,  "LatentDQNAgent (pretrained)", n_episodes=n_eval_episodes),
        "oos": evaluate_agent(latent_pretrained_agent, env_oos, "LatentDQNAgent (pretrained)", n_episodes=n_eval_episodes),
    }

    env_is.close()
    env_oos.close()

    return results
