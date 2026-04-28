"""Ejemplo de entrenamiento de DQNAgent.

Este ejemplo demuestra cómo entrenar un agente DQN simple en el entorno financiero.
Es un ejemplo experimental y educativo, no optimizado para producción.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from latent_rl.envs import FinancialEnv
from latent_rl.agents import DQNAgent


def create_synthetic_data(n_steps: int = 200) -> pd.DataFrame:
    """
    Crea datos sintéticos con tendencia alcista.

    Args:
        n_steps: Número de pasos de datos

    Returns:
        DataFrame con datos OHLCV
    """
    np.random.seed(42)

    # Crear tendencia alcista
    trend = np.linspace(0, 20, n_steps)
    noise = np.random.randn(n_steps) * 2
    prices = 100 + trend + noise

    return pd.DataFrame({
        "open": prices + np.random.randn(n_steps),
        "high": prices + np.random.randn(n_steps) + 2,
        "low": prices + np.random.randn(n_steps) - 2,
        "close": prices,
        "volume": np.random.randint(1000, 10000, n_steps)
    })


def main():
    """Función principal de entrenamiento."""
    print("=== Ejemplo de Entrenamiento DQN ===\n")

    # 1. Crear datos sintéticos
    print("1. Creando datos sintéticos...")
    data = create_synthetic_data(n_steps=200)
    print(f"   - Datos creados: {len(data)} filas\n")

    # 2. Crear entorno financiero
    print("2. Creando entorno financiero...")
    env = FinancialEnv(data, lookback_window=10, initial_balance=10000)
    print(f"   - Espacio de acciones: {env.action_space}")
    print(f"   - Espacio de observaciones: {env.observation_space.shape}\n")

    # 3. Crear agente DQN
    print("3. Creando agente DQN...")
    agent = DQNAgent(
        action_space=env.action_space,
        observation_shape=env.observation_space.shape,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        batch_size=32,
        buffer_capacity=5000,
        target_update_freq=50,
        device="cpu"
    )
    print("   - Agente DQN creado\n")

    # 4. Hiperparámetros de entrenamiento
    n_episodes = 10  # Entrenamiento corto para demostración
    max_steps_per_episode = 200

    print(f"4. Entrenando por {n_episodes} episodios...\n")

    # 5. Bucle de entrenamiento
    episode_rewards = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step = 0

        while not done and step < max_steps_per_episode:
            # Seleccionar acción
            action = agent.select_action(obs, training=True)

            # Ejecutar acción
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Almacenar transición
            agent.store_transition(obs, action, reward, next_obs, done)

            # Actualizar modelo si hay suficientes datos
            if len(agent.replay_buffer) >= agent.batch_size:
                metrics = agent.update()

            episode_reward += reward
            obs = next_obs
            step += 1

        episode_rewards.append(episode_reward)

        # Imprimir progreso
        print(f"   Episodio {episode + 1}/{n_episodes}: "
              f"Reward={episode_reward:.4f}, "
              f"Epsilon={agent.epsilon:.3f}, "
              f"Steps={step}")

    print(f"\n   Entrenamiento completado")
    print(f"   - Reward medio: {np.mean(episode_rewards):.4f}")
    print(f"   - Reward std: {np.std(episode_rewards):.4f}")
    print(f"   - Mejor episodio: {np.max(episode_rewards):.4f}")
    print(f"   - Peor episodio: {np.min(episode_rewards):.4f}\n")

    # 6. Evaluación final con política greedy
    print("5. Evaluando política greedy...")
    agent.set_epsilon(0.0)  # Política greedy

    test_rewards = []
    for i in range(5):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step = 0

        while not done and step < max_steps_per_episode:
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1

        test_rewards.append(episode_reward)
        print(f"   Test {i + 1}: Reward={episode_reward:.4f}, Steps={step}")

    print(f"\n   - Reward medio test: {np.mean(test_rewards):.4f}")
    print(f"   - Reward std test: {np.std(test_rewards):.4f}\n")

    # 7. Guardar modelo
    print("6. Guardando modelo...")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "dqn_agent.pth"
    agent.save(str(model_path))
    print(f"   - Modelo guardado en: {model_path}\n")

    # 8. Cerrar entorno
    env.close()

    print("=== Ejemplo completado ===")
    print("\nNotas:")
    print("- Este es un agente DQN experimental y educativo.")
    print("- El entrenamiento es corto (10 episodios) para demostración.")
    print("- No se afirma que DQN supere a RandomAgent o BuyAndHoldAgent.")
    print("- Para resultados de producción, se requiere más entrenamiento y tuning.")
    print("- Este ejemplo sirve como base para experimentar con agentes RL entrenables.")


if __name__ == "__main__":
    main()