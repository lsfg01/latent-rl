"""Ejemplo de uso de RandomAgent con FinancialEnv.

Este ejemplo demuestra cómo usar el agente aleatorio como línea base
en el entorno financiero.
"""

import numpy as np
import pandas as pd
from latent_rl.envs import FinancialEnv
from latent_rl.agents import RandomAgent


def main():
    """Función principal del ejemplo."""
    print("=== Ejemplo de RandomAgent ===\n")

    # 1. Crear datos sintéticos para el ejemplo
    print("1. Creando datos sintéticos...")
    np.random.seed(42)
    n_steps = 200

    data = pd.DataFrame({
        "open": np.random.randn(n_steps).cumsum() + 100,
        "high": np.random.randn(n_steps).cumsum() + 105,
        "low": np.random.randn(n_steps).cumsum() + 95,
        "close": np.random.randn(n_steps).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, n_steps)
    })

    print(f"   - Datos creados: {len(data)} filas\n")

    # 2. Crear entorno financiero
    print("2. Creando entorno financiero...")
    env = FinancialEnv(data, lookback_window=10, initial_balance=10000)
    print(f"   - Espacio de acciones: {env.action_space}")
    print(f"   - Espacio de observaciones: {env.observation_space}\n")

    # 3. Crear agente aleatorio
    print("3. Creando RandomAgent...")
    agent = RandomAgent(env.action_space, seed=42)
    print("   - Agente creado con semilla 42\n")

    # 4. Ejecutar episodio
    print("4. Ejecutando episodio...")
    obs, info = env.reset(seed=42)
    done = False
    total_reward = 0.0
    step_count = 0

    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

    print(f"   - Pasos ejecutados: {step_count}")
    print(f"   - Recompensa total: {total_reward:.4f}")
    print(f"   - Balance final: {info['balance']:.2f}")
    print(f"   - Profit realizado: {info['realized_profit']:.2f}")
    print(f"   - Número de trades: {info['n_trades']}\n")

    # 5. Ejecutar múltiples episodios para análisis
    print("5. Ejecutando múltiples episodios para análisis...")
    n_episodes = 10
    episode_rewards = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=episode)
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        print(f"   - Episodio {episode + 1}: {episode_reward:.4f}")

    print(f"\n   - Media de recompensas: {np.mean(episode_rewards):.4f}")
    print(f"   - Desviación estándar: {np.std(episode_rewards):.4f}")
    print(f"   - Mejor episodio: {np.max(episode_rewards):.4f}")
    print(f"   - Peor episodio: {np.min(episode_rewards):.4f}\n")

    # 6. Cerrar entorno
    env.close()

    print("=== Ejemplo completado ===")
    print("\nRandomAgent sirve como línea base para comparar con")
    print("futuros agentes de aprendizaje por refuerzo (DQN, PPO, etc.).")


if __name__ == "__main__":
    main()