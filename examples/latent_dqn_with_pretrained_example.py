"""Ejemplo de entrenamiento de LatentDQNAgent con encoder preentrenado.

Este ejemplo demuestra el flujo completo:
1. Preentrenar un encoder latente como autoencoder
2. Cargar el encoder preentrenado en LatentDQNAgent
3. Entrenar el agente con el encoder congelado

Es un ejemplo experimental y educativo, no optimizado para producción.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from latent_rl.envs import FinancialEnv
from latent_rl.agents import LatentDQNAgent
from latent_rl.representations import MLPLatentEncoder
from latent_rl.pretraining import AutoencoderTrainer


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
    """Función principal del pipeline completo."""
    print("=== Ejemplo de LatentDQN con Encoder Preentrenado ===\n")

    # 1. Crear datos sintéticos
    print("1. Creando datos sintéticos...")
    data = create_synthetic_data(n_steps=200)
    print(f"   - Datos creados: {len(data)} filas\n")

    # 2. Configurar parámetros
    lookback_window = 10
    latent_dim = 16
    n_samples = 100  # Número de observaciones para preentrenamiento
    n_pretrain_epochs = 10  # Entrenamiento corto para demostración

    print("2. Configurando parámetros...")
    print(f"   - Lookback window: {lookback_window}")
    print(f"   - Dimensión latente: {latent_dim}")
    print(f"   - Muestras de preentrenamiento: {n_samples}")
    print(f"   - Epochs de preentrenamiento: {n_pretrain_epochs}\n")

    # 3. Crear encoder latente
    print("3. Creando encoder latente...")
    input_dim = lookback_window * 5  # 5 features OHLCV
    encoder = MLPLatentEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=[64, 32],
        activation="relu",
        dropout=0.0
    )
    print("   - Encoder MLPLatentEncoder creado\n")

    # 4. Crear entrenador de autoencoder
    print("4. Creando entrenador de autoencoder...")
    trainer = AutoencoderTrainer(
        encoder=encoder,
        learning_rate=1e-3,
        batch_size=32,
        device="cpu"
    )
    print("   - AutoencoderTrainer creado\n")

    # 5. Extraer observaciones para preentrenamiento
    print("5. Extrayendo observaciones...")
    observations = trainer.collect_observations(
        data=data,
        lookback_window=lookback_window,
        n_samples=n_samples
    )
    print(f"   - Observaciones extraídas: {observations.shape}")
    print(f"   - Forma: (n_samples={observations.shape[0]}, lookback={observations.shape[1]}, features={observations.shape[2]})\n")

    # 6. Normalizar observaciones
    print("6. Normalizando observaciones...")
    observations_normalized = trainer.fit_transform_observations(observations)
    print(f"   - Observaciones normalizadas\n")

    # 7. Entrenar autoencoder
    print("7. Entrenando autoencoder...")
    loss_history = trainer.train(observations_normalized, n_epochs=n_pretrain_epochs)

    print(f"   - Entrenamiento completado")
    print(f"   - Loss inicial: {loss_history[0]:.6f}")
    print(f"   - Loss final: {loss_history[-1]:.6f}")
    print(f"   - Mejora: {loss_history[0] - loss_history[-1]:.6f}\n")

    # 8. Guardar encoder preentrenado
    print("8. Guardando encoder preentrenado...")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = models_dir / "pretrained_encoder.pth"
    trainer.save_encoder(str(encoder_path))
    print(f"   - Encoder guardado en: {encoder_path}\n")

    # 9. Crear entorno financiero
    print("9. Creando entorno financiero...")
    env = FinancialEnv(data, lookback_window=lookback_window, initial_balance=10000)
    print(f"   - Espacio de acciones: {env.action_space}")
    print(f"   - Espacio de observaciones: {env.observation_space.shape}\n")

    # 10. Crear agente LatentDQN
    print("10. Creando agente LatentDQN...")
    agent = LatentDQNAgent(
        action_space=env.action_space,
        observation_shape=env.observation_space.shape,
        latent_dim=latent_dim,
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
        device="cpu"
    )
    print("   - Agente LatentDQN creado")
    print(f"   - Dimensión latente: {agent.latent_dim}\n")

    # 11. Cargar encoder preentrenado
    print("11. Cargando encoder preentrenado...")
    agent.load_pretrained_encoder(str(encoder_path))
    print("   - Encoder preentrenado cargado en LatentDQNAgent")
    print("   - Parámetros del encoder congelados\n")

    # 12. Hiperparámetros de entrenamiento RL
    n_episodes = 10  # Entrenamiento corto para demostración
    max_steps_per_episode = 200

    print(f"12. Entrenando LatentDQN por {n_episodes} episodios...\n")

    # 13. Bucle de entrenamiento
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

    # 14. Evaluación final con política greedy
    print("13. Evaluando política greedy...")
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

    # 15. Guardar agente
    print("14. Guardando agente entrenado...")
    agent_path = models_dir / "latent_dqn_pretrained_agent.pth"
    agent.save(str(agent_path))
    print(f"   - Agente guardado en: {agent_path}\n")

    # 16. Cerrar entorno
    env.close()

    print("=== Ejemplo completado ===")
    print("\nNotas:")
    print("- Este es un pipeline experimental y educativo.")
    print("- El preentrenamiento es corto (10 epochs) para demostración.")
    print("- El entrenamiento RL es corto (10 episodios) para demostración.")
    print("- No se afirma que el encoder preentrenado mejore el rendimiento.")
    print("- Una comparación formal requeriría:")
    print("  * Múltiples semillas aleatorias")
    print("  * Múltiples activos financieros")
    print("  * Diferentes periodos temporales")
    print("  * Métricas financieras (Sharpe ratio, drawdown, etc.)")
    print("- Este ejemplo sirve como base para experimentar con transfer learning en RL financiera.")


if __name__ == "__main__":
    main()
