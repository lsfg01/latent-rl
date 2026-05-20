"""Script experimental de comparación de agentes para el TFM.

Este script compara de forma homogénea los siguientes agentes:
1. RandomAgent
2. BuyAndHoldAgent
3. DQNAgent
4. LatentDQNAgent sin preentrenamiento
5. LatentDQNAgent con encoder preentrenado

Es un experimento sintético y corto, no optimizado para producción.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from latent_rl.envs import FinancialEnv
from latent_rl.agents import RandomAgent, BuyAndHoldAgent, DQNAgent, LatentDQNAgent
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


def select_action_for_evaluation(agent: Any, obs: np.ndarray) -> int:
    """
    Selecciona una acción para evaluación, manejando diferentes firmas de select_action.

    Args:
        agent: Agente a evaluar
        obs: Observación actual

    Returns:
        Acción seleccionada
    """
    try:
        return agent.select_action(obs, training=False)
    except TypeError:
        return agent.select_action(obs)


def evaluate_agent(agent: Any, env: FinancialEnv, name: str, n_episodes: int = 5) -> Dict[str, float]:
    """
    Evalúa un agente en el entorno y recoge métricas.

    Args:
        agent: Agente a evaluar
        env: Entorno financiero
        name: Nombre del agente
        n_episodes: Número de episodios de evaluación

    Returns:
        Diccionario con métricas agregadas
    """
    print(f"\nEvaluando {name}...")

    metrics_list = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        # Configurar agente para evaluación determinista
        if hasattr(agent, 'reset'):
            agent.reset()
        if hasattr(agent, 'set_epsilon'):
            agent.set_epsilon(0.0)

        while not done:
            # Seleccionar acción
            action = select_action_for_evaluation(agent, obs)

            # Ejecutar acción
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1

        # Recoger métricas del episodio
        episode_metrics = {
            "total_reward": episode_reward,
            "final_equity": info["equity"],
            "total_return": (info["equity"] - env.initial_balance) / env.initial_balance,
            "realized_profit": info["realized_profit"],
            "n_trades": info["n_trades"],
            "steps": steps
        }

        metrics_list.append(episode_metrics)

    # Calcular métricas promedio
    avg_metrics = {
        "name": name,
        "total_reward": np.mean([m["total_reward"] for m in metrics_list]),
        "final_equity": np.mean([m["final_equity"] for m in metrics_list]),
        "total_return": np.mean([m["total_return"] for m in metrics_list]),
        "realized_profit": np.mean([m["realized_profit"] for m in metrics_list]),
        "n_trades": np.mean([m["n_trades"] for m in metrics_list]),
        "steps": np.mean([m["steps"] for m in metrics_list])
    }

    print(f"  - Reward medio: {avg_metrics['total_reward']:.4f}")
    print(f"  - Return medio: {avg_metrics['total_return']:.4f}")
    print(f"  - Equity medio: ${avg_metrics['final_equity']:.2f}")

    return avg_metrics


def train_dqn_agent(env: FinancialEnv, n_episodes: int = 10) -> DQNAgent:
    """
    Entrena un agente DQN.

    Args:
        env: Entorno financiero
        n_episodes: Número de episodios de entrenamiento

    Returns:
        Agente DQN entrenado
    """
    print("\nEntrenando DQNAgent...")

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
        hidden_dim=128,
        device="cpu"
    )

    max_steps_per_episode = 200

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step = 0

        while not done and step < max_steps_per_episode:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)

            if len(agent.replay_buffer) >= agent.batch_size:
                agent.update()

            episode_reward += reward
            obs = next_obs
            step += 1

        print(f"  Episodio {episode + 1}/{n_episodes}: Reward={episode_reward:.4f}, Epsilon={agent.epsilon:.3f}")

    return agent


def train_latent_dqn_agent(env: FinancialEnv, n_episodes: int = 10, pretrained: bool = False, pretraining_data: pd.DataFrame = None) -> LatentDQNAgent:
    """
    Entrena un agente LatentDQN.

    Args:
        env: Entorno financiero
        n_episodes: Número de episodios de entrenamiento
        pretrained: Si True, usa encoder preentrenado
        pretraining_data: DataFrame con datos OHLCV para preentrenamiento (requerido si pretrained=True)

    Returns:
        Agente LatentDQN entrenado
    """
    agent_name = "LatentDQNAgent (pretrained)" if pretrained else "LatentDQNAgent (no pretrained)"
    print(f"\nEntrenando {agent_name}...")

    agent = LatentDQNAgent(
        action_space=env.action_space,
        observation_shape=env.observation_space.shape,
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
        device="cpu"
    )

    # Preentrenar encoder si se solicita
    if pretrained:
        if pretraining_data is None:
            raise ValueError("pretraining_data es requerido cuando pretrained=True")

        print("  Preentrenando encoder...")

        # Crear encoder y trainer
        lookback_window = 10
        input_dim = lookback_window * 5  # 5 features OHLCV
        encoder = MLPLatentEncoder(
            input_dim=input_dim,
            latent_dim=16,
            hidden_dims=[64, 32],
            activation="relu",
            dropout=0.0
        )

        trainer = AutoencoderTrainer(
            encoder=encoder,
            learning_rate=1e-3,
            batch_size=32,
            device="cpu"
        )

        # Extraer observaciones
        n_samples = 100
        observations = trainer.collect_observations(
            data=pretraining_data,
            lookback_window=lookback_window,
            n_samples=n_samples
        )

        # Normalizar y entrenar
        observations_normalized = trainer.fit_transform_observations(observations)
        loss_history = trainer.train(observations_normalized, n_epochs=10)

        print(f"  - Loss inicial: {loss_history[0]:.6f}")
        print(f"  - Loss final: {loss_history[-1]:.6f}")

        # Guardar encoder temporalmente
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        encoder_path = models_dir / "temp_pretrained_encoder.pth"
        trainer.save_encoder(str(encoder_path))

        # Cargar encoder en el agente
        agent.load_pretrained_encoder(str(encoder_path))
        print("  - Encoder preentrenado cargado")

    # Entrenar agente
    max_steps_per_episode = 200

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step = 0

        while not done and step < max_steps_per_episode:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)

            if len(agent.replay_buffer) >= agent.batch_size:
                agent.update()

            episode_reward += reward
            obs = next_obs
            step += 1

        print(f"  Episodio {episode + 1}/{n_episodes}: Reward={episode_reward:.4f}, Epsilon={agent.epsilon:.3f}")

    return agent


def print_comparison_table(results: list):
    """
    Imprime una tabla comparativa con los resultados.

    Args:
        results: Lista de diccionarios con métricas de cada agente
    """
    print("\n" + "=" * 100)
    print("TABLA COMPARATIVA DE AGENTES")
    print("=" * 100)

    # Encabezado
    header = f"{'Agente':<30} {'Total Reward':<15} {'Total Return':<15} {'Final Equity':<15} {'N Trades':<10}"
    print(header)
    print("-" * 100)

    # Filas
    for result in results:
        row = (f"{result['name']:<30} "
               f"{result['total_reward']:<15.4f} "
               f"{result['total_return']:<15.4f} "
               f"${result['final_equity']:<14.2f} "
               f"{result['n_trades']:<10.0f}")
        print(row)

    print("=" * 100)


def print_ranking(results: list):
    """
    Imprime el ranking de agentes por total_return.

    Args:
        results: Lista de diccionarios con métricas de cada agente
    """
    print("\n" + "=" * 60)
    print("RANKING POR TOTAL RETURN")
    print("=" * 60)

    # Ordenar por total_return
    sorted_results = sorted(results, key=lambda x: x['total_return'], reverse=True)

    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result['name']:<30} {result['total_return']:.4f}")

    print("=" * 60)


def main():
    """Función principal del experimento de comparación."""
    print("=" * 100)
    print("EXPERIMENTO DE COMPARACIÓN DE AGENTES")
    print("=" * 100)

    # 1. Configuración del experimento
    print("\n1. CONFIGURACIÓN DEL EXPERIMENTO")
    print("-" * 100)
    print("Datos: Sintéticos OHLCV con tendencia alcista")
    print("Entorno: FinancialEnv")
    print("  - lookback_window: 10")
    print("  - initial_balance: $10,000")
    print("  - transaction_cost: 0.001 (0.1%)")
    print("\nEntrenamiento:")
    print("  - DQNAgent: 10 episodios")
    print("  - LatentDQNAgent (no pretrained): 10 episodios")
    print("  - LatentDQNAgent (pretrained): 10 epochs pretraining + 10 episodios RL")
    print("\nEvaluación:")
    print("  - 5 episodios por agente")
    print("  - Política determinista cuando aplica")

    # 2. Crear datos sintéticos
    print("\n2. CREANDO DATOS SINTÉTICOS")
    print("-" * 100)
    data = create_synthetic_data(n_steps=200)
    print(f"Datos creados: {len(data)} filas")

    # 3. Crear entorno financiero
    print("\n3. CREANDO ENTORNO FINANCIERO")
    print("-" * 100)
    env = FinancialEnv(
        data=data,
        lookback_window=10,
        initial_balance=10000,
        transaction_cost=0.001
    )
    print(f"Espacio de acciones: {env.action_space}")
    print(f"Espacio de observaciones: {env.observation_space.shape}")

    # 4. Crear y evaluar agentes
    print("\n4. FASE DE ENTRENAMIENTO Y EVALUACIÓN")
    print("=" * 100)

    results = []

    # 4.1 RandomAgent
    print("\n--- RandomAgent ---")
    random_agent = RandomAgent(action_space=env.action_space, seed=42)
    random_metrics = evaluate_agent(random_agent, env, "RandomAgent")
    results.append(random_metrics)

    # 4.2 BuyAndHoldAgent
    print("\n--- BuyAndHoldAgent ---")
    buy_and_hold_agent = BuyAndHoldAgent(action_space=env.action_space)
    buy_and_hold_metrics = evaluate_agent(buy_and_hold_agent, env, "BuyAndHoldAgent")
    results.append(buy_and_hold_metrics)

    # 4.3 DQNAgent
    dqn_agent = train_dqn_agent(env, n_episodes=10)
    dqn_metrics = evaluate_agent(dqn_agent, env, "DQNAgent")
    results.append(dqn_metrics)

    # 4.4 LatentDQNAgent sin preentrenamiento
    latent_dqn_agent = train_latent_dqn_agent(env, n_episodes=10, pretrained=False)
    latent_dqn_metrics = evaluate_agent(latent_dqn_agent, env, "LatentDQNAgent (no pretrained)")
    results.append(latent_dqn_metrics)

    # 4.5 LatentDQNAgent con encoder preentrenado
    latent_dqn_pretrained_agent = train_latent_dqn_agent(env, n_episodes=10, pretrained=True, pretraining_data=data)
    latent_dqn_pretrained_metrics = evaluate_agent(latent_dqn_pretrained_agent, env, "LatentDQNAgent (pretrained)")
    results.append(latent_dqn_pretrained_metrics)

    # 5. Imprimir resultados
    print("\n5. RESULTADOS")
    print("=" * 100)

    print_comparison_table(results)
    print_ranking(results)

    # 6. Notas finales
    print("\n" + "=" * 100)
    print("NOTAS FINALES")
    print("=" * 100)
    print("- Este es un experimento sintético y corto.")
    print("- Los datos son sintéticos con una sola semilla aleatoria.")
    print("- El entrenamiento es corto (10 episodios) para demostración.")
    print("- No se prueba superioridad estadística de ningún agente.")
    print("- No se afirma que ningún agente sea mejor de forma general.")
    print("\nUna comparación formal requeriría:")
    print("  * Múltiples semillas aleatorias")
    print("  * Múltiples activos financieros")
    print("  * Diferentes periodos temporales")
    print("  * Métricas financieras (Sharpe ratio, drawdown, etc.)")
    print("  * Validación cruzada temporal")
    print("  * Análisis de significancia estadística")
    print("\nEste experimento sirve como base para investigar:")
    print("  * El efecto del preentrenamiento en el rendimiento RL")
    print("  * La comparación entre representaciones latentes y estándar")
    print("  * El impacto de diferentes arquitecturas de agentes")

    # 7. Cerrar entorno
    env.close()

    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()
