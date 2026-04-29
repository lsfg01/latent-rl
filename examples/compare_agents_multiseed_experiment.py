"""Script experimental de comparación multi-semilla de agentes para el TFM.

Este script compara de forma homogénea los siguientes agentes con múltiples semillas:
1. RandomAgent
2. BuyAndHoldAgent
3. DQNAgent
4. LatentDQNAgent sin preentrenamiento
5. LatentDQNAgent con encoder preentrenado

Es un experimento sintético y corto, no optimizado para producción.
"""

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

from latent_rl.envs import FinancialEnv
from latent_rl.agents import RandomAgent, BuyAndHoldAgent, DQNAgent, LatentDQNAgent
from latent_rl.representations import MLPLatentEncoder
from latent_rl.pretraining import AutoencoderTrainer


def create_synthetic_data(n_steps: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Crea datos sintéticos con tendencia alcista.

    Args:
        n_steps: Número de pasos de datos
        seed: Semilla aleatoria para reproducibilidad

    Returns:
        DataFrame con datos OHLCV
    """
    np.random.seed(seed)

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


def evaluate_agent(agent: Any, env: FinancialEnv, name: str, n_episodes: int = 3) -> Dict[str, float]:
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
    metrics_list = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        # Configurar agente para evaluación determinista
        if hasattr(agent, 'set_epsilon'):
            agent.set_epsilon(0.0)
        if hasattr(agent, 'reset'):
            agent.reset()

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

        # Guardar encoder temporalmente en directorio temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            encoder_path = Path(temp_dir) / "temp_pretrained_encoder.pth"
            trainer.save_encoder(str(encoder_path))

            # Cargar encoder en el agente
            agent.load_pretrained_encoder(str(encoder_path))

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

    return agent


def run_single_seed(seed: int, n_training_episodes: int = 10, n_eval_episodes: int = 3) -> Dict[str, Dict[str, float]]:
    """
    Ejecuta el experimento completo para una semilla específica.

    Args:
        seed: Semilla aleatoria
        n_training_episodes: Número de episodios de entrenamiento
        n_eval_episodes: Número de episodios de evaluación

    Returns:
        Diccionario con métricas de cada agente para esta semilla
    """
    print(f"\n--- Ejecutando semilla {seed} ---")

    # Crear datos sintéticos con esta semilla
    data = create_synthetic_data(n_steps=200, seed=seed)

    # Crear entorno financiero
    env = FinancialEnv(
        data=data,
        lookback_window=10,
        initial_balance=10000,
        transaction_cost=0.001
    )

    results = {}

    # 1. RandomAgent
    random_agent = RandomAgent(action_space=env.action_space, seed=seed)
    random_metrics = evaluate_agent(random_agent, env, "RandomAgent", n_episodes=n_eval_episodes)
    results["RandomAgent"] = random_metrics

    # 2. BuyAndHoldAgent
    buy_and_hold_agent = BuyAndHoldAgent(action_space=env.action_space)
    buy_and_hold_metrics = evaluate_agent(buy_and_hold_agent, env, "BuyAndHoldAgent", n_episodes=n_eval_episodes)
    results["BuyAndHoldAgent"] = buy_and_hold_metrics

    # 3. DQNAgent
    dqn_agent = train_dqn_agent(env, n_episodes=n_training_episodes)
    dqn_metrics = evaluate_agent(dqn_agent, env, "DQNAgent", n_episodes=n_eval_episodes)
    results["DQNAgent"] = dqn_metrics

    # 4. LatentDQNAgent sin preentrenamiento
    latent_dqn_agent = train_latent_dqn_agent(env, n_episodes=n_training_episodes, pretrained=False)
    latent_dqn_metrics = evaluate_agent(latent_dqn_agent, env, "LatentDQNAgent (no pretrained)", n_episodes=n_eval_episodes)
    results["LatentDQNAgent (no pretrained)"] = latent_dqn_metrics

    # 5. LatentDQNAgent con encoder preentrenado
    latent_dqn_pretrained_agent = train_latent_dqn_agent(env, n_episodes=n_training_episodes, pretrained=True, pretraining_data=data)
    latent_dqn_pretrained_metrics = evaluate_agent(latent_dqn_pretrained_agent, env, "LatentDQNAgent (pretrained)", n_episodes=n_eval_episodes)
    results["LatentDQNAgent (pretrained)"] = latent_dqn_pretrained_metrics

    # Cerrar entorno
    env.close()

    return results


def aggregate_results(all_results: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    Agrega resultados de múltiples semillas.

    Args:
        all_results: Lista de resultados por semilla

    Returns:
        Diccionario con métricas agregadas por agente
    """
    agent_names = list(all_results[0].keys())
    aggregated = {}

    for agent_name in agent_names:
        # Recoger todas las métricas de este agente
        returns = []
        equities = []
        n_trades_list = []

        for seed_results in all_results:
            agent_metrics = seed_results[agent_name]
            returns.append(agent_metrics["total_return"])
            equities.append(agent_metrics["final_equity"])
            n_trades_list.append(agent_metrics["n_trades"])

        # Calcular estadísticas
        aggregated[agent_name] = {
            "name": agent_name,
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "min_return": np.min(returns),
            "max_return": np.max(returns),
            "mean_equity": np.mean(equities),
            "mean_n_trades": np.mean(n_trades_list)
        }

    return aggregated


def print_summary_table(aggregated: Dict[str, Dict[str, float]]):
    """
    Imprime una tabla resumen con estadísticas agregadas.

    Args:
        aggregated: Diccionario con métricas agregadas por agente
    """
    print("\n" + "=" * 120)
    print("TABLA RESUMEN DE AGENTES (MULTI-SEMILLA)")
    print("=" * 120)

    # Encabezado
    header = (f"{'Agente':<35} {'Mean Return':<15} {'Std Return':<15} {'Min Return':<15} "
              f"{'Max Return':<15} {'Mean Equity':<15} {'Mean N Trades':<15}")
    print(header)
    print("-" * 120)

    # Filas
    for agent_name, metrics in aggregated.items():
        row = (f"{metrics['name']:<35} "
               f"{metrics['mean_return']:<15.4f} "
               f"{metrics['std_return']:<15.4f} "
               f"{metrics['min_return']:<15.4f} "
               f"{metrics['max_return']:<15.4f} "
               f"${metrics['mean_equity']:<14.2f} "
               f"{metrics['mean_n_trades']:<15.0f}")
        print(row)

    print("=" * 120)


def print_ranking(aggregated: Dict[str, Dict[str, float]]):
    """
    Imprime el ranking de agentes por mean_return.

    Args:
        aggregated: Diccionario con métricas agregadas por agente
    """
    print("\n" + "=" * 60)
    print("RANKING POR MEAN RETURN")
    print("=" * 60)

    # Ordenar por mean_return
    sorted_results = sorted(aggregated.items(), key=lambda x: x[1]['mean_return'], reverse=True)

    for i, (agent_name, metrics) in enumerate(sorted_results, 1):
        print(f"{i}. {metrics['name']:<35} {metrics['mean_return']:.4f} (±{metrics['std_return']:.4f})")

    print("=" * 60)


def export_dashboard_results(all_results: List[Dict[str, Dict[str, float]]], aggregated: Dict[str, Dict[str, float]]):
    """
    Exporta los resultados en formato CSV para el dashboard.

    Args:
        all_results: Lista de resultados por semilla
        aggregated: Diccionario con métricas agregadas por agente
    """
    print("\n" + "=" * 120)
    print("EXPORTANDO RESULTADOS PARA EL DASHBOARD")
    print("=" * 120)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Exportar agent_summary.csv (una fila por agente)
    print("\n1. Exportando agent_summary.csv...")

    summary_data = []
    for agent_name, metrics in aggregated.items():
        summary_data.append({
            "agent_name": agent_name,
            "mean_return": metrics["mean_return"],
            "std_return": metrics["std_return"],
            "min_return": metrics["min_return"],
            "max_return": metrics["max_return"],
            "mean_equity": metrics["mean_equity"],
            "mean_n_trades": metrics["mean_n_trades"]
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = results_dir / "agent_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"   - Guardado en: {summary_path}")
    print(f"   - {len(summary_df)} agentes")

    # 2. Exportar agent_seed_metrics.csv (una fila por agente y semilla)
    print("\n2. Exportando agent_seed_metrics.csv...")

    seed_metrics_data = []
    for seed_idx, seed_results in enumerate(all_results):
        for agent_name, metrics in seed_results.items():
            seed_metrics_data.append({
                "agent_name": agent_name,
                "seed": seed_idx,
                "total_reward": metrics["total_reward"],
                "total_return": metrics["total_return"],
                "final_equity": metrics["final_equity"],
                "realized_profit": metrics["realized_profit"],
                "n_trades": metrics["n_trades"],
                "steps": metrics["steps"]
            })

    seed_metrics_df = pd.DataFrame(seed_metrics_data)
    seed_metrics_path = results_dir / "agent_seed_metrics.csv"
    seed_metrics_df.to_csv(seed_metrics_path, index=False)
    print(f"   - Guardado en: {seed_metrics_path}")
    print(f"   - {len(seed_metrics_df)} filas (agentes × semillas)")

    print("\n" + "=" * 120)
    print("RESULTADOS EXPORTADOS CORRECTAMENTE")
    print("=" * 120)


def main():
    """Función principal del experimento de comparación multi-semilla."""
    print("=" * 120)
    print("EXPERIMENTO DE COMPARACIÓN MULTI-SEMILLA DE AGENTES")
    print("=" * 120)

    # 1. Configuración del experimento
    print("\n1. CONFIGURACIÓN DEL EXPERIMENTO")
    print("-" * 120)
    print("Datos: Sintéticos OHLCV con tendencia alcista")
    print("Semillas: [0, 1, 2, 3, 4]")
    print("Entorno: FinancialEnv")
    print("  - lookback_window: 10")
    print("  - initial_balance: $10,000")
    print("  - transaction_cost: 0.001 (0.1%)")
    print("\nEntrenamiento:")
    print("  - DQNAgent: 10 episodios")
    print("  - LatentDQNAgent (no pretrained): 10 episodios")
    print("  - LatentDQNAgent (pretrained): 10 epochs pretraining + 10 episodios RL")
    print("\nEvaluación:")
    print("  - 3 episodios por agente y semilla")
    print("  - Política determinista cuando aplica")

    # 2. Ejecutar experimento con múltiples semillas
    print("\n2. EJECUTANDO EXPERIMENTO MULTI-SEMILLA")
    print("=" * 120)

    seeds = [0, 1, 2, 3, 4]
    n_training_episodes = 10
    n_eval_episodes = 3

    all_results = []

    for seed in seeds:
        seed_results = run_single_seed(seed, n_training_episodes, n_eval_episodes)
        all_results.append(seed_results)

        # Imprimir resumen de esta semilla
        print(f"\nResumen semilla {seed}:")
        for agent_name, metrics in seed_results.items():
            print(f"  {agent_name}: Return={metrics['total_return']:.4f}, Equity=${metrics['final_equity']:.2f}")

    # 3. Agregar resultados
    print("\n3. AGREGANDO RESULTADOS")
    print("=" * 120)

    aggregated = aggregate_results(all_results)

    # 4. Imprimir resultados
    print("\n4. RESULTADOS")
    print("=" * 120)

    print_summary_table(aggregated)
    print_ranking(aggregated)

    # 5. Exportar resultados para el dashboard
    export_dashboard_results(all_results, aggregated)

    # 6. Notas finales
    print("\n" + "=" * 120)
    print("NOTAS FINALES")
    print("=" * 120)
    print("- Este es un experimento sintético y corto.")
    print("- Los datos son sintéticos con múltiples semillas aleatorias.")
    print("- El entrenamiento es corto (10 episodios) para demostración.")
    print("- No se prueba superioridad estadística de ningún agente.")
    print("- No se afirma que ningún agente sea mejor de forma general.")
    print("\nMejoras respecto a experimento de semilla única:")
    print("  * Múltiples semillas aleatorias aumentan la robustez")
    print("  * Se observan variaciones en el rendimiento entre semillas")
    print("  * Se calculan estadísticas (media, desviación estándar, min, max)")
    print("\nUna evaluación formal requeriría:")
    print("  * Activos financieros reales (no sintéticos)")
    print("  * Splits temporales apropiados (train/validation/test)")
    print("  * Métricas financieras (Sharpe ratio, drawdown, etc.)")
    print("  * Validación cruzada temporal")
    print("  * Tests de significancia estadística")
    print("  * Comparación con baselines financieros estándar")
    print("\nEste experimento sirve como base para investigar:")
    print("  * La variabilidad del rendimiento entre semillas")
    print("  * El efecto del preentrenamiento en diferentes condiciones")
    print("  * La consistencia de las representaciones latentes vs estándar")

    print("\n" + "=" * 120)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 120)


if __name__ == "__main__":
    main()
