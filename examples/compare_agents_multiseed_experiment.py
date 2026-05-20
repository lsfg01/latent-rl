"""Script experimental de comparación multi-semilla de agentes para el TFM.

Este script compara de forma homogénea los siguientes agentes con múltiples semillas:
1. RandomAgent
2. BuyAndHoldAgent
3. DQNAgent
4. LatentDQNAgent sin preentrenamiento
5. LatentDQNAgent con encoder preentrenado

Utiliza datos reales descargados con yfinance.
El entrenamiento se realiza sobre datos IS (In-Sample) y la evaluación sobre IS y OOS (Out-of-Sample).
Los datos se descargan una sola vez; las semillas controlan únicamente la inicialización de los agentes.
Los resultados se exportan en formato CSV para el dashboard y para compute_ivl_from_results.py.
"""

import numpy as np
import pandas as pd
import tempfile
import yfinance as yf
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

from latent_rl.envs import FinancialEnv
from latent_rl.agents import RandomAgent, BuyAndHoldAgent, DQNAgent, LatentDQNAgent
from latent_rl.representations import MLPLatentEncoder
from latent_rl.pretraining import AutoencoderTrainer
from latent_rl.evaluation.metrics import FinancialMetrics


def load_yfinance_data(
    ticker: str = "SPY",
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    n_obs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Descarga datos OHLCV reales desde Yahoo Finance y los normaliza.

    Args:
        ticker: Símbolo del activo (p. ej. "SPY", "AAPL", "BTC-USD")
        start: Fecha de inicio en formato "YYYY-MM-DD"
        end: Fecha de fin en formato "YYYY-MM-DD"
        n_obs: Si se especifica, recorta las últimas n_obs filas.
               Útil para controlar el tamaño del experimento sin cambiar fechas.

    Returns:
        DataFrame con columnas lowercase [open, high, low, close, volume],
        índice numérico reseteado y sin NaN.

    Raises:
        ValueError: Si no se descarga ningún dato para el ticker/periodo indicado.
    """
    print(f"  Descargando {ticker} ({start} -> {end})...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No se obtuvieron datos para ticker='{ticker}' en el rango {start}–{end}.")

    # yfinance devuelve columnas capitalizadas: Open, High, Low, Close, Volume
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df = df.reset_index(drop=True)

    if n_obs is not None:
        df = df.tail(n_obs).reset_index(drop=True)

    print(f"  Datos descargados: {len(df)} filas")
    return df


def split_data(data: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide los datos en In-Sample (IS) y Out-of-Sample (OOS).

    Args:
        data: DataFrame completo con datos OHLCV
        train_ratio: Fracción de datos para IS

    Returns:
        Tupla (data_is, data_oos)
    """
    split_idx = int(len(data) * train_ratio)
    return data.iloc[:split_idx].reset_index(drop=True), data.iloc[split_idx:].reset_index(drop=True)


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
    Evalúa un agente en el entorno y recoge métricas financieras.

    Registra la equity paso a paso para derivar Sharpe y MDD a partir de la
    serie de retornos del episodio.

    Args:
        agent: Agente a evaluar
        env: Entorno financiero
        name: Nombre del agente
        n_episodes: Número de episodios de evaluación

    Returns:
        Diccionario con métricas agregadas (incluye sharpe y max_drawdown)
    """
    metrics_list = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        equity_history = [env.initial_balance]

        # Configurar agente para evaluación determinista
        if hasattr(agent, 'reset'):
            agent.reset()
        if hasattr(agent, 'set_epsilon'):
            agent.set_epsilon(0.0)

        while not done:
            action = select_action_for_evaluation(agent, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1
            equity_history.append(info["equity"])

        # Serie de retornos paso a paso
        equity_arr = np.array(equity_history)
        step_returns = np.diff(equity_arr) / equity_arr[:-1]

        n = len(step_returns)
        sharpe = FinancialMetrics.sharpe_ratio(step_returns, periods_per_year=n) if n >= 2 else 0.0
        mdd = FinancialMetrics.max_drawdown(step_returns) if n >= 1 else 0.0

        episode_metrics = {
            "total_reward": episode_reward,
            "final_equity": info["equity"],
            "total_return": (info["equity"] - env.initial_balance) / env.initial_balance,
            "realized_profit": info["realized_profit"],
            "n_trades": info["n_trades"],
            "steps": steps,
            "sharpe": sharpe,
            "max_drawdown": mdd,
        }

        metrics_list.append(episode_metrics)

    avg_metrics = {
        "name": name,
        "total_reward": np.mean([m["total_reward"] for m in metrics_list]),
        "final_equity": np.mean([m["final_equity"] for m in metrics_list]),
        "total_return": np.mean([m["total_return"] for m in metrics_list]),
        "realized_profit": np.mean([m["realized_profit"] for m in metrics_list]),
        "n_trades": np.mean([m["n_trades"] for m in metrics_list]),
        "steps": np.mean([m["steps"] for m in metrics_list]),
        "sharpe": np.mean([m["sharpe"] for m in metrics_list]),
        "max_drawdown": np.mean([m["max_drawdown"] for m in metrics_list]),
    }

    return avg_metrics


def train_dqn_agent(env: FinancialEnv, n_episodes: int = 10) -> DQNAgent:
    """
    Entrena un agente DQN.

    Args:
        env: Entorno financiero (IS)
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


def train_latent_dqn_agent(
    env: FinancialEnv,
    n_episodes: int = 10,
    pretrained: bool = False,
    pretraining_data: pd.DataFrame = None
) -> LatentDQNAgent:
    """
    Entrena un agente LatentDQN.

    Args:
        env: Entorno financiero (IS)
        n_episodes: Número de episodios de entrenamiento
        pretrained: Si True, usa encoder preentrenado
        pretraining_data: Datos IS para preentrenamiento (requerido si pretrained=True)

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

    # Preentrenar encoder solo con datos IS para evitar data leakage
    if pretrained:
        if pretraining_data is None:
            raise ValueError("pretraining_data es requerido cuando pretrained=True")

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

        n_samples = 100
        observations = trainer.collect_observations(
            data=pretraining_data,
            lookback_window=lookback_window,
            n_samples=n_samples
        )

        observations_normalized = trainer.fit_transform_observations(observations)
        trainer.train(observations_normalized, n_epochs=10)

        with tempfile.TemporaryDirectory() as temp_dir:
            encoder_path = Path(temp_dir) / "temp_pretrained_encoder.pth"
            trainer.save_encoder(str(encoder_path))
            agent.load_pretrained_encoder(str(encoder_path))

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


def run_single_seed(
    seed: int,
    data_is: pd.DataFrame,
    data_oos: pd.DataFrame,
    n_training_episodes: int = 10,
    n_eval_episodes: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Ejecuta el experimento completo para una semilla específica.

    Los datos IS y OOS se reciben ya divididos desde main() para que el split
    temporal sea idéntico en todas las semillas. La semilla controla únicamente
    la inicialización estocástica de los agentes (pesos, epsilon-greedy, etc.).

    Flujo:
    1. Fija las semillas de numpy/torch para reproducibilidad del agente.
    2. Entrena cada agente solo en IS (sin ver datos OOS).
    3. Evalúa cada agente por separado en IS y OOS.
    4. Devuelve métricas IS y OOS para cada agente.

    Args:
        seed: Semilla aleatoria (afecta solo a los agentes, no a los datos)
        data_is: DataFrame IS con columnas OHLCV lowercase
        data_oos: DataFrame OOS con columnas OHLCV lowercase
        n_training_episodes: Número de episodios de entrenamiento
        n_eval_episodes: Número de episodios de evaluación

    Returns:
        Diccionario con métricas IS y OOS de cada agente para esta semilla.
        Claves: agent_name -> {"is": {...}, "oos": {...}}
    """
    print(f"\n--- Ejecutando semilla {seed} ---")

    # Fijar semilla para reproducibilidad de los agentes
    np.random.seed(seed)

    env_is = FinancialEnv(data=data_is, lookback_window=10, initial_balance=10000, transaction_cost=0.001)
    env_oos = FinancialEnv(data=data_oos, lookback_window=10, initial_balance=10000, transaction_cost=0.001)

    results = {}

    # 1. RandomAgent
    random_agent = RandomAgent(action_space=env_is.action_space, seed=seed)
    results["RandomAgent"] = {
        "is": evaluate_agent(random_agent, env_is, "RandomAgent", n_episodes=n_eval_episodes),
        "oos": evaluate_agent(random_agent, env_oos, "RandomAgent", n_episodes=n_eval_episodes),
    }

    # 2. BuyAndHoldAgent
    buy_and_hold_agent = BuyAndHoldAgent(action_space=env_is.action_space)
    results["BuyAndHoldAgent"] = {
        "is": evaluate_agent(buy_and_hold_agent, env_is, "BuyAndHoldAgent", n_episodes=n_eval_episodes),
        "oos": evaluate_agent(buy_and_hold_agent, env_oos, "BuyAndHoldAgent", n_episodes=n_eval_episodes),
    }

    # 3. DQNAgent — entrenado solo en IS
    dqn_agent = train_dqn_agent(env_is, n_episodes=n_training_episodes)
    results["DQNAgent"] = {
        "is": evaluate_agent(dqn_agent, env_is, "DQNAgent", n_episodes=n_eval_episodes),
        "oos": evaluate_agent(dqn_agent, env_oos, "DQNAgent", n_episodes=n_eval_episodes),
    }

    # 4. LatentDQNAgent sin preentrenamiento — entrenado solo en IS
    latent_dqn_agent = train_latent_dqn_agent(env_is, n_episodes=n_training_episodes, pretrained=False)
    results["LatentDQNAgent (no pretrained)"] = {
        "is": evaluate_agent(latent_dqn_agent, env_is, "LatentDQNAgent (no pretrained)", n_episodes=n_eval_episodes),
        "oos": evaluate_agent(latent_dqn_agent, env_oos, "LatentDQNAgent (no pretrained)", n_episodes=n_eval_episodes),
    }

    # 5. LatentDQNAgent con encoder preentrenado — preentrenamiento y RL solo en IS
    latent_dqn_pretrained_agent = train_latent_dqn_agent(
        env_is,
        n_episodes=n_training_episodes,
        pretrained=True,
        pretraining_data=data_is,  # Solo IS para evitar data leakage
    )
    results["LatentDQNAgent (pretrained)"] = {
        "is": evaluate_agent(latent_dqn_pretrained_agent, env_is, "LatentDQNAgent (pretrained)", n_episodes=n_eval_episodes),
        "oos": evaluate_agent(latent_dqn_pretrained_agent, env_oos, "LatentDQNAgent (pretrained)", n_episodes=n_eval_episodes),
    }

    env_is.close()
    env_oos.close()

    return results


def aggregate_results(all_results: List[Dict[str, Dict[str, Dict[str, float]]]]) -> Dict[str, Dict[str, float]]:
    """
    Agrega resultados de múltiples semillas.

    Args:
        all_results: Lista de resultados por semilla.
            Estructura: seed_results[agent_name]["is" | "oos"][metric_name]

    Returns:
        Diccionario con métricas agregadas por agente (media sobre semillas)
    """
    agent_names = list(all_results[0].keys())
    aggregated = {}

    for agent_name in agent_names:
        is_returns, oos_returns = [], []
        is_sharpes, oos_sharpes = [], []
        is_mdds, oos_mdds = [], []
        is_equities, oos_equities = [], []
        n_trades_list = []

        for seed_results in all_results:
            m_is = seed_results[agent_name]["is"]
            m_oos = seed_results[agent_name]["oos"]

            is_returns.append(m_is["total_return"])
            oos_returns.append(m_oos["total_return"])
            is_sharpes.append(m_is["sharpe"])
            oos_sharpes.append(m_oos["sharpe"])
            is_mdds.append(m_is["max_drawdown"])
            oos_mdds.append(m_oos["max_drawdown"])
            is_equities.append(m_is["final_equity"])
            oos_equities.append(m_oos["final_equity"])
            n_trades_list.append(m_is["n_trades"])

        aggregated[agent_name] = {
            "name": agent_name,
            # IS
            "mean_return_is": np.mean(is_returns),
            "std_return_is": np.std(is_returns),
            "mean_sharpe_is": np.mean(is_sharpes),
            "mean_mdd_is": np.mean(is_mdds),
            "mean_equity_is": np.mean(is_equities),
            # OOS
            "mean_return_oos": np.mean(oos_returns),
            "std_return_oos": np.std(oos_returns),
            "mean_sharpe_oos": np.mean(oos_sharpes),
            "mean_mdd_oos": np.mean(oos_mdds),
            "mean_equity_oos": np.mean(oos_equities),
            # Cross-sectional
            "mean_n_trades": np.mean(n_trades_list),
            "seed_std_return_is": np.std(is_returns),
        }

    return aggregated


def print_summary_table(aggregated: Dict[str, Dict[str, float]]):
    """
    Imprime una tabla resumen con estadísticas agregadas IS y OOS.

    Args:
        aggregated: Diccionario con métricas agregadas por agente
    """
    print("\n" + "=" * 140)
    print("TABLA RESUMEN DE AGENTES (MULTI-SEMILLA, IS vs OOS)")
    print("=" * 140)

    header = (f"{'Agente':<35} "
              f"{'IS Return':<12} {'IS Sharpe':<12} {'IS MDD':<12} "
              f"{'OOS Return':<12} {'OOS Sharpe':<12} {'OOS MDD':<12}")
    print(header)
    print("-" * 140)

    for agent_name, metrics in aggregated.items():
        row = (f"{metrics['name']:<35} "
               f"{metrics['mean_return_is']:<12.4f} "
               f"{metrics['mean_sharpe_is']:<12.4f} "
               f"{metrics['mean_mdd_is']:<12.4f} "
               f"{metrics['mean_return_oos']:<12.4f} "
               f"{metrics['mean_sharpe_oos']:<12.4f} "
               f"{metrics['mean_mdd_oos']:<12.4f}")
        print(row)

    print("=" * 140)


def print_ranking(aggregated: Dict[str, Dict[str, float]]):
    """
    Imprime el ranking de agentes por mean_return_oos.

    Args:
        aggregated: Diccionario con métricas agregadas por agente
    """
    print("\n" + "=" * 60)
    print("RANKING POR MEAN RETURN OOS")
    print("=" * 60)

    sorted_results = sorted(aggregated.items(), key=lambda x: x[1]['mean_return_oos'], reverse=True)

    for i, (agent_name, metrics) in enumerate(sorted_results, 1):
        print(f"{i}. {metrics['name']:<35} OOS={metrics['mean_return_oos']:.4f}  IS={metrics['mean_return_is']:.4f}")

    print("=" * 60)


def export_dashboard_results(
    all_results: List[Dict[str, Dict[str, Dict[str, float]]]],
    aggregated: Dict[str, Dict[str, float]]
):
    """
    Exporta los resultados en formato CSV para el dashboard y compute_ivl_from_results.py.

    Archivos generados:
    - agent_summary.csv: una fila por agente con métricas IS y OOS agregadas
    - agent_seed_metrics.csv: una fila por (agente, semilla, split) con métricas individuales

    Args:
        all_results: Lista de resultados por semilla
        aggregated: Diccionario con métricas agregadas por agente
    """
    print("\n" + "=" * 120)
    print("EXPORTANDO RESULTADOS PARA EL DASHBOARD")
    print("=" * 120)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. agent_summary.csv
    print("\n1. Exportando agent_summary.csv...")

    summary_data = []
    for agent_name, metrics in aggregated.items():
        summary_data.append({
            "agent_name": agent_name,
            # IS
            "mean_return_is": metrics["mean_return_is"],
            "std_return_is": metrics["std_return_is"],
            "mean_sharpe_is": metrics["mean_sharpe_is"],
            "mean_mdd_is": metrics["mean_mdd_is"],
            "mean_equity_is": metrics["mean_equity_is"],
            # OOS
            "mean_return_oos": metrics["mean_return_oos"],
            "std_return_oos": metrics["std_return_oos"],
            "mean_sharpe_oos": metrics["mean_sharpe_oos"],
            "mean_mdd_oos": metrics["mean_mdd_oos"],
            "mean_equity_oos": metrics["mean_equity_oos"],
            # Cross-sectional
            "mean_n_trades": metrics["mean_n_trades"],
            "seed_std_return_is": metrics["seed_std_return_is"],
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = results_dir / "agent_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"   - Guardado en: {summary_path} ({len(summary_df)} agentes)")

    # 2. agent_seed_metrics.csv
    print("\n2. Exportando agent_seed_metrics.csv...")

    seed_metrics_data = []
    for seed_idx, seed_results in enumerate(all_results):
        for agent_name, splits in seed_results.items():
            for split_name, metrics in splits.items():
                seed_metrics_data.append({
                    "agent_name": agent_name,
                    "seed": seed_idx,
                    "split": split_name,
                    "total_reward": metrics["total_reward"],
                    "total_return": metrics["total_return"],
                    "final_equity": metrics["final_equity"],
                    "realized_profit": metrics["realized_profit"],
                    "n_trades": metrics["n_trades"],
                    "steps": metrics["steps"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                })

    seed_metrics_df = pd.DataFrame(seed_metrics_data)
    seed_metrics_path = results_dir / "agent_seed_metrics.csv"
    seed_metrics_df.to_csv(seed_metrics_path, index=False)
    print(f"   - Guardado en: {seed_metrics_path} ({len(seed_metrics_df)} filas)")

    print("\n" + "=" * 120)
    print("RESULTADOS EXPORTADOS CORRECTAMENTE")
    print("=" * 120)


def main():
    """Función principal del experimento de comparación multi-semilla."""
    # ------------------------------------------------------------------ #
    # Parámetros del experimento — ajustar aquí para cambiar el activo,   #
    # el periodo o el número de observaciones.                             #
    # ------------------------------------------------------------------ #
    TICKER = "SPY"
    START_DATE = "2020-01-01"
    END_DATE = "2023-12-31"
    N_OBS = None          # None = todos los datos del rango; int = últimas N filas
    TRAIN_RATIO = 0.7     # Fracción IS
    SEEDS = [0, 1, 2, 3, 4]
    N_TRAINING_EPISODES = 10
    N_EVAL_EPISODES = 3
    # ------------------------------------------------------------------ #

    print("=" * 120)
    print("EXPERIMENTO DE COMPARACIÓN MULTI-SEMILLA DE AGENTES (IS/OOS)")
    print("=" * 120)

    print("\n1. CONFIGURACIÓN DEL EXPERIMENTO")
    print("-" * 120)
    print(f"Fuente de datos: Yahoo Finance  |  Ticker: {TICKER}")
    print(f"Periodo: {START_DATE} -> {END_DATE}  |  N_OBS: {'todos' if N_OBS is None else N_OBS}")
    print(f"Split IS/OOS: {int(TRAIN_RATIO*100)}% / {int((1-TRAIN_RATIO)*100)}%")
    print(f"Semillas (solo agentes): {SEEDS}")
    print("Entorno: FinancialEnv (lookback_window=10, initial_balance=$10,000, tc=0.1%)")
    print("\nEntrenamiento (solo sobre IS):")
    print("  - DQNAgent: 10 episodios")
    print("  - LatentDQNAgent (no pretrained): 10 episodios")
    print("  - LatentDQNAgent (pretrained): 10 epochs pretraining IS + 10 episodios RL")
    print("\nEvaluación (IS y OOS separados):")
    print(f"  - {N_EVAL_EPISODES} episodios por agente, split y semilla")
    print("  - Política determinista cuando aplica")
    print("  - Métricas: Return, Sharpe, MDD, N Trades")

    # --- Cargar y dividir datos UNA SOLA VEZ para todas las semillas --- #
    print("\n2. CARGA Y DIVISIÓN DE DATOS")
    print("=" * 120)
    data = load_yfinance_data(ticker=TICKER, start=START_DATE, end=END_DATE, n_obs=N_OBS)
    data_is, data_oos = split_data(data, train_ratio=TRAIN_RATIO)
    print(f"  IS:  {len(data_is)} filas  |  OOS: {len(data_oos)} filas")

    print("\n3. EJECUTANDO EXPERIMENTO MULTI-SEMILLA")
    print("=" * 120)

    all_results = []

    for seed in SEEDS:
        seed_results = run_single_seed(
            seed=seed,
            data_is=data_is,
            data_oos=data_oos,
            n_training_episodes=N_TRAINING_EPISODES,
            n_eval_episodes=N_EVAL_EPISODES,
        )
        all_results.append(seed_results)

        print(f"\nResumen semilla {seed}:")
        for agent_name, splits in seed_results.items():
            m_is = splits["is"]
            m_oos = splits["oos"]
            print(f"  {agent_name}: IS Return={m_is['total_return']:.4f} (Sharpe={m_is['sharpe']:.3f}) | "
                  f"OOS Return={m_oos['total_return']:.4f} (Sharpe={m_oos['sharpe']:.3f})")

    print("\n4. AGREGANDO RESULTADOS")
    print("=" * 120)

    aggregated = aggregate_results(all_results)

    print("\n5. RESULTADOS")
    print("=" * 120)

    print_summary_table(aggregated)
    print_ranking(aggregated)

    export_dashboard_results(all_results, aggregated)

    print("\n" + "=" * 120)
    print("NOTAS FINALES")
    print("=" * 120)
    print(f"- Datos reales de {TICKER} descargados desde Yahoo Finance.")
    print("- El split IS/OOS es temporal: IS precede siempre a OOS.")
    print("- Los datos son idénticos para todas las semillas; solo varía la init de los agentes.")
    print("- El preentrenamiento del encoder usa solo datos IS (sin data leakage).")
    print("- Para calcular el IVL, ejecutar compute_ivl_from_results.py.")

    print("\n" + "=" * 120)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 120)


if __name__ == "__main__":
    main()
