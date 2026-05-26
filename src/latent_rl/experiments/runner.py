"""
Orquestación de experimentos: entrenamiento, ejecución por semilla y pipeline completo.

Todas las funciones aceptan un ExperimentConfig que centraliza los hiperparámetros.
No quedan valores hardcodeados en este módulo.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from latent_rl.envs import FinancialEnv
from latent_rl.agents import DQNAgent, LatentDQNAgent, RandomAgent, BuyAndHoldAgent
from latent_rl.representations import MLPLatentEncoder
from latent_rl.pretraining import AutoencoderTrainer
from latent_rl.evaluation.latent_advantage import LatentAdvantageIndex

from .config import ExperimentConfig
from .utils import (
    load_yfinance_data,
    split_data,
    evaluate_agent,
    aggregate_results,
    print_summary_table,
    print_ranking,
    export_dashboard_results,
)


# ---------------------------------------------------------------------------
# Entrenamiento de agentes
# ---------------------------------------------------------------------------

def train_dqn_agent(
    env: FinancialEnv,
    n_episodes: int,
    cfg: ExperimentConfig,
) -> DQNAgent:
    """
    Crea y entrena un DQNAgent sobre el entorno dado.

    El entorno debe corresponder al split IS para evitar data leakage.

    Args:
        env: Entorno financiero (IS).
        n_episodes: Número de episodios de entrenamiento.
        cfg: Configuración del experimento con los hiperparámetros del DQN.

    Returns:
        Agente DQN entrenado.
    """
    agent = DQNAgent(
        action_space=env.action_space,
        observation_shape=env.observation_space.shape,
        learning_rate=cfg.dqn_lr,
        gamma=cfg.dqn_gamma,
        epsilon_start=cfg.dqn_epsilon_start,
        epsilon_end=cfg.dqn_epsilon_end,
        epsilon_decay=cfg.dqn_epsilon_decay,
        batch_size=cfg.dqn_batch_size,
        buffer_capacity=cfg.dqn_buffer_capacity,
        target_update_freq=cfg.dqn_target_update,
        hidden_dim=cfg.dqn_hidden_dim,
        device=cfg.device,
    )

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done and step < cfg.max_steps_per_episode:
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
    n_episodes: int,
    cfg: ExperimentConfig,
    pretrained: bool = False,
    pretraining_data: Optional[pd.DataFrame] = None,
) -> LatentDQNAgent:
    """
    Crea y entrena un LatentDQNAgent, opcionalmente con encoder preentrenado.

    El preentrenamiento usa SOLO datos IS (pretraining_data) para evitar
    data leakage: el encoder nunca ve datos OOS antes de la evaluación.

    Args:
        env: Entorno financiero (IS).
        n_episodes: Número de episodios de entrenamiento RL.
        cfg: Configuración del experimento con todos los hiperparámetros.
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
        latent_dim=cfg.latent_dim,
        encoder_hidden_dims=cfg.encoder_hidden_dims,
        encoder_dropout=cfg.encoder_dropout,
        encoder_activation=cfg.encoder_activation,
        learning_rate=cfg.latent_lr,
        gamma=cfg.latent_gamma,
        epsilon_start=cfg.latent_epsilon_start,
        epsilon_end=cfg.latent_epsilon_end,
        epsilon_decay=cfg.latent_epsilon_decay,
        batch_size=cfg.latent_batch_size,
        buffer_capacity=cfg.latent_buffer_capacity,
        target_update_freq=cfg.latent_target_update,
        q_hidden_dim=cfg.latent_q_hidden_dim,
        device=cfg.device,
    )

    if pretrained:
        if pretraining_data is None:
            raise ValueError("pretraining_data es requerido cuando pretrained=True")

        input_dim = cfg.lookback_window * 5  # 5 features OHLCV
        encoder = MLPLatentEncoder(
            input_dim=input_dim,
            latent_dim=cfg.latent_dim,
            hidden_dims=cfg.encoder_hidden_dims,
            activation=cfg.encoder_activation,
            dropout=cfg.encoder_dropout,
        )

        trainer = AutoencoderTrainer(
            encoder=encoder,
            learning_rate=cfg.pretrain_lr,
            batch_size=cfg.pretrain_batch_size,
            device=cfg.device,
        )

        observations = trainer.collect_observations(
            data=pretraining_data,
            lookback_window=cfg.lookback_window,
            n_samples=cfg.pretrain_n_samples,
        )
        observations_normalized = trainer.fit_transform_observations(observations)
        trainer.train(observations_normalized, n_epochs=cfg.pretrain_n_epochs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            encoder_path = Path(tmp_dir) / "pretrained_encoder.pth"
            trainer.save_encoder(str(encoder_path))
            agent.load_pretrained_encoder(str(encoder_path))

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done and step < cfg.max_steps_per_episode:
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
    cfg: ExperimentConfig,
) -> Dict[str, Dict[str, Any]]:
    """
    Ejecuta el experimento completo para una semilla específica.

    Los datos IS y OOS deben estar ya divididos antes de llamar a esta función
    (split temporal idéntico para todas las semillas). La semilla controla
    únicamente la inicialización estocástica de los agentes.

    Flujo:
        1. Fija ``np.random.seed(seed)`` para reproducibilidad.
        2. Entrena cada agente **solo en IS** (sin ver datos OOS).
        3. Evalúa cada agente por separado en IS y OOS.
        4. Devuelve métricas IS y OOS para cada agente.

    Args:
        seed: Semilla aleatoria (afecta solo a los agentes, no a los datos).
        data_is: DataFrame IS con columnas OHLCV en minúsculas.
        data_oos: DataFrame OOS con columnas OHLCV en minúsculas.
        cfg: Configuración del experimento.

    Returns:
        ``results[agent_name]["is" | "oos"]`` con el dict de métricas de evaluate_agent.
    """
    print(f"\n--- Ejecutando semilla {seed} ---")
    np.random.seed(seed)

    env_kwargs = dict(
        lookback_window=cfg.lookback_window,
        initial_balance=cfg.initial_balance,
        transaction_cost=cfg.transaction_cost,
    )
    env_is  = FinancialEnv(data=data_is,  **env_kwargs)
    env_oos = FinancialEnv(data=data_oos, **env_kwargs)

    n_train = cfg.n_training_episodes
    n_eval  = cfg.n_eval_episodes
    results: Dict[str, Dict[str, Any]] = {}

    # 1. RandomAgent
    random_agent = RandomAgent(action_space=env_is.action_space, seed=seed)
    results["RandomAgent"] = {
        "is":  evaluate_agent(random_agent, env_is,  "RandomAgent", n_eval),
        "oos": evaluate_agent(random_agent, env_oos, "RandomAgent", n_eval),
    }

    # 2. BuyAndHoldAgent
    bah_agent = BuyAndHoldAgent(action_space=env_is.action_space)
    results["BuyAndHoldAgent"] = {
        "is":  evaluate_agent(bah_agent, env_is,  "BuyAndHoldAgent", n_eval),
        "oos": evaluate_agent(bah_agent, env_oos, "BuyAndHoldAgent", n_eval),
    }

    # 3. DQNAgent — entrenado solo en IS
    dqn_agent = train_dqn_agent(env_is, n_episodes=n_train, cfg=cfg)
    results["DQNAgent"] = {
        "is":  evaluate_agent(dqn_agent, env_is,  "DQNAgent", n_eval),
        "oos": evaluate_agent(dqn_agent, env_oos, "DQNAgent", n_eval),
    }

    # 4. LatentDQNAgent sin preentrenamiento — entrenado solo en IS
    latent_agent = train_latent_dqn_agent(
        env_is, n_episodes=n_train, cfg=cfg, pretrained=False
    )
    results["LatentDQNAgent (no pretrained)"] = {
        "is":  evaluate_agent(latent_agent, env_is,  "LatentDQNAgent (no pretrained)", n_eval),
        "oos": evaluate_agent(latent_agent, env_oos, "LatentDQNAgent (no pretrained)", n_eval),
    }

    # 5. LatentDQNAgent con encoder preentrenado — preentrenamiento y RL solo en IS
    latent_pretrained = train_latent_dqn_agent(
        env_is,
        n_episodes=n_train,
        cfg=cfg,
        pretrained=True,
        pretraining_data=data_is,  # solo IS — sin data leakage
    )
    results["LatentDQNAgent (pretrained)"] = {
        "is":  evaluate_agent(latent_pretrained, env_is,  "LatentDQNAgent (pretrained)", n_eval),
        "oos": evaluate_agent(latent_pretrained, env_oos, "LatentDQNAgent (pretrained)", n_eval),
    }

    env_is.close()
    env_oos.close()

    return results


# ---------------------------------------------------------------------------
# Cálculo del IVL
# ---------------------------------------------------------------------------

def compute_ivl(
    aggregated: Dict[str, Dict[str, float]],
    cfg: ExperimentConfig,
) -> List[Dict[str, Any]]:
    """
    Calcula el IVL comparando cada agente latente contra el agente directo.

    Args:
        aggregated: Resultado de aggregate_results().
        cfg: Configuración del experimento (define direct_agent, latent_agents,
             e ivl_weights).

    Returns:
        Lista de dicts con los resultados del IVL (uno por par directo-latente).
    """
    ivl_calc = LatentAdvantageIndex(weights=cfg.ivl_weights)

    def _metrics(name: str, split: str) -> Dict[str, float]:
        row = aggregated[name]
        return {
            "sharpe":       row[f"mean_sharpe_{split}"],
            "max_drawdown": row[f"mean_mdd_{split}"],
            "seed_std":     row["seed_std_return_is"],
            "perf_is":      row["mean_return_is"],
            "perf_oos":     row["mean_return_oos"],
        }

    direct_metrics = _metrics(cfg.direct_agent, "is")
    records = []

    for latent_name in cfg.latent_agents:
        if latent_name not in aggregated:
            print(f"  Advertencia: '{latent_name}' no encontrado en resultados, omitiendo.")
            continue

        latent_metrics = _metrics(latent_name, "is")
        result = ivl_calc.compute(direct_metrics, latent_metrics)

        direct_row = aggregated[cfg.direct_agent]
        latent_row = aggregated[latent_name]
        records.append({
            "direct_agent":      cfg.direct_agent,
            "latent_agent":      latent_name,
            "ivl":               result["ivl"],
            "delta_sharpe":      result["delta_sharpe"],
            "delta_mdd":         result["delta_mdd"],
            "delta_seed_std":    result["delta_seed_std"],
            "delta_is_oos_gap":  result["delta_is_oos_gap"],
            "interpretation":    result["interpretation"],
            # Métricas base para trazabilidad
            "direct_sharpe_is":  direct_row["mean_sharpe_is"],
            "direct_return_is":  direct_row["mean_return_is"],
            "direct_return_oos": direct_row["mean_return_oos"],
            "latent_sharpe_is":  latent_row["mean_sharpe_is"],
            "latent_return_is":  latent_row["mean_return_is"],
            "latent_return_oos": latent_row["mean_return_oos"],
        })

        print(f"\nIVL: {cfg.direct_agent} vs {latent_name}")
        print(f"  IVL          = {result['ivl']:.4f}")
        print(f"  d Sharpe     = {result['delta_sharpe']:.4f}")
        print(f"  d MDD        = {result['delta_mdd']:.4f}")
        print(f"  d Seed Std   = {result['delta_seed_std']:.4f}")
        print(f"  d IS/OOS Gap = {result['delta_is_oos_gap']:.4f}")
        print(f"  Interpretacion: {result['interpretation']}")

    return records


def export_ivl_results(
    records: List[Dict[str, Any]],
    cfg: ExperimentConfig,
) -> None:
    """Guarda los resultados del IVL en results_dir/ivl_results.csv."""
    if not records:
        print("No se calculo ningun IVL.")
        return

    results_path = Path(cfg.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    output_path = results_path / "ivl_results.csv"
    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"\nResultados IVL guardados en: {output_path}")


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def run_experiment(cfg: Optional[ExperimentConfig] = None) -> Dict[str, Any]:
    """
    Ejecuta el pipeline experimental completo con la configuración dada.

    Pasos:
        1. Descarga datos reales de Yahoo Finance.
        2. Divide temporalmente en IS y OOS (split único para todas las semillas).
        3. Para cada semilla: entrena todos los agentes en IS y evalúa en IS+OOS.
        4. Agrega resultados y calcula el IVL.
        5. Exporta CSVs al directorio results_dir.

    Args:
        cfg: Configuración del experimento. Si None, usa ExperimentConfig() por defecto.

    Returns:
        Diccionario con ``all_results``, ``aggregated`` e ``ivl_records``.
    """
    if cfg is None:
        cfg = ExperimentConfig()

    print("=" * 120)
    print("EXPERIMENTO DE COMPARACION MULTI-SEMILLA DE AGENTES (IS/OOS)")
    print("=" * 120)
    print(f"\nFuente de datos: Yahoo Finance  |  Ticker: {cfg.ticker}")
    print(f"Periodo: {cfg.start_date} -> {cfg.end_date}  |  N_OBS: {'todos' if cfg.n_obs is None else cfg.n_obs}")
    print(f"Split IS/OOS: {int(cfg.train_ratio*100)}% / {int((1-cfg.train_ratio)*100)}%")
    print(f"Semillas: {cfg.seeds}")
    print(f"Lookback: {cfg.lookback_window}  |  Balance inicial: ${cfg.initial_balance:,.0f}  |  TC: {cfg.transaction_cost:.3%}")
    print(f"Entrenamiento: {cfg.n_training_episodes} episodios  |  Evaluacion: {cfg.n_eval_episodes} episodios")
    print(f"LatentDQN: latent_dim={cfg.latent_dim}, encoder={cfg.encoder_hidden_dims}, dropout={cfg.encoder_dropout}")
    print(f"IVL pesos: {cfg.ivl_weights}")

    # 1. Cargar y dividir datos UNA SOLA VEZ para todas las semillas
    print("\n" + "=" * 120)
    print("CARGA Y DIVISION DE DATOS")
    print("=" * 120)
    data = load_yfinance_data(
        ticker=cfg.ticker,
        start=cfg.start_date,
        end=cfg.end_date,
        n_obs=cfg.n_obs,
    )
    data_is, data_oos = split_data(data, train_ratio=cfg.train_ratio)
    print(f"  IS: {len(data_is)} filas  |  OOS: {len(data_oos)} filas")

    # 2. Experimento multi-semilla
    print("\n" + "=" * 120)
    print("EXPERIMENTO MULTI-SEMILLA")
    print("=" * 120)
    all_results = []
    for seed in cfg.seeds:
        seed_results = run_single_seed(
            seed=seed, data_is=data_is, data_oos=data_oos, cfg=cfg
        )
        all_results.append(seed_results)

        print(f"\nResumen semilla {seed}:")
        for agent_name, splits in seed_results.items():
            m_is  = splits["is"]
            m_oos = splits["oos"]
            print(
                f"  {agent_name}: IS Return={m_is['total_return']:.4f}"
                f" (Sharpe={m_is['sharpe']:.3f}) | "
                f"OOS Return={m_oos['total_return']:.4f}"
                f" (Sharpe={m_oos['sharpe']:.3f})"
            )

    # 3. Agregacion y presentacion
    print("\n" + "=" * 120)
    print("RESULTADOS AGREGADOS")
    print("=" * 120)
    aggregated = aggregate_results(all_results)
    print_summary_table(aggregated)
    print_ranking(aggregated)

    # 4. Exportar CSVs
    export_dashboard_results(all_results, aggregated, results_dir=cfg.results_dir)

    # 5. IVL
    print("\n" + "=" * 120)
    print("INDICE DE VENTAJA LATENTE (IVL)")
    print("=" * 120)
    ivl_records = compute_ivl(aggregated, cfg)
    export_ivl_results(ivl_records, cfg)

    print("\n" + "=" * 120)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 120)
    print(f"- Datos reales de {cfg.ticker} descargados desde Yahoo Finance.")
    print("- El split IS/OOS es temporal: IS precede siempre a OOS.")
    print("- Los datos son identicos para todas las semillas; solo varia la init de los agentes.")
    print("- El preentrenamiento del encoder usa solo datos IS (sin data leakage).")
    print(f"- Resultados en: {cfg.results_dir}/")

    return {
        "all_results": all_results,
        "aggregated":  aggregated,
        "ivl_records": ivl_records,
    }
