"""
Orquestación de experimentos: entrenamiento, ejecución por semilla y pipeline completo.

Todas las funciones aceptan un ExperimentConfig que centraliza los hiperparámetros.
Soporta uno o varios tickers: cada ticker ejecuta el pipeline de forma independiente
y los resultados se agregan en un resumen cross-ticker.

Brazos experimentales:
  A  DQNAgent directo
  B  LatentDQNAgent (encoder random, finetune)
  C  LatentDQNAgent (encoder ligero pretrained en IS, frozen)
  D  LatentDQNAgent (encoder heavy offline, frozen) — requiere cfg.heavy_encoder_path
"""

from __future__ import annotations

import copy
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from latent_rl.envs import FinancialEnv, LatentObservationWrapper, precompute_latent_series
from latent_rl.agents import DQNAgent, LatentDQNAgent, RandomAgent, BuyAndHoldAgent
from latent_rl.representations import MLPLatentEncoder, build_encoder
from latent_rl.pretraining import AutoencoderTrainer
from latent_rl.evaluation.latent_advantage import LatentAdvantageIndex

from .config import ExperimentConfig
from .utils import (
    load_yfinance_data,
    load_ticker_with_config,
    split_data,
    normalize_is_oos,
    normalize_train_val_oos,
    split_internal_validation,
    walk_forward_splits,
    evaluate_agent,
    aggregate_results,
    print_summary_table,
    print_ranking,
    export_dashboard_results,
)


# ---------------------------------------------------------------------------
# Contexto de encoders congelados (precomputación)
# ---------------------------------------------------------------------------

@dataclass
class FrozenEncoderContext:
    """Latentes precomputados para los brazos C y D.

    Calculado una vez por ticker/split, reutilizado en cada semilla.
    Para encoders congelados, latente(encoder(obs)) es idéntico en cada
    iteración, por lo que precomputarlo no altera los resultados numéricos.
    """
    c_encoder: Optional[Any]
    c_latents_train: Optional[np.ndarray]
    c_latents_val: Optional[np.ndarray]
    c_latents_is: Optional[np.ndarray]    # (T_is, latent_dim)
    c_latents_oos: Optional[np.ndarray]   # (T_oos, latent_dim)
    d_encoder: Optional[Any]
    d_latents_train: Optional[np.ndarray]
    d_latents_val: Optional[np.ndarray]
    d_latents_is: Optional[np.ndarray]    # (T_is, latent_dim)
    d_latents_oos: Optional[np.ndarray]   # (T_oos, latent_dim)


def prepare_arm_c_encoder(
    data_is: pd.DataFrame,
    cfg: ExperimentConfig,
) -> Optional[Any]:
    """Entrena el encoder ligero del brazo C sobre IS una única vez.

    Returns:
        Encoder entrenado y congelado, o None si no hay features/log_return.
    """
    from latent_rl.pretraining.encoder_pretrainer import EncoderPretrainer

    features = list(cfg.features) if cfg.features else []
    if not features or "log_return" not in features:
        return None

    L = cfg.lookback_window
    F = len(features)
    encoder = build_encoder(
        cfg.encoder_type,
        input_len=L,
        n_features=F,
        **cfg.encoder_kwargs(),
    )
    pretrainer = EncoderPretrainer(
        encoder=encoder,
        learning_rate=cfg.pretrain_lr,
        batch_size=cfg.pretrain_batch_size,
        lambda_forecast=cfg.pretrain_lambda_forecast,
        k=cfg.pretrain_k_forecast,
        device=cfg.device,
    )
    X, Y = pretrainer.make_windows(data_is, features, L)
    if len(X) > 0:
        pretrainer.train(
            X=X, Y=Y,
            n_epochs=cfg.pretrain_n_epochs,
            val_ratio=0.1,
            early_stopping_patience=5,
        )
    encoder.freeze()
    encoder.eval()
    return encoder


def _build_frozen_ctx(
    data_is: pd.DataFrame,
    data_oos: pd.DataFrame,
    cfg: ExperimentConfig,
    d_encoder: Optional[Any],
    data_train: Optional[pd.DataFrame] = None,
    data_val: Optional[pd.DataFrame] = None,
) -> FrozenEncoderContext:
    """Precomputa latentes para C y D antes del bucle de semillas."""
    features = list(cfg.features) if cfg.features else []
    device = torch.device(cfg.device)
    L = cfg.lookback_window

    train_source = data_train if data_train is not None else data_is
    c_encoder = c_latents_train = c_latents_val = None
    c_latents_is = c_latents_oos = None
    d_latents_train = d_latents_val = None
    d_latents_is = d_latents_oos = None

    if "C" in cfg.run_arms and features:
        print("  [ctx] Entrenando encoder ligero (brazo C)...")
        c_encoder = prepare_arm_c_encoder(train_source, cfg)
        if c_encoder is not None:
            c_encoder = c_encoder.to(device)
            print("  [ctx] Precomputando latentes C (train/val + IS/OOS)...")
            c_latents_train = precompute_latent_series(
                train_source, L, features, c_encoder, device
            )
            if data_val is not None:
                c_latents_val = precompute_latent_series(
                    data_val, L, features, c_encoder, device
                )
            c_latents_is  = precompute_latent_series(data_is,  L, features, c_encoder, device)
            c_latents_oos = precompute_latent_series(data_oos, L, features, c_encoder, device)

    if "D" in cfg.run_arms and d_encoder is not None and features:
        print("  [ctx] Precomputando latentes D (IS + OOS)...")
        d_enc = d_encoder.to(device)
        d_latents_train = precompute_latent_series(
            train_source, L, features, d_enc, device
        )
        if data_val is not None:
            d_latents_val = precompute_latent_series(
                data_val, L, features, d_enc, device
            )
        d_latents_is  = precompute_latent_series(data_is,  L, features, d_enc, device)
        d_latents_oos = precompute_latent_series(data_oos, L, features, d_enc, device)

    return FrozenEncoderContext(
        c_encoder=c_encoder,
        c_latents_train=c_latents_train,
        c_latents_val=c_latents_val,
        c_latents_is=c_latents_is,
        c_latents_oos=c_latents_oos,
        d_encoder=d_encoder,
        d_latents_train=d_latents_train,
        d_latents_val=d_latents_val,
        d_latents_is=d_latents_is,
        d_latents_oos=d_latents_oos,
    )


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _build_env_kwargs(cfg: ExperimentConfig, data: pd.DataFrame) -> dict:
    """Construye kwargs para FinancialEnv respetando el desacople obs/ejecución."""
    feature_cols = list(cfg.features) if cfg.features else None
    return dict(
        lookback_window=cfg.lookback_window,
        initial_balance=cfg.initial_balance,
        transaction_cost=cfg.transaction_cost,
        feature_cols=feature_cols,
        price_col="close",
        reward_mode=cfg.reward_mode,
        reward_clip=cfg.reward_clip,
        trade_penalty=cfg.trade_penalty,
    )


def _make_latent_agent(
    env: FinancialEnv,
    cfg: ExperimentConfig,
    freeze_encoder: bool = False,
) -> LatentDQNAgent:
    """Crea un LatentDQNAgent con el encoder_type de cfg."""
    kwargs = cfg.encoder_kwargs()
    return LatentDQNAgent(
        action_space=env.action_space,
        observation_shape=env.observation_space.shape,
        latent_dim=cfg.latent_dim,
        encoder_type=cfg.encoder_type,
        encoder_hidden_dims=cfg.encoder_hidden_dims,
        encoder_dropout=cfg.encoder_dropout,
        encoder_activation=cfg.encoder_activation,
        tcn_kernel_size=cfg.tcn_kernel_size,
        tcn_dilations=cfg.tcn_dilations,
        tcn_channels=cfg.tcn_channels,
        gru_hidden_dim=cfg.gru_hidden_dim,
        gru_num_layers=cfg.gru_num_layers,
        learning_rate=cfg.latent_lr,
        gamma=cfg.latent_gamma,
        epsilon_start=cfg.latent_epsilon_start,
        epsilon_end=cfg.latent_epsilon_end,
        epsilon_decay=cfg.latent_epsilon_decay,
        batch_size=cfg.latent_batch_size,
        buffer_capacity=cfg.latent_buffer_capacity,
        target_update_freq=cfg.latent_target_update,
        q_hidden_dim=cfg.latent_q_hidden_dim,
        weight_decay=cfg.latent_weight_decay,
        grad_clip_norm=cfg.latent_grad_clip_norm,
        q_dropout=cfg.latent_q_dropout,
        freeze_encoder=freeze_encoder,
        device=cfg.device,
    )


def _make_precomputed_latent_agent(
    env,
    cfg: ExperimentConfig,
) -> LatentDQNAgent:
    """Crea un LatentDQNAgent en modo precomputed_latents (sin encoder interno)."""
    return LatentDQNAgent(
        action_space=env.action_space,
        observation_shape=(cfg.latent_dim,),
        latent_dim=cfg.latent_dim,
        precomputed_latents=True,
        learning_rate=cfg.latent_lr,
        gamma=cfg.latent_gamma,
        epsilon_start=cfg.latent_epsilon_start,
        epsilon_end=cfg.latent_epsilon_end,
        epsilon_decay=cfg.latent_epsilon_decay,
        batch_size=cfg.latent_batch_size,
        buffer_capacity=cfg.latent_buffer_capacity,
        target_update_freq=cfg.latent_target_update,
        q_hidden_dim=cfg.latent_q_hidden_dim,
        weight_decay=cfg.latent_weight_decay,
        grad_clip_norm=cfg.latent_grad_clip_norm,
        q_dropout=cfg.latent_q_dropout,
        device=cfg.device,
    )


def _capture_agent_state(agent) -> Dict[str, Any]:
    """Copia el estado entrenable sin serializarlo a disco."""
    state: Dict[str, Any] = {}
    for name in ("encoder", "q_network", "target_network"):
        module = getattr(agent, name, None)
        if isinstance(module, torch.nn.Module):
            state[f"{name}_state"] = copy.deepcopy(module.state_dict())
            state[f"{name}_training"] = module.training
    optimizer = getattr(agent, "optimizer", None)
    if optimizer is not None:
        state["optimizer_state"] = copy.deepcopy(optimizer.state_dict())
    for name in ("epsilon", "update_step"):
        if hasattr(agent, name):
            state[name] = getattr(agent, name)
    return state


def _restore_agent_state(agent, state: Dict[str, Any]) -> None:
    for name in ("encoder", "q_network", "target_network"):
        module = getattr(agent, name, None)
        key = f"{name}_state"
        if isinstance(module, torch.nn.Module) and key in state:
            module.load_state_dict(state[key])
            module.train(state[f"{name}_training"])
    if "optimizer_state" in state and getattr(agent, "optimizer", None) is not None:
        agent.optimizer.load_state_dict(state["optimizer_state"])
    for name in ("epsilon", "update_step"):
        if name in state:
            setattr(agent, name, state[name])


def _validation_score(metrics: Dict[str, float], cfg: ExperimentConfig) -> float:
    trade_rate = metrics["n_trades"] / max(metrics["steps"], 1.0)
    return float(
        metrics["sharpe"]
        - cfg.validation_score_mdd_weight * abs(metrics["max_drawdown"])
        - cfg.validation_score_trade_weight * trade_rate
    )


def _train_agent(
    env,
    agent,
    n_episodes: int,
    cfg: ExperimentConfig,
    validation_env=None,
    seed: Optional[int] = None,
):
    """Bucle RL con random starts y seleccion opcional por validacion temporal."""
    best_state: Optional[Dict[str, Any]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_score = -np.inf
    validations_without_improvement = 0

    for episode in range(1, n_episodes + 1):
        reset_kwargs: Dict[str, Any] = {}
        if cfg.random_start_train:
            reset_kwargs["options"] = {
                "random_start": True,
                "max_steps": cfg.max_steps_per_episode,
            }
        if episode == 1 and seed is not None:
            reset_kwargs["seed"] = seed
        obs, _ = env.reset(**reset_kwargs)
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

        should_validate = validation_env is not None and (
            episode % cfg.validation_eval_freq == 0 or episode == n_episodes
        )
        if should_validate:
            training_state = _capture_agent_state(agent)
            val_metrics = evaluate_agent(
                agent, validation_env, "validation", cfg.n_eval_episodes
            )
            _restore_agent_state(agent, training_state)
            score = _validation_score(val_metrics, cfg)
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(training_state)
                best_metrics = dict(val_metrics)
                best_metrics["episode"] = episode
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1

            if (
                cfg.validation_patience is not None
                and validations_without_improvement >= cfg.validation_patience
            ):
                break

    if best_state is not None and best_metrics is not None:
        _restore_agent_state(agent, best_state)
        agent.validation_metrics = {
            "best_val_episode": int(best_metrics["episode"]),
            "best_val_score": float(best_score),
            "best_val_sharpe": float(best_metrics["sharpe"]),
            "best_val_mdd": float(best_metrics["max_drawdown"]),
            "best_val_return": float(best_metrics["total_return"]),
            "best_val_n_trades": float(best_metrics["n_trades"]),
        }
    return agent


# ---------------------------------------------------------------------------
# Entrenamiento de agentes (API pública)
# ---------------------------------------------------------------------------

def train_dqn_agent(
    env: FinancialEnv,
    n_episodes: int,
    cfg: ExperimentConfig,
    validation_env: Optional[FinancialEnv] = None,
    seed: Optional[int] = None,
) -> DQNAgent:
    """Crea y entrena un DQNAgent sobre el entorno IS."""
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
        weight_decay=cfg.dqn_weight_decay,
        grad_clip_norm=cfg.dqn_grad_clip_norm,
        dropout=cfg.dqn_dropout,
        device=cfg.device,
    )
    return _train_agent(
        env, agent, n_episodes, cfg, validation_env=validation_env, seed=seed
    )


def train_latent_dqn_agent(
    env: FinancialEnv,
    n_episodes: int,
    cfg: ExperimentConfig,
    pretrained: bool = False,
    pretraining_data: Optional[pd.DataFrame] = None,
    validation_env: Optional[FinancialEnv] = None,
    seed: Optional[int] = None,
) -> LatentDQNAgent:
    """Crea y entrena un LatentDQNAgent (legado, mantiene pretrained=bool API).

    Para los brazos C/D con el nuevo protocolo usar las funciones
    ``_train_arm_c`` / ``_train_arm_d`` directamente.
    """
    agent = _make_latent_agent(env, cfg, freeze_encoder=False)

    if pretrained:
        if pretraining_data is None:
            raise ValueError("pretraining_data es requerido cuando pretrained=True")

        # Preentrenamiento legado con AutoencoderTrainer (MLP flat)
        L = cfg.lookback_window
        F = len(cfg.features) if cfg.features else 5
        input_dim = L * F
        encoder_mlp = MLPLatentEncoder(
            input_dim=input_dim,
            latent_dim=cfg.latent_dim,
            hidden_dims=cfg.encoder_hidden_dims,
            activation=cfg.encoder_activation,
            dropout=cfg.encoder_dropout,
            input_len=L,
            n_features=F,
        )
        trainer = AutoencoderTrainer(
            encoder=encoder_mlp,
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

    return _train_agent(
        env, agent, n_episodes, cfg, validation_env=validation_env, seed=seed
    )


# ---------------------------------------------------------------------------
# Brazos individuales
# ---------------------------------------------------------------------------

def _train_arm_a(
    env: FinancialEnv,
    n_episodes: int,
    cfg: ExperimentConfig,
    validation_env: Optional[FinancialEnv] = None,
    seed: Optional[int] = None,
) -> DQNAgent:
    """Brazo A: DQNAgent directo."""
    return train_dqn_agent(env, n_episodes, cfg, validation_env, seed)


def _train_arm_b(
    env: FinancialEnv,
    n_episodes: int,
    cfg: ExperimentConfig,
    validation_env: Optional[FinancialEnv] = None,
    seed: Optional[int] = None,
) -> LatentDQNAgent:
    """Brazo B: LatentDQNAgent encoder random, finetune."""
    agent = _make_latent_agent(env, cfg, freeze_encoder=False)
    return _train_agent(env, agent, n_episodes, cfg, validation_env, seed)


def _train_arm_c(
    env: FinancialEnv,
    n_episodes: int,
    cfg: ExperimentConfig,
    data_is: pd.DataFrame,
    validation_env: Optional[FinancialEnv] = None,
    seed: Optional[int] = None,
) -> LatentDQNAgent:
    """Brazo C: encoder ligero pretrained en IS del pliegue, frozen."""
    from latent_rl.pretraining.encoder_pretrainer import EncoderPretrainer
    from latent_rl.representations.artifact import save_encoder_artifact, load_encoder_artifact

    L = cfg.lookback_window
    F = len(cfg.features) if cfg.features else env.n_features

    encoder = build_encoder(
        cfg.encoder_type,
        input_len=L,
        n_features=F,
        **cfg.encoder_kwargs(),
    )
    pretrainer = EncoderPretrainer(
        encoder=encoder,
        learning_rate=cfg.pretrain_lr,
        batch_size=cfg.pretrain_batch_size,
        lambda_forecast=cfg.pretrain_lambda_forecast,
        k=cfg.pretrain_k_forecast,
        device=cfg.device,
    )

    features = list(cfg.features) if cfg.features else []
    if features and "log_return" in features:
        X, Y = pretrainer.make_windows(data_is, features, L)
        if len(X) > 0:
            pretrainer.train(
                X=X,
                Y=Y,
                n_epochs=cfg.pretrain_n_epochs,
                val_ratio=0.1,
                early_stopping_patience=5,
            )

    # Guardar y cargar vía artefacto para construir el agente correctamente
    with tempfile.TemporaryDirectory() as tmp_dir:
        enc_path = Path(tmp_dir) / "arm_c_encoder.pt"
        save_encoder_artifact(encoder, norm_stats={}, provenance={}, path=enc_path)

        agent = _make_latent_agent(env, cfg, freeze_encoder=True)
        loaded_enc, _, _ = load_encoder_artifact(enc_path)
        agent.encoder = loaded_enc.to(agent.device)
        agent.encoder.freeze()

    return _train_agent(env, agent, n_episodes, cfg, validation_env, seed)


def _train_arm_d(
    env: FinancialEnv,
    n_episodes: int,
    cfg: ExperimentConfig,
    validation_env: Optional[FinancialEnv] = None,
    seed: Optional[int] = None,
) -> Optional[LatentDQNAgent]:
    """Brazo D: encoder heavy offline, frozen. Omite si no hay ruta configurada."""
    if not cfg.heavy_encoder_path:
        print("  Brazo D omitido: heavy_encoder_path no configurado.")
        return None

    agent = _make_latent_agent(env, cfg, freeze_encoder=True)
    try:
        agent.load_artifact_encoder(cfg.heavy_encoder_path, cfg=cfg)
    except Exception as exc:
        print(f"  Brazo D omitido: error cargando artefacto ({exc})")
        return None

    return _train_agent(env, agent, n_episodes, cfg, validation_env, seed)


# ---------------------------------------------------------------------------
# Ejecución por semilla
# ---------------------------------------------------------------------------

def run_single_seed(
    seed: int,
    data_is: pd.DataFrame,
    data_oos: pd.DataFrame,
    cfg: ExperimentConfig,
    frozen_ctx: Optional[FrozenEncoderContext] = None,
    data_train: Optional[pd.DataFrame] = None,
    data_val: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Ejecuta el experimento completo para una semilla específica.

    Construye los brazos definidos en ``cfg.run_arms``.  Los resultados se
    indexan por "A", "B", "C", "D" (plus "RandomAgent", "BuyAndHoldAgent").

    Args:
        seed: Semilla aleatoria (afecta solo a los agentes, no a los datos).
        data_is: DataFrame IS con columnas OHLCV + features en minúsculas.
        data_oos: DataFrame OOS con columnas OHLCV + features en minúsculas.
        cfg: Configuración del experimento.
        frozen_ctx: Latentes precomputados para brazos C y D. Si None, los
            brazos C y D usan el path original (entrena encoder + recalcula
            en cada paso de RL).

    Returns:
        ``results[arm_name]["is" | "oos"]`` con el dict de métricas.
    """
    print(f"\n--- Ejecutando semilla {seed} ---")
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_kw = _build_env_kwargs(cfg, data_is)
    env_is  = FinancialEnv(data=data_is,  **env_kw)
    env_oos = FinancialEnv(data=data_oos, **env_kw)
    train_data = data_train if data_train is not None else data_is
    env_train = FinancialEnv(data=train_data, **env_kw)
    env_val = FinancialEnv(data=data_val, **env_kw) if data_val is not None else None
    for env in (env_is, env_oos, env_train, env_val):
        if env is not None:
            env.action_space.seed(seed)

    n_train = cfg.n_training_episodes
    n_eval  = cfg.n_eval_episodes
    results: Dict[str, Dict[str, Any]] = {}

    # Agentes de referencia (siempre)
    random_agent = RandomAgent(action_space=env_is.action_space, seed=seed)
    results["RandomAgent"] = {
        "is":  evaluate_agent(random_agent, env_is,  "RandomAgent", n_eval),
        "oos": evaluate_agent(random_agent, env_oos, "RandomAgent", n_eval),
    }
    bah_agent = BuyAndHoldAgent(action_space=env_is.action_space)
    results["BuyAndHoldAgent"] = {
        "is":  evaluate_agent(bah_agent, env_is,  "BuyAndHoldAgent", n_eval),
        "oos": evaluate_agent(bah_agent, env_oos, "BuyAndHoldAgent", n_eval),
    }

    arms = cfg.run_arms

    def store_agent_results(name: str, agent, eval_is, eval_oos) -> None:
        results[name] = {
            "is": evaluate_agent(agent, eval_is, name, n_eval),
            "oos": evaluate_agent(agent, eval_oos, name, n_eval),
        }
        if hasattr(agent, "validation_metrics"):
            results[name]["validation"] = dict(agent.validation_metrics)

    if "A" in arms:
        agent_a = _train_arm_a(env_train, n_train, cfg, env_val, seed)
        store_agent_results("A", agent_a, env_is, env_oos)

    if "B" in arms:
        agent_b = _train_arm_b(env_train, n_train, cfg, env_val, seed)
        store_agent_results("B", agent_b, env_is, env_oos)

    if "C" in arms:
        if frozen_ctx is not None and frozen_ctx.c_latents_is is not None:
            # Ruta optimizada: latentes precomputados, sin correr el encoder en RL
            w_c_train = LatentObservationWrapper(
                FinancialEnv(data=train_data, **env_kw), frozen_ctx.c_latents_train
            )
            w_c_val = (
                LatentObservationWrapper(
                    FinancialEnv(data=data_val, **env_kw), frozen_ctx.c_latents_val
                )
                if data_val is not None and frozen_ctx.c_latents_val is not None
                else None
            )
            w_c_is = LatentObservationWrapper(
                FinancialEnv(data=data_is, **env_kw), frozen_ctx.c_latents_is
            )
            w_c_oos = LatentObservationWrapper(
                FinancialEnv(data=data_oos, **env_kw), frozen_ctx.c_latents_oos
            )
            for wrapped_env in (w_c_train, w_c_val, w_c_is, w_c_oos):
                if wrapped_env is not None:
                    wrapped_env.action_space.seed(seed)
            agent_c = _make_precomputed_latent_agent(w_c_train, cfg)
            _train_agent(w_c_train, agent_c, n_train, cfg, w_c_val, seed)
            store_agent_results("C", agent_c, w_c_is, w_c_oos)
            w_c_train.close()
            if w_c_val is not None:
                w_c_val.close()
            w_c_is.close()
            w_c_oos.close()
        else:
            agent_c = _train_arm_c(
                env_train, n_train, cfg, train_data, env_val, seed
            )
            store_agent_results("C", agent_c, env_is, env_oos)

    if "D" in arms:
        if frozen_ctx is not None and frozen_ctx.d_latents_is is not None:
            # Ruta optimizada: latentes precomputados del encoder gordo offline
            w_d_train = LatentObservationWrapper(
                FinancialEnv(data=train_data, **env_kw), frozen_ctx.d_latents_train
            )
            w_d_val = (
                LatentObservationWrapper(
                    FinancialEnv(data=data_val, **env_kw), frozen_ctx.d_latents_val
                )
                if data_val is not None and frozen_ctx.d_latents_val is not None
                else None
            )
            w_d_is = LatentObservationWrapper(
                FinancialEnv(data=data_is, **env_kw), frozen_ctx.d_latents_is
            )
            w_d_oos = LatentObservationWrapper(
                FinancialEnv(data=data_oos, **env_kw), frozen_ctx.d_latents_oos
            )
            for wrapped_env in (w_d_train, w_d_val, w_d_is, w_d_oos):
                if wrapped_env is not None:
                    wrapped_env.action_space.seed(seed)
            agent_d = _make_precomputed_latent_agent(w_d_train, cfg)
            _train_agent(w_d_train, agent_d, n_train, cfg, w_d_val, seed)
            store_agent_results("D", agent_d, w_d_is, w_d_oos)
            w_d_train.close()
            if w_d_val is not None:
                w_d_val.close()
            w_d_is.close()
            w_d_oos.close()
        else:
            agent_d = _train_arm_d(env_train, n_train, cfg, env_val, seed)
            if agent_d is not None:
                store_agent_results("D", agent_d, env_is, env_oos)

    env_is.close()
    env_oos.close()
    env_train.close()
    if env_val is not None:
        env_val.close()

    return results


# ---------------------------------------------------------------------------
# IVL
# ---------------------------------------------------------------------------

def compute_ivl(
    aggregated: Dict[str, Dict[str, float]],
    cfg: ExperimentConfig,
    ticker: str = "",
    window: int = -1,
) -> List[Dict[str, Any]]:
    """
    Recoge los registros IVL crudos (sin normalización) para un ticker.

    La normalización por componente se aplica a nivel de experimento en
    ``_finalize_ivl_records``, una vez disponibles todos los pares
    (ticker, ventana). Las métricas usan Sharpe OOS y MDD OOS (no IS).

    Args:
        aggregated: Resultado de aggregate_results().
        cfg: Configuración del experimento.
        ticker: Nombre del ticker (incluido en cada registro para trazabilidad).
        window: Índice de ventana WF (-1 cuando no se usa WF).

    Returns:
        Lista de dicts con deltas crudos (sin IVL ni normalización finales).
    """
    if cfg.direct_agent not in aggregated:
        print(f"  Advertencia: agente directo '{cfg.direct_agent}' no encontrado, omitiendo IVL.")
        return []

    def _metrics(name: str) -> Dict[str, float]:
        row = aggregated[name]
        return {
            "sharpe_oos":          row["mean_sharpe_oos"],
            "mdd_oos":             row["mean_mdd_oos"],
            "seed_std_sharpe_oos": row["seed_std_sharpe_oos"],
            "sharpe_is":           row["mean_sharpe_is"],
        }

    direct_m = _metrics(cfg.direct_agent)
    direct_row = aggregated[cfg.direct_agent]
    records = []
    ivl_calc = LatentAdvantageIndex(weights=cfg.ivl_weights)

    for latent_name in cfg.latent_agents:
        if latent_name not in aggregated:
            print(f"  Advertencia: '{latent_name}' no encontrado en resultados, omitiendo.")
            continue

        latent_m = _metrics(latent_name)
        # Calcula deltas crudos (sin scales -> sin normalización)
        result = ivl_calc.compute(direct_m, latent_m, scales=None)
        latent_row = aggregated[latent_name]

        records.append({
            "ticker":             ticker,
            "window":             window,
            "direct_agent":       cfg.direct_agent,
            "latent_agent":       latent_name,
            # Deltas crudos (antes de normalización)
            "delta_sharpe":       result["delta_sharpe"],
            "delta_mdd":          result["delta_mdd"],
            "delta_seed_std":     result["delta_seed_std"],
            "delta_is_oos_gap":   result["delta_is_oos_gap"],
            # Placeholders — rellenados por _finalize_ivl_records
            "delta_sharpe_norm":     float("nan"),
            "delta_mdd_norm":        float("nan"),
            "delta_seed_std_norm":   float("nan"),
            "delta_is_oos_gap_norm": float("nan"),
            "ivl":                   float("nan"),
            "interpretation":        "pending",
            # Contexto OOS para el dashboard
            "direct_sharpe_oos":  direct_row["mean_sharpe_oos"],
            "direct_sharpe_is":   direct_row["mean_sharpe_is"],
            "direct_return_oos":  direct_row["mean_return_oos"],
            "latent_sharpe_oos":  latent_row["mean_sharpe_oos"],
            "latent_sharpe_is":   latent_row["mean_sharpe_is"],
            "latent_return_oos":  latent_row["mean_return_oos"],
        })

    return records


def _finalize_ivl_records(
    all_raw: List[Dict[str, Any]],
    ivl_calc: LatentAdvantageIndex,
) -> List[Dict[str, Any]]:
    """
    Normaliza por componente y calcula el IVL final para todos los registros.

    La escala de cada componente es la std agrupada sobre todos los pares
    (ticker, ventana) disponibles. Si la std < 1e-8 (p. ej. comparación única
    o componente constante), esa componente se fija a 0 en el IVL normalizado.

    Args:
        all_raw: Lista de registros crudos devueltos por compute_ivl.
        ivl_calc: Instancia de LatentAdvantageIndex (con los pesos del experimento).

    Returns:
        Lista de registros finalizados con deltas crudos, normalizados e IVL.
    """
    if not all_raw:
        return []

    component_keys = ["delta_sharpe", "delta_mdd", "delta_seed_std", "delta_is_oos_gap"]

    # Escala = std de cada componente sobre todos los registros
    scales: Dict[str, float] = {}
    for k in component_keys:
        vals = [r[k] for r in all_raw if not (r[k] != r[k])]  # excluir NaN
        scales[k] = float(np.std(vals)) if len(vals) >= 2 else 0.0

    finalized = []
    for raw in all_raw:
        def _norm(k: str) -> float:
            s = scales[k]
            return raw[k] / s if s > 1e-8 else 0.0

        n_sharpe   = _norm("delta_sharpe")
        n_mdd      = _norm("delta_mdd")
        n_seed_std = _norm("delta_seed_std")
        n_gap      = _norm("delta_is_oos_gap")

        ivl = (
            ivl_calc.weights["sharpe"]       * n_sharpe
            - ivl_calc.weights["mdd"]        * n_mdd
            - ivl_calc.weights["seed_std"]   * n_seed_std
            - ivl_calc.weights["is_oos_gap"] * n_gap
        )

        rec = {
            **raw,
            "delta_sharpe_norm":     n_sharpe,
            "delta_mdd_norm":        n_mdd,
            "delta_seed_std_norm":   n_seed_std,
            "delta_is_oos_gap_norm": n_gap,
            "ivl":                   ivl,
            "interpretation":        ivl_calc.interpret(ivl),
        }
        finalized.append(rec)

    return finalized


def export_ivl_results(
    records: List[Dict[str, Any]],
    results_dir: str | Path,
) -> None:
    """Guarda los resultados del IVL en results_dir/ivl_results.csv."""
    if not records:
        print("No se calculo ningun IVL.")
        return
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(results_path / "ivl_results.csv", index=False)
    print(f"\nResultados IVL guardados en: {results_path / 'ivl_results.csv'}")


def _print_ivl_summary(ticker: str, records: List[Dict[str, Any]]) -> None:
    """Imprime el resumen IVL normalizado para un ticker."""
    for rec in records:
        d, l = rec["direct_agent"], rec["latent_agent"]
        print(f"\nIVL: {d} vs {l}  [{ticker}]")
        print(f"  IVL (norm)   = {rec['ivl']:+.4f}")
        print(f"  d Sharpe     = {rec['delta_sharpe']:.4f}  (norm: {rec['delta_sharpe_norm']:+.4f})")
        print(f"  d MDD        = {rec['delta_mdd']:.4f}  (norm: {rec['delta_mdd_norm']:+.4f})")
        print(f"  d Seed Std   = {rec['delta_seed_std']:.4f}  (norm: {rec['delta_seed_std_norm']:+.4f})")
        print(f"  d IS/OOS Gap = {rec['delta_is_oos_gap']:.4f}  (norm: {rec['delta_is_oos_gap_norm']:+.4f})")
        print(f"  Interpretacion: {rec['interpretation']}")


# ---------------------------------------------------------------------------
# Walk-Forward helpers
# ---------------------------------------------------------------------------

def _aggregate_wf_windows(
    wf_window_aggregated: List[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Promedia las métricas de N ventanas WF para obtener un resumen global.

    Cada ventana ya tiene sus semillas agregadas. Este paso promedia entre
    ventanas. Solo se promedian las claves numéricas presentes en todas las
    ventanas; ``"name"`` se conserva de la primera ventana.
    """
    if not wf_window_aggregated:
        return {}

    agent_names = list(wf_window_aggregated[0].keys())
    result: Dict[str, Dict[str, float]] = {}

    for agent in agent_names:
        all_windows = [w[agent] for w in wf_window_aggregated if agent in w]
        if not all_windows:
            continue
        numeric_keys = [k for k, v in all_windows[0].items()
                        if isinstance(v, (int, float)) and k != "name"]
        row: Dict[str, float] = {"name": all_windows[0].get("name", agent)}
        for k in numeric_keys:
            vals = [w[k] for w in all_windows]
            row[k] = float(np.mean(vals))
        result[agent] = row

    return result


def _export_wf_window_metrics(
    wf_window_aggregated: List[Dict[str, Dict[str, float]]],
    results_dir: "str | Path",
    cfg: ExperimentConfig,
    ivl_records_per_window: Optional[List[List[Dict[str, Any]]]] = None,
) -> None:
    """Exporta ``wf_window_metrics.csv`` con una fila por (ventana, agente).

    Cada fila incluye las métricas IS/OOS agregadas entre semillas para esa
    ventana, más el IVL de esa ventana si está disponible.
    """
    rows = []
    for wf_idx, window_agg in enumerate(wf_window_aggregated):
        # IVL por agente latente para esta ventana
        ivl_lookup: Dict[str, float] = {}
        if ivl_records_per_window and wf_idx < len(ivl_records_per_window):
            for rec in ivl_records_per_window[wf_idx]:
                ivl_lookup[rec["latent_agent"]] = rec.get("ivl", float("nan"))

        for agent_name, metrics in window_agg.items():
            row = {
                "window":      wf_idx,
                "agent":       agent_name,
            }
            for k, v in metrics.items():
                if k != "name":
                    row[k] = v
            # Añadir IVL de ventana si disponible
            if agent_name in ivl_lookup:
                row["ivl"] = ivl_lookup[agent_name]
            rows.append(row)

    if not rows:
        return

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    out_path = results_path / "wf_window_metrics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  wf_window_metrics.csv: {len(rows)} filas -> {out_path}")


# ---------------------------------------------------------------------------
# Pipeline para un único ticker
# ---------------------------------------------------------------------------

def _prepare_train_val_eval_splits(
    data_is: pd.DataFrame,
    data_oos: pd.DataFrame,
    cfg: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Prepara IS de evaluacion, OOS, IS_train e IS_val sin leakage."""
    if cfg.use_internal_validation:
        data_train, data_val = split_internal_validation(
            data_is, cfg.internal_val_ratio, cfg.lookback_window
        )
        if cfg.normalize_features and cfg.features:
            data_train, data_val, data_oos, normalizer = normalize_train_val_oos(
                data_train, data_val, data_oos
            )
            data_is = normalizer.transform(data_is)
        return data_is, data_oos, data_train, data_val

    if cfg.normalize_features and cfg.features:
        data_is, data_oos, _ = normalize_is_oos(data_is, data_oos)
    return data_is, data_oos, data_is, None


def _export_validation_metrics(
    all_results: List[Dict[str, Dict[str, Any]]],
    seeds: List[int],
    ticker: str,
    hidden_dim: int,
    results_dir: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    for seed, seed_results in zip(seeds, all_results):
        for agent_name, splits in seed_results.items():
            metrics = splits.get("validation")
            if metrics is None:
                continue
            rows.append({
                "ticker": ticker,
                "hidden_dim": hidden_dim,
                "agent_name": agent_name,
                "seed": seed,
                **metrics,
            })
    if rows:
        results_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(results_dir / "validation_metrics.csv", index=False)

def _run_single_ticker(
    ticker: str,
    cfg: ExperimentConfig,
    d_encoder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Ejecuta el pipeline completo para un ticker."""
    ticker_results_dir = Path(cfg.results_dir) / ticker

    print(f"\n{'='*60}")
    print(f"TICKER: {ticker}")
    print(f"{'='*60}")

    params = cfg.resolve_ticker_params(ticker)
    print(
        f"  Periodo: {params['start']} -> {params['end']}  "
        f"interval={params['interval']}  "
        f"features={cfg.features if cfg.features else 'OHLCV'}"
    )
    data = load_ticker_with_config(ticker, cfg)
    n_features = data.shape[1]

    if cfg.wf_enabled:
        print(
            f"  Walk-Forward ({cfg.wf_mode}): {cfg.wf_n_windows} ventanas, "
            f"is_ratio={cfg.wf_is_ratio} | semillas={cfg.seeds} | "
            f"normalizacion={'si' if cfg.normalize_features and cfg.features else 'no'}"
        )
        splits_wf = walk_forward_splits(
            data, cfg.wf_n_windows, cfg.wf_is_ratio,
            lookback=cfg.lookback_window,
        )

        # Resultados estructurados por ventana
        wf_window_aggregated: List[Dict] = []
        wf_raw_ivl_per_window: List[List[Dict]] = []
        all_seed_results_flat: List[Dict] = []  # para export_dashboard_results

        for wf_idx, (wf_is, wf_oos) in enumerate(splits_wf):
            print(
                f"\n  WF ventana {wf_idx + 1}/{cfg.wf_n_windows}: "
                f"IS={len(wf_is)} filas  OOS={len(wf_oos)} filas  n_features={n_features}"
            )
            wf_is, wf_oos, wf_train, wf_val = _prepare_train_val_eval_splits(
                wf_is, wf_oos, cfg
            )

            # Encoder de D no se reentrena; el encoder de C se entrena una vez
            # por ventana (IS crece, stats cambian). Anti-fuga de D: el encoder
            # gordo fue preentrenado con un universo disjunto por activo, así que
            # no ha visto este activo en ninguna ventana.
            frozen_ctx = _build_frozen_ctx(
                wf_is, wf_oos, cfg, d_encoder, wf_train, wf_val
            )

            # Todas las semillas dentro de esta ventana
            window_seed_results: List[Dict] = []
            for seed in cfg.seeds:
                sr = run_single_seed(
                    seed=seed, data_is=wf_is, data_oos=wf_oos,
                    cfg=cfg, frozen_ctx=frozen_ctx,
                    data_train=wf_train, data_val=wf_val,
                )
                window_seed_results.append(sr)
                all_seed_results_flat.append(sr)

            # Agregar entre semillas dentro de la ventana
            window_agg = aggregate_results(window_seed_results)
            wf_window_aggregated.append(window_agg)

            print(f"  Resumen ventana {wf_idx + 1} (media semillas):")
            for arm_name, metrics in window_agg.items():
                print(
                    f"    {arm_name}: "
                    f"IS={metrics['mean_return_is']:.4f} (Sharpe={metrics['mean_sharpe_is']:.3f}) | "
                    f"OOS={metrics['mean_return_oos']:.4f} (Sharpe={metrics['mean_sharpe_oos']:.3f})"
                )

            # IVL crudo de esta ventana
            window_ivl = compute_ivl(window_agg, cfg, ticker=ticker, window=wf_idx)
            wf_raw_ivl_per_window.append(window_ivl)

        # Agregado global (promedio entre ventanas) para los CSV del dashboard
        aggregated = _aggregate_wf_windows(wf_window_aggregated)
        print_summary_table(aggregated)
        print_ranking(aggregated)
        export_dashboard_results(
            all_seed_results_flat,
            aggregated,
            results_dir=ticker_results_dir,
            seeds=cfg.seeds * cfg.wf_n_windows,
        )

        # Exportar métricas por ventana (incluyendo IVL provisional)
        _export_wf_window_metrics(
            wf_window_aggregated, ticker_results_dir, cfg,
            ivl_records_per_window=wf_raw_ivl_per_window,
        )

        # Aplanar registros IVL de todas las ventanas para normalización cross-ticker
        raw_ivl_records = [rec for win in wf_raw_ivl_per_window for rec in win]

        return {
            "ticker":                ticker,
            "all_results":           all_seed_results_flat,
            "aggregated":            aggregated,
            "raw_ivl_records":       raw_ivl_records,
            "wf_window_aggregated":  wf_window_aggregated,
        }
    else:
        data_is, data_oos = split_data(data, train_ratio=cfg.train_ratio)
        print(f"  IS: {len(data_is)} filas  |  OOS: {len(data_oos)} filas  |  n_features: {n_features}")

        data_is, data_oos, data_train, data_val = _prepare_train_val_eval_splits(
            data_is, data_oos, cfg
        )
        if cfg.use_internal_validation:
            print(
                f"  Validacion interna: train={len(data_train)} filas | "
                f"val={len(data_val)} filas."
            )
        if cfg.normalize_features and cfg.features:
            fit_split = "IS_train" if cfg.use_internal_validation else "IS"
            print(
                f"  Normalizacion z-score aplicada (ajuste en {fit_split}, "
                "sin leakage a validacion/OOS)."
            )

        # Precomputa latentes para C y D una sola vez (antes del bucle de semillas)
        frozen_ctx = _build_frozen_ctx(
            data_is, data_oos, cfg, d_encoder, data_train, data_val
        )

        all_results = []
        for seed in cfg.seeds:
            seed_results = run_single_seed(
                seed=seed, data_is=data_is, data_oos=data_oos, cfg=cfg,
                frozen_ctx=frozen_ctx,
                data_train=data_train, data_val=data_val,
            )
            all_results.append(seed_results)

            print(f"\nResumen semilla {seed}:")
            for arm_name, splits in seed_results.items():
                m_is  = splits["is"]
                m_oos = splits["oos"]
                print(
                    f"  {arm_name}: "
                    f"IS={m_is['total_return']:.4f} (Sharpe={m_is['sharpe']:.3f}) | "
                    f"OOS={m_oos['total_return']:.4f} (Sharpe={m_oos['sharpe']:.3f})"
                )

    aggregated = aggregate_results(all_results)
    print_summary_table(aggregated)
    print_ranking(aggregated)
    export_dashboard_results(
        all_results, aggregated, results_dir=ticker_results_dir, seeds=cfg.seeds
    )
    if cfg.use_internal_validation:
        _export_validation_metrics(
            all_results, cfg.seeds, ticker, cfg.dqn_hidden_dim, ticker_results_dir
        )

    # IVL: solo recoge componentes crudas. La normalización y el export
    # se realizan en run_experiment, tras tener todos los tickers.
    raw_ivl_records = compute_ivl(aggregated, cfg, ticker=ticker)

    return {
        "ticker":          ticker,
        "all_results":     all_results,
        "aggregated":      aggregated,
        "raw_ivl_records": raw_ivl_records,
    }


# ---------------------------------------------------------------------------
# Comparación cross-ticker
# ---------------------------------------------------------------------------

def aggregate_ticker_results(
    results_per_ticker: Dict[str, Dict[str, Any]],
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """Genera un resumen cross-ticker del IVL por par (directo, latente).

    Espera registros ya finalizados (con campo ``ticker`` y ``ivl`` numérico).
    """
    rows = []
    for ticker_result in results_per_ticker.values():
        for record in ticker_result.get("ivl_records", []):
            # Los registros finalizados ya contienen el campo "ticker"
            rows.append(record)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary_rows = []
    for latent_agent in df["latent_agent"].unique():
        sub = df[df["latent_agent"] == latent_agent]
        summary_rows.append({
            "ticker":          "MEAN",
            "latent_agent":    latent_agent,
            "ivl":             sub["ivl"].mean(),
            "delta_sharpe":    sub["delta_sharpe"].mean(),
            "delta_mdd":       sub["delta_mdd"].mean(),
            "delta_seed_std":  sub["delta_seed_std"].mean(),
            "delta_is_oos_gap":sub["delta_is_oos_gap"].mean(),
            "interpretation":  "mean_across_tickers",
        })
        summary_rows.append({
            "ticker":          "STD",
            "latent_agent":    latent_agent,
            "ivl":             sub["ivl"].std(),
            "delta_sharpe":    sub["delta_sharpe"].std(),
            "delta_mdd":       sub["delta_mdd"].std(),
            "delta_seed_std":  sub["delta_seed_std"].std(),
            "delta_is_oos_gap":sub["delta_is_oos_gap"].std(),
            "interpretation":  "std_across_tickers",
        })

    return pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)


def export_cross_ticker_results(
    cross_df: pd.DataFrame,
    cfg: ExperimentConfig,
) -> None:
    """Guarda el resumen cross-ticker en {results_dir}/ticker_comparison.csv."""
    if cross_df.empty:
        return
    results_path = Path(cfg.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    cross_df.to_csv(results_path / "ticker_comparison.csv", index=False)
    print(f"\nComparacion cross-ticker guardada en: {results_path / 'ticker_comparison.csv'}")


def _print_cross_ticker_summary(cross_df: pd.DataFrame) -> None:
    if cross_df.empty:
        return
    detail = cross_df[~cross_df["ticker"].isin(["MEAN", "STD"])]
    print("\n" + "=" * 80)
    print("RESUMEN CROSS-TICKER (IVL por ticker y agente latente)")
    print("=" * 80)
    print(f"{'Ticker':<12} {'Agente latente':<38} {'IVL':>8}  {'Interpretacion'}")
    print("-" * 80)
    for _, row in detail.iterrows():
        print(
            f"{row['ticker']:<12} {row['latent_agent']:<38} "
            f"{row['ivl']:>+8.4f}  {row['interpretation']}"
        )
    mean_rows = cross_df[cross_df["ticker"] == "MEAN"]
    if not mean_rows.empty:
        print("\nMedia sobre tickers:")
        for _, row in mean_rows.iterrows():
            print(f"  {row['latent_agent']}: IVL={row['ivl']:+.4f}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Pipeline completo (entry point público)
# ---------------------------------------------------------------------------

def run_experiment(cfg: Optional[ExperimentConfig] = None) -> Dict[str, Any]:
    """Ejecuta el pipeline experimental completo para todos los tickers."""
    if cfg is None:
        cfg = ExperimentConfig()

    print("=" * 120)
    print("EXPERIMENTO DE COMPARACION MULTI-SEMILLA DE AGENTES (IS/OOS)")
    print("=" * 120)
    print(f"Tickers: {cfg.tickers}")
    print(f"Periodo: {cfg.start_date} -> {cfg.end_date}  |  N_OBS: {'todos' if cfg.n_obs is None else cfg.n_obs}")
    print(f"Split IS/OOS: {int(cfg.train_ratio*100)}% / {int((1-cfg.train_ratio)*100)}%")
    print(f"Semillas: {cfg.seeds}  |  Episodios: train={cfg.n_training_episodes}, eval={cfg.n_eval_episodes}")
    print(f"Brazos: {cfg.run_arms}  |  encoder_type: {cfg.encoder_type}  |  latent_dim: {cfg.latent_dim}")
    print(f"IVL pesos: {cfg.ivl_weights}  |  Resultados en: {cfg.results_dir}/")

    # Carga el encoder gordo (brazo D) una sola vez para todos los tickers
    d_encoder: Optional[Any] = None
    if "D" in cfg.run_arms and cfg.heavy_encoder_path:
        try:
            from latent_rl.representations.artifact import load_encoder_artifact
            d_encoder, _, _ = load_encoder_artifact(cfg.heavy_encoder_path, cfg=cfg)
            d_encoder.eval()
            for p in d_encoder.parameters():
                p.requires_grad_(False)
            print(f"  Encoder gordo (brazo D) cargado desde: {cfg.heavy_encoder_path}")
        except Exception as exc:
            print(f"  Brazo D: error cargando artefacto ({exc}). Brazo D sera omitido.")

    results_per_ticker: Dict[str, Any] = {}
    for ticker in cfg.tickers:
        results_per_ticker[ticker] = _run_single_ticker(ticker, cfg, d_encoder=d_encoder)

    # ---------------------------------------------------------------------------
    # IVL: normalización cross-ticker y exportación
    # ---------------------------------------------------------------------------
    ivl_calc = LatentAdvantageIndex(weights=cfg.ivl_weights)

    # Recolectar todos los registros crudos de todos los tickers
    all_raw_ivl = []
    for tr in results_per_ticker.values():
        all_raw_ivl.extend(tr.get("raw_ivl_records", []))

    # Normalizar y calcular IVL (escala = std agrupado cross-ticker)
    finalized_ivl = _finalize_ivl_records(all_raw_ivl, ivl_calc)

    # Distribuir registros finalizados de vuelta a cada ticker y exportar
    for tr in results_per_ticker.values():
        t = tr["ticker"]
        ticker_records = [r for r in finalized_ivl if r.get("ticker") == t]
        tr["ivl_records"] = ticker_records
        if ticker_records:
            ticker_results_dir = Path(cfg.results_dir) / t
            export_ivl_results(ticker_records, results_dir=ticker_results_dir)
            _print_ivl_summary(t, ticker_records)

    cross_df = pd.DataFrame()
    if len(cfg.tickers) > 1:
        print("\n" + "=" * 120)
        print("COMPARACION CROSS-TICKER")
        print("=" * 120)
        cross_df = aggregate_ticker_results(results_per_ticker, cfg)
        _print_cross_ticker_summary(cross_df)
        export_cross_ticker_results(cross_df, cfg)

    print("\n" + "=" * 120)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 120)
    for ticker in cfg.tickers:
        print(f"  {ticker}: resultados en {cfg.results_dir}/{ticker}/")
    if len(cfg.tickers) > 1:
        print(f"  Cross-ticker: {cfg.results_dir}/ticker_comparison.csv")

    return {**results_per_ticker, "cross_ticker": cross_df}
