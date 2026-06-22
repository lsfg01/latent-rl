"""Cobertura del protocolo DQN anti-overfitting."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch.nn as nn

from latent_rl.agents import DQNAgent, LatentDQNAgent
from latent_rl.envs import FinancialEnv
from latent_rl.experiments import ExperimentConfig
from latent_rl.experiments.runner import _train_agent
from latent_rl.experiments.utils import (
    export_dashboard_results,
    normalize_train_val_oos,
)
from scripts.experiments.a_hidden_sweep import build_config


def _data(n: int = 100, price_step: float = 1.0) -> pd.DataFrame:
    close = 100.0 + np.arange(n) * price_step
    return pd.DataFrame({
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(n, 1_000.0),
        "feature": np.arange(n, dtype=float),
    })


def test_reset_is_deterministic_by_default_and_random_start_is_seeded():
    env = FinancialEnv(_data(), lookback_window=5, feature_cols=["feature"])
    env.reset()
    assert env.current_step == 5

    env.reset(seed=7, options={"random_start": True, "max_steps": 20})
    first = env.current_step
    env.reset(seed=7, options={"random_start": True, "max_steps": 20})
    assert env.current_step == first
    assert 5 <= first <= len(env.data) - 2 - 20


def test_random_start_uses_available_range_when_requested_margin_does_not_fit():
    env = FinancialEnv(_data(10), lookback_window=5, feature_cols=["feature"])
    starts = {
        env.reset(seed=seed, options={"random_start": True, "max_steps": 100})[1][
            "current_step"
        ]
        for seed in range(20)
    }
    assert starts.issubset(set(range(5, 9)))
    assert len(starts) > 1


def test_log_return_reward_matches_equity_change():
    env = FinancialEnv(
        _data(8, price_step=10.0),
        lookback_window=1,
        transaction_cost=0.0,
        reward_mode="log_return",
        feature_cols=["feature"],
    )
    env.reset()
    env.step(1)
    _, reward, _, _, _ = env.step(0)
    assert math.isclose(reward, math.log(120.0 / 110.0), rel_tol=1e-9)


def test_trade_penalty_only_applies_to_executed_trades_and_clips_afterward():
    env = FinancialEnv(
        _data(8, price_step=0.0),
        lookback_window=1,
        transaction_cost=0.0,
        trade_penalty=0.5,
        reward_clip=0.1,
        feature_cols=["feature"],
    )
    env.reset()
    assert env.step(1)[1] == -0.1
    assert env.step(1)[1] == 0.0
    assert env.step(2)[1] == -0.1
    assert env.step(2)[1] == 0.0


def test_regularization_is_wired_to_both_agents():
    env = FinancialEnv(_data(20), lookback_window=2, feature_cols=["feature"])
    direct = DQNAgent(
        env.action_space,
        env.observation_space.shape,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        dropout=0.2,
    )
    latent = LatentDQNAgent(
        env.action_space,
        env.observation_space.shape,
        weight_decay=2e-4,
        grad_clip_norm=2.0,
        q_dropout=0.3,
    )
    assert direct.optimizer.param_groups[0]["weight_decay"] == 1e-4
    assert direct.grad_clip_norm == 1.0
    assert isinstance(direct.q_network.dropout, nn.Dropout)
    assert direct.q_network.dropout.p == 0.2
    assert latent.optimizer.param_groups[0]["weight_decay"] == 2e-4
    assert latent.grad_clip_norm == 2.0
    assert latent.q_network.dropout.p == 0.3


def test_align_latent_q_with_dqn_is_explicit_and_complete():
    cfg = ExperimentConfig(
        dqn_lr=2e-4,
        dqn_hidden_dim=64,
        dqn_batch_size=17,
        dqn_weight_decay=3e-4,
        dqn_grad_clip_norm=0.7,
        dqn_dropout=0.1,
        align_latent_q_with_dqn=True,
    )
    assert cfg.latent_lr == cfg.dqn_lr
    assert cfg.latent_q_hidden_dim == cfg.dqn_hidden_dim
    assert cfg.latent_batch_size == cfg.dqn_batch_size
    assert cfg.latent_weight_decay == cfg.dqn_weight_decay
    assert cfg.latent_grad_clip_norm == cfg.dqn_grad_clip_norm
    assert cfg.latent_q_dropout == cfg.dqn_dropout


def test_validation_training_restores_best_checkpoint_and_records_metrics():
    train_env = FinancialEnv(_data(35), lookback_window=2, feature_cols=["feature"])
    val_env = FinancialEnv(_data(20), lookback_window=2, feature_cols=["feature"])
    cfg = ExperimentConfig(
        run_arms=["A"],
        n_training_episodes=2,
        n_eval_episodes=1,
        max_steps_per_episode=5,
        use_internal_validation=True,
        validation_eval_freq=1,
        dqn_batch_size=2,
        dqn_hidden_dim=8,
    )
    agent = DQNAgent(
        train_env.action_space,
        train_env.observation_space.shape,
        batch_size=2,
        hidden_dim=8,
    )
    _train_agent(train_env, agent, 2, cfg, validation_env=val_env, seed=0)
    assert set(agent.validation_metrics) == {
        "best_val_episode", "best_val_score", "best_val_sharpe",
        "best_val_mdd", "best_val_return", "best_val_n_trades",
    }
    assert agent.validation_metrics["best_val_episode"] in {1, 2}


def test_normalization_fits_only_train():
    train = _data(10)
    val = _data(5)
    oos = _data(5)
    val["feature"] += 1_000
    train_n, val_n, _, normalizer = normalize_train_val_oos(train, val, oos)
    assert abs(train_n["feature"].mean()) < 1e-12
    assert val_n["feature"].mean() > 100
    assert normalizer.mean_["feature"] == train["feature"].mean()


def test_export_preserves_real_seed_values(tmp_path):
    metrics = {
        "name": "A", "total_reward": 0.0, "total_return": 0.0,
        "final_equity": 10_000.0, "realized_profit": 0.0,
        "n_trades": 0.0, "steps": 1.0, "sharpe": 0.0,
        "max_drawdown": 0.0,
    }
    all_results = [{"A": {"is": metrics, "oos": metrics}}]
    aggregated = {"A": {
        "name": "A", "mean_return_is": 0.0, "std_return_is": 0.0,
        "mean_sharpe_is": 0.0, "mean_mdd_is": 0.0,
        "mean_equity_is": 10_000.0, "mean_return_oos": 0.0,
        "std_return_oos": 0.0, "mean_sharpe_oos": 0.0,
        "mean_mdd_oos": 0.0, "mean_equity_oos": 10_000.0,
        "mean_n_trades": 0.0, "seed_std_return_is": 0.0,
        "seed_std_sharpe_oos": 0.0,
    }}
    export_dashboard_results(all_results, aggregated, tmp_path, seeds=[42])
    exported = pd.read_csv(tmp_path / "agent_seed_metrics.csv")
    assert set(exported["seed"]) == {42}


def test_hidden_sweep_smoke_config_matches_protocol(tmp_path):
    cfg = build_config(64, smoke=True, results_dir=str(tmp_path))
    assert cfg.run_arms == ["A"]
    assert cfg.seeds == [0]
    assert cfg.random_start_train is True
    assert cfg.reward_mode == "log_return"
    assert cfg.use_internal_validation is True
    assert cfg.dqn_weight_decay == 1e-4
    assert cfg.dqn_grad_clip_norm == 1.0
    assert cfg.results_dir.endswith("h64")
