"""
Script de verificación de las mejoras implementadas.
Usa datos sintéticos (sin internet) y un experimento corto para comprobar
que todos los componentes nuevos funcionan correctamente.
"""

import numpy as np
import pandas as pd
import pytest
import torch

pytestmark = pytest.mark.integration

# ── 1. Datos sintéticos ───────────────────────────────────────────────────────

def make_synthetic_data(n=300, seed=42):
    rng = np.random.default_rng(seed)
    price = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    noise = lambda s: rng.normal(0, s, n)
    return pd.DataFrame({
        "open":   price * (1 + noise(0.003)),
        "high":   price * (1 + np.abs(noise(0.007))),
        "low":    price * (1 - np.abs(noise(0.007))),
        "close":  price,
        "volume": rng.integers(5_000, 50_000, n).astype(float),
    })


# ── Helpers ───────────────────────────────────────────────────────────────────

def ok(_msg):
    """Conserva las anotaciones del antiguo verificador sin producir salida."""


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_qnetwork():
    from latent_rl.agents.dqn_agent import QNetwork
    net = QNetwork(input_dim=50, output_dim=3, hidden_dim=64)
    x = torch.randn(8, 50)
    q = net(x)
    assert q.shape == (8, 3), f"Shape inesperado: {q.shape}"
    ok(f"output shape {q.shape} correcto")

    # Verificar que la red produce valores no constantes.
    assert q.std().item() > 0, "Q-values son constantes, algo falla"
    ok("Q-values no constantes")


def test_latent_qnetwork():
    from latent_rl.agents.latent_dqn_agent import LatentQNetwork
    net = LatentQNetwork(latent_dim=32, output_dim=3, hidden_dim=64)
    z = torch.randn(8, 32)
    q = net(z)
    assert q.shape == (8, 3)
    ok(f"output shape {q.shape} correcto")


def test_mlp_encoder_gelu():
    from latent_rl.representations import MLPLatentEncoder
    enc = MLPLatentEncoder(input_dim=50, latent_dim=16, hidden_dims=[32, 16], activation="gelu")
    x = torch.randn(4, 50)
    z = enc(x)
    assert z.shape == (4, 16)
    ok(f"encode shape {z.shape} correcto")
    r = enc.decode(z)
    assert r.shape == (4, 50)
    ok(f"decode shape {r.shape} correcto")


def test_tcn_encoder():
    from latent_rl.representations import TCNLatentEncoder

    enc = TCNLatentEncoder(
        input_len=10,
        n_features=5,
        latent_dim=16,
        channels=16,
        dilations=[1, 2, 4],
        dropout=0.0,
    )
    x = torch.randn(8, 10, 5)
    z = enc(x)
    assert z.shape == (8, 16), f"Encode shape inesperado: {z.shape}"
    ok(f"encode shape {z.shape} correcto")

    r = enc.decode(z)
    assert r.shape == (8, 10, 5), f"Decode shape inesperado: {r.shape}"
    ok(f"decode shape {r.shape} correcto")

    # Pérdida de reconstrucción debe ser finita
    loss = enc.reconstruction_loss(x)
    assert torch.isfinite(loss), "Reconstruction loss no es finita"
    ok(f"reconstruction_loss = {loss.item():.4f}")


def test_dqn_agent_training():
    from gymnasium import spaces
    from latent_rl.agents import DQNAgent
    from latent_rl.envs import FinancialEnv

    data = make_synthetic_data(200)
    env = FinancialEnv(data, lookback_window=10)

    agent = DQNAgent(
        action_space=env.action_space,
        observation_shape=env.observation_space.shape,
        learning_rate=1e-3,
        hidden_dim=64,
        batch_size=16,
        buffer_capacity=500,
        target_update_freq=20,
        epsilon_decay=0.99,
    )

    obs, _ = env.reset()
    losses = []
    for step in range(100):
        action = agent.select_action(obs, training=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(obs, action, reward, next_obs, done)
        metrics = agent.update()
        if metrics["loss"] > 0:
            losses.append(metrics["loss"])
        obs = next_obs if not done else env.reset()[0]

    assert len(losses) > 0, "No se calculó ningún loss"
    ok(f"Entrenamiento OK — {len(losses)} updates, loss_final={losses[-1]:.6f}")


def test_latent_dqn_tcn_training():
    from latent_rl.agents import LatentDQNAgent
    from latent_rl.envs import FinancialEnv

    data = make_synthetic_data(200)
    env = FinancialEnv(data, lookback_window=10)

    agent = LatentDQNAgent(
        action_space=env.action_space,
        observation_shape=env.observation_space.shape,
        latent_dim=16,
        encoder_type="tcn",
        tcn_dilations=[1, 2, 4],
        tcn_channels=16,
        learning_rate=1e-3,
        q_hidden_dim=64,
        batch_size=16,
        buffer_capacity=500,
        target_update_freq=20,
        epsilon_decay=0.99,
    )

    obs, _ = env.reset()
    losses = []
    for step in range(100):
        action = agent.select_action(obs, training=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(obs, action, reward, next_obs, done)
        metrics = agent.update()
        if metrics["loss"] > 0:
            losses.append(metrics["loss"])
        obs = next_obs if not done else env.reset()[0]

    assert len(losses) > 0
    ok(f"Entrenamiento OK — {len(losses)} updates, loss_final={losses[-1]:.6f}")

    # También probar get_latent_representation
    latent = agent.get_latent_representation(obs)
    assert latent.shape == (16,)
    ok(f"get_latent_representation shape {latent.shape} correcto")


def test_latent_dqn_mlp_training():
    from latent_rl.agents import LatentDQNAgent
    from latent_rl.envs import FinancialEnv

    data = make_synthetic_data(200)
    env = FinancialEnv(data, lookback_window=10)

    agent = LatentDQNAgent(
        action_space=env.action_space,
        observation_shape=env.observation_space.shape,
        latent_dim=16,
        encoder_hidden_dims=[32, 16],
        encoder_type="mlp",
        encoder_activation="gelu",
        learning_rate=1e-3,
        q_hidden_dim=64,
        batch_size=16,
        buffer_capacity=500,
        target_update_freq=20,
        epsilon_decay=0.99,
    )

    obs, _ = env.reset()
    losses = []
    for step in range(100):
        action = agent.select_action(obs, training=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(obs, action, reward, next_obs, done)
        metrics = agent.update()
        if metrics["loss"] > 0:
            losses.append(metrics["loss"])
        obs = next_obs if not done else env.reset()[0]

    assert len(losses) > 0
    ok(f"Entrenamiento OK — {len(losses)} updates, loss_final={losses[-1]:.6f}")


def test_full_pipeline_synthetic():
    from latent_rl.experiments.config import ExperimentConfig
    from latent_rl.experiments.runner import run_single_seed

    data = make_synthetic_data(300)
    split = int(0.7 * len(data))
    data_is  = data.iloc[:split].reset_index(drop=True)
    data_oos = data.iloc[split:].reset_index(drop=True)

    cfg = ExperimentConfig(
        tickers=["SYNTHETIC"],
        seeds=[0],
        n_training_episodes=3,
        n_eval_episodes=1,
        max_steps_per_episode=50,
        lookback_window=10,
        dqn_hidden_dim=64,
        dqn_buffer_capacity=500,
        dqn_batch_size=16,
        latent_dim=16,
        encoder_hidden_dims=[16, 32],
        encoder_type="tcn",
        tcn_dilations=[1, 2, 4],
        tcn_channels=16,
        latent_q_hidden_dim=64,
        latent_buffer_capacity=500,
        latent_batch_size=16,
        pretrain_n_samples=50,
        pretrain_n_epochs=3,
        run_arms=["A", "B", "C"],
    )

    results = run_single_seed(seed=0, data_is=data_is, data_oos=data_oos, cfg=cfg)

    expected_agents = [
        "RandomAgent", "BuyAndHoldAgent", "A", "B", "C",
    ]
    for name in expected_agents:
        assert name in results, f"Agente '{name}' no en resultados"
        assert "is"  in results[name], f"Falta split IS para {name}"
        assert "oos" in results[name], f"Falta split OOS para {name}"
        m = results[name]["is"]
        assert "total_return" in m and "sharpe" in m
        ok(f"{name}: IS return={m['total_return']:.4f}, Sharpe={m['sharpe']:.3f}")


def test_config_new_fields():
    from latent_rl.experiments.config import ExperimentConfig

    cfg = ExperimentConfig()
    assert cfg.encoder_type == "tcn"
    assert cfg.dqn_hidden_dim == 128
    assert cfg.latent_dim == 16
    assert cfg.n_training_episodes == 10
    assert cfg.pretrain_n_samples == 100
    ok("Valores por defecto correctos")

    # Verificar que mlp también funciona
    cfg2 = ExperimentConfig(encoder_type="mlp", encoder_activation="gelu")
    assert cfg2.encoder_type == "mlp"
    ok("ExperimentConfig acepta encoder_type='mlp'")


# ── Runner ────────────────────────────────────────────────────────────────────

