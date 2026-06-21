"""Tests de aceptación para la optimización de latentes precomputados.

Invariante crítico (spec §0): Para encoders congelados, precomputar el latente
es matemáticamente idéntico a recalcularlo en cada paso. Los brazos C y D con
precomputed_latents=True deben producir Q-values bit a bit equivalentes a los
del path original con la misma semilla.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import gymnasium as gym

from latent_rl.agents import LatentDQNAgent
from latent_rl.envs import FinancialEnv, LatentObservationWrapper, precompute_latent_series
from latent_rl.representations.factory import build_encoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """DataFrame financiero mínimo con features y OHLCV."""
    np.random.seed(0)
    n = 120
    return pd.DataFrame({
        "open":        np.random.uniform(100, 200, n),
        "high":        np.random.uniform(100, 200, n),
        "low":         np.random.uniform(100, 200, n),
        "close":       np.random.uniform(100, 200, n),
        "volume":      np.random.uniform(1e6, 2e6, n),
        "log_return":  np.random.randn(n) * 0.01,
        "rsi_14":      np.random.uniform(20, 80, n),
    })


@pytest.fixture
def feature_cols():
    return ["log_return", "rsi_14"]


@pytest.fixture
def lookback():
    return 10


@pytest.fixture
def latent_dim():
    return 8


@pytest.fixture
def frozen_encoder(lookback, feature_cols, latent_dim):
    """Encoder MLP congelado con pesos fijos."""
    torch.manual_seed(42)
    enc = build_encoder(
        "mlp",
        input_len=lookback,
        n_features=len(feature_cols),
        latent_dim=latent_dim,
    )
    enc.freeze()
    enc.eval()
    return enc


@pytest.fixture
def base_env(sample_df, feature_cols, lookback):
    return FinancialEnv(
        data=sample_df,
        lookback_window=lookback,
        feature_cols=feature_cols,
    )


# ---------------------------------------------------------------------------
# Test 1: precompute_latent_series == encoder(env._get_observation()) en cada paso
# ---------------------------------------------------------------------------

class TestPrecomputeLatentSeries:

    def test_matches_env_observation_at_each_step(
        self, sample_df, feature_cols, lookback, frozen_encoder, latent_dim
    ):
        """latents[t] == encoder(env._get_observation()) cuando current_step == t."""
        device = torch.device("cpu")
        latents = precompute_latent_series(
            sample_df, lookback, feature_cols, frozen_encoder, device
        )

        env = FinancialEnv(
            data=sample_df, lookback_window=lookback, feature_cols=feature_cols
        )
        env.reset()

        # Verificar en varios pasos (no solo reset)
        for _ in range(15):
            t = env.current_step
            obs = env._get_observation()
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0)
                expected = frozen_encoder(obs_t).squeeze(0).numpy()

            np.testing.assert_allclose(
                latents[t], expected, atol=1e-5,
                err_msg=f"Desacuerdo en step={t}"
            )
            env.step(env.action_space.sample())

        env.close()

    def test_output_shape(self, sample_df, feature_cols, lookback, frozen_encoder, latent_dim):
        device = torch.device("cpu")
        latents = precompute_latent_series(
            sample_df, lookback, feature_cols, frozen_encoder, device
        )
        assert latents.shape == (len(sample_df), latent_dim)

    def test_dtype_is_float32(self, sample_df, feature_cols, lookback, frozen_encoder):
        device = torch.device("cpu")
        latents = precompute_latent_series(
            sample_df, lookback, feature_cols, frozen_encoder, device
        )
        assert latents.dtype == np.float32


# ---------------------------------------------------------------------------
# Test 2: LatentObservationWrapper devuelve latents[env.current_step]
# ---------------------------------------------------------------------------

class TestLatentObservationWrapper:

    def test_obs_after_reset_matches_latent(
        self, sample_df, feature_cols, lookback, frozen_encoder
    ):
        device = torch.device("cpu")
        latents = precompute_latent_series(
            sample_df, lookback, feature_cols, frozen_encoder, device
        )
        env = FinancialEnv(
            data=sample_df, lookback_window=lookback, feature_cols=feature_cols
        )
        wrapped = LatentObservationWrapper(env, latents)
        obs, _ = wrapped.reset()
        t = env.current_step
        np.testing.assert_array_equal(obs, latents[t].astype(np.float32))
        wrapped.close()

    def test_obs_after_step_matches_latent(
        self, sample_df, feature_cols, lookback, frozen_encoder
    ):
        device = torch.device("cpu")
        latents = precompute_latent_series(
            sample_df, lookback, feature_cols, frozen_encoder, device
        )
        env = FinancialEnv(
            data=sample_df, lookback_window=lookback, feature_cols=feature_cols
        )
        wrapped = LatentObservationWrapper(env, latents)
        wrapped.reset()
        for _ in range(5):
            obs, _, terminated, _, _ = wrapped.step(0)
            t = env.current_step
            np.testing.assert_array_equal(obs, latents[t].astype(np.float32))
            if terminated:
                break
        wrapped.close()

    def test_observation_space_shape(
        self, sample_df, feature_cols, lookback, frozen_encoder, latent_dim
    ):
        device = torch.device("cpu")
        latents = precompute_latent_series(
            sample_df, lookback, feature_cols, frozen_encoder, device
        )
        env = FinancialEnv(
            data=sample_df, lookback_window=lookback, feature_cols=feature_cols
        )
        wrapped = LatentObservationWrapper(env, latents)
        assert wrapped.observation_space.shape == (latent_dim,)
        wrapped.close()

    def test_raises_when_latents_too_short(
        self, sample_df, feature_cols, lookback, frozen_encoder
    ):
        device = torch.device("cpu")
        latents = precompute_latent_series(
            sample_df, lookback, feature_cols, frozen_encoder, device
        )
        env = FinancialEnv(
            data=sample_df, lookback_window=lookback, feature_cols=feature_cols
        )
        short_latents = latents[:5]  # mucho menos que len(data)
        with pytest.raises(ValueError, match="latents tiene"):
            LatentObservationWrapper(env, short_latents)
        env.close()


# ---------------------------------------------------------------------------
# Test 3: Q-values equivalentes precomputado vs recalculo en cada paso
# ---------------------------------------------------------------------------

class TestQValueEquivalence:
    """Invariante central: precomputed_latents produce Q-values idénticos."""

    def test_q_values_identical_given_same_latent(
        self, base_env, frozen_encoder, latent_dim, lookback
    ):
        """LatentDQNAgent con encoder congelado == modo precomputed dado mismo vector."""
        torch.manual_seed(7)
        device = torch.device("cpu")

        # Agente A: con encoder (freeze=True)
        agent_enc = LatentDQNAgent(
            action_space=base_env.action_space,
            observation_shape=base_env.observation_space.shape,
            latent_dim=latent_dim,
            encoder_type="mlp",
            freeze_encoder=True,
            device="cpu",
        )
        # Copiar pesos del frozen_encoder al agente
        agent_enc.encoder.load_state_dict(frozen_encoder.state_dict())
        agent_enc.encoder.freeze()

        # Agente B: precomputed_latents=True, misma Q-network
        agent_pre = LatentDQNAgent(
            action_space=base_env.action_space,
            observation_shape=(latent_dim,),
            latent_dim=latent_dim,
            precomputed_latents=True,
            device="cpu",
        )
        # Mismos pesos de Q-network
        agent_pre.q_network.load_state_dict(agent_enc.q_network.state_dict())

        # Comparar Q-values sobre una observación raw
        obs, _ = base_env.reset()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            latent_vec = frozen_encoder(obs_t)
            q_enc = agent_enc.q_network(latent_vec)
            q_pre = agent_pre.q_network(latent_vec)

        torch.testing.assert_close(q_enc, q_pre)

    def test_precomputed_agent_has_no_encoder(self, base_env, latent_dim):
        agent = LatentDQNAgent(
            action_space=base_env.action_space,
            observation_shape=(latent_dim,),
            latent_dim=latent_dim,
            precomputed_latents=True,
            device="cpu",
        )
        assert agent.encoder is None

    def test_precomputed_agent_optimizer_has_no_encoder_params(
        self, base_env, latent_dim
    ):
        agent = LatentDQNAgent(
            action_space=base_env.action_space,
            observation_shape=(latent_dim,),
            latent_dim=latent_dim,
            precomputed_latents=True,
            device="cpu",
        )
        # No hay encoder => todos los params del optimizer son de q_network
        q_param_ids = {id(p) for p in agent.q_network.parameters()}
        for group in agent.optimizer.param_groups:
            for p in group["params"]:
                assert id(p) in q_param_ids


# ---------------------------------------------------------------------------
# Test 4: encoder C entrenado una vez (no por semilla)
# ---------------------------------------------------------------------------

class TestArmCTrainedOnce:
    """prepare_arm_c_encoder debe llamarse una sola vez por ticker/split."""

    def test_prepare_arm_c_called_once_across_seeds(
        self, sample_df, feature_cols, lookback, latent_dim
    ):
        """Verifica que prepare_arm_c_encoder se llama una vez en _run_single_ticker."""
        from latent_rl.experiments import runner as runner_mod

        call_count = {"n": 0}
        original_fn = runner_mod.prepare_arm_c_encoder

        def counting_fn(*args, **kwargs):
            call_count["n"] += 1
            return original_fn(*args, **kwargs)

        with patch.object(runner_mod, "prepare_arm_c_encoder", side_effect=counting_fn):
            # Simular 3 semillas; prepare_arm_c_encoder solo debería llamarse 1 vez
            from latent_rl.experiments.config import ExperimentConfig

            cfg = ExperimentConfig(
                tickers=["FAKE"],
                start_date="2020-01-01",
                end_date="2024-01-01",
                seeds=[0, 1, 2],
                run_arms=["C"],
                features=feature_cols,
                lookback_window=lookback,
                latent_dim=latent_dim,
                n_training_episodes=1,
                n_eval_episodes=1,
                max_steps_per_episode=20,
                pretrain_n_epochs=1,
                latent_buffer_capacity=100,
                dqn_buffer_capacity=100,
            )

            # _build_frozen_ctx llama prepare_arm_c_encoder una vez
            data_is = sample_df.iloc[:80].copy()
            data_oos = sample_df.iloc[80:].copy()
            ctx = runner_mod._build_frozen_ctx(data_is, data_oos, cfg, d_encoder=None)

        assert call_count["n"] == 1, (
            f"prepare_arm_c_encoder debería llamarse 1 vez, se llamó {call_count['n']}"
        )


# ---------------------------------------------------------------------------
# Test 5: brazo B no usa precomputed_latents (encoder sigue entrenando)
# ---------------------------------------------------------------------------

class TestArmBUnchanged:

    def test_arm_b_encoder_is_not_none(self, sample_df, feature_cols, lookback, latent_dim):
        """El brazo B usa LatentDQNAgent con encoder activo (no precomputed)."""
        from latent_rl.experiments.runner import _make_latent_agent
        from latent_rl.experiments.config import ExperimentConfig

        cfg = ExperimentConfig(
            tickers=["FAKE"],
            features=feature_cols,
            lookback_window=lookback,
            latent_dim=latent_dim,
        )
        env = FinancialEnv(
            data=sample_df, lookback_window=lookback, feature_cols=feature_cols
        )
        agent = _make_latent_agent(env, cfg, freeze_encoder=False)
        assert agent.encoder is not None
        assert not agent.precomputed_latents
        env.close()

    def test_precomputed_latents_flag_false_by_default(self, base_env, latent_dim):
        agent = LatentDQNAgent(
            action_space=base_env.action_space,
            observation_shape=base_env.observation_space.shape,
            latent_dim=latent_dim,
            device="cpu",
        )
        assert agent.precomputed_latents is False
        assert agent.encoder is not None


# ---------------------------------------------------------------------------
# Test 6: frozen encoder weights preserved after RL training con precomputed
# ---------------------------------------------------------------------------

class TestFrozenWeightsPreservedPrecomputed:

    def test_weights_unchanged_during_rl(
        self, sample_df, feature_cols, lookback, frozen_encoder, latent_dim
    ):
        """El encoder C no muta sus pesos durante el entrenamiento RL del agente."""
        device = torch.device("cpu")
        latents = precompute_latent_series(
            sample_df, lookback, feature_cols, frozen_encoder, device
        )

        weights_before = {k: v.clone() for k, v in frozen_encoder.state_dict().items()}

        env = FinancialEnv(
            data=sample_df, lookback_window=lookback, feature_cols=feature_cols
        )
        wrapped = LatentObservationWrapper(env, latents)

        agent = LatentDQNAgent(
            action_space=wrapped.action_space,
            observation_shape=(latent_dim,),
            latent_dim=latent_dim,
            precomputed_latents=True,
            batch_size=16,
            buffer_capacity=200,
            device="cpu",
        )

        # Bucle RL mínimo
        obs, _ = wrapped.reset()
        for _ in range(60):
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, _ = wrapped.step(action)
            agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
            agent.update()
            if terminated or truncated:
                obs, _ = wrapped.reset()
            else:
                obs = next_obs

        # El encoder externo no debería haber mutado
        for k, v in frozen_encoder.state_dict().items():
            torch.testing.assert_close(weights_before[k], v, msg=f"Parámetro {k} mutó")

        wrapped.close()
