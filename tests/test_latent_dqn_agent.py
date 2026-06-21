"""Tests para LatentDQNAgent."""

import pytest
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import tempfile
import os

from latent_rl.agents import LatentDQNAgent, ReplayBuffer
from latent_rl.envs import FinancialEnv
from latent_rl.pretraining import AutoencoderTrainer


class TestLatentDQNAgent:
    """Tests para LatentDQNAgent."""

    @pytest.fixture
    def sample_data(self):
        """Datos de ejemplo para tests."""
        return pd.DataFrame({
            "open": np.random.randn(100) + 100,
            "high": np.random.randn(100) + 105,
            "low": np.random.randn(100) + 95,
            "close": np.random.randn(100) + 100,
            "volume": np.random.randint(1000, 10000, 100)
        })

    @pytest.fixture
    def env(self, sample_data):
        """Entorno de ejemplo para tests."""
        return FinancialEnv(sample_data, lookback_window=10)

    @pytest.fixture
    def agent(self, env):
        """Agente LatentDQN de ejemplo para tests."""
        return LatentDQNAgent(
            action_space=env.action_space,
            observation_shape=env.observation_space.shape,
            latent_dim=16,
            learning_rate=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            batch_size=32,
            buffer_capacity=1000,
            target_update_freq=10,
            device="cpu"
        )

    def test_init(self, env):
        """Test de inicialización del agente."""
        agent = LatentDQNAgent(
            action_space=env.action_space,
            observation_shape=env.observation_space.shape,
            latent_dim=16
        )

        assert agent.n_actions == 3
        assert agent.latent_dim == 16
        assert agent.epsilon == 1.0
        assert agent.batch_size == 64
        assert len(agent.replay_buffer) == 0

    def test_init_with_custom_latent_dim(self, env):
        """Test de inicialización con dimensión latente personalizada."""
        agent = LatentDQNAgent(
            action_space=env.action_space,
            observation_shape=env.observation_space.shape,
            latent_dim=32
        )

        assert agent.latent_dim == 32

    def test_select_action_greedy(self, agent):
        """Test de selección de acción greedy."""
        obs = np.random.randn(10, 5).astype(np.float32)

        # Establecer epsilon a 0 para forzar greedy
        agent.set_epsilon(0.0)

        action = agent.select_action(obs, training=False)

        assert action in [0, 1, 2]
        assert agent.action_space.contains(action)

    def test_select_action_epsilon_greedy(self, agent):
        """Test de selección de acción epsilon-greedy."""
        obs = np.random.randn(10, 5).astype(np.float32)

        # Establecer epsilon a 1.0 para forzar aleatorio
        agent.set_epsilon(1.0)

        actions = []
        for _ in range(100):
            action = agent.select_action(obs, training=True)
            actions.append(action)

        # Con epsilon=1.0, todas las acciones deben ser aleatorias
        # Debería haber variedad en las acciones
        unique_actions = set(actions)
        assert len(unique_actions) > 1

    def test_select_action_training_vs_eval(self, agent):
        """Test de diferencia entre training y evaluación."""
        obs = np.random.randn(10, 5).astype(np.float32)

        # En training con epsilon alto, puede ser aleatorio
        agent.set_epsilon(0.5)
        action_train = agent.select_action(obs, training=True)

        # En evaluación, debe ser greedy
        action_eval = agent.select_action(obs, training=False)

        # Ambas acciones deben ser válidas
        assert agent.action_space.contains(action_train)
        assert agent.action_space.contains(action_eval)

    def test_get_latent_representation(self, agent):
        """Test de obtención de representación latente."""
        obs = np.random.randn(10, 5).astype(np.float32)

        latent = agent.get_latent_representation(obs)

        # Verificar forma
        assert latent.shape == (agent.latent_dim,)
        # Verificar que es un array numpy
        assert isinstance(latent, np.ndarray)

    def test_get_latent_representation_batch(self, agent):
        """Test de representación latente con múltiples observaciones."""
        obs1 = np.random.randn(10, 5).astype(np.float32)
        obs2 = np.random.randn(10, 5).astype(np.float32)

        latent1 = agent.get_latent_representation(obs1)
        latent2 = agent.get_latent_representation(obs2)

        # Verificar que son diferentes (probablemente)
        assert not np.allclose(latent1, latent2)

    def test_store_transition(self, agent):
        """Test de almacenar transiciones."""
        state = np.random.randn(10, 5).astype(np.float32)
        next_state = np.random.randn(10, 5).astype(np.float32)

        agent.store_transition(state, 1, 0.5, next_state, False)

        assert len(agent.replay_buffer) == 1

    def test_store_multiple_transitions(self, agent):
        """Test de almacenar múltiples transiciones."""
        for i in range(10):
            state = np.random.randn(10, 5).astype(np.float32)
            next_state = np.random.randn(10, 5).astype(np.float32)
            agent.store_transition(state, i % 3, float(i), next_state, i == 9)

        assert len(agent.replay_buffer) == 10

    def test_update_with_insufficient_data(self, agent):
        """Test de actualización con datos insuficientes."""
        # Añadir menos transiciones que el batch_size
        for i in range(10):
            state = np.random.randn(10, 5).astype(np.float32)
            next_state = np.random.randn(10, 5).astype(np.float32)
            agent.store_transition(state, 0, 0.0, next_state, False)

        # Actualizar con buffer insuficiente
        metrics = agent.update()

        # Debe devolver métricas cero sin romper
        assert metrics["loss"] == 0.0
        assert metrics["q_mean"] == 0.0
        assert metrics["q_std"] == 0.0

    def test_update_with_sufficient_data(self, agent):
        """Test de actualización con datos suficientes."""
        # Añadir suficientes transiciones
        for i in range(100):
            state = np.random.randn(10, 5).astype(np.float32)
            next_state = np.random.randn(10, 5).astype(np.float32)
            agent.store_transition(state, i % 3, float(i) / 100, next_state, i == 99)

        # Actualizar con buffer suficiente
        metrics = agent.update()

        # Debe devolver métricas válidas
        assert "loss" in metrics
        assert "q_mean" in metrics
        assert "q_std" in metrics
        assert metrics["loss"] >= 0.0  # Loss no puede ser negativo

    def test_epsilon_decay(self, agent):
        """Test de decaimiento de epsilon."""
        initial_epsilon = agent.epsilon

        # Simular varias selecciones de acciones
        obs = np.random.randn(10, 5).astype(np.float32)
        for _ in range(10):
            agent.select_action(obs, training=True)

        # Epsilon debe haber decaído
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_end

    def test_reset(self, agent):
        """Test de reset del agente."""
        # Modificar estado del agente
        agent.set_epsilon(0.5)
        agent.update_step = 50

        # Resetear
        agent.reset()

        # Verificar reset
        assert agent.epsilon == agent.epsilon_start
        assert agent.update_step == 0

    def test_save_load(self, agent):
        """Test de guardar y cargar modelo."""
        # Añadir algunas transiciones
        for i in range(10):
            state = np.random.randn(10, 5).astype(np.float32)
            next_state = np.random.randn(10, 5).astype(np.float32)
            agent.store_transition(state, 0, 0.0, next_state, False)

        # Modificar estado
        agent.set_epsilon(0.3)
        agent.update_step = 25

        # Guardar
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name

        try:
            agent.save(temp_path)

            # Crear nuevo agente
            new_agent = LatentDQNAgent(
                action_space=agent.action_space,
                observation_shape=agent.observation_shape,
                latent_dim=agent.latent_dim
            )

            # Cargar
            new_agent.load(temp_path)

            # Verificar que se cargó correctamente
            assert new_agent.epsilon == 0.3
            assert new_agent.update_step == 25

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_set_epsilon(self, agent):
        """Test de establecer epsilon manualmente."""
        agent.set_epsilon(0.5)
        assert agent.epsilon == 0.5

        agent.set_epsilon(0.0)
        assert agent.epsilon == 0.0

        agent.set_epsilon(1.0)
        assert agent.epsilon == 1.0

    def test_integration_with_financial_env(self, env):
        """Test de integración corta con FinancialEnv."""
        agent = LatentDQNAgent(
            action_space=env.action_space,
            observation_shape=env.observation_space.shape,
            latent_dim=16,
            batch_size=16,
            buffer_capacity=500,
            target_update_freq=5
        )

        # Ejecutar episodio corto
        obs, info = env.reset()
        done = False
        steps = 0
        max_steps = 20

        while not done and steps < max_steps:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, terminated or truncated)

            obs = next_obs
            done = terminated or truncated
            steps += 1

        # Verificar que se ejecutó el episodio
        assert steps > 0
        assert len(agent.replay_buffer) == steps

        # Intentar actualizar
        if len(agent.replay_buffer) >= agent.batch_size:
            metrics = agent.update()
            assert "loss" in metrics

        env.close()

    def test_encoder_q_network_integration(self, agent):
        """Test de integración entre encoder y QNetwork."""
        # Añadir suficientes transiciones
        for i in range(100):
            state = np.random.randn(10, 5).astype(np.float32)
            next_state = np.random.randn(10, 5).astype(np.float32)
            agent.store_transition(state, i % 3, float(i) / 100, next_state, i == 99)

        # Actualizar varias veces
        for _ in range(5):
            metrics = agent.update()
            assert metrics["loss"] >= 0.0

        # Verificar que el encoder y la QNetwork tienen parámetros entrenables
        encoder_params = list(agent.encoder.parameters())
        q_params = list(agent.q_network.parameters())

        assert len(encoder_params) > 0
        assert len(q_params) > 0

        # Verificar que los parámetros se están actualizando
        # (al menos algunos deberían tener gradients)
        for param in encoder_params + q_params:
            assert param.requires_grad

    def test_load_pretrained_encoder(self, agent):
        """Test de cargar encoder preentrenado."""
        # Crear un encoder temporal y guardarlo
        trainer = AutoencoderTrainer(encoder=agent.encoder, device="cpu")

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name

        try:
            # Guardar encoder
            trainer.save_encoder(temp_path)

            # Crear nuevo agente
            new_agent = LatentDQNAgent(
                action_space=agent.action_space,
                observation_shape=agent.observation_shape,
                latent_dim=agent.latent_dim
            )

            # Cargar encoder preentrenado
            new_agent.load_pretrained_encoder(temp_path)

            # Verificar que se cargó sin errores
            assert new_agent.latent_dim == agent.latent_dim

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_encoder_type_tcn_no_error(self, env):
        """LatentDQNAgent debe aceptar encoder_type='tcn' sin error."""
        agent = LatentDQNAgent(
            action_space=env.action_space,
            observation_shape=env.observation_space.shape,
            latent_dim=8,
            encoder_type="tcn",
        )
        obs, _ = env.reset()
        action = agent.select_action(obs, training=False)
        assert env.action_space.contains(action)

    def test_freeze_encoder_weights_unchanged(self, env):
        """Con freeze_encoder=True, los pesos del encoder no cambian al hacer update()."""
        obs_shape = env.observation_space.shape
        agent = LatentDQNAgent(
            action_space=env.action_space,
            observation_shape=obs_shape,
            latent_dim=8,
            freeze_encoder=True,
            batch_size=16,
            buffer_capacity=500,
        )

        # Guardar pesos iniciales del encoder
        weights_before = {
            k: v.clone() for k, v in agent.encoder.state_dict().items()
        }

        # Añadir transiciones y actualizar
        for _ in range(50):
            state = np.random.randn(*obs_shape).astype(np.float32)
            next_state = np.random.randn(*obs_shape).astype(np.float32)
            agent.store_transition(state, 0, 0.0, next_state, False)

        for _ in range(5):
            agent.update()

        # Los pesos del encoder deben ser idénticos
        for k, v in agent.encoder.state_dict().items():
            assert torch.allclose(weights_before[k], v), \
                f"El peso {k} del encoder cambió con freeze_encoder=True"

    def test_freeze_encoder_optimizer_has_no_encoder_params(self, env):
        """Con freeze_encoder=True, el optimizador sólo contiene parámetros de Q-network."""
        agent = LatentDQNAgent(
            action_space=env.action_space,
            observation_shape=env.observation_space.shape,
            latent_dim=8,
            freeze_encoder=True,
        )

        # Recopilar IDs de parámetros del encoder
        encoder_param_ids = {id(p) for p in agent.encoder.parameters()}

        # Comprobar que ningún grupo de parámetros del optimizador incluye encoder params
        for group in agent.optimizer.param_groups:
            for p in group["params"]:
                assert id(p) not in encoder_param_ids, \
                    "El optimizador contiene parámetros del encoder en modo freeze"

    def test_set_normalization(self, agent):
        """Test de set_normalization."""
        lookback_window = 10
        n_features = 5

        # Crear parámetros de normalización
        mean = np.random.randn(lookback_window, n_features).astype(np.float32)
        std = np.random.randn(lookback_window, n_features).astype(np.float32) + 1.0

        # Establecer normalización
        agent.set_normalization(mean, std)

        # Verificar que se guardaron
        assert agent.normalization_mean is not None
        assert agent.normalization_std is not None
        assert np.array_equal(agent.normalization_mean, mean)
        assert np.array_equal(agent.normalization_std, std)

    def test_load_pretrained_encoder_with_normalization(self, agent, sample_data):
        """Test de que load_pretrained_encoder carga normalización si existe."""
        # Crear trainer y guardar encoder con normalización
        trainer = AutoencoderTrainer(encoder=agent.encoder, device="cpu")

        # Extraer observaciones y fit normalization
        lookback_window = 10
        observations = trainer.collect_observations(
            data=sample_data,
            lookback_window=lookback_window,
            n_samples=50
        )
        trainer.fit_normalization(observations)

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name

        try:
            # Guardar encoder con normalización
            trainer.save_encoder(temp_path)

            # Crear nuevo agente
            new_agent = LatentDQNAgent(
                action_space=agent.action_space,
                observation_shape=agent.observation_shape,
                latent_dim=agent.latent_dim
            )

            # Cargar encoder preentrenado
            new_agent.load_pretrained_encoder(temp_path)

            # Verificar que se cargó la normalización
            assert new_agent.normalization_mean is not None
            assert new_agent.normalization_std is not None
            assert np.array_equal(new_agent.normalization_mean, trainer.normalization_mean)
            assert np.array_equal(new_agent.normalization_std, trainer.normalization_std)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
