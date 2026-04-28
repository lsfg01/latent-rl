"""Tests para DQNAgent."""

import pytest
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import tempfile
import os

from latent_rl.agents import DQNAgent, ReplayBuffer
from latent_rl.envs import FinancialEnv


class TestReplayBuffer:
    """Tests para ReplayBuffer."""

    def test_init(self):
        """Test de inicialización del buffer."""
        buffer = ReplayBuffer(capacity=100, observation_shape=(10, 5))

        assert buffer.capacity == 100
        assert buffer.observation_shape == (10, 5)
        assert len(buffer) == 0
        assert buffer.size == 0

    def test_add(self):
        """Test de añadir transiciones al buffer."""
        buffer = ReplayBuffer(capacity=100, observation_shape=(10, 5))

        state = np.random.randn(10, 5).astype(np.float32)
        action = 1
        reward = 0.5
        next_state = np.random.randn(10, 5).astype(np.float32)
        done = False

        buffer.add(state, action, reward, next_state, done)

        assert len(buffer) == 1
        assert buffer.size == 1

    def test_add_multiple(self):
        """Test de añadir múltiples transiciones."""
        buffer = ReplayBuffer(capacity=100, observation_shape=(10, 5))

        for i in range(10):
            state = np.random.randn(10, 5).astype(np.float32)
            action = i % 3
            reward = np.random.randn()
            next_state = np.random.randn(10, 5).astype(np.float32)
            done = i == 9
            buffer.add(state, action, reward, next_state, done)

        assert len(buffer) == 10

    def test_sample(self):
        """Test de muestreo del buffer."""
        buffer = ReplayBuffer(capacity=100, observation_shape=(10, 5))

        # Añadir transiciones
        for i in range(20):
            state = np.random.randn(10, 5).astype(np.float32)
            action = i % 3
            reward = np.random.randn()
            next_state = np.random.randn(10, 5).astype(np.float32)
            done = i == 19
            buffer.add(state, action, reward, next_state, done)

        # Muestrear batch
        states, actions, rewards, next_states, dones = buffer.sample(batch_size=5)

        assert states.shape == (5, 10, 5)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)
        assert next_states.shape == (5, 10, 5)
        assert dones.shape == (5,)

    def test_sample_insufficient_data(self):
        """Test de muestreo con datos insuficientes."""
        buffer = ReplayBuffer(capacity=100, observation_shape=(10, 5))

        # Añadir solo 5 transiciones
        for i in range(5):
            state = np.random.randn(10, 5).astype(np.float32)
            buffer.add(state, 0, 0.0, state, False)

        # Intentar muestrear batch más grande que el buffer
        # np.random.choice con replace=False lanza ValueError si size > population
        try:
            buffer.sample(batch_size=10)
            assert False, "Debería lanzar ValueError"
        except ValueError:
            pass  # Comportamiento esperado

    def test_capacity_limit(self):
        """Test de límite de capacidad del buffer."""
        buffer = ReplayBuffer(capacity=10, observation_shape=(10, 5))

        # Añadir más transiciones que la capacidad
        for i in range(15):
            state = np.random.randn(10, 5).astype(np.float32)
            buffer.add(state, i % 3, float(i), state, i == 14)

        # El buffer no debe exceder la capacidad
        assert len(buffer) == 10
        assert buffer.size == 10

    def test_len(self):
        """Test del método __len__."""
        buffer = ReplayBuffer(capacity=100, observation_shape=(10, 5))

        assert len(buffer) == 0

        for i in range(5):
            state = np.random.randn(10, 5).astype(np.float32)
            buffer.add(state, 0, 0.0, state, False)

        assert len(buffer) == 5


class TestDQNAgent:
    """Tests para DQNAgent."""

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
        """Agente DQN de ejemplo para tests."""
        return DQNAgent(
            action_space=env.action_space,
            observation_shape=env.observation_space.shape,
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
        agent = DQNAgent(
            action_space=env.action_space,
            observation_shape=env.observation_space.shape
        )

        assert agent.n_actions == 3
        assert agent.epsilon == 1.0
        assert agent.batch_size == 64
        assert len(agent.replay_buffer) == 0

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
            new_agent = DQNAgent(
                action_space=agent.action_space,
                observation_shape=agent.observation_shape
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
        agent = DQNAgent(
            action_space=env.action_space,
            observation_shape=env.observation_space.shape,
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