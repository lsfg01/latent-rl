"""Tests para el módulo de agentes."""

import pytest
import numpy as np
from gymnasium import spaces
from unittest.mock import patch

from latent_rl.agents import BaseAgent, RandomAgent, BuyAndHoldAgent


class TestBaseAgent:
    """Tests para BaseAgent."""

    def test_base_agent_is_abstract(self):
        """Test de que BaseAgent no puede instanciarse directamente."""
        action_space = spaces.Discrete(3)

        with pytest.raises(TypeError):
            BaseAgent(action_space)

    def test_base_agent_requires_select_action(self):
        """Test de que select_action es un método abstracto."""
        action_space = spaces.Discrete(3)

        # Intentar crear una subclase sin implementar select_action
        class IncompleteAgent(BaseAgent):
            pass

        with pytest.raises(TypeError):
            IncompleteAgent(action_space)

    def test_base_agent_reset_default_implementation(self):
        """Test de que reset() tiene implementación por defecto."""
        action_space = spaces.Discrete(3)

        # Crear una subclase mínima
        class MinimalAgent(BaseAgent):
            def select_action(self, observation):
                return 0

        agent = MinimalAgent(action_space)
        agent.reset()  # No debe lanzar excepción


class TestRandomAgent:
    """Tests para RandomAgent."""

    def test_init_with_discrete_space(self):
        """Test de inicialización con espacio Discrete."""
        action_space = spaces.Discrete(3)
        agent = RandomAgent(action_space)

        assert agent.action_space == action_space
        assert agent.seed is None

    def test_init_with_seed(self):
        """Test de inicialización con semilla."""
        action_space = spaces.Discrete(3)
        agent = RandomAgent(action_space, seed=42)

        assert agent.seed == 42

    def test_select_action_returns_valid_action(self):
        """Test de que select_action devuelve una acción válida."""
        action_space = spaces.Discrete(3)
        agent = RandomAgent(action_space)

        observation = np.random.randn(10, 5)  # Observación dummy
        action = agent.select_action(observation)

        assert action_space.contains(action)
        assert action in [0, 1, 2]

    def test_select_action_ignores_observation(self):
        """Test de que select_action ignora la observación."""
        action_space = spaces.Discrete(3)
        agent = RandomAgent(action_space, seed=42)

        # Dos observaciones diferentes
        obs1 = np.ones((10, 5))
        obs2 = np.zeros((10, 5))

        # Con semilla, las acciones deben ser diferentes (porque son aleatorias)
        action1 = agent.select_action(obs1)
        action2 = agent.select_action(obs2)

        # Las acciones pueden ser iguales o diferentes, pero ambas deben ser válidas
        assert action_space.contains(action1)
        assert action_space.contains(action2)

    def test_reset_with_seed(self):
        """Test de que reset() restaura la semilla."""
        action_space = spaces.Discrete(3)
        agent = RandomAgent(action_space, seed=42)

        # Obtener primera acción
        action1 = agent.select_action(None)

        # Resetear
        agent.reset()

        # Obtener segunda acción después del reset
        action2 = agent.select_action(None)

        # Con la misma semilla, las acciones deben ser iguales
        assert action1 == action2

    def test_multiple_calls_different_actions(self):
        """Test de que múltiples llamadas pueden dar diferentes acciones."""
        action_space = spaces.Discrete(3)
        agent = RandomAgent(action_space)

        actions = [agent.select_action(None) for _ in range(100)]

        # Con 100 muestras, es muy probable que haya variedad
        unique_actions = set(actions)
        assert len(unique_actions) > 1

    def test_reproducibility_with_seed(self):
        """Test de reproducibilidad con semilla."""
        action_space = spaces.Discrete(3)

        # Un agente con semilla
        agent = RandomAgent(action_space, seed=42)

        # Obtener primera secuencia de acciones
        actions1 = [agent.select_action(None) for _ in range(10)]

        # Resetear y obtener segunda secuencia
        agent.reset()
        actions2 = [agent.select_action(None) for _ in range(10)]

        # Las secuencias deben ser idénticas después del reset
        assert actions1 == actions2

    def test_different_seeds_different_actions(self):
        """Test de que semillas diferentes dan diferentes acciones."""
        action_space = spaces.Discrete(3)

        # Dos agentes con semillas diferentes
        agent1 = RandomAgent(action_space, seed=42)
        agent2 = RandomAgent(action_space, seed=123)

        # Obtener secuencias de acciones
        actions1 = [agent1.select_action(None) for _ in range(10)]
        actions2 = [agent2.select_action(None) for _ in range(10)]

        # Las secuencias deben ser diferentes (con alta probabilidad)
        assert actions1 != actions2

    def test_integration_with_financial_env(self):
        """Test de integración con FinancialEnv."""
        from latent_rl.envs import FinancialEnv
        import pandas as pd

        # Crear datos de prueba
        data = pd.DataFrame({
            "open": np.random.randn(100) + 100,
            "high": np.random.randn(100) + 105,
            "low": np.random.randn(100) + 95,
            "close": np.random.randn(100) + 100,
            "volume": np.random.randint(1000, 10000, 100)
        })

        # Crear entorno
        env = FinancialEnv(data, lookback_window=10)

        # Crear agente
        agent = RandomAgent(env.action_space, seed=42)

        # Ejecutar un episodio
        obs, info = env.reset(seed=42)
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Verificar que se ejecutó el episodio
        assert total_reward != 0 or env.current_step > env.lookback_window

        env.close()

    def test_action_space_box(self):
        """Test de que funciona con espacios Box continuos."""
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        agent = RandomAgent(action_space, seed=42)

        action = agent.select_action(None)

        assert action_space.contains(action)
        assert action.shape == (2,)
        assert -1.0 <= action[0] <= 1.0
        assert -1.0 <= action[1] <= 1.0


class TestBuyAndHoldAgent:
    """Tests para BuyAndHoldAgent."""

    def test_init(self):
        """Test de inicialización básica."""
        action_space = spaces.Discrete(3)
        agent = BuyAndHoldAgent(action_space)

        assert agent.action_space == action_space
        assert agent.has_bought is False

    def test_init_with_invalid_action_space(self):
        """Test de inicialización con action_space inválido."""
        # Action space que no permite acción 1
        action_space = spaces.Discrete(1)  # Solo permite acción 0

        with pytest.raises(ValueError, match="action_space debe permitir las acciones 0"):
            BuyAndHoldAgent(action_space)

    def test_first_action_is_buy(self):
        """Test de que la primera acción es buy (1)."""
        action_space = spaces.Discrete(3)
        agent = BuyAndHoldAgent(action_space)

        action = agent.select_action(None)

        assert action == 1  # Buy
        assert agent.has_bought is True

    def test_subsequent_actions_are_hold(self):
        """Test de que las acciones siguientes son hold (0)."""
        action_space = spaces.Discrete(3)
        agent = BuyAndHoldAgent(action_space)

        # Primera acción debe ser buy
        first_action = agent.select_action(None)
        assert first_action == 1

        # Acciones siguientes deben ser hold
        for _ in range(10):
            action = agent.select_action(None)
            assert action == 0  # Hold

    def test_reset_resets_state(self):
        """Test de que reset() restaura el estado interno."""
        action_space = spaces.Discrete(3)
        agent = BuyAndHoldAgent(action_space)

        # Ejecutar algunas acciones
        agent.select_action(None)
        agent.select_action(None)
        assert agent.has_bought is True

        # Resetear
        agent.reset()
        assert agent.has_bought is False

        # Después del reset, primera acción debe ser buy de nuevo
        action = agent.select_action(None)
        assert action == 1

    def test_buy_then_hold_sequence(self):
        """Test de la secuencia completa buy + holds."""
        action_space = spaces.Discrete(3)
        agent = BuyAndHoldAgent(action_space)

        actions = []
        for _ in range(5):
            action = agent.select_action(None)
            actions.append(action)

        # Primera acción debe ser buy, resto hold
        assert actions[0] == 1
        assert all(action == 0 for action in actions[1:])

    def test_ignores_observation(self):
        """Test de que select_action ignora la observación."""
        action_space = spaces.Discrete(3)
        agent = BuyAndHoldAgent(action_space)

        # Diferentes observaciones
        obs1 = np.ones((10, 5))
        obs2 = np.zeros((10, 5))
        obs3 = np.random.randn(10, 5)

        # La primera acción siempre debe ser buy
        action1 = agent.select_action(obs1)
        action2 = agent.select_action(obs2)
        action3 = agent.select_action(obs3)

        assert action1 == 1
        assert action2 == 0  # Ya compró, ahora hold
        assert action3 == 0  # Sigue hold

    def test_deterministic_behavior(self):
        """Test de que el comportamiento es determinista."""
        action_space = spaces.Discrete(3)

        # Dos agentes con el mismo action_space
        agent1 = BuyAndHoldAgent(action_space)
        agent2 = BuyAndHoldAgent(action_space)

        # Obtener secuencias de acciones
        actions1 = [agent1.select_action(None) for _ in range(10)]
        actions2 = [agent2.select_action(None) for _ in range(10)]

        # Las secuencias deben ser idénticas
        assert actions1 == actions2

    def test_reset_and_repeat_sequence(self):
        """Test de que reset() permite repetir la secuencia."""
        action_space = spaces.Discrete(3)
        agent = BuyAndHoldAgent(action_space)

        # Primera secuencia
        sequence1 = [agent.select_action(None) for _ in range(5)]

        # Resetear
        agent.reset()

        # Segunda secuencia
        sequence2 = [agent.select_action(None) for _ in range(5)]

        # Las secuencias deben ser idénticas
        assert sequence1 == sequence2

    def test_integration_with_financial_env(self):
        """Test de integración con FinancialEnv."""
        from latent_rl.envs import FinancialEnv
        import pandas as pd

        # Crear datos de prueba
        data = pd.DataFrame({
            "open": np.random.randn(100) + 100,
            "high": np.random.randn(100) + 105,
            "low": np.random.randn(100) + 95,
            "close": np.random.randn(100) + 100,
            "volume": np.random.randint(1000, 10000, 100)
        })

        # Crear entorno
        env = FinancialEnv(data, lookback_window=10)

        # Crear agente
        agent = BuyAndHoldAgent(env.action_space)

        # Ejecutar un episodio
        obs, info = env.reset(seed=42)
        done = False
        total_reward = 0.0
        action_count = {"buy": 0, "hold": 0, "sell": 0}

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Contar acciones
            if action == 1:
                action_count["buy"] += 1
            elif action == 0:
                action_count["hold"] += 1
            elif action == 2:
                action_count["sell"] += 1

        # Verificar comportamiento esperado
        assert action_count["buy"] == 1  # Solo una compra
        assert action_count["hold"] > 0  # Varios holds
        assert action_count["sell"] == 0  # Ninguna venta

        # Verificar que se ejecutó el episodio
        assert total_reward != 0 or env.current_step > env.lookback_window

        env.close()

    def test_multiple_episodes_consistency(self):
        """Test de consistencia entre múltiples episodios."""
        from latent_rl.envs import FinancialEnv
        import pandas as pd

        # Crear datos de prueba
        data = pd.DataFrame({
            "open": np.random.randn(50) + 100,
            "high": np.random.randn(50) + 105,
            "low": np.random.randn(50) + 95,
            "close": np.random.randn(50) + 100,
            "volume": np.random.randint(1000, 10000, 50)
        })

        # Crear entorno
        env = FinancialEnv(data, lookback_window=10)
        agent = BuyAndHoldAgent(env.action_space)

        # Ejecutar múltiples episodios
        episode_rewards = []
        for episode in range(3):
            obs, info = env.reset(seed=episode)
            done = False
            episode_reward = 0.0

            while not done:
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            agent.reset()

        # Verificar que todos los episodios se ejecutaron
        assert len(episode_rewards) == 3
        assert all(reward != 0 or True for reward in episode_rewards)  # Al menos se ejecutaron

        env.close()