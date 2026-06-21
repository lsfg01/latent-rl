"""Tests para el módulo de entornos."""

import pytest
import numpy as np
import pandas as pd
import gymnasium as gym

from latent_rl.envs import FinancialEnv, BaseFinancialEnv


class TestFinancialEnv:
    """Tests para FinancialEnv."""

    @pytest.fixture
    def sample_data(self):
        """Datos de ejemplo para tests."""
        return pd.DataFrame({
            "Open": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
            "High": [105.0, 107.0, 109.0, 111.0, 113.0, 115.0],
            "Low": [95.0, 97.0, 99.0, 101.0, 103.0, 105.0],
            "Close": [102.0, 104.0, 106.0, 108.0, 110.0, 112.0],
            "Volume": [1000, 1100, 1200, 1300, 1400, 1500]
        })

    @pytest.fixture
    def env(self, sample_data):
        """Entorno de ejemplo para tests."""
        return FinancialEnv(sample_data, lookback_window=2)

    def test_init(self, sample_data):
        """Test de inicialización del entorno."""
        env = FinancialEnv(sample_data, lookback_window=2)

        assert env.initial_balance == 10000.0
        assert env.transaction_cost == 0.001
        assert env.lookback_window == 2
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Discrete)

    def test_init_insufficient_data(self):
        """Test de inicialización con datos insuficientes."""
        data = pd.DataFrame({
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            "Close": [102.0],
            "Volume": [1000]
        })

        with pytest.raises(ValueError, match="Datos insuficientes"):
            FinancialEnv(data, lookback_window=5)

    def test_reset(self, env):
        """Test de reset del entorno."""
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)
        assert "balance" in info
        assert "position" in info
        assert env.current_step == env.lookback_window

    def test_step_valid_action(self, env):
        """Test de step con acción válida."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)  # Hold

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_invalid_action(self, env):
        """Test de step con acción inválida."""
        env.reset()

        with pytest.raises(ValueError, match="Acción inválida"):
            env.step(5)  # Acción fuera de rango

    def test_step_hold_action(self, env):
        """Test de acción hold."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        assert info["position"] == 0  # Sin posición

    def test_step_buy_action(self, env):
        """Test de acción buy."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)  # Buy

        assert info["position"] == 1  # Long

    def test_step_sell_action(self, env):
        """Test de acción sell."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(2)  # Sell sin posición

        assert info["position"] == 0  # Sin posición (mantener)

    def test_observation_shape(self, env):
        """Test de forma de observación."""
        env.reset()
        obs, _, _, _, _ = env.step(0)

        assert obs.shape == (env.lookback_window, env.n_features)

    def test_termination_condition(self, env):
        """Test de condición de terminación."""
        env.reset()

        # Ejecutar hasta el final
        for _ in range(len(env.data) - env.lookback_window):
            obs, reward, terminated, truncated, info = env.step(0)

        # Último paso debe terminar
        assert terminated or truncated

    def test_info_dict(self, env):
        """Test de diccionario info."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        required_keys = ["balance", "cash", "equity", "portfolio_value", "position",
                        "position_units", "current_price", "entry_price",
                        "realized_profit", "total_profit",  # Alias temporal
                        "current_step", "n_trades"]

        for key in required_keys:
            assert key in info

    def test_action_space(self, env):
        """Test de espacio de acciones."""
        assert env.action_space.n == 3  # 0=hold, 1=buy, 2=sell

        for action in [0, 1, 2]:
            assert env.action_space.contains(action)

    def test_observation_space(self, env):
        """Test de espacio de observaciones."""
        assert env.observation_space.shape == (env.lookback_window, env.n_features)
        assert env.observation_space.dtype == np.float32

    def test_render(self, env):
        """Test de renderizado."""
        env.reset()
        env.render()  # No debe lanzar error

    def test_close(self, env):
        """Test de cierre del entorno."""
        env.reset()
        env.close()  # No debe lanzar error

    def test_buy_creates_position_units(self, env):
        """Test de que buy crea position_units > 0."""
        env.reset()

        # Estado inicial
        assert env.position_units == 0.0
        assert env.cash == env.initial_balance

        # Ejecutar buy
        obs, reward, terminated, truncated, info = env.step(1)  # Buy

        # Verificar que se creó posición
        assert env.position_units > 0.0
        assert env.cash == 0.0  # Todo el cash invertido
        assert env.position == 1
        assert info["position_units"] > 0.0
        assert info["cash"] == 0.0

    def test_hold_increases_equity_with_rising_price(self, env):
        """Test de que hold con precio creciente aumenta equity."""
        env.reset()

        # Comprar al inicio
        obs, reward, terminated, truncated, info = env.step(1)  # Buy
        initial_equity = info["equity"]

        # Hold mientras el precio sube
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(0)  # Hold
            current_equity = info["equity"]

            # Equity debe aumentar con el precio
            assert current_equity >= initial_equity

    def test_sell_closes_position(self, env):
        """Test de que sell cierra posición y vuelve a cash."""
        env.reset()

        # Comprar
        obs, reward, terminated, truncated, info = env.step(1)  # Buy
        assert env.position == 1
        assert env.position_units > 0.0

        # Vender
        obs, reward, terminated, truncated, info = env.step(2)  # Sell

        # Verificar que se cerró la posición
        assert env.position == 0
        assert env.position_units == 0.0
        assert env.cash > 0.0  # Cash disponible después de vender
        assert info["position"] == 0
        assert info["position_units"] == 0.0
        assert info["cash"] > 0.0

    def test_equity_calculation(self, env):
        """Test de cálculo correcto de equity."""
        env.reset()

        # Equity inicial = cash
        initial_equity = env._get_equity()
        assert initial_equity == env.cash == env.initial_balance

        # Después de buy: equity = position_units * price
        obs, reward, terminated, truncated, info = env.step(1)  # Buy
        expected_equity = env.position_units * info["current_price"]
        assert abs(env._get_equity() - expected_equity) < 0.01

        # Después de hold: equity debe cambiar con el precio
        obs, reward, terminated, truncated, info = env.step(0)  # Hold
        expected_equity = env.position_units * info["current_price"]
        assert abs(env._get_equity() - expected_equity) < 0.01

    def test_trade_history_updates(self, env):
        """Test de que trade_history se actualiza correctamente."""
        env.reset()

        # Inicialmente sin trades
        assert len(env.trade_history) == 0

        # Buy debe añadir trade
        obs, reward, terminated, truncated, info = env.step(1)  # Buy
        assert len(env.trade_history) == 1
        assert env.trade_history[0]["type"] == "buy"
        assert env.trade_history[0]["units"] > 0

        # Hold no añade trade
        obs, reward, terminated, truncated, info = env.step(0)  # Hold
        assert len(env.trade_history) == 1

        # Sell debe añadir trade
        obs, reward, terminated, truncated, info = env.step(2)  # Sell
        assert len(env.trade_history) == 2
        assert env.trade_history[1]["type"] == "sell"

    def test_buy_when_already_long(self, env):
        """Test de que buy cuando ya es long se comporta como hold."""
        env.reset()

        # Primer buy
        obs, reward1, terminated, truncated, info1 = env.step(1)  # Buy
        position_after_first_buy = env.position_units
        equity_after_first_buy = info1["equity"]

        # Segundo buy (debe comportarse como hold)
        obs, reward2, terminated, truncated, info2 = env.step(1)  # Buy
        assert env.position_units == position_after_first_buy  # No cambia
        assert env.position == 1  # Sigue long

    def test_sell_when_no_position(self, env):
        """Test de que sell sin posición se comporta como hold."""
        env.reset()

        # Sell sin posición (debe comportarse como hold)
        obs, reward, terminated, truncated, info = env.step(2)  # Sell
        assert env.position == 0  # Sigue sin posición
        assert env.position_units == 0.0
        assert env.cash == env.initial_balance  # Cash sin cambios

    def test_buy_and_hold_positive_return(self):
        """Test de que BuyAndHoldAgent en datos crecientes obtiene retorno positivo."""
        from latent_rl.agents import BuyAndHoldAgent

        # Crear datos con tendencia alcista clara
        np.random.seed(42)
        n_steps = 50
        trend = np.linspace(0, 30, n_steps)  # Tendencia alcista
        noise = np.random.randn(n_steps) * 1
        prices = 100 + trend + noise

        data = pd.DataFrame({
            "Open": prices + np.random.randn(n_steps),
            "High": prices + np.random.randn(n_steps) + 1,
            "Low": prices + np.random.randn(n_steps) - 1,
            "Close": prices,
            "Volume": np.random.randint(1000, 10000, n_steps)
        })

        # Crear entorno y agente
        env = FinancialEnv(data, lookback_window=5, initial_balance=10000)
        agent = BuyAndHoldAgent(env.action_space)

        # Ejecutar episodio
        obs, info = env.reset(seed=42)
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Verificar retorno positivo
        final_equity = info["equity"]
        total_return = (final_equity - env.initial_balance) / env.initial_balance

        assert total_return > 0, f"Retorno debe ser positivo, fue {total_return:.2%}"
        assert final_equity > env.initial_balance

        env.close()

    def test_realized_profit_single_trade(self):
        """Un ciclo buy→sell produce realized_profit correcto."""
        np.random.seed(0)
        prices = np.full(20, 100.0)
        prices[10:] = 110.0  # sube a 110 en el paso 10
        data = pd.DataFrame({
            "Open": prices, "High": prices + 1, "Low": prices - 1,
            "Close": prices, "Volume": np.ones(20) * 1000
        })
        env = FinancialEnv(data, initial_balance=10000.0, transaction_cost=0.0, lookback_window=2)
        env.reset()

        # Buy en paso 2 (precio 100)
        env.step(1)
        assert env._cash_at_last_buy == pytest.approx(10000.0)

        # Hold hasta que el precio sube
        for _ in range(8):
            env.step(0)

        # Sell en precio 110
        env.step(2)

        expected_profit = env.position_units_before_sell if hasattr(env, 'position_units_before_sell') else None
        # profit = units * 110 - cash_invested = (10000/100)*110 - 10000 = 11000 - 10000 = 1000
        assert env.realized_profit == pytest.approx(1000.0, rel=1e-4)
        assert env._cash_at_last_buy == pytest.approx(0.0)

    def test_realized_profit_two_trades_accumulates(self):
        """Dos ciclos buy→sell acumulan realized_profit correctamente."""
        prices = np.array([100.0] * 5 + [110.0] * 5 + [110.0] * 5 + [120.0] * 5)
        data = pd.DataFrame({
            "Open": prices, "High": prices + 1, "Low": prices - 1,
            "Close": prices, "Volume": np.ones(20) * 1000
        })
        env = FinancialEnv(data, initial_balance=10000.0, transaction_cost=0.0, lookback_window=2)
        env.reset()

        # Trade 1: buy a 100, sell a 110
        env.step(1)   # buy
        for _ in range(3):
            env.step(0)   # hold
        env.step(2)   # sell a 110
        profit_after_trade1 = env.realized_profit
        assert profit_after_trade1 > 0

        # Trade 2: buy a 110, sell a 120
        env.step(1)   # buy
        cash_invested_trade2 = env._cash_at_last_buy
        for _ in range(3):
            env.step(0)   # hold
        env.step(2)   # sell a 120
        profit_after_trade2 = env.realized_profit

        # realized_profit debe ser la suma de ambos trades
        expected_trade2_profit = env.cash - cash_invested_trade2
        assert profit_after_trade2 == pytest.approx(profit_after_trade1 + expected_trade2_profit, rel=1e-4)

    def test_realized_profit_losing_trade(self):
        """Un trade perdedor produce realized_profit negativo."""
        prices = np.array([100.0] * 5 + [80.0] * 5 + [80.0] * 10)
        data = pd.DataFrame({
            "Open": prices, "High": prices + 1, "Low": prices - 1,
            "Close": prices, "Volume": np.ones(20) * 1000
        })
        env = FinancialEnv(data, initial_balance=10000.0, transaction_cost=0.0, lookback_window=2)
        env.reset()

        env.step(1)   # buy a 100
        for _ in range(3):
            env.step(0)
        env.step(2)   # sell a 80

        # profit = 10000*(80/100) - 10000 = -2000
        assert env.realized_profit == pytest.approx(-2000.0, rel=1e-4)

    def test_realized_profit_resets_on_env_reset(self):
        """env.reset() reinicia realized_profit y _cash_at_last_buy a 0."""
        data = pd.DataFrame({
            "Open": [100.0] * 10, "High": [101.0] * 10, "Low": [99.0] * 10,
            "Close": [100.0] * 10, "Volume": [1000.0] * 10
        })
        env = FinancialEnv(data, initial_balance=10000.0, transaction_cost=0.0, lookback_window=2)
        env.reset()

        env.step(1)   # buy
        env.step(2)   # sell
        assert env.realized_profit != 0 or env.position == 0

        env.reset()
        assert env.realized_profit == pytest.approx(0.0)
        assert env._cash_at_last_buy == pytest.approx(0.0)

    def test_realized_profit_zero_when_position_never_closed(self):
        """Si el agente compra y nunca vende, realized_profit es 0 al final del episodio."""
        prices = np.linspace(100, 120, 20)
        data = pd.DataFrame({
            "Open": prices, "High": prices + 1, "Low": prices - 1,
            "Close": prices, "Volume": np.ones(20) * 1000
        })
        env = FinancialEnv(data, initial_balance=10000.0, transaction_cost=0.0, lookback_window=2)
        env.reset()

        # Buy y luego solo hold hasta el final
        env.step(1)
        done = False
        info = {}
        while not done:
            _, _, terminated, truncated, info = env.step(0)
            done = terminated or truncated

        # Posición abierta → realized_profit = 0, pero equity > initial_balance
        assert info["realized_profit"] == pytest.approx(0.0)
        assert info["equity"] > env.initial_balance


class TestFinancialEnvFeatureCols:
    """Tests para feature_cols y price_col (Parte 0 del spec)."""

    @pytest.fixture
    def rich_data(self):
        """DataFrame con OHLCV + features adicionales."""
        np.random.seed(0)
        n = 20
        return pd.DataFrame({
            "open":  np.full(n, 100.0),
            "high":  np.full(n, 105.0),
            "low":   np.full(n, 95.0),
            "close": np.linspace(100, 110, n),
            "volume": np.full(n, 1000.0),
            "rsi_14":       np.random.rand(n) * 100,
            "log_return":   np.random.randn(n) * 0.01,
        })

    def test_observation_shape_with_feature_cols(self, rich_data):
        """observation_space.shape debe ser (L, len(feature_cols)) cuando se especifican."""
        L = 5
        feature_cols = ["rsi_14", "log_return"]
        env = FinancialEnv(rich_data, lookback_window=L, feature_cols=feature_cols)
        assert env.observation_space.shape == (L, len(feature_cols))
        obs, _ = env.reset()
        assert obs.shape == (L, len(feature_cols))

    def test_observation_excludes_ohlcv_when_feature_cols_given(self, rich_data):
        """Con feature_cols, la obs NO incluye columnas OHLCV."""
        L = 3
        feature_cols = ["rsi_14"]
        env = FinancialEnv(rich_data, lookback_window=L, feature_cols=feature_cols)
        obs, _ = env.reset()
        # La observación debe tener exactamente 1 columna (rsi_14)
        assert obs.shape == (L, 1)
        # Verificar que los valores de la obs corresponden a rsi_14
        expected = rich_data["rsi_14"].values[0:L].reshape(L, 1)
        assert np.allclose(obs, expected)

    def test_price_col_used_for_execution(self, rich_data):
        """El precio de ejecución refleja price_col, no el índice 3 hardcoded."""
        env = FinancialEnv(
            rich_data, lookback_window=3, price_col="close",
            feature_cols=["rsi_14"]
        )
        env.reset()
        _, _, _, _, info = env.step(0)
        # El precio actual debe venir de close, que comienza en ~100 y sube a 110
        assert 99.0 < info["current_price"] < 111.0

    def test_auto_detects_non_ohlcv_features(self, rich_data):
        """Sin feature_cols, el entorno usa automáticamente las columnas no-OHLCV."""
        env = FinancialEnv(rich_data, lookback_window=3)
        # rsi_14 y log_return son las únicas no-OHLCV
        assert env.n_features == 2
        assert env.observation_space.shape == (3, 2)

    def test_price_col_fallback_when_missing(self):
        """Si price_col no existe en el DataFrame, se usa el índice 3 como fallback."""
        data = pd.DataFrame({
            "Open": [100.0] * 10,
            "High": [105.0] * 10,
            "Low":  [95.0] * 10,
            "Close": [102.0] * 10,
            "Volume": [1000.0] * 10,
        })
        # price_col="close" (minúsculas) no existe → fallback a índice 3
        env = FinancialEnv(data, lookback_window=2, price_col="close")
        env.reset()
        _, _, _, _, info = env.step(0)
        # El precio debe ser Close (índice 3), que es 102.0
        assert info["current_price"] == pytest.approx(102.0)

    def test_column_order_irrelevant_for_equity(self, rich_data):
        """El cálculo de equity no depende del orden de columnas, sólo de price_col."""
        # Crear versión con columnas en orden diferente
        cols_reversed = list(rich_data.columns[::-1])
        data_reversed = rich_data[cols_reversed]

        env = FinancialEnv(data_reversed, lookback_window=3,
                           price_col="close", feature_cols=["rsi_14"])
        env.reset()
        env.step(1)  # buy
        _, _, _, _, info = env.step(0)  # hold
        # El precio debe seguir siendo close, independientemente del orden
        assert 99.0 < info["current_price"] < 111.0


class TestBaseFinancialEnv:
    """Tests para BaseFinancialEnv."""

    def test_abstract_methods(self):
        """Test de que los métodos abstractos deben ser implementados."""
        from gymnasium import spaces

        obs_space = spaces.Box(low=0, high=1, shape=(10,))
        action_space = spaces.Discrete(3)

        env = BaseFinancialEnv(obs_space, action_space)

        with pytest.raises(NotImplementedError):
            env.step(0)

        with pytest.raises(NotImplementedError):
            env.reset()