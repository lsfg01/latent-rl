"""Entorno financiero concreto compatible con Gymnasium."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from latent_rl.envs.base import BaseFinancialEnv

# Columnas OHLCV estándar (minúsculas)
_OHLCV_COLS = {"open", "high", "low", "close", "volume"}


class FinancialEnv(BaseFinancialEnv):
    """Entorno financiero para trading algorítmico.

    La observación contiene únicamente las columnas de features (estacionarias,
    normalizadas por la capa de datos).  El precio de ejecución se resuelve
    por nombre de columna (``price_col``), no por índice fijo.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        lookback_window: int = 10,
        render_mode: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        price_col: str = "close",
        reward_mode: str = "equity_delta_initial",
        reward_clip: Optional[float] = None,
        trade_penalty: float = 0.0,
    ):
        """
        Inicializa el entorno financiero.

        Args:
            data: DataFrame con datos OHLCV (+ features opcionales).
            initial_balance: Balance inicial del agente.
            transaction_cost: Coste de transacción (porcentaje).
            lookback_window: Ventana de observación histórica.
            render_mode: Modo de renderizado.
            feature_cols: Columnas que forman la observación. Si None, usa
                todas las columnas no-OHLCV; si no hay ninguna, usa OHLCV
                completo (comportamiento heredado).
            price_col: Columna de precio para ejecución/equity (default "close").
            reward_mode: "equity_delta_initial" (comportamiento heredado) o
                "log_return" para usar el log-retorno del equity.
            reward_clip: Si no es None, limita la recompensa a
                ``[-reward_clip, reward_clip]``.
            trade_penalty: Penalización aplicada solo a operaciones ejecutadas.
        """
        if len(data) < lookback_window:
            raise ValueError(
                f"Datos insuficientes: se necesitan al menos {lookback_window} filas"
            )
        if reward_mode not in {"equity_delta_initial", "log_return"}:
            raise ValueError(
                "reward_mode debe ser 'equity_delta_initial' o 'log_return', "
                f"got {reward_mode!r}"
            )
        if reward_clip is not None and reward_clip <= 0:
            raise ValueError("reward_clip debe ser > 0 cuando se especifica")
        if trade_penalty < 0:
            raise ValueError("trade_penalty debe ser >= 0")

        self.data = data.values
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.reward_mode = reward_mode
        self.reward_clip = reward_clip
        self.trade_penalty = trade_penalty

        # --- Resolver índice de precio -------------------------------------------
        if price_col in data.columns:
            self._price_idx = data.columns.get_loc(price_col)
        else:
            # Fallback: índice 3 (posición clásica de Close en OHLCV)
            self._price_idx = 3

        # --- Resolver índices de features ----------------------------------------
        if feature_cols is not None:
            self._feature_idx = [data.columns.get_loc(c) for c in feature_cols]
        else:
            # Usar columnas no-OHLCV si existen, sino todas
            non_ohlcv = [
                c for c in data.columns if c.lower() not in _OHLCV_COLS
            ]
            if non_ohlcv:
                self._feature_idx = [data.columns.get_loc(c) for c in non_ohlcv]
            else:
                self._feature_idx = list(range(data.shape[1]))

        self.n_features = len(self._feature_idx)

        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_window, self.n_features),
            dtype=np.float32,
        )
        action_space = spaces.Discrete(3)

        super().__init__(
            observation_space=obs_space,
            action_space=action_space,
            render_mode=render_mode,
        )

        # Estado interno
        self.current_step = 0
        self.cash = initial_balance
        self.position_units = 0.0
        self.position = 0
        self.entry_price = 0.0
        self.realized_profit = 0.0
        self._cash_at_last_buy = 0.0
        self.trade_history: List[Dict] = []
        self.previous_equity = initial_balance

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        # BaseFinancialEnv.reset es abstracto y lanza NotImplementedError;
        # llamamos directamente al reset de Gymnasium para inicializar np_random.
        gym.Env.reset(self, seed=seed)
        options = options or {}

        self.current_step = self.lookback_window
        if options.get("random_start", False):
            last_valid_start = len(self.data) - 2
            max_steps = options.get("max_steps")
            if max_steps is not None:
                max_steps = max(int(max_steps), 0)
                preferred_max = last_valid_start - max_steps
            else:
                preferred_max = last_valid_start

            start_max = (
                preferred_max
                if preferred_max >= self.lookback_window
                else last_valid_start
            )
            if start_max >= self.lookback_window:
                self.current_step = int(
                    self.np_random.integers(self.lookback_window, start_max + 1)
                )
        self.cash = self.initial_balance
        self.position_units = 0.0
        self.position = 0
        self.entry_price = 0.0
        self.realized_profit = 0.0
        self._cash_at_last_buy = 0.0
        self.trade_history = []
        self.previous_equity = self.initial_balance

        return self._get_observation(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Acción inválida: {action}")

        current_price = self._current_price()
        reward = self._execute_action(action, current_price)

        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _current_price(self) -> float:
        idx = min(self.current_step, len(self.data) - 1)
        return float(self.data[idx, self._price_idx])

    def _get_equity(self) -> float:
        price = self._current_price()
        if self.position == 1:
            return self.cash + self.position_units * price
        return self.cash

    def _get_observation(self) -> np.ndarray:
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        window = self.data[start_idx:end_idx][:, self._feature_idx]

        if len(window) < self.lookback_window:
            padding = np.zeros(
                (self.lookback_window - len(window), self.n_features)
            )
            window = np.vstack([padding, window])

        return window.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        price = self._current_price()
        equity = self._get_equity()
        return {
            "balance": self.cash,
            "cash": self.cash,
            "equity": equity,
            "portfolio_value": equity,
            "position": self.position,
            "position_units": self.position_units,
            "current_price": price,
            "entry_price": self.entry_price,
            "realized_profit": self.realized_profit,
            "total_profit": self.realized_profit,
            "current_step": self.current_step,
            "n_trades": len(self.trade_history),
        }

    def _execute_action(self, action: int, current_price: float) -> float:
        trade_executed = False

        if action == 1:  # Buy
            if self.position == 0:
                self._cash_at_last_buy = self.cash
                cost = self.cash * self.transaction_cost
                self.position_units = (self.cash - cost) / current_price
                self.cash = 0.0
                self.position = 1
                self.entry_price = current_price
                self.trade_history.append(
                    {"type": "buy", "price": current_price,
                     "units": self.position_units, "step": self.current_step}
                )
                trade_executed = True

        elif action == 2:  # Sell
            if self.position == 1:
                proceeds = self.position_units * current_price
                cost = proceeds * self.transaction_cost
                self.cash = proceeds - cost
                self.realized_profit += self.cash - self._cash_at_last_buy
                self._cash_at_last_buy = 0.0
                self.trade_history.append(
                    {"type": "sell", "price": current_price,
                     "units": self.position_units, "step": self.current_step}
                )
                self.position_units = 0.0
                self.position = 0
                self.entry_price = 0.0
                trade_executed = True

        new_equity = self._get_equity()
        if self.reward_mode == "log_return":
            eps = 1e-8
            reward = float(
                np.log((new_equity + eps) / (self.previous_equity + eps))
            )
        else:
            reward = (
                (new_equity - self.previous_equity) / self.initial_balance
                if self.previous_equity > 0
                else 0.0
            )

        if trade_executed:
            reward -= self.trade_penalty
        if self.reward_clip is not None:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        self.previous_equity = new_equity
        return float(reward)

    def _calculate_pnl(self, current_price: float) -> float:
        if self.position == 1:
            return (current_price - self.entry_price) * self.position_units
        return 0.0

    def render(self):
        if self.render_mode is None:
            return None
        info = self._get_info()
        print(
            f"Step: {info['current_step']}, Cash: ${info['cash']:.2f}, "
            f"Equity: ${info['equity']:.2f}, Position: {info['position']}, "
            f"Units: {info['position_units']:.4f}, "
            f"Total Profit: ${info['total_profit']:.2f}"
        )

    def close(self):
        pass
