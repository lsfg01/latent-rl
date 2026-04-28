"""Entorno financiero concreto compatible con Gymnasium."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from latent_rl.envs.base import BaseFinancialEnv


class FinancialEnv(BaseFinancialEnv):
    """Entorno financiero para trading algorítmico."""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        lookback_window: int = 10,
        render_mode: Optional[str] = None
    ):
        """
        Inicializa el entorno financiero.

        Args:
            data: DataFrame con datos OHLCV
            initial_balance: Balance inicial del agente
            transaction_cost: Coste de transacción (porcentaje)
            lookback_window: Ventana de observación histórica
            render_mode: Modo de renderizado
        """
        # Validar datos
        if len(data) < lookback_window:
            raise ValueError(f"Datos insuficientes: se necesitan al menos {lookback_window} filas")

        self.data = data.values
        self.n_features = data.shape[1]
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window

        # Espacio de observaciones: ventana histórica de features
        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_window, self.n_features),
            dtype=np.float32
        )

        # Espacio de acciones: 0=hold, 1=buy, 2=sell
        action_space = spaces.Discrete(3)

        super().__init__(observation_space=obs_space, action_space=action_space, render_mode=render_mode)

        # Estado interno
        self.current_step = 0
        self.cash = initial_balance
        self.position_units = 0.0
        self.position = 0  # 0=flat, 1=long, -1=short (no implementado)
        self.entry_price = 0.0
        self.realized_profit = 0.0  # Profit/pérdida realizada tras cerrar posición
        self.trade_history = []
        self.previous_equity = initial_balance

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """
        Resetea el entorno al estado inicial.

        Args:
            seed: Semilla aleatoria
            options: Opciones adicionales

        Returns:
            tuple: (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)

        self.current_step = self.lookback_window
        self.cash = self.initial_balance
        self.position_units = 0.0
        self.position = 0
        self.entry_price = 0.0
        self.realized_profit = 0.0
        self.trade_history = []
        self.previous_equity = self.initial_balance

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Ejecuta un paso del entorno.

        Args:
            action: Acción a ejecutar (0=hold, 1=buy, 2=sell)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Validar acción
        if not self.action_space.contains(action):
            raise ValueError(f"Acción inválida: {action}")

        # Obtener precio actual
        current_price = self.data[self.current_step, 3]  # Columna Close

        # Ejecutar acción y calcular recompensa
        reward = self._execute_action(action, current_price)

        # Avanzar paso
        self.current_step += 1

        # Verificar condiciones de terminación
        terminated = self.current_step >= len(self.data) - 1
        truncated = False

        # Obtener nueva observación e info
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_equity(self) -> float:
        """
        Calcula el equity total (cash + valor de posición).

        Returns:
            Equity total de la cuenta
        """
        current_price = self.data[self.current_step, 3] if self.current_step < len(self.data) else self.data[-1, 3]

        if self.position == 0:
            return self.cash
        elif self.position == 1:  # Long
            position_value = self.position_units * current_price
            return self.cash + position_value
        else:  # Short (no implementado)
            return self.cash

    def _get_observation(self) -> np.ndarray:
        """Obtiene la observación actual (ventana histórica)."""
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step

        window = self.data[start_idx:end_idx]

        # Si la ventana es más pequeña que lookback_window, pad con ceros
        if len(window) < self.lookback_window:
            padding = np.zeros((self.lookback_window - len(window), self.n_features))
            window = np.vstack([padding, window])

        return window.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Obtiene información adicional del estado actual."""
        current_price = self.data[self.current_step, 3] if self.current_step < len(self.data) else self.data[-1, 3]
        equity = self._get_equity()

        return {
            "balance": self.cash,  # Mantenido por compatibilidad, ahora es cash
            "cash": self.cash,
            "equity": equity,
            "portfolio_value": equity,
            "position": self.position,
            "position_units": self.position_units,
            "current_price": current_price,
            "entry_price": self.entry_price,
            "realized_profit": self.realized_profit,  # Profit/pérdida realizada
            "total_profit": self.realized_profit,  # Alias temporal por compatibilidad
            "current_step": self.current_step,
            "n_trades": len(self.trade_history)
        }

    def _execute_action(self, action: int, current_price: float) -> float:
        """
        Ejecuta una acción y calcula la recompensa.

        Args:
            action: Acción a ejecutar
            current_price: Precio actual

        Returns:
            Recompensa obtenida (cambio de equity normalizado)
        """
        current_equity = self._get_equity()

        # Acción 0: Hold
        if action == 0:
            # Mantener posición actual, no se modifica nada
            pass

        # Acción 1: Buy
        elif action == 1:
            if self.position == 0:
                # Abrir posición long: invertir todo el cash disponible
                cost = self.cash * self.transaction_cost
                cash_after_cost = self.cash - cost
                self.position_units = cash_after_cost / current_price
                self.cash = 0.0
                self.position = 1
                self.entry_price = current_price

                # Registrar trade
                self.trade_history.append({
                    "type": "buy",
                    "price": current_price,
                    "units": self.position_units,
                    "step": self.current_step
                })

            elif self.position == 1:
                # Ya tenemos posición long, mantener (hold)
                pass

        # Acción 2: Sell
        elif action == 2:
            if self.position == 1:
                # Cerrar posición long
                proceeds = self.position_units * current_price
                cost = proceeds * self.transaction_cost
                self.cash = proceeds - cost

                # Calcular profit realizado
                self.realized_profit = self.cash - self.initial_balance

                # Registrar trade
                self.trade_history.append({
                    "type": "sell",
                    "price": current_price,
                    "units": self.position_units,
                    "step": self.current_step
                })

                # Resetear posición
                self.position_units = 0.0
                self.position = 0
                self.entry_price = 0.0

            elif self.position == 0:
                # No hay posición, mantener (hold)
                pass

        # Calcular reward basado en cambio de equity
        new_equity = self._get_equity()
        if self.previous_equity > 0:
            reward = (new_equity - self.previous_equity) / self.initial_balance
        else:
            reward = 0.0

        # Actualizar equity anterior para siguiente paso
        self.previous_equity = new_equity

        return reward

    def _calculate_pnl(self, current_price: float) -> float:
        """
        Calcula el profit/loss de la posición actual.

        Args:
            current_price: Precio actual

        Returns:
            Profit/loss calculado
        """
        if self.position == 0:
            return 0.0

        if self.position == 1:  # Long
            return (current_price - self.entry_price) * self.position_units
        else:  # Short (no implementado)
            return 0.0

    def render(self):
        """Renderiza el entorno (no implementado en MVP)."""
        if self.render_mode is None:
            return None

        # En MVP, simplemente imprimir estado
        info = self._get_info()
        print(f"Step: {info['current_step']}, Cash: ${info['cash']:.2f}, "
              f"Equity: ${info['equity']:.2f}, Position: {info['position']}, "
              f"Units: {info['position_units']:.4f}, Total Profit: ${info['total_profit']:.2f}")

    def close(self):
        """Cierra el entorno."""
        pass