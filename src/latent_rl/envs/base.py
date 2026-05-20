"""Entorno financiero base compatible con Gymnasium."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional


class BaseFinancialEnv(gym.Env):
    """Entorno financiero base compatible con Gymnasium."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        render_mode: Optional[str] = None
    ):
        """
        Inicializa el entorno base.

        Args:
            observation_space: Espacio de observaciones
            action_space: Espacio de acciones
            render_mode: Modo de renderizado ('human', 'rgb_array', None)
        """
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.render_mode = render_mode

    def step(self, action):
        """
        Ejecuta un paso del entorno.

        Args:
            action: Acción a ejecutar

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        raise NotImplementedError("El método step debe ser implementado por la subclase")

    def reset(self, seed=None, options=None):
        """
        Resetea el entorno.

        Args:
            seed: Semilla aleatoria
            options: Opciones adicionales

        Returns:
            tuple: (observation, info)
        """
        raise NotImplementedError("El método reset debe ser implementado por la subclase")

    def render(self):
        """Renderiza el entorno."""
        if self.render_mode is None:
            return None

        raise NotImplementedError("El método render debe ser implementado por la subclase")

    def close(self):
        """Cierra el entorno."""
        pass