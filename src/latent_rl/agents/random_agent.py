"""Agente aleatorio para línea base."""

from typing import Any, Optional
from gymnasium import spaces
from latent_rl.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """Agente que selecciona acciones aleatorias."""

    def __init__(self, action_space: spaces.Space, seed: Optional[int] = None):
        """
        Inicializa el agente aleatorio.

        Args:
            action_space: Espacio de acciones de Gymnasium
            seed: Semilla aleatoria opcional para reproducibilidad
        """
        super().__init__(action_space)
        self.seed = seed

        if seed is not None:
            self.action_space.seed(seed)

    def select_action(self, observation: Any) -> Any:
        """
        Selecciona una acción aleatoria del action_space.

        Args:
            observation: Observación del entorno (ignorada)

        Returns:
            Acción aleatoria del espacio de acciones
        """
        return self.action_space.sample()

    def reset(self) -> None:
        """Resetea el generador aleatorio con la semilla original."""
        if self.seed is not None:
            self.action_space.seed(self.seed)