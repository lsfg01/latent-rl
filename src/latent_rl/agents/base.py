"""Clase base abstracta para agentes de aprendizaje por refuerzo."""

from abc import ABC, abstractmethod
from typing import Any
from gymnasium import spaces


class BaseAgent(ABC):
    """Clase base abstracta para todos los agentes."""

    def __init__(self, action_space: spaces.Space):
        """
        Inicializa el agente.

        Args:
            action_space: Espacio de acciones de Gymnasium
        """
        self.action_space = action_space

    @abstractmethod
    def select_action(self, observation: Any) -> Any:
        """
        Selecciona una acción dado una observación.

        Args:
            observation: Observación del entorno

        Returns:
            Acción seleccionada
        """
        pass

    def reset(self) -> None:
        """Resetea el estado interno del agente."""
        pass