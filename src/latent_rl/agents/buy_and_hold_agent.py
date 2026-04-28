"""Agente buy-and-hold para línea base financiera."""

from typing import Any
from gymnasium import spaces
from latent_rl.agents.base import BaseAgent


class BuyAndHoldAgent(BaseAgent):
    """Agente que implementa estrategia buy-and-hold."""

    def __init__(self, action_space: spaces.Space):
        """
        Inicializa el agente buy-and-hold.

        Args:
            action_space: Espacio de acciones de Gymnasium

        Raises:
            ValueError: Si el action_space no permite las acciones 0 y 1
        """
        super().__init__(action_space)

        # Validar que el action_space permita las acciones necesarias
        if not (action_space.contains(0) and action_space.contains(1)):
            raise ValueError(
                f"action_space debe permitir las acciones 0 (hold) y 1 (buy). "
                f"Acciones permitidas: {self._get_allowed_actions()}"
            )

        self.has_bought = False

    def _get_allowed_actions(self) -> str:
        """Obtiene una representación de las acciones permitidas."""
        if hasattr(self.action_space, 'n'):
            return f"0-{self.action_space.n - 1}"
        else:
            return "desconocido"

    def select_action(self, observation: Any) -> Any:
        """
        Selecciona acción según estrategia buy-and-hold.

        Lógica:
        - Si no ha comprado aún: acción 1 (buy)
        - Si ya compró: acción 0 (hold)

        Args:
            observation: Observación del entorno (ignorada)

        Returns:
            Acción 1 (buy) o 0 (hold)
        """
        if not self.has_bought:
            self.has_bought = True
            return 1  # Buy
        else:
            return 0  # Hold

    def reset(self) -> None:
        """Resetea el estado interno del agente."""
        self.has_bought = False