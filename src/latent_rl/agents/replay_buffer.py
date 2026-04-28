"""Replay Buffer simple para DQN."""

import numpy as np
from typing import Tuple


class ReplayBuffer:
    """Buffer de experiencia simple para DQN."""

    def __init__(self, capacity: int, observation_shape: tuple):
        """
        Inicializa el buffer de experiencia.

        Args:
            capacity: Tamaño máximo del buffer
            observation_shape: Forma de las observaciones
        """
        self.capacity = capacity
        self.observation_shape = observation_shape

        # Arrays para almacenar transiciones
        self.states = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.position = 0
        self.size = 0

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        """
        Añade una transición al buffer.

        Args:
            state: Estado actual
            action: Acción ejecutada
            reward: Recompensa obtenida
            next_state: Estado siguiente
            done: Si el episodio terminó
        """
        # Almacenar en la posición actual
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        # Actualizar posición y tamaño
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Muestrea un batch de transiciones.

        Args:
            batch_size: Tamaño del batch a muestrear

        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        # Muestrear índices aleatorios
        indices = np.random.choice(self.size, size=batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self) -> int:
        """Devuelve el número de transiciones en el buffer."""
        return self.size