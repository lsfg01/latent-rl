"""Agente DQN simple para aprendizaje por refuerzo."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
from gymnasium import spaces

from latent_rl.agents.base import BaseAgent
from latent_rl.agents.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """Red neuronal Q simple MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        """
        Inicializa la red Q.

        Args:
            input_dim: Dimensión de entrada (observación aplanada)
            output_dim: Dimensión de salida (número de acciones)
            hidden_dim: Dimensión de las capas ocultas
        """
        super(QNetwork, self).__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red.

        Args:
            x: Tensor de entrada (batch, *observation_shape)

        Returns:
            Tensor de salida (batch, n_actions)
        """
        x = self.flatten(x)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class DQNAgent(BaseAgent):
    """Agente DQN simple para aprendizaje por refuerzo."""

    def __init__(
        self,
        action_space: spaces.Discrete,
        observation_shape: tuple,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 10000,
        target_update_freq: int = 100,
        hidden_dim: int = 128,
        device: str = "cpu",
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        dropout: float = 0.0,
    ):
        """
        Inicializa el agente DQN.

        Args:
            action_space: Espacio de acciones (Discrete)
            observation_shape: Forma de las observaciones
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            epsilon_start: Valor inicial de epsilon
            epsilon_end: Valor final de epsilon
            epsilon_decay: Factor de decaimiento de epsilon
            batch_size: Tamaño del batch de entrenamiento
            buffer_capacity: Capacidad del replay buffer
            target_update_freq: Frecuencia de actualización de target network
            hidden_dim: Dimensión de las capas ocultas
            device: Dispositivo de PyTorch ("cpu" o "cuda")
        """
        super().__init__(action_space)

        self.observation_shape = observation_shape
        self.n_actions = action_space.n
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm

        # Calcular dimensión de entrada (aplanar observación)
        self.input_dim = int(np.prod(observation_shape))

        # Crear redes Q
        self.q_network = QNetwork(
            self.input_dim, self.n_actions, hidden_dim, dropout=dropout
        ).to(self.device)
        self.target_network = QNetwork(
            self.input_dim, self.n_actions, hidden_dim, dropout=dropout
        ).to(self.device)

        # Inicializar target network con los mismos pesos
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizador y loss
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.SmoothL1Loss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, observation_shape)

        # Contador de pasos para actualización de target network
        self.update_step = 0

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Selecciona una acción usando política epsilon-greedy.

        Args:
            observation: Observación actual
            training: Si es True, usa epsilon-greedy; si False, usa greedy

        Returns:
            Acción seleccionada
        """
        # Decaimiento de epsilon durante entrenamiento
        if training:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Selección epsilon-greedy
        if training and np.random.random() < self.epsilon:
            # Acción aleatoria
            return self.action_space.sample()
        else:
            # Acción greedy
            was_training = self.q_network.training
            if not training:
                self.q_network.eval()
            try:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                    q_values = self.q_network(obs_tensor)
                    return q_values.argmax().item()
            finally:
                self.q_network.train(was_training)

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                       next_state: np.ndarray, done: bool) -> None:
        """
        Almacena una transición en el replay buffer.

        Args:
            state: Estado actual
            action: Acción ejecutada
            reward: Recompensa obtenida
            next_state: Estado siguiente
            done: Si el episodio terminó
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """
        Actualiza la red Q usando un batch del replay buffer.

        Returns:
            dict: Métricas de entrenamiento (loss, q_values, etc.)
        """
        # Verificar si hay suficientes datos
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "q_mean": 0.0, "q_std": 0.0}

        # Muestrear batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convertir a tensores
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Calcular Q-values actuales
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Calcular Q-values target
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Calcular loss
        loss = self.criterion(q_values, target_q_values)

        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), self.grad_clip_norm
            )
        self.optimizer.step()

        # Actualizar target network periódicamente
        self.update_step += 1
        if self.update_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Calcular métricas
        with torch.no_grad():
            all_q_values = self.q_network(states)
            q_mean = all_q_values.mean().item()
            q_std = all_q_values.std().item()

        return {
            "loss": loss.item(),
            "q_mean": q_mean,
            "q_std": q_std
        }

    def reset(self) -> None:
        """Resetea el estado interno del agente."""
        self.epsilon = self.epsilon_start
        self.update_step = 0
        # Nota: No reseteamos el replay buffer para acumular experiencia

    def save(self, path: str) -> None:
        """
        Guarda el modelo del agente.

        Args:
            path: Ruta donde guardar el modelo
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_step': self.update_step
        }, path)

    def load(self, path: str) -> None:
        """
        Carga el modelo del agente.

        Args:
            path: Ruta desde donde cargar el modelo
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_step = checkpoint['update_step']

    def set_epsilon(self, epsilon: float) -> None:
        """
        Establece el valor de epsilon manualmente.

        Args:
            epsilon: Nuevo valor de epsilon
        """
        self.epsilon = epsilon
