"""Agente DQN con encoder latente para aprendizaje por refuerzo."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
from gymnasium import spaces

from latent_rl.agents.base import BaseAgent
from latent_rl.agents.replay_buffer import ReplayBuffer
from latent_rl.representations import MLPLatentEncoder


class LatentQNetwork(nn.Module):
    """Red neuronal Q que opera sobre representaciones latentes."""

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 128):
        """
        Inicializa la red Q latente.

        Args:
            latent_dim: Dimensión de la representación latente
            output_dim: Dimensión de salida (número de acciones)
            hidden_dim: Dimensión de las capas ocultas
        """
        super(LatentQNetwork, self).__init__()

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red.

        Args:
            x: Tensor de entrada (batch, latent_dim)

        Returns:
            Tensor de salida (batch, n_actions)
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LatentDQNAgent(BaseAgent):
    """Agente DQN con encoder latente para aprendizaje por refuerzo."""

    def __init__(
        self,
        action_space: spaces.Discrete,
        observation_shape: tuple,
        latent_dim: int = 16,
        encoder_hidden_dims: Optional[list] = None,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 10000,
        target_update_freq: int = 100,
        q_hidden_dim: int = 128,
        device: str = "cpu"
    ):
        """
        Inicializa el agente LatentDQN.

        Args:
            action_space: Espacio de acciones (Discrete)
            observation_shape: Forma de las observaciones
            latent_dim: Dimensión de la representación latente
            encoder_hidden_dims: Dimensiones de capas ocultas del encoder
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            epsilon_start: Valor inicial de epsilon
            epsilon_end: Valor final de epsilon
            epsilon_decay: Factor de decaimiento de epsilon
            batch_size: Tamaño del batch de entrenamiento
            buffer_capacity: Capacidad del replay buffer
            target_update_freq: Frecuencia de actualización de target network
            q_hidden_dim: Dimensión de las capas ocultas de QNetwork
            device: Dispositivo de PyTorch ("cpu" o "cuda")
        """
        super().__init__(action_space)

        self.observation_shape = observation_shape
        self.n_actions = action_space.n
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Calcular dimensión de entrada (aplanar observación)
        self.input_dim = int(np.prod(observation_shape))

        # Crear encoder latente
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [64, 32]

        self.encoder = MLPLatentEncoder(
            input_dim=self.input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation="relu",
            dropout=0.0
        ).to(self.device)

        # Crear red Q latente
        self.q_network = LatentQNetwork(latent_dim, self.n_actions, q_hidden_dim).to(self.device)
        self.target_network = LatentQNetwork(latent_dim, self.n_actions, q_hidden_dim).to(self.device)

        # Inicializar target network con los mismos pesos
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizador y loss
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.q_network.parameters()),
            lr=learning_rate
        )
        self.criterion = nn.SmoothL1Loss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, observation_shape)

        # Parámetros de normalización (para encoder preentrenado)
        self.normalization_mean = None
        self.normalization_std = None

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
            with torch.no_grad():
                obs_normalized = self._normalize_observation(observation)
                obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0).to(self.device)
                obs_flat = obs_tensor.reshape(obs_tensor.size(0), -1)
                latent = self.encoder(obs_flat)
                q_values = self.q_network(latent)
                return q_values.argmax().item()

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
        Actualiza el encoder y la red Q usando un batch del replay buffer.

        Returns:
            dict: Métricas de entrenamiento (loss, q_values, etc.)
        """
        # Verificar si hay suficientes datos
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "q_mean": 0.0, "q_std": 0.0}

        # Muestrear batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Aplicar normalización si hay parámetros
        if self.normalization_mean is not None and self.normalization_std is not None:
            epsilon = 1e-8
            states = (states - self.normalization_mean) / (self.normalization_std + epsilon)
            next_states = (next_states - self.normalization_mean) / (self.normalization_std + epsilon)

        # Convertir a tensores
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Aplanar observaciones
        states_flat = states.reshape(states.size(0), -1)
        next_states_flat = next_states.reshape(next_states.size(0), -1)

        # Calcular representaciones latentes
        latents = self.encoder(states_flat)
        next_latents = self.encoder(next_states_flat)

        # Calcular Q-values actuales
        q_values = self.q_network(latents).gather(1, actions.unsqueeze(1)).squeeze()

        # Calcular Q-values target
        with torch.no_grad():
            next_q_values = self.target_network(next_latents).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Calcular loss
        loss = self.criterion(q_values, target_q_values)

        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Actualizar target network periódicamente
        self.update_step += 1
        if self.update_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Calcular métricas
        with torch.no_grad():
            all_q_values = self.q_network(latents)
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
            'encoder_state_dict': self.encoder.state_dict(),
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
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
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

    def get_latent_representation(self, observation: np.ndarray) -> np.ndarray:
        """
        Obtiene la representación latente de una observación.

        Args:
            observation: Observación actual

        Returns:
            Representación latente como array numpy
        """
        with torch.no_grad():
            obs_normalized = self._normalize_observation(observation)
            obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0).to(self.device)
            obs_flat = obs_tensor.reshape(obs_tensor.size(0), -1)
            latent = self.encoder(obs_flat)
            return latent.cpu().numpy().squeeze()

    def load_pretrained_encoder(self, path: str) -> None:
        """
        Carga pesos de un encoder preentrenado desde un checkpoint.

        Args:
            path: Ruta del checkpoint del encoder preentrenado
        """
        # Cargar checkpoint con weights_only=True
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        # Verificar compatibilidad de dimensiones
        if 'latent_dim' in checkpoint:
            assert checkpoint['latent_dim'] == self.latent_dim, \
                f"Latent dimension mismatch: checkpoint has {checkpoint['latent_dim']}, agent has {self.latent_dim}"

        if 'input_dim' in checkpoint:
            assert checkpoint['input_dim'] == self.input_dim, \
                f"Input dimension mismatch: checkpoint has {checkpoint['input_dim']}, agent has {self.input_dim}"

        # Cargar pesos del encoder
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

        # Cargar parámetros de normalización si existen
        if 'normalization_mean' in checkpoint and checkpoint['normalization_mean'] is not None:
            # Convertir tensor a numpy si es necesario
            if isinstance(checkpoint['normalization_mean'], torch.Tensor):
                self.normalization_mean = checkpoint['normalization_mean'].cpu().numpy()
            else:
                self.normalization_mean = np.asarray(checkpoint['normalization_mean'], dtype=np.float32)

        if 'normalization_std' in checkpoint and checkpoint['normalization_std'] is not None:
            # Convertir tensor a numpy si es necesario
            if isinstance(checkpoint['normalization_std'], torch.Tensor):
                self.normalization_std = checkpoint['normalization_std'].cpu().numpy()
            else:
                self.normalization_std = np.asarray(checkpoint['normalization_std'], dtype=np.float32)

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        """
        Establece manualmente los parámetros de normalización.

        Args:
            mean: Array de medias con forma (lookback_window, n_features)
            std: Array de desviaciones estándar con forma (lookback_window, n_features)
        """
        self.normalization_mean = mean
        self.normalization_std = std

    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Aplica normalización a una observación si hay parámetros de normalización.

        Args:
            observation: Observación sin normalizar

        Returns:
            Observación normalizada (o original si no hay parámetros)
        """
        if self.normalization_mean is not None and self.normalization_std is not None:
            epsilon = 1e-8
            return (observation - self.normalization_mean) / (self.normalization_std + epsilon)
        return observation
