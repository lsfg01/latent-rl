"""Agente DQN con encoder latente para aprendizaje por refuerzo."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional
from gymnasium import spaces

from latent_rl.agents.base import BaseAgent
from latent_rl.agents.replay_buffer import ReplayBuffer
from latent_rl.representations.factory import build_encoder


class LatentQNetwork(nn.Module):
    """Red neuronal Q que opera sobre representaciones latentes."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        return self.fc3(x)


class LatentDQNAgent(BaseAgent):
    """Agente DQN con encoder latente.

    El encoder recibe observaciones de forma (N, L, F) directamente (sin aplanar
    externamente).  La capa de datos ya normaliza los features; el agente no
    vuelve a normalizar por defecto (``encoder_internal_norm=False``).

    Parámetros de dos ejes ortogonales del encoder:
    - ``encoder_type``: "mlp" | "tcn" | "gru"
    - ``freeze_encoder``: si True, el encoder se congela y el optimizador
      sólo actualiza la QNetwork.
    """

    def __init__(
        self,
        action_space: spaces.Discrete,
        observation_shape: tuple,
        latent_dim: int = 16,
        encoder_type: str = "mlp",
        encoder_hidden_dims: Optional[List[int]] = None,
        encoder_dropout: float = 0.0,
        encoder_activation: str = "relu",
        # TCN kwargs
        tcn_kernel_size: int = 3,
        tcn_dilations: Optional[List[int]] = None,
        tcn_channels: int = 32,
        # GRU kwargs
        gru_hidden_dim: int = 64,
        gru_num_layers: int = 1,
        # RL hyperparams
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 10000,
        target_update_freq: int = 100,
        q_hidden_dim: int = 128,
        # Encoder usage
        freeze_encoder: bool = False,
        precomputed_latents: bool = False,
        device: str = "cpu",
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        q_dropout: float = 0.0,
    ):
        """
        Args:
            precomputed_latents: Si True, el agente espera que las observaciones
                ya sean vectores latentes de forma (latent_dim,). No construye
                ni ejecuta encoder. Útil con LatentObservationWrapper para
                brazos con encoder congelado (C y D): precomputa los latentes
                una vez fuera del bucle RL.
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
        self.freeze_encoder = freeze_encoder
        self.precomputed_latents = precomputed_latents
        self.grad_clip_norm = grad_clip_norm

        # Dimensiones de la observación (L, F)
        L, F = observation_shape[0], observation_shape[1] if len(observation_shape) > 1 else 1
        self.input_dim = int(np.prod(observation_shape))  # backward compat

        if precomputed_latents:
            # La observación ya es el vector latente; no se construye encoder.
            if len(observation_shape) != 1 or observation_shape[0] != latent_dim:
                raise ValueError(
                    f"precomputed_latents=True requiere observation_shape=({latent_dim},), "
                    f"recibido {observation_shape}"
                )
            self.encoder = None
        else:
            # Construir encoder via factory
            encoder_kwargs: Dict[str, Any] = {
                "input_len": L,
                "n_features": F,
                "latent_dim": latent_dim,
                "activation": encoder_activation,
                "dropout": encoder_dropout,
            }
            if encoder_type == "mlp":
                encoder_kwargs["input_dim"] = self.input_dim
                if encoder_hidden_dims is not None:
                    encoder_kwargs["hidden_dims"] = encoder_hidden_dims
            elif encoder_type == "tcn":
                encoder_kwargs["kernel_size"] = tcn_kernel_size
                if tcn_dilations is not None:
                    encoder_kwargs["dilations"] = tcn_dilations
                encoder_kwargs["channels"] = tcn_channels
            elif encoder_type == "gru":
                encoder_kwargs["hidden_dim"] = gru_hidden_dim
                encoder_kwargs["num_layers"] = gru_num_layers

            self.encoder = build_encoder(encoder_type, **encoder_kwargs).to(self.device)
            if freeze_encoder:
                self.encoder.freeze()

        # Redes Q (idénticas independientemente del modo)
        self.q_network = LatentQNetwork(
            latent_dim, self.n_actions, q_hidden_dim, dropout=q_dropout
        ).to(
            self.device
        )
        self.target_network = LatentQNetwork(
            latent_dim, self.n_actions, q_hidden_dim, dropout=q_dropout
        ).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizador: solo QNetwork si encoder congelado o precomputado
        if precomputed_latents or freeze_encoder:
            trainable_params = list(self.q_network.parameters())
        else:
            trainable_params = list(self.encoder.parameters()) + list(
                self.q_network.parameters()
            )
        self._trainable_params = trainable_params
        self.optimizer = optim.Adam(
            trainable_params, lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(buffer_capacity, observation_shape)

        # Normalización interna (legado; por defecto no se aplica si la capa
        # de datos ya normalizó los features)
        self.normalization_mean = None
        self.normalization_std = None

        self.update_step = 0

    # ------------------------------------------------------------------
    # Política
    # ------------------------------------------------------------------

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        if training:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if training and np.random.random() < self.epsilon:
            return self.action_space.sample()

        q_was_training = self.q_network.training
        encoder_was_training = (
            self.encoder.training if isinstance(self.encoder, nn.Module) else None
        )
        if not training:
            self.q_network.eval()
            if isinstance(self.encoder, nn.Module):
                self.encoder.eval()
        try:
            with torch.no_grad():
                obs = self._normalize_observation(observation)
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                if self.precomputed_latents:
                    # obs ya es el vector latente (latent_dim,); shape (1, latent_dim)
                    latent = obs_t
                else:
                    latent = self.encoder(obs_t)
                return self.q_network(latent).argmax().item()
        finally:
            self.q_network.train(q_was_training)
            if isinstance(self.encoder, nn.Module):
                self.encoder.train(encoder_was_training)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "q_mean": 0.0, "q_std": 0.0}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        if self.normalization_mean is not None and self.normalization_std is not None:
            eps = 1e-8
            states = (states - self.normalization_mean) / (self.normalization_std + eps)
            next_states = (
                next_states - self.normalization_mean
            ) / (self.normalization_std + eps)

        states_t = torch.FloatTensor(states).to(self.device)       # (B, L, F) o (B, latent_dim)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Latentes
        if self.precomputed_latents:
            # Las observaciones ya son vectores latentes; se pasan directamente.
            latents = states_t
            next_latents = next_t
        elif self.freeze_encoder:
            with torch.no_grad():
                latents = self.encoder(states_t)
                next_latents = self.encoder(next_t)
        else:
            latents = self.encoder(states_t)
            next_latents = self.encoder(next_t)

        q_values = self.q_network(latents).gather(1, actions_t.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.target_network(next_latents).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.criterion(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._trainable_params, self.grad_clip_norm
            )
        self.optimizer.step()

        self.update_step += 1
        if self.update_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        with torch.no_grad():
            all_q = self.q_network(latents)
        return {
            "loss": loss.item(),
            "q_mean": all_q.mean().item(),
            "q_std": all_q.std().item(),
        }

    def reset(self) -> None:
        self.epsilon = self.epsilon_start
        self.update_step = 0

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "encoder_state_dict": self.encoder.state_dict() if self.encoder is not None else None,
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "update_step": self.update_step,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if self.encoder is not None and ckpt.get("encoder_state_dict") is not None:
            self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.q_network.load_state_dict(ckpt["q_network_state_dict"])
        self.target_network.load_state_dict(ckpt["target_network_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon = ckpt["epsilon"]
        self.update_step = ckpt["update_step"]

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def get_latent_representation(self, observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs = self._normalize_observation(observation)
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            return self.encoder(obs_t).cpu().numpy().squeeze()

    def load_pretrained_encoder(self, path: str) -> None:
        """Carga un encoder preentrenado desde un checkpoint legacy (AutoencoderTrainer)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)

        if "latent_dim" in ckpt:
            assert ckpt["latent_dim"] == self.latent_dim, (
                f"Latent dim mismatch: ckpt={ckpt['latent_dim']}, agent={self.latent_dim}"
            )
        if "input_dim" in ckpt:
            assert ckpt["input_dim"] == self.input_dim, (
                f"Input dim mismatch: ckpt={ckpt['input_dim']}, agent={self.input_dim}"
            )

        self.encoder.load_state_dict(ckpt["encoder_state_dict"])

        for key, attr in (("normalization_mean", "normalization_mean"),
                          ("normalization_std", "normalization_std")):
            val = ckpt.get(key)
            if val is not None:
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                setattr(self, attr, np.asarray(val, dtype=np.float32))

    def load_artifact_encoder(self, path: str, cfg=None) -> None:
        """Carga un encoder desde un artefacto v2 (save_encoder_artifact)."""
        from latent_rl.representations.artifact import load_encoder_artifact

        encoder, _, _ = load_encoder_artifact(path, cfg=cfg)
        self.encoder = encoder.to(self.device)
        if self.freeze_encoder:
            self.encoder.freeze()

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.normalization_mean = mean
        self.normalization_std = std

    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        if self.normalization_mean is not None and self.normalization_std is not None:
            return (observation - self.normalization_mean) / (
                self.normalization_std + 1e-8
            )
        return observation
