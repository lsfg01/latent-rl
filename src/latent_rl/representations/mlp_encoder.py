"""Encoder latente MLP concreto."""

import torch
import torch.nn as nn
from typing import Optional, List
from latent_rl.representations.base import LatentEncoder


class MLPLatentEncoder(LatentEncoder):
    """Encoder latente basado en MLP (Multi-Layer Perceptron)."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """
        Inicializa el encoder MLP.

        Args:
            input_dim: Dimensión de entrada
            latent_dim: Dimensión de la representación latente
            hidden_dims: Lista de dimensiones de capas ocultas
            activation: Función de activación ('relu', 'tanh', 'sigmoid')
            dropout: Tasa de dropout
        """
        super().__init__(input_dim, latent_dim)

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.dropout_rate = dropout

        # Construir encoder
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(self._get_activation(activation))
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Capa latente
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Construir decoder
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(self._get_activation(activation))
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Capa de salida
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """
        Obtiene la función de activación especificada.

        Args:
            activation: Nombre de la activación

        Returns:
            Módulo de activación
        """
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }

        if activation.lower() not in activations:
            raise ValueError(f"Activación no soportada: {activation}")

        return activations[activation.lower()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Codifica la entrada a representación latente.

        Args:
            x: Tensor de entrada de forma (batch_size, input_dim)

        Returns:
            Tensor latente de forma (batch_size, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodifica la representación latente.

        Args:
            z: Tensor latente de forma (batch_size, latent_dim)

        Returns:
            Tensor reconstruido de forma (batch_size, input_dim)
        """
        return self.decoder(z)

    def get_encoder_parameters(self) -> List[nn.Parameter]:
        """
        Obtiene los parámetros del encoder.

        Returns:
            Lista de parámetros del encoder
        """
        return list(self.encoder.parameters())

    def get_decoder_parameters(self) -> List[nn.Parameter]:
        """
        Obtiene los parámetros del decoder.

        Returns:
            Lista de parámetros del decoder
        """
        return list(self.decoder.parameters())

    def freeze_encoder(self):
        """Congela los parámetros del encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Descongela los parámetros del encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_decoder(self):
        """Congela los parámetros del decoder."""
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        """Descongela los parámetros del decoder."""
        for param in self.decoder.parameters():
            param.requires_grad = True