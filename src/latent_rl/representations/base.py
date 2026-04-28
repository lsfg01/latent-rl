"""Encoder latente abstracto."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class LatentEncoder(nn.Module, ABC):
    """Encoder latente abstracto para representaciones de mercado."""

    def __init__(self, input_dim: int, latent_dim: int):
        """
        Inicializa el encoder latente.

        Args:
            input_dim: Dimensión de entrada
            latent_dim: Dimensión de la representación latente
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Codifica la entrada a representación latente.

        Args:
            x: Tensor de entrada de forma (batch_size, input_dim)

        Returns:
            Tensor latente de forma (batch_size, latent_dim)
        """
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodifica la representación latente.

        Args:
            z: Tensor latente de forma (batch_size, latent_dim)

        Returns:
            Tensor reconstruido de forma (batch_size, input_dim)
        """
        pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alias para el método forward.

        Args:
            x: Tensor de entrada

        Returns:
            Tensor latente
        """
        return self.forward(x)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida de reconstrucción.

        Args:
            x: Tensor de entrada

        Returns:
            Pérdida de reconstrucción (MSE)
        """
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return nn.functional.mse_loss(x_reconstructed, x)