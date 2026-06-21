"""Encoder latente abstracto."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict


class LatentEncoder(nn.Module, ABC):
    """Encoder latente abstracto para representaciones de mercado."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.is_frozen = False

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Codifica la entrada a representación latente.

        Args:
            x: Tensor de forma (N, L, F) o (N, input_dim)

        Returns:
            Tensor latente de forma (N, latent_dim)
        """
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodifica la representación latente.

        Args:
            z: Tensor latente de forma (N, latent_dim)

        Returns:
            Tensor reconstruido
        """
        pass

    @abstractmethod
    def get_arch_config(self) -> Dict:
        """Devuelve la configuración de arquitectura para el sistema de artefactos."""
        pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias para forward."""
        return self.forward(x)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula la pérdida de reconstrucción (MSE)."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        if x_reconstructed.shape != x.shape:
            x_flat = x.reshape(x.size(0), -1)
            x_recon_flat = x_reconstructed.reshape(x_reconstructed.size(0), -1)
            return nn.functional.mse_loss(x_recon_flat, x_flat)
        return nn.functional.mse_loss(x_reconstructed, x)

    def freeze(self) -> None:
        """Congela todos los parámetros del módulo."""
        for param in self.parameters():
            param.requires_grad = False
        self.is_frozen = True

    def unfreeze(self) -> None:
        """Descongela todos los parámetros del módulo."""
        for param in self.parameters():
            param.requires_grad = True
        self.is_frozen = False
