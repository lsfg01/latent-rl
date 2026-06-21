"""GRULatentEncoder: encoder temporal recurrente."""

import torch
import torch.nn as nn
from typing import Dict
from latent_rl.representations.base import LatentEncoder


class GRULatentEncoder(LatentEncoder):
    """Encoder latente basado en GRU.

    Procesa secuencias (N, L, F) con un GRU y proyecta el último estado
    oculto al espacio latente.
    """

    def __init__(
        self,
        input_len: int,
        n_features: int,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(input_dim=input_len * n_features, latent_dim=latent_dim)
        self.input_len = input_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, latent_dim)
        self.decoder_lin = nn.Linear(latent_dim, input_len * n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, L, F)
        Returns:
            (N, latent_dim)
        """
        _, h_n = self.gru(x)          # h_n: (num_layers, N, hidden_dim)
        h_last = h_n[-1]              # (N, hidden_dim)
        return self.proj(h_last)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (N, latent_dim)
        Returns:
            (N, L, F)
        """
        x_flat = self.decoder_lin(z)
        return x_flat.reshape(x_flat.size(0), self.input_len, self.n_features)

    def get_arch_config(self) -> Dict:
        return {
            "encoder_type": "gru",
            "L": self.input_len,
            "F": self.n_features,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "feature_names": [],
        }
