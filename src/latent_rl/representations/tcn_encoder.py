"""TCNLatentEncoder: encoder temporal con convoluciones causales dilatadas."""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from latent_rl.representations.base import LatentEncoder


class _CausalResBlock(nn.Module):
    """Bloque residual con convolución 1D causal y dilatada."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=0
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = self._make_activation(activation)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}.get(
            name, nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L)
        residual = self.skip(x)
        out = F.pad(x, (self.causal_pad, 0))  # sólo relleno izquierdo
        out = self.conv(out)                   # (N, out_channels, L)
        # LayerNorm sobre la dimensión de canal: transponer temporalmente
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class TCNLatentEncoder(LatentEncoder):
    """Encoder temporal basado en TCN (convoluciones causales y dilatadas).

    Entrada: (N, L, F) → transpone a (N, F, L) para Conv1d.
    Salida:  (N, latent_dim) tomando el último paso temporal (causal).
    """

    def __init__(
        self,
        input_len: int,
        n_features: int,
        latent_dim: int,
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        channels: int = 32,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__(input_dim=input_len * n_features, latent_dim=latent_dim)
        self.input_len = input_len
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.dilations = dilations if dilations is not None else [1, 2, 4]
        self.channels = channels
        self.activation_name = activation
        self.dropout_rate = dropout

        # Campo receptivo: 1 + (K-1) * sum(dilations)
        receptive_field = 1 + (kernel_size - 1) * sum(self.dilations)
        if receptive_field < input_len:
            warnings.warn(
                f"TCN receptive field ({receptive_field}) < input_len ({input_len}). "
                "Considera aumentar dilations o kernel_size.",
                stacklevel=2,
            )

        # Pila de bloques residuales
        blocks: List[nn.Module] = []
        in_ch = n_features
        for d in self.dilations:
            blocks.append(
                _CausalResBlock(in_ch, channels, kernel_size, d, dropout, activation)
            )
            in_ch = channels
        self.tcn = nn.Sequential(*blocks)

        # Proyección al espacio latente (último paso temporal)
        self.proj = nn.Linear(channels, latent_dim)

        # Decoder MVP: linear
        self.decoder_lin = nn.Linear(latent_dim, input_len * n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, L, F)
        Returns:
            (N, latent_dim)
        """
        x = x.transpose(1, 2)          # (N, F, L)
        x = self.tcn(x)                # (N, channels, L)
        z = self.proj(x[:, :, -1])     # tomar último paso: (N, latent_dim)
        return z

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
            "encoder_type": "tcn",
            "L": self.input_len,
            "F": self.n_features,
            "latent_dim": self.latent_dim,
            "kernel_size": self.kernel_size,
            "dilations": self.dilations,
            "channels": self.channels,
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
            "feature_names": [],
        }
