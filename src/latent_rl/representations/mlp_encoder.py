"""Encoder latente MLP concreto."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from latent_rl.representations.base import LatentEncoder


class MLPLatentEncoder(LatentEncoder):
    """Encoder latente basado en MLP.

    Acepta entrada plana (N, input_dim) o secuencial (N, L, F); en el segundo
    caso aplana internamente antes de codificar.  El decoder devuelve (N, L, F)
    cuando se construye con input_len y n_features, o (N, input_dim) en caso
    contrario (comportamiento heredado).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        input_len: Optional[int] = None,
        n_features: Optional[int] = None,
    ):
        super().__init__(input_dim, latent_dim)

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.dropout_rate = dropout
        self.input_len = input_len
        self.n_features = n_features

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(self._get_activation(activation))
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(self._get_activation(activation))
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "gelu": nn.GELU(),
        }
        if activation.lower() not in activations:
            raise ValueError(f"Activación no soportada: {activation}")
        return activations[activation.lower()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Codifica la entrada.

        Args:
            x: (N, L, F) o (N, input_dim)

        Returns:
            (N, latent_dim)
        """
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodifica desde el espacio latente.

        Args:
            z: (N, latent_dim)

        Returns:
            (N, L, F) si input_len/n_features conocidos, sino (N, input_dim)
        """
        x_flat = self.decoder(z)
        if self.input_len is not None and self.n_features is not None:
            return x_flat.reshape(x_flat.size(0), self.input_len, self.n_features)
        return x_flat

    def get_arch_config(self) -> Dict:
        return {
            "encoder_type": "mlp",
            "L": self.input_len,
            "F": self.n_features,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
            "feature_names": [],
        }

    # ------------------------------------------------------------------
    # Métodos de congelado individuales (compatibilidad con AutoencoderTrainer)
    # ------------------------------------------------------------------

    def get_encoder_parameters(self) -> List[nn.Parameter]:
        return list(self.encoder.parameters())

    def get_decoder_parameters(self) -> List[nn.Parameter]:
        return list(self.decoder.parameters())

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True
