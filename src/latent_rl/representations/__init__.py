"""Módulo de representaciones latentes."""

from latent_rl.representations.base import LatentEncoder
from latent_rl.representations.mlp_encoder import MLPLatentEncoder

__all__ = ["LatentEncoder", "MLPLatentEncoder"]