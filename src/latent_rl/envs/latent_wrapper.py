"""Wrapper que sustituye la observación raw por el vector latente precomputado.

Para brazos con encoder congelado (C y D) el latente es matemáticamente
idéntico al que produciría el encoder en cada paso.  Precomputarlo una vez
evita correr el TCN/GRU en cada llamada a step/select_action durante el bucle
de RL, reduciendo el tiempo de experimento sin alterar los resultados.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from latent_rl.envs.financial_env import FinancialEnv


def precompute_latent_series(
    data: "pd.DataFrame",
    lookback_window: int,
    feature_cols: List[str],
    encoder: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Precomputa todos los vectores latentes de una serie temporal.

    Replica exactamente la lógica de `FinancialEnv._get_observation()`:
    - Ventana deslizante de longitud `lookback_window` sobre `feature_cols`.
    - Zero-padding en los primeros pasos (igual que en el entorno).

    El índice devuelto coincide con `env.current_step` en el entorno
    correspondiente, empezando desde `lookback_window` (primer paso tras reset).

    Args:
        data: DataFrame con al menos las columnas `feature_cols`.
        lookback_window: L en la arquitectura del encoder.
        feature_cols: Columnas de features (deben coincidir con las del encoder).
        encoder: Encoder ya entrenado y en modo eval.
        device: Dispositivo de cómputo.

    Returns:
        Array de forma (T, latent_dim) donde T = len(data).
        latents[t] es el latente correspondiente a `current_step == t`.
    """
    import pandas as pd

    vals = data[feature_cols].values.astype(np.float32)
    T = len(vals)
    n_features = len(feature_cols)

    encoder.eval()
    latents_list = []

    with torch.no_grad():
        for t in range(T):
            start_idx = max(0, t - lookback_window)
            window = vals[start_idx:t]
            if len(window) < lookback_window:
                padding = np.zeros(
                    (lookback_window - len(window), n_features), dtype=np.float32
                )
                window = np.vstack([padding, window])
            # (1, L, F)
            obs_t = torch.from_numpy(window).unsqueeze(0).to(device)
            latent = encoder(obs_t)
            latents_list.append(latent.squeeze(0).cpu().numpy())

    return np.stack(latents_list, axis=0)  # (T, latent_dim)


class LatentObservationWrapper(gym.Wrapper):
    """Envuelve FinancialEnv devolviendo latentes precomputados como observación.

    El espacio de observación cambia de (L, F) a (latent_dim,).
    Todos los demás métodos (step, reset, render, equity info) se delegan al
    entorno base sin modificación.

    Args:
        env: FinancialEnv ya construido.
        latents: Array (T, latent_dim) generado por `precompute_latent_series`.
            Debe cubrir al menos len(env.data) posiciones.
    """

    def __init__(self, env: FinancialEnv, latents: np.ndarray) -> None:
        super().__init__(env)
        if len(latents) < len(env.data):
            raise ValueError(
                f"latents tiene {len(latents)} filas pero env.data tiene {len(env.data)}. "
                "Precomputa los latentes sobre el DataFrame completo."
            )
        self._latents = latents
        latent_dim = latents.shape[1]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(latent_dim,),
            dtype=np.float32,
        )

    # Propiedades del entorno base expuestas para compatibilidad ---------

    @property
    def initial_balance(self) -> float:
        return self.env.initial_balance

    @property
    def data(self):
        return self.env.data

    # Gymnassium API ---------------------------------------------------------

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        _, info = self.env.reset(**kwargs)
        return self._latents[self.env.current_step].astype(np.float32), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        _, reward, terminated, truncated, info = self.env.step(action)
        obs = self._latents[self.env.current_step].astype(np.float32)
        return obs, reward, terminated, truncated, info
