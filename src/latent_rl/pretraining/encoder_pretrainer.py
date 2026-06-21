"""Entrenador multitarea para encoders latentes (reconstrucción + forecasting)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from latent_rl.representations.base import LatentEncoder
from latent_rl.representations.artifact import save_encoder_artifact

logger = logging.getLogger(__name__)


class EncoderPretrainer:
    """Preentrenador de encoders con objetivo multitarea.

    Pérdida total:  L = L_recon + λ · L_forecast
      - L_recon:    MSE entre decode(z) y la ventana de entrada (L, F).
      - L_forecast: MSE entre head(z) y los retornos de los k pasos siguientes.

    La cabeza de forecasting vive sólo en el trainer; **no** se guarda en el
    artefacto del encoder.  Sólo el encoder (sin cabeza) se exporta.
    """

    def __init__(
        self,
        encoder: LatentEncoder,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        lambda_forecast: float = 0.5,
        k: int = 5,
        device: str = "cpu",
    ):
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lambda_forecast = lambda_forecast
        self.k = k
        self.device = torch.device(device)

        self.encoder.to(self.device)

        # Cabeza de forecasting: latent_dim → k (predice log_return de t+1..t+k)
        self.forecast_head = nn.Linear(encoder.latent_dim, k).to(self.device)

        self.optimizer = optim.Adam(
            list(encoder.parameters()) + list(self.forecast_head.parameters()),
            lr=learning_rate,
        )
        self.mse = nn.MSELoss()

        # Estadísticas de normalización (ajustadas en IS)
        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None
        # Estadísticas de targets de forecasting
        self._target_mean: Optional[float] = None
        self._target_std: Optional[float] = None

    # ------------------------------------------------------------------
    # Construcción del dataset de ventanas
    # ------------------------------------------------------------------

    def make_windows(
        self,
        df: pd.DataFrame,
        features: List[str],
        lookback: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera ventanas (X) y targets de forecasting (Y) para un DataFrame.

        Args:
            df: DataFrame con features normalizados y columna "log_return".
            features: Lista de columnas de features (F columnas).
            lookback: Longitud de la ventana de entrada (L pasos).

        Returns:
            X: (N, L, F)  ventanas de entrada.
            Y: (N, k)     retornos futuros log_return normalizados.
        """
        if "log_return" not in df.columns:
            raise ValueError(
                "El DataFrame debe contener la columna 'log_return' para "
                "construir targets de forecasting."
            )

        feat_arr = df[features].values.astype(np.float32)
        lr_arr = df["log_return"].values.astype(np.float32)
        n = len(df)

        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []

        max_start = n - lookback - self.k
        for i in range(max_start + 1):
            x_window = feat_arr[i : i + lookback]          # (L, F)
            y_targets = lr_arr[i + lookback : i + lookback + self.k]  # (k,)
            X_list.append(x_window)
            Y_list.append(y_targets)

        X = np.stack(X_list, axis=0) if X_list else np.empty((0, lookback, len(features)))
        Y = np.stack(Y_list, axis=0) if Y_list else np.empty((0, self.k))
        return X, Y

    # ------------------------------------------------------------------
    # Bucle de entrenamiento
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_epochs: int = 10,
        val_ratio: float = 0.1,
        early_stopping_patience: int = 5,
        seed: int = 0,
    ) -> Dict[str, List[float]]:
        """
        Entrena el encoder con pérdida recon + forecast.

        Args:
            X: (N, L, F) ventanas de entrada ya normalizadas.
            Y: (N, k) targets de log_return ya estandarizados.
            n_epochs: Épocas máximas.
            val_ratio: Fracción de validación (split temporal, sin barajar).
            early_stopping_patience: Épocas sin mejora antes de detener.
            seed: Semilla para reproducibilidad de mini-batches.

        Returns:
            Dict con "train_loss" y "val_loss" por época.
        """
        rng = np.random.default_rng(seed)

        # Estandarizar targets
        self._target_mean = float(Y.mean())
        self._target_std = float(Y.std()) + 1e-8
        Y_norm = (Y - self._target_mean) / self._target_std

        # Split temporal train/val
        n_val = max(1, int(len(X) * val_ratio))
        n_train = len(X) - n_val
        X_train, Y_train = X[:n_train], Y_norm[:n_train]
        X_val, Y_val = X[n_train:], Y_norm[n_train:]

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        Y_train_t = torch.FloatTensor(Y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        Y_val_t = torch.FloatTensor(Y_val).to(self.device)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        patience_counter = 0
        best_state = {
            "encoder": {k: v.clone() for k, v in self.encoder.state_dict().items()},
            "head": {k: v.clone() for k, v in self.forecast_head.state_dict().items()},
        }

        for epoch in range(n_epochs):
            self.encoder.train()
            self.forecast_head.train()

            indices = rng.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, self.batch_size):
                idx = indices[start : start + self.batch_size]
                xb = X_train_t[idx]
                yb = Y_train_t[idx]

                z = self.encoder(xb)
                x_recon = self.encoder.decode(z)
                y_pred = self.forecast_head(z)

                # Aplanar si decode retorna 3D
                xb_flat = xb.reshape(xb.size(0), -1)
                x_recon_flat = x_recon.reshape(x_recon.size(0), -1)

                loss_recon = self.mse(x_recon_flat, xb_flat)
                loss_forecast = self.mse(y_pred, yb)
                loss = loss_recon + self.lambda_forecast * loss_forecast

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)
            avg_val = self._eval_loss(X_val_t, Y_val_t)

            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)

            logger.debug("Epoch %d/%d  train=%.4f  val=%.4f", epoch + 1, n_epochs, avg_train, avg_val)

            if avg_val < best_val:
                best_val = avg_val
                patience_counter = 0
                best_state = {
                    "encoder": {k: v.clone() for k, v in self.encoder.state_dict().items()},
                    "head": {k: v.clone() for k, v in self.forecast_head.state_dict().items()},
                }
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping en época %d", epoch + 1)
                    break

        # Restaurar mejor checkpoint
        self.encoder.load_state_dict(best_state["encoder"])
        self.forecast_head.load_state_dict(best_state["head"])
        return history

    def _eval_loss(self, X_t: torch.Tensor, Y_t: torch.Tensor) -> float:
        self.encoder.eval()
        self.forecast_head.eval()
        with torch.no_grad():
            z = self.encoder(X_t)
            x_recon = self.encoder.decode(z)
            y_pred = self.forecast_head(z)

            xb_flat = X_t.reshape(X_t.size(0), -1)
            x_recon_flat = x_recon.reshape(x_recon.size(0), -1)

            loss = self.mse(x_recon_flat, xb_flat) + self.lambda_forecast * self.mse(
                y_pred, Y_t
            )
        return loss.item()

    # ------------------------------------------------------------------
    # Guardado
    # ------------------------------------------------------------------

    def save_encoder(
        self,
        path: str | Path,
        norm_stats: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Guarda el encoder (sin cabeza de forecasting) como artefacto.

        Args:
            path: Ruta de destino.
            norm_stats: Estadísticas de normalización del corpus.
            provenance: Metadatos de procedencia.
            feature_names: Nombres de los features (se inyectan en arch_config).
        """
        if feature_names:
            self.encoder.get_arch_config()["feature_names"] = feature_names

        save_encoder_artifact(
            encoder=self.encoder,
            norm_stats=norm_stats or {},
            provenance=provenance or {},
            path=path,
        )
