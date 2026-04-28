"""Entrenador de autoencoder para preentrenamiento de encoders latentes."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Optional, List

from latent_rl.representations import MLPLatentEncoder


class AutoencoderTrainer:
    """Entrenador de autoencoder para preentrenamiento de encoders latentes."""

    def __init__(
        self,
        encoder: MLPLatentEncoder,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        """
        Inicializa el entrenador de autoencoder.

        Args:
            encoder: Encoder latente a entrenar
            learning_rate: Tasa de aprendizaje
            batch_size: Tamaño del batch
            device: Dispositivo de PyTorch ("cpu" o "cuda")
        """
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Parámetros de normalización
        self.normalization_mean = None
        self.normalization_std = None

        # Mover encoder al dispositivo
        self.encoder.to(self.device)

        # Optimizador solo para encoder (incluye decoder)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)

        # Loss function (MSE para reconstrucción)
        self.criterion = nn.MSELoss()

    def collect_observations(
        self,
        data: pd.DataFrame,
        lookback_window: int,
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Extrae observaciones mediante ventanas directas sobre datos OHLCV.

        Args:
            data: DataFrame con datos OHLCV
            lookback_window: Tamaño de la ventana de observación
            n_samples: Número de muestras a extraer (None = todas las posibles)

        Returns:
            Array de observaciones con forma (n_samples, lookback_window, n_features)
        """
        # Obtener columnas de features (OHLCV)
        feature_cols = [col for col in data.columns if col in ['open', 'high', 'low', 'close', 'volume']]
        n_features = len(feature_cols)

        # Calcular número máximo de muestras posibles
        max_samples = len(data) - lookback_window + 1

        if n_samples is None:
            n_samples = max_samples
        else:
            n_samples = min(n_samples, max_samples)

        # Extraer ventanas
        observations = []
        for i in range(n_samples):
            window = data[feature_cols].iloc[i:i + lookback_window].values
            observations.append(window)

        return np.array(observations, dtype=np.float32)

    def fit_normalization(self, observations: np.ndarray) -> None:
        """
        Calcula parámetros de normalización (mean y std) a partir de las observaciones.

        Args:
            observations: Array de observaciones con forma (n_samples, lookback_window, n_features)
        """
        # Calcular mean y std por posición/feature
        # observations shape: (n_samples, lookback_window, n_features)
        # Resultado: (lookback_window, n_features)
        self.normalization_mean = observations.mean(axis=0)
        self.normalization_std = observations.std(axis=0)

    def transform_observations(self, observations: np.ndarray) -> np.ndarray:
        """
        Aplica normalización a las observaciones usando parámetros ya calculados.

        Args:
            observations: Array de observaciones con forma (n_samples, lookback_window, n_features)

        Returns:
            Observaciones normalizadas con el mismo shape
        """
        if self.normalization_mean is None or self.normalization_std is None:
            raise ValueError("Normalization parameters not fitted. Call fit_normalization() first.")

        # Normalización: (x - mean) / (std + epsilon)
        epsilon = 1e-8
        normalized = (observations - self.normalization_mean) / (self.normalization_std + epsilon)
        return normalized.astype(np.float32)

    def fit_transform_observations(self, observations: np.ndarray) -> np.ndarray:
        """
        Calcula parámetros de normalización y aplica la normalización en un solo paso.

        Args:
            observations: Array de observaciones con forma (n_samples, lookback_window, n_features)

        Returns:
            Observaciones normalizadas con el mismo shape
        """
        self.fit_normalization(observations)
        return self.transform_observations(observations)

    def train(
        self,
        observations: np.ndarray,
        n_epochs: int = 10
    ) -> List[float]:
        """
        Entrena el autoencoder con las observaciones proporcionadas.

        Args:
            observations: Array de observaciones con forma (n_samples, lookback_window, n_features)
            n_epochs: Número de epochs de entrenamiento

        Returns:
            Lista de losses por epoch
        """
        # Aplanar observaciones: (n_samples, lookback_window, n_features) -> (n_samples, lookback_window * n_features)
        n_samples = observations.shape[0]
        flattened_observations = observations.reshape(n_samples, -1)

        # Convertir a tensor
        observations_tensor = torch.FloatTensor(flattened_observations).to(self.device)

        # Historial de losses
        loss_history = []

        # Bucle de entrenamiento
        self.encoder.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Crear batches
            for i in range(0, n_samples, self.batch_size):
                # Obtener batch
                batch = observations_tensor[i:i + self.batch_size]

                # Forward pass
                latent = self.encoder(batch)
                reconstructed = self.encoder.decode(latent)

                # Calcular loss
                loss = self.criterion(reconstructed, batch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Calcular loss media del epoch
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            loss_history.append(avg_loss)

        return loss_history

    def save_encoder(self, path: str) -> None:
        """
        Guarda el encoder entrenado en un archivo.

        Args:
            path: Ruta donde guardar el encoder
        """
        # Crear directorio si no existe
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Convertir numpy arrays a torch tensors si existen
        norm_mean = None
        norm_std = None
        if self.normalization_mean is not None:
            norm_mean = torch.as_tensor(self.normalization_mean, dtype=torch.float32)
        if self.normalization_std is not None:
            norm_std = torch.as_tensor(self.normalization_std, dtype=torch.float32)

        # Guardar checkpoint
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'input_dim': self.encoder.input_dim,
            'latent_dim': self.encoder.latent_dim,
            'hidden_dims': self.encoder.hidden_dims,
            'activation': self.encoder.activation_name,
            'dropout': self.encoder.dropout_rate,
            'normalization_mean': norm_mean,
            'normalization_std': norm_std
        }, path)

    @staticmethod
    def load_encoder(path: str, device: str = "cpu") -> MLPLatentEncoder:
        """
        Carga un encoder entrenado desde un archivo.

        Args:
            path: Ruta desde donde cargar el encoder
            device: Dispositivo de PyTorch ("cpu" o "cuda")

        Returns:
            Encoder cargado
        """
        # Cargar checkpoint con weights_only=True
        checkpoint = torch.load(path, map_location=torch.device(device), weights_only=True)

        # Crear encoder con la misma configuración
        encoder = MLPLatentEncoder(
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            activation=checkpoint['activation'],
            dropout=checkpoint['dropout']
        )

        # Cargar pesos
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder.to(torch.device(device))

        return encoder

    @staticmethod
    def load_checkpoint_metadata(path: str) -> dict:
        """
        Carga metadatos del checkpoint sin cargar el encoder completo.

        Args:
            path: Ruta del checkpoint

        Returns:
            Diccionario con metadatos (incluye normalization_mean y normalization_std)
        """
        # Cargar checkpoint con weights_only=True
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        # Convertir tensors a numpy arrays si existen
        norm_mean = None
        norm_std = None
        if checkpoint.get('normalization_mean') is not None:
            norm_mean = checkpoint['normalization_mean'].cpu().numpy()
        if checkpoint.get('normalization_std') is not None:
            norm_std = checkpoint['normalization_std'].cpu().numpy()

        return {
            'input_dim': checkpoint.get('input_dim'),
            'latent_dim': checkpoint.get('latent_dim'),
            'hidden_dims': checkpoint.get('hidden_dims'),
            'activation': checkpoint.get('activation'),
            'dropout': checkpoint.get('dropout'),
            'normalization_mean': norm_mean,
            'normalization_std': norm_std
        }
