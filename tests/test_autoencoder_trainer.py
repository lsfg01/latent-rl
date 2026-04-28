"""Tests para AutoencoderTrainer."""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import os

from latent_rl.pretraining import AutoencoderTrainer
from latent_rl.representations import MLPLatentEncoder


class TestAutoencoderTrainer:
    """Tests para AutoencoderTrainer."""

    @pytest.fixture
    def sample_data(self):
        """Datos de ejemplo para tests."""
        np.random.seed(42)
        return pd.DataFrame({
            "open": np.random.randn(100) + 100,
            "high": np.random.randn(100) + 105,
            "low": np.random.randn(100) + 95,
            "close": np.random.randn(100) + 100,
            "volume": np.random.randint(1000, 10000, 100)
        })

    @pytest.fixture
    def encoder(self):
        """Encoder de ejemplo para tests."""
        return MLPLatentEncoder(
            input_dim=50,  # 10 lookback * 5 features
            latent_dim=16,
            hidden_dims=[32, 16],
            activation="relu",
            dropout=0.0
        )

    @pytest.fixture
    def trainer(self, encoder):
        """Entrenador de ejemplo para tests."""
        return AutoencoderTrainer(
            encoder=encoder,
            learning_rate=1e-3,
            batch_size=32,
            device="cpu"
        )

    def test_init(self, encoder):
        """Test de inicialización del entrenador."""
        trainer = AutoencoderTrainer(
            encoder=encoder,
            learning_rate=1e-3,
            batch_size=64,
            device="cpu"
        )

        assert trainer.encoder is encoder
        assert trainer.learning_rate == 1e-3
        assert trainer.batch_size == 64
        assert trainer.device == torch.device("cpu")

    def test_collect_observations_shape(self, trainer, sample_data):
        """Test de que collect_observations devuelve shape correcta."""
        lookback_window = 10
        n_samples = 50

        observations = trainer.collect_observations(
            data=sample_data,
            lookback_window=lookback_window,
            n_samples=n_samples
        )

        # Verificar forma
        assert observations.shape == (n_samples, lookback_window, 5)  # 5 features OHLCV

    def test_collect_observations_all_samples(self, trainer, sample_data):
        """Test de collect_observations con todas las muestras posibles."""
        lookback_window = 10

        observations = trainer.collect_observations(
            data=sample_data,
            lookback_window=lookback_window,
            n_samples=None
        )

        # Verificar que se extraen todas las muestras posibles
        max_samples = len(sample_data) - lookback_window + 1
        assert observations.shape[0] == max_samples

    def test_collect_observations_n_samples_limited(self, trainer, sample_data):
        """Test de collect_observations con n_samples limitado."""
        lookback_window = 10
        n_samples = 20

        observations = trainer.collect_observations(
            data=sample_data,
            lookback_window=lookback_window,
            n_samples=n_samples
        )

        # Verificar que se extraen exactamente n_samples
        assert observations.shape[0] == n_samples

    def test_collect_observations_features(self, trainer, sample_data):
        """Test de que collect_observations extrae las features correctas."""
        lookback_window = 10
        n_samples = 10

        observations = trainer.collect_observations(
            data=sample_data,
            lookback_window=lookback_window,
            n_samples=n_samples
        )

        # Verificar que hay 5 features (OHLCV)
        assert observations.shape[2] == 5

    def test_train_basic(self, trainer, sample_data):
        """Test de entrenamiento básico."""
        lookback_window = 10
        n_samples = 100
        n_epochs = 2

        # Extraer observaciones
        observations = trainer.collect_observations(
            data=sample_data,
            lookback_window=lookback_window,
            n_samples=n_samples
        )

        # Entrenar
        loss_history = trainer.train(observations, n_epochs=n_epochs)

        # Verificar que se devuelve historial de losses
        assert len(loss_history) == n_epochs
        assert all(isinstance(loss, float) for loss in loss_history)

    def test_train_returns_losses(self, trainer, sample_data):
        """Test de que train devuelve losses válidos."""
        lookback_window = 10
        n_samples = 50
        n_epochs = 1

        observations = trainer.collect_observations(
            data=sample_data,
            lookback_window=lookback_window,
            n_samples=n_samples
        )

        loss_history = trainer.train(observations, n_epochs=n_epochs)

        # Verificar que los losses son números válidos
        assert len(loss_history) == 1
        assert isinstance(loss_history[0], float)
        assert not np.isnan(loss_history[0])
        assert not np.isinf(loss_history[0])

    def test_train_with_small_batch(self, trainer, sample_data):
        """Test de entrenamiento con batch pequeño."""
        lookback_window = 10
        n_samples = 10
        n_epochs = 1

        observations = trainer.collect_observations(
            data=sample_data,
            lookback_window=lookback_window,
            n_samples=n_samples
        )

        loss_history = trainer.train(observations, n_epochs=n_epochs)

        # Debe funcionar incluso con pocos datos
        assert len(loss_history) == 1

    def test_save_encoder(self, trainer):
        """Test de guardar encoder."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name

        try:
            # Guardar encoder
            trainer.save_encoder(temp_path)

            # Verificar que el archivo existe
            assert os.path.exists(temp_path)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_encoder(self, encoder):
        """Test de cargar encoder."""
        # Crear trainer y guardar encoder
        trainer = AutoencoderTrainer(encoder=encoder, device="cpu")

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name

        try:
            # Guardar encoder
            trainer.save_encoder(temp_path)

            # Cargar encoder
            loaded_encoder = AutoencoderTrainer.load_encoder(temp_path, device="cpu")

            # Verificar que es un MLPLatentEncoder
            assert isinstance(loaded_encoder, MLPLatentEncoder)

            # Verificar que tiene la misma configuración
            assert loaded_encoder.latent_dim == encoder.latent_dim
            assert loaded_encoder.input_dim == encoder.input_dim

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_encoder_compatible(self, encoder):
        """Test de que el encoder cargado es compatible."""
        # Crear trainer y guardar encoder
        trainer = AutoencoderTrainer(encoder=encoder, device="cpu")

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name

        try:
            # Guardar encoder
            trainer.save_encoder(temp_path)

            # Cargar encoder
            loaded_encoder = AutoencoderTrainer.load_encoder(temp_path, device="cpu")

            # Verificar que se puede usar para forward pass
            test_input = torch.randn(1, encoder.input_dim)
            with torch.no_grad():
                latent = loaded_encoder(test_input)

            # Verificar forma de salida
            assert latent.shape == (1, encoder.latent_dim)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_load_roundtrip(self, encoder):
        """Test de roundtrip de guardar/cargar encoder."""
        # Crear trainer
        trainer = AutoencoderTrainer(encoder=encoder, device="cpu")

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name

        try:
            # Guardar encoder
            trainer.save_encoder(temp_path)

            # Cargar encoder
            loaded_encoder = AutoencoderTrainer.load_encoder(temp_path, device="cpu")

            # Verificar que los pesos son iguales
            for param1, param2 in zip(encoder.parameters(), loaded_encoder.parameters()):
                assert torch.allclose(param1, param2)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_train_save_load_integration(self, trainer, sample_data):
        """Test de integración completa: entrenar, guardar, cargar."""
        lookback_window = 10
        n_samples = 50
        n_epochs = 2

        # Extraer observaciones
        observations = trainer.collect_observations(
            data=sample_data,
            lookback_window=lookback_window,
            n_samples=n_samples
        )

        # Entrenar
        loss_history = trainer.train(observations, n_epochs=n_epochs)

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name

        try:
            # Guardar encoder entrenado
            trainer.save_encoder(temp_path)

            # Cargar encoder
            loaded_encoder = AutoencoderTrainer.load_encoder(temp_path, device="cpu")

            # Verificar que se puede usar
            test_input = torch.randn(1, trainer.encoder.input_dim)
            with torch.no_grad():
                latent = loaded_encoder(test_input)

            assert latent.shape == (1, trainer.encoder.latent_dim)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_fit_normalization(self, trainer, sample_data):
        """Test de fit_normalization."""
        lookback_window = 10
        n_samples = 50

        observations = trainer.collect_observations(
            data=sample_data,
            lookback_window=lookback_window,
            n_samples=n_samples
        )

        # Fit normalization
        trainer.fit_normalization(observations)

        # Verificar que se calcularon mean y std
        assert trainer.normalization_mean is not None
        assert trainer.normalization_std is not None
        assert trainer.normalization_mean.shape == (lookback_window, 5)
        assert trainer.normalization_std.shape == (lookback_window, 5)

    def test_transform_observations(self, trainer, sample_data):
        """Test de transform_observations."""
        lookback_window = 10
        n_samples = 50

        observations = trainer.collect_observations(
            data=sample_data,
            lookback_window=lookback_window,
            n_samples=n_samples
        )

        # Fit normalization primero
        trainer.fit_normalization(observations)

        # Transform observaciones
        normalized = trainer.transform_observations(observations)

        # Verificar forma
        assert normalized.shape == observations.shape

        # Verificar que las observaciones normalizadas tienen media ~0 y std ~1
        # Usamos tolerancia más realista para float32
        assert np.allclose(normalized.mean(axis=0), 0, atol=1e-4)
        assert np.allclose(normalized.std(axis=0), 1, atol=1e-3)

    def test_fit_transform_observations(self, trainer, sample_data):
        """Test de fit_transform_observations."""
        lookback_window = 10
        n_samples = 50

        observations = trainer.collect_observations(
            data=sample_data,
            lookback_window=lookback_window,
            n_samples=n_samples
        )

        # Fit y transform en un solo paso
        normalized = trainer.fit_transform_observations(observations)

        # Verificar forma
        assert normalized.shape == observations.shape

        # Verificar que se calcularon parámetros
        assert trainer.normalization_mean is not None
        assert trainer.normalization_std is not None

    def test_transform_observations_fails_without_fit(self, trainer, sample_data):
        """Test de que transform_observations falla si no se llamó fit_normalization."""
        lookback_window = 10
        n_samples = 10

        observations = trainer.collect_observations(
            data=sample_data,
            lookback_window=lookback_window,
            n_samples=n_samples
        )

        # Intentar transform sin fit debe fallar
        try:
            trainer.transform_observations(observations)
            assert False, "Debería lanzar ValueError"
        except ValueError as e:
            assert "not fitted" in str(e).lower()

    def test_save_encoder_saves_normalization(self, trainer, sample_data):
        """Test de que save_encoder guarda normalization_mean y normalization_std."""
        lookback_window = 10
        n_samples = 50

        observations = trainer.collect_observations(
            data=sample_data,
            lookback_window=lookback_window,
            n_samples=n_samples
        )

        # Fit normalization
        trainer.fit_normalization(observations)

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name

        try:
            # Guardar encoder
            trainer.save_encoder(temp_path)

            # Cargar checkpoint
            checkpoint = torch.load(temp_path, map_location="cpu")

            # Verificar que se guardaron parámetros de normalización
            assert 'normalization_mean' in checkpoint
            assert 'normalization_std' in checkpoint
            assert checkpoint['normalization_mean'] is not None
            assert checkpoint['normalization_std'] is not None

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_checkpoint_metadata(self, trainer, sample_data):
        """Test de load_checkpoint_metadata recupera normalization_mean y normalization_std."""
        lookback_window = 10
        n_samples = 50

        observations = trainer.collect_observations(
            data=sample_data,
            lookback_window=lookback_window,
            n_samples=n_samples
        )

        # Fit normalization
        trainer.fit_normalization(observations)

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name

        try:
            # Guardar encoder
            trainer.save_encoder(temp_path)

            # Cargar metadatos
            metadata = AutoencoderTrainer.load_checkpoint_metadata(temp_path)

            # Verificar que se recuperaron parámetros de normalización
            assert metadata['normalization_mean'] is not None
            assert metadata['normalization_std'] is not None
            assert metadata['normalization_mean'].shape == (lookback_window, 5)
            assert metadata['normalization_std'].shape == (lookback_window, 5)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
