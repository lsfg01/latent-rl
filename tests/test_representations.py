"""Tests para el módulo de representaciones."""

import pytest
import torch
import numpy as np

from latent_rl.representations import MLPLatentEncoder, LatentEncoder


class TestMLPLatentEncoder:
    """Tests para MLPLatentEncoder."""

    @pytest.fixture
    def encoder(self):
        """Encoder de ejemplo para tests."""
        return MLPLatentEncoder(
            input_dim=10,
            latent_dim=5,
            hidden_dims=[8, 6],
            activation="relu",
            dropout=0.0
        )

    def test_init(self, encoder):
        """Test de inicialización."""
        assert encoder.input_dim == 10
        assert encoder.latent_dim == 5
        assert encoder.hidden_dims == [8, 6]
        assert encoder.activation_name == "relu"

    def test_forward(self, encoder):
        """Test de codificación (forward)."""
        batch_size = 4
        x = torch.randn(batch_size, 10)

        z = encoder(x)

        assert z.shape == (batch_size, 5)
        assert isinstance(z, torch.Tensor)

    def test_decode(self, encoder):
        """Test de decodificación."""
        batch_size = 4
        z = torch.randn(batch_size, 5)

        x_reconstructed = encoder.decode(z)

        assert x_reconstructed.shape == (batch_size, 10)
        assert isinstance(x_reconstructed, torch.Tensor)

    def test_encode_alias(self, encoder):
        """Test de alias encode."""
        batch_size = 4
        x = torch.randn(batch_size, 10)

        z_forward = encoder(x)
        z_encode = encoder.encode(x)

        # Deben ser iguales
        assert torch.allclose(z_forward, z_encode)

    def test_reconstruction_loss(self, encoder):
        """Test de pérdida de reconstrucción."""
        batch_size = 4
        x = torch.randn(batch_size, 10)

        loss = encoder.reconstruction_loss(x)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # MSE es no negativo

    def test_different_activations(self):
        """Test de diferentes activaciones."""
        activations = ["relu", "tanh", "sigmoid", "leaky_relu"]

        for activation in activations:
            encoder = MLPLatentEncoder(
                input_dim=10,
                latent_dim=5,
                activation=activation
            )

            x = torch.randn(2, 10)
            z = encoder(x)

            assert z.shape == (2, 5)

    def test_invalid_activation(self):
        """Test de activación inválida."""
        with pytest.raises(ValueError, match="Activación no soportada"):
            MLPLatentEncoder(
                input_dim=10,
                latent_dim=5,
                activation="invalid_activation"
            )

    def test_dropout(self):
        """Test de dropout."""
        encoder = MLPLatentEncoder(
            input_dim=10,
            latent_dim=5,
            dropout=0.5
        )

        x = torch.randn(4, 10)
        encoder.train()  # Modo entrenamiento
        z = encoder(x)

        assert z.shape == (4, 5)

    def test_freeze_encoder(self, encoder):
        """Test de congelar encoder."""
        encoder.freeze_encoder()

        for param in encoder.get_encoder_parameters():
            assert not param.requires_grad

    def test_unfreeze_encoder(self, encoder):
        """Test de descongelar encoder."""
        encoder.freeze_encoder()
        encoder.unfreeze_encoder()

        for param in encoder.get_encoder_parameters():
            assert param.requires_grad

    def test_freeze_decoder(self, encoder):
        """Test de congelar decoder."""
        encoder.freeze_decoder()

        for param in encoder.get_decoder_parameters():
            assert not param.requires_grad

    def test_unfreeze_decoder(self, encoder):
        """Test de descongelar decoder."""
        encoder.freeze_decoder()
        encoder.unfreeze_decoder()

        for param in encoder.get_decoder_parameters():
            assert param.requires_grad

    def test_get_encoder_parameters(self, encoder):
        """Test de obtener parámetros del encoder."""
        params = encoder.get_encoder_parameters()

        assert len(params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in params)

    def test_get_decoder_parameters(self, encoder):
        """Test de obtener parámetros del decoder."""
        params = encoder.get_decoder_parameters()

        assert len(params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in params)

    def test_default_hidden_dims(self):
        """Test de dimensiones ocultas por defecto."""
        encoder = MLPLatentEncoder(input_dim=10, latent_dim=5)

        assert encoder.hidden_dims == [64, 32]

    def test_reconstruction_quality(self, encoder):
        """Test básico de calidad de reconstrucción."""
        # Crear datos simples
        x = torch.randn(10, 10)

        # Reconstruir
        z = encoder.encode(x)
        x_reconstructed = encoder.decode(z)

        # Verificar forma
        assert x_reconstructed.shape == x.shape

        # La pérdida no debe ser infinita
        loss = torch.nn.functional.mse_loss(x_reconstructed, x)
        assert torch.isfinite(loss)


class TestLatentEncoder:
    """Tests para LatentEncoder (clase abstracta)."""

    def test_abstract_class(self):
        """Test de que LatentEncoder es abstracta."""
        from latent_rl.representations.base import LatentEncoder

        with pytest.raises(TypeError):
            # No se puede instanciar la clase abstracta
            LatentEncoder(input_dim=10, latent_dim=5)

    def test_concrete_implementation(self):
        """Test de implementación concreta."""
        # MLPLatentEncoder debe ser instanciable
        encoder = MLPLatentEncoder(input_dim=10, latent_dim=5)

        assert isinstance(encoder, LatentEncoder)

    def test_reconstruction_loss_method(self):
        """Test del método reconstruction_loss."""
        encoder = MLPLatentEncoder(input_dim=10, latent_dim=5)
        x = torch.randn(4, 10)

        loss = encoder.reconstruction_loss(x)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Escalar