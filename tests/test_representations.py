"""Tests para el módulo de representaciones."""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from latent_rl.representations import (
    MLPLatentEncoder,
    LatentEncoder,
    TCNLatentEncoder,
    GRULatentEncoder,
    build_encoder,
    save_encoder_artifact,
    load_encoder_artifact,
)


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


# ---------------------------------------------------------------------------
# TCNLatentEncoder
# ---------------------------------------------------------------------------

class TestTCNLatentEncoder:

    @pytest.fixture
    def encoder(self):
        return TCNLatentEncoder(
            input_len=10, n_features=5, latent_dim=8,
            kernel_size=3, dilations=[1, 2, 4], channels=16,
        )

    def test_forward_shape(self, encoder):
        x = torch.randn(4, 10, 5)
        z = encoder(x)
        assert z.shape == (4, 8)

    def test_decode_shape(self, encoder):
        z = torch.randn(4, 8)
        x_recon = encoder.decode(z)
        assert x_recon.shape == (4, 10, 5)

    def test_forward_3d_required(self, encoder):
        """Pasamos un tensor 3D (N, L, F)."""
        x = torch.randn(2, 10, 5)
        z = encoder(x)
        assert z.shape == (2, 8)

    def test_freeze_all_params(self, encoder):
        encoder.freeze()
        assert encoder.is_frozen
        for param in encoder.parameters():
            assert not param.requires_grad

    def test_unfreeze_all_params(self, encoder):
        encoder.freeze()
        encoder.unfreeze()
        assert not encoder.is_frozen
        for param in encoder.parameters():
            assert param.requires_grad

    def test_causality(self):
        """Alterar el último paso de la ventana no cambia salidas de pasos anteriores."""
        encoder = TCNLatentEncoder(
            input_len=10, n_features=3, latent_dim=8,
            kernel_size=3, dilations=[1, 2], channels=8,
        )
        encoder.eval()

        x = torch.randn(1, 10, 3)
        x_mod = x.clone()
        x_mod[:, -1, :] = torch.randn(1, 3)  # modificar solo el último paso

        with torch.no_grad():
            x_t = x.transpose(1, 2)          # (1, F, L)
            x_mod_t = x_mod.transpose(1, 2)

            out = encoder.tcn(x_t)            # (1, channels, L)
            out_mod = encoder.tcn(x_mod_t)

        # Los pasos anteriores (0..L-2) no deben cambiar
        assert torch.allclose(out[:, :, :-1], out_mod[:, :, :-1], atol=1e-5), \
            "TCN no es causal: el último input afecta pasos anteriores en la salida"

    def test_receptive_field_warning(self):
        """Debe lanzar warning si el campo receptivo no cubre input_len."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # kernel_size=2, dilations=[1] → RF = 1 + 1*(2-1) = 2 < L=20
            TCNLatentEncoder(input_len=20, n_features=3, latent_dim=4,
                             kernel_size=2, dilations=[1], channels=4)
        assert any("receptive field" in str(warning.message).lower() for warning in w)

    def test_get_arch_config(self, encoder):
        cfg = encoder.get_arch_config()
        assert cfg["encoder_type"] == "tcn"
        assert cfg["L"] == 10
        assert cfg["F"] == 5
        assert cfg["latent_dim"] == 8


# ---------------------------------------------------------------------------
# GRULatentEncoder
# ---------------------------------------------------------------------------

class TestGRULatentEncoder:

    @pytest.fixture
    def encoder(self):
        return GRULatentEncoder(
            input_len=10, n_features=5, latent_dim=8,
            hidden_dim=16, num_layers=1,
        )

    def test_forward_shape(self, encoder):
        x = torch.randn(4, 10, 5)
        z = encoder(x)
        assert z.shape == (4, 8)

    def test_decode_shape(self, encoder):
        z = torch.randn(4, 8)
        x_recon = encoder.decode(z)
        assert x_recon.shape == (4, 10, 5)

    def test_freeze_unfreeze(self, encoder):
        encoder.freeze()
        for p in encoder.parameters():
            assert not p.requires_grad
        encoder.unfreeze()
        for p in encoder.parameters():
            assert p.requires_grad

    def test_get_arch_config(self, encoder):
        cfg = encoder.get_arch_config()
        assert cfg["encoder_type"] == "gru"
        assert cfg["L"] == 10
        assert cfg["F"] == 5


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestBuildEncoder:

    def test_build_mlp(self):
        enc = build_encoder("mlp", input_len=10, n_features=5, latent_dim=8)
        assert isinstance(enc, MLPLatentEncoder)
        z = enc(torch.randn(2, 10, 5))
        assert z.shape == (2, 8)

    def test_build_tcn(self):
        enc = build_encoder("tcn", input_len=10, n_features=5, latent_dim=8)
        assert isinstance(enc, TCNLatentEncoder)
        z = enc(torch.randn(2, 10, 5))
        assert z.shape == (2, 8)

    def test_build_gru(self):
        enc = build_encoder("gru", input_len=10, n_features=5, latent_dim=8)
        assert isinstance(enc, GRULatentEncoder)
        z = enc(torch.randn(2, 10, 5))
        assert z.shape == (2, 8)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="desconocido"):
            build_encoder("lstm", input_len=10, n_features=5, latent_dim=8)


# ---------------------------------------------------------------------------
# Artifact system
# ---------------------------------------------------------------------------

class TestArtifactSystem:

    def _save_and_load(self, encoder, tmp_path):
        path = tmp_path / "enc.pt"
        save_encoder_artifact(encoder, norm_stats={"a": 1}, provenance={"b": 2}, path=path)
        enc_loaded, ns, prov = load_encoder_artifact(path)
        return enc_loaded, ns, prov

    def test_tcn_round_trip(self, tmp_path):
        enc = TCNLatentEncoder(10, 5, 8)
        enc_loaded, ns, prov = self._save_and_load(enc, tmp_path)
        assert isinstance(enc_loaded, TCNLatentEncoder)
        assert ns == {"a": 1}
        assert prov == {"b": 2}

    def test_gru_round_trip(self, tmp_path):
        enc = GRULatentEncoder(10, 5, 8)
        enc_loaded, _, _ = self._save_and_load(enc, tmp_path)
        assert isinstance(enc_loaded, GRULatentEncoder)

    def test_mlp_round_trip(self, tmp_path):
        enc = MLPLatentEncoder(50, 8, input_len=10, n_features=5)
        enc_loaded, _, _ = self._save_and_load(enc, tmp_path)
        assert isinstance(enc_loaded, MLPLatentEncoder)

    def test_outputs_reproduced(self, tmp_path):
        enc = TCNLatentEncoder(10, 5, 8)
        x = torch.randn(2, 10, 5)
        with torch.no_grad():
            z_before = enc(x)

        path = tmp_path / "enc.pt"
        save_encoder_artifact(enc, {}, {}, path)
        enc_loaded, _, _ = load_encoder_artifact(path)

        with torch.no_grad():
            z_after = enc_loaded(x)
        assert torch.allclose(z_before, z_after)

    def test_incompatible_L_raises(self, tmp_path):
        """load_encoder_artifact debe lanzar ValueError si L no coincide con cfg."""
        from dataclasses import dataclass

        @dataclass
        class FakeCfg:
            lookback_window: int = 20  # distinto al artefacto
            features: list = None
            def __post_init__(self): self.features = self.features or []

        enc = TCNLatentEncoder(10, 5, 8)
        path = tmp_path / "enc.pt"
        save_encoder_artifact(enc, {}, {}, path)

        with pytest.raises(ValueError, match="L del artefacto"):
            load_encoder_artifact(path, cfg=FakeCfg())

    def test_incompatible_features_raises(self, tmp_path):
        """Debe fallar si feature_names del artefacto no coincide con cfg.features."""
        from dataclasses import dataclass

        @dataclass
        class FakeCfg:
            lookback_window: int = 10
            features: list = None
            def __post_init__(self): self.features = self.features or ["a", "b", "c", "d", "e"]

        enc = TCNLatentEncoder(10, 5, 8)
        # Guardar con feature_names
        arch = enc.get_arch_config()
        arch["feature_names"] = ["x1", "x2", "x3", "x4", "x5"]  # distinto
        import torch as _torch
        path = tmp_path / "enc.pt"
        _torch.save({
            "encoder_state_dict": enc.state_dict(),
            "arch_config": arch,
            "norm_stats": {},
            "provenance": {},
        }, path)

        with pytest.raises(ValueError, match="feature_names"):
            load_encoder_artifact(path, cfg=FakeCfg())