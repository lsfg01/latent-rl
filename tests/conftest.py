"""Configuración de pytest para tests de latent-rl."""

import pytest
import numpy as np
import torch

# Configurar semillas aleatorias para reproducibilidad
@pytest.fixture(autouse=True)
def setup_random_seeds():
    """Configura semillas aleatorias antes de cada test."""
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

# Configuración de dispositivos
@pytest.fixture
def device():
    """Proporciona el dispositivo adecuado (CPU o CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tolerancia para comparaciones numéricas
@pytest.fixture
def tolerance():
    """Tolerancia para comparaciones numéricas."""
    return 1e-6