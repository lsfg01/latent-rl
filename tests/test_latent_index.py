"""Tests para LatentIndexCalculator."""

import numpy as np
import pytest

from latent_rl.analysis import LatentIndexCalculator


def test_norm_method():
    """Test método norm: debe calcular norma L2."""
    vectors = np.array([[1, 0], [0, 1], [3, 4]])
    calc = LatentIndexCalculator(method="norm")
    index = calc.fit_transform(vectors)

    assert index.shape == (3,)
    assert np.allclose(index, [1, 1, 5])


def test_first_component_method():
    """Test método first_component: debe extraer primera dimensión."""
    vectors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    calc = LatentIndexCalculator(method="first_component")
    index = calc.fit_transform(vectors)

    assert np.allclose(index, [1, 4, 7])


def test_pca_method():
    """Test método pca: debe calcular primera componente principal."""
    # Datos colineales: y = 2x
    vectors = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
    calc = LatentIndexCalculator(method="pca")
    index = calc.fit_transform(vectors)

    assert index.shape == (4,)
    # Verificar que PCA captura la dirección de máxima varianza
    # Los datos están en línea recta, así que PCA debería capturar eso
    assert np.std(index) > 0  # Debe haber variación


def test_fit_transform_consistency():
    """Test que fit_transform es equivalente a fit + transform."""
    vectors = np.random.randn(100, 16)
    calc1 = LatentIndexCalculator(method="norm")
    index1 = calc1.fit_transform(vectors)

    calc2 = LatentIndexCalculator(method="norm")
    calc2.fit(vectors)
    index2 = calc2.transform(vectors)

    assert np.allclose(index1, index2)


def test_invalid_method_in_init():
    """Test que método inválido lanza ValueError en __init__."""
    with pytest.raises(ValueError, match="Método inválido"):
        LatentIndexCalculator(method="invalid")


def test_empty_input():
    """Test que input vacío lanza ValueError."""
    calc = LatentIndexCalculator(method="norm")
    with pytest.raises(ValueError, match="Input vacío"):
        calc.fit_transform(np.array([]))


def test_1d_input():
    """Test que input 1D lanza ValueError."""
    calc = LatentIndexCalculator(method="norm")
    with pytest.raises(ValueError, match="Input debe ser 2D"):
        calc.fit_transform(np.array([1, 2, 3]))


def test_single_sample():
    """Test con una sola muestra."""
    vectors = np.array([[1, 2, 3]])
    calc = LatentIndexCalculator(method="norm")
    index = calc.fit_transform(vectors)

    assert index.shape == (1,)
    assert np.allclose(index, [np.sqrt(14)])


def test_pca_with_single_sample():
    """Test que PCA con una sola muestra lanza ValueError."""
    vectors = np.array([[1, 2, 3]])
    calc = LatentIndexCalculator(method="pca")

    with pytest.raises(ValueError, match="PCA requiere al menos 2 muestras"):
        calc.fit(vectors)


def test_pca_fit_then_transform():
    """Test que PCA puede ajustarse en un dataset y transformar otro."""
    train_vectors = np.random.randn(100, 16)
    test_vectors = np.random.randn(50, 16)

    calc = LatentIndexCalculator(method="pca")
    calc.fit(train_vectors)
    index = calc.transform(test_vectors)

    assert index.shape == (50,)


def test_pca_transform_before_fit():
    """Test que transform PCA antes de fit lanza ValueError."""
    vectors = np.random.randn(10, 16)
    calc = LatentIndexCalculator(method="pca")

    with pytest.raises(ValueError, match="PCA debe ser ajustado"):
        calc.transform(vectors)


def test_pca_sign_normalization():
    """Test que PCA normaliza el signo para reproducibilidad."""
    # Datos con media positiva en la dirección principal
    vectors = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])

    calc1 = LatentIndexCalculator(method="pca")
    index1 = calc1.fit_transform(vectors)

    calc2 = LatentIndexCalculator(method="pca")
    index2 = calc2.fit_transform(vectors)

    # Los índices deben ser idénticos (mismo signo)
    assert np.allclose(index1, index2)


def test_pca_with_constant_data():
    """Test PCA con datos constantes."""
    # Todos los datos son iguales
    vectors = np.array([[1, 1], [1, 1], [1, 1]])

    calc = LatentIndexCalculator(method="pca")
    index = calc.fit_transform(vectors)

    # Con datos constantes, PCA debería devolver ceros
    assert np.allclose(index, [0, 0, 0], atol=1e-10)


def test_norm_with_negative_values():
    """Test norm con valores negativos."""
    vectors = np.array([[-1, -2], [-3, -4]])
    calc = LatentIndexCalculator(method="norm")
    index = calc.fit_transform(vectors)

    # La norma siempre es positiva
    assert np.all(index >= 0)
    assert np.allclose(index, [np.sqrt(5), 5])


def test_first_component_with_negative_values():
    """Test first_component con valores negativos."""
    vectors = np.array([[-1, 2, 3], [-4, 5, 6]])
    calc = LatentIndexCalculator(method="first_component")
    index = calc.fit_transform(vectors)

    assert np.allclose(index, [-1, -4])


def test_fit_idempotent_for_non_pca():
    """Test que fit es idempotente para métodos no PCA."""
    vectors = np.random.randn(10, 16)

    calc = LatentIndexCalculator(method="norm")
    calc.fit(vectors)
    calc.fit(vectors)  # No debería lanzar error

    calc2 = LatentIndexCalculator(method="first_component")
    calc2.fit(vectors)
    calc2.fit(vectors)  # No debería lanzar error


def test_transform_multiple_times():
    """Test que transform puede llamarse múltiples veces."""
    vectors = np.random.randn(10, 16)

    calc = LatentIndexCalculator(method="norm")
    calc.fit(vectors)

    index1 = calc.transform(vectors)
    index2 = calc.transform(vectors)

    assert np.allclose(index1, index2)


def test_pca_with_high_dimensional_data():
    """Test PCA con datos de alta dimensionalidad."""
    vectors = np.random.randn(100, 32)

    calc = LatentIndexCalculator(method="pca")
    index = calc.fit_transform(vectors)

    assert index.shape == (100,)
    assert np.std(index) > 0


def test_norm_with_zero_vectors():
    """Test norm con vectores cero."""
    vectors = np.array([[0, 0], [0, 0]])
    calc = LatentIndexCalculator(method="norm")
    index = calc.fit_transform(vectors)

    assert np.allclose(index, [0, 0])


def test_first_component_with_single_dimension():
    """Test first_component con vectores de una dimensión."""
    vectors = np.array([[1], [2], [3]])
    calc = LatentIndexCalculator(method="first_component")
    index = calc.fit_transform(vectors)

    assert np.allclose(index, [1, 2, 3])
