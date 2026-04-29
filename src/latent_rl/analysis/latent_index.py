"""Calculador de índice latente a partir de vectores latentes."""

import numpy as np
from typing import Literal


class LatentIndexCalculator:
    """Calcula un índice unidimensional a partir de vectores latentes."""

    VALID_METHODS = ["norm", "first_component", "pca"]

    def __init__(self, method: str = "norm"):
        """
        Inicializa el calculador de índice latente.

        Args:
            method: Método para calcular el índice ("norm", "first_component", "pca")

        Raises:
            ValueError: Si el método no es válido
        """
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Método inválido: {method}. "
                f"Métodos válidos: {self.VALID_METHODS}"
            )
        self.method = method
        self.pca_mean_ = None
        self.pca_component_ = None

    def fit(self, latent_vectors: np.ndarray) -> "LatentIndexCalculator":
        """
        Ajusta el calculador a los vectores latentes.

        Para "pca", calcula la primera componente principal.
        Para otros métodos, no hace nada (idempotente).

        Args:
            latent_vectors: Array de shape (n_samples, latent_dim)

        Returns:
            self

        Raises:
            ValueError: Si el input no tiene la forma correcta
        """
        self._validate_input(latent_vectors)

        if self.method == "pca":
            self._fit_pca(latent_vectors)

        return self

    def transform(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Transforma vectores latentes en índice latente.

        Args:
            latent_vectors: Array de shape (n_samples, latent_dim)

        Returns:
            Array de shape (n_samples,)

        Raises:
            ValueError: Si el input no tiene la forma correcta o si PCA no está ajustado
        """
        self._validate_input(latent_vectors)

        if self.method == "norm":
            return self._transform_norm(latent_vectors)
        elif self.method == "first_component":
            return self._transform_first_component(latent_vectors)
        elif self.method == "pca":
            return self._transform_pca(latent_vectors)
        else:
            raise ValueError(f"Método no implementado: {self.method}")

    def fit_transform(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Ajusta y transforma vectores latentes.

        Args:
            latent_vectors: Array de shape (n_samples, latent_dim)

        Returns:
            Array de shape (n_samples,)
        """
        self.fit(latent_vectors)
        return self.transform(latent_vectors)

    def _validate_input(self, latent_vectors: np.ndarray) -> None:
        """
        Valida que el input tenga la forma correcta.

        Args:
            latent_vectors: Array a validar

        Raises:
            ValueError: Si el input no tiene la forma correcta
        """
        if latent_vectors.size == 0:
            raise ValueError("Input vacío")

        if latent_vectors.ndim != 2:
            raise ValueError(
                f"Input debe ser 2D, got {latent_vectors.ndim}D. "
                f"Shape esperado: (n_samples, latent_dim)"
            )

        if latent_vectors.shape[0] < 1:
            raise ValueError("Debe haber al menos una muestra")

    def _transform_norm(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Calcula la norma L2 de cada vector latente.

        Args:
            latent_vectors: Array de shape (n_samples, latent_dim)

        Returns:
            Array de shape (n_samples,)
        """
        return np.linalg.norm(latent_vectors, axis=1)

    def _transform_first_component(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Extrae la primera componente de cada vector latente.

        Args:
            latent_vectors: Array de shape (n_samples, latent_dim)

        Returns:
            Array de shape (n_samples,)
        """
        return latent_vectors[:, 0]

    def _fit_pca(self, latent_vectors: np.ndarray) -> None:
        """
        Calcula la primera componente principal usando NumPy.

        Algoritmo:
        1. Centrar los datos (restar la media)
        2. Calcular la matriz de covarianza: C = (X^T @ X) / (n-1)
        3. Calcular eigenvectores y eigenvalores usando np.linalg.eigh
        4. Seleccionar el eigenvector con mayor eigenvalor
        5. Normalizar el signo para reproducibilidad

        Args:
            latent_vectors: Array de shape (n_samples, latent_dim)

        Raises:
            ValueError: Si hay menos de 2 muestras
        """
        n_samples = latent_vectors.shape[0]

        if n_samples < 2:
            raise ValueError("PCA requiere al menos 2 muestras")

        # 1. Centrar datos
        self.pca_mean_ = np.mean(latent_vectors, axis=0)
        centered = latent_vectors - self.pca_mean_

        # 2. Matriz de covarianza
        cov_matrix = (centered.T @ centered) / (n_samples - 1)

        # 3. Eigendecomposición
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Seleccionar componente principal (mayor eigenvalor)
        idx = np.argmax(eigenvalues)
        self.pca_component_ = eigenvectors[:, idx]

        # 5. Normalizar signo para reproducibilidad
        # Asegurar que el componente tenga media positiva en los datos
        projections = centered @ self.pca_component_
        if np.mean(projections) < 0:
            self.pca_component_ = -self.pca_component_

    def _transform_pca(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Proyecta vectores latentes en la primera componente principal.

        Args:
            latent_vectors: Array de shape (n_samples, latent_dim)

        Returns:
            Array de shape (n_samples,)

        Raises:
            ValueError: Si PCA no ha sido ajustado previamente
        """
        if self.pca_mean_ is None or self.pca_component_ is None:
            raise ValueError("PCA debe ser ajustado con fit() antes de transform()")

        # Centrar y proyectar
        centered = latent_vectors - self.pca_mean_
        return centered @ self.pca_component_
