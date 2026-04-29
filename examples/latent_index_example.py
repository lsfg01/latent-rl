"""Ejemplo de cálculo de índice latente.

Este ejemplo demuestra cómo:
1. Crear datos sintéticos OHLCV
2. Entrenar un MLPLatentEncoder con AutoencoderTrainer
3. Extraer vectores latentes
4. Calcular índices latentes con diferentes métodos
5. Guardar resultados en CSV

Es un ejemplo experimental y educativo, no optimizado para producción.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from latent_rl.representations import MLPLatentEncoder
from latent_rl.pretraining import AutoencoderTrainer
from latent_rl.analysis import LatentIndexCalculator


def create_synthetic_data(n_steps: int = 200) -> pd.DataFrame:
    """
    Crea datos sintéticos con tendencia alcista.

    Args:
        n_steps: Número de pasos de datos

    Returns:
        DataFrame con datos OHLCV
    """
    np.random.seed(42)

    # Crear tendencia alcista
    trend = np.linspace(0, 20, n_steps)
    noise = np.random.randn(n_steps) * 2
    prices = 100 + trend + noise

    return pd.DataFrame({
        "open": prices + np.random.randn(n_steps),
        "high": prices + np.random.randn(n_steps) + 2,
        "low": prices + np.random.randn(n_steps) - 2,
        "close": prices,
        "volume": np.random.randint(1000, 10000, n_steps)
    })


def main():
    """Función principal."""
    print("=== Ejemplo de Índice Latente ===\n")

    # 1. Crear datos sintéticos
    print("1. Creando datos sintéticos...")
    data = create_synthetic_data(n_steps=200)
    print(f"   - Datos creados: {len(data)} filas\n")

    # 2. Configurar encoder
    lookback_window = 10
    latent_dim = 16
    n_samples = 100

    print("2. Configurando encoder...")
    print(f"   - Lookback window: {lookback_window}")
    print(f"   - Dimensión latente: {latent_dim}")
    print(f"   - Muestras: {n_samples}\n")

    # 3. Crear y entrenar encoder
    print("3. Entrenando encoder...")
    input_dim = lookback_window * 5  # 5 features OHLCV
    encoder = MLPLatentEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=[64, 32],
        activation="relu",
        dropout=0.0
    )

    trainer = AutoencoderTrainer(
        encoder=encoder,
        learning_rate=1e-3,
        batch_size=32,
        device="cpu"
    )

    observations = trainer.collect_observations(
        data=data,
        lookback_window=lookback_window,
        n_samples=n_samples
    )

    observations_normalized = trainer.fit_transform_observations(observations)
    loss_history = trainer.train(observations_normalized, n_epochs=10)

    print(f"   - Loss inicial: {loss_history[0]:.6f}")
    print(f"   - Loss final: {loss_history[-1]:.6f}\n")

    # 4. Extraer vectores latentes
    print("4. Extrayendo vectores latentes...")
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(observations_normalized)
        obs_flat = obs_tensor.reshape(obs_tensor.size(0), -1)
        latent_vectors = encoder(obs_flat).cpu().numpy()

    print(f"   - Forma de vectores latentes: {latent_vectors.shape}")
    print(f"   - (n_samples={latent_vectors.shape[0]}, latent_dim={latent_vectors.shape[1]})\n")

    # 5. Calcular índices latentes con diferentes métodos
    print("5. Calculando índices latentes...")

    methods = ["norm", "first_component", "pca"]
    indices = {}

    for method in methods:
        calc = LatentIndexCalculator(method=method)
        index = calc.fit_transform(latent_vectors)
        indices[method] = index

        print(f"\n   Método: {method}")
        print(f"   - Forma: {index.shape}")
        print(f"   - Media: {np.mean(index):.4f}")
        print(f"   - Std: {np.std(index):.4f}")
        print(f"   - Min: {np.min(index):.4f}")
        print(f"   - Max: {np.max(index):.4f}")

    # 6. Guardar resultados en CSV
    print("\n6. Guardando resultados...")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Obtener precios de cierre correspondientes a las observaciones
    # Las observaciones empiezan en lookback_window, así que obtenemos los precios
    # desde lookback_window hasta lookback_window + n_samples
    close_prices = data["close"].values[lookback_window:lookback_window + n_samples]

    results_df = pd.DataFrame({
        "sample": range(n_samples),
        "close": close_prices,
        "norm_index": indices["norm"],
        "first_component_index": indices["first_component"],
        "pca_index": indices["pca"]
    })

    results_path = results_dir / "latent_index_example.csv"
    results_df.to_csv(results_path, index=False)
    print(f"   - Resultados guardados en: {results_path}\n")

    print("=== Ejemplo completado ===")
    print("\nNotas:")
    print("- El índice latente es una señal unidimensional derivada de representaciones latentes.")
    print("- 'norm' captura la magnitud del vector latente.")
    print("- 'first_component' usa la primera dimensión latente.")
    print("- 'pca' captura la dirección de máxima varianza.")
    print("- La interpretación financiera depende del contexto y del entrenamiento del encoder.")
    print("- Este es un ejemplo experimental y educativo.")
    print("- Para análisis financiero serio, se requiere validación con datos reales.")


if __name__ == "__main__":
    main()
