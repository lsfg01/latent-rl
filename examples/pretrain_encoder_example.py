"""Ejemplo de preentrenamiento de encoder latente con autoencoder.

Este ejemplo demuestra cómo preentrenar un encoder latente usando un autoencoder
con datos financieros. Es un ejemplo experimental y educativo, no optimizado
para producción.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from latent_rl.representations import MLPLatentEncoder
from latent_rl.pretraining import AutoencoderTrainer


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
    """Función principal de preentrenamiento."""
    print("=== Ejemplo de Preentrenamiento de Encoder ===\n")

    # 1. Crear datos sintéticos
    print("1. Creando datos sintéticos...")
    data = create_synthetic_data(n_steps=200)
    print(f"   - Datos creados: {len(data)} filas\n")

    # 2. Configurar parámetros
    lookback_window = 10
    latent_dim = 16
    n_samples = 100  # Número de observaciones para preentrenamiento
    n_epochs = 10  # Entrenamiento corto para demostración

    print("2. Configurando parámetros...")
    print(f"   - Lookback window: {lookback_window}")
    print(f"   - Dimensión latente: {latent_dim}")
    print(f"   - Muestras de preentrenamiento: {n_samples}")
    print(f"   - Epochs de entrenamiento: {n_epochs}\n")

    # 3. Crear encoder latente
    print("3. Creando encoder latente...")
    input_dim = lookback_window * 5  # 5 features OHLCV
    encoder = MLPLatentEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=[64, 32],
        activation="relu",
        dropout=0.0
    )
    print("   - Encoder MLPLatentEncoder creado\n")

    # 4. Crear entrenador de autoencoder
    print("4. Creando entrenador de autoencoder...")
    trainer = AutoencoderTrainer(
        encoder=encoder,
        learning_rate=1e-3,
        batch_size=32,
        device="cpu"
    )
    print("   - AutoencoderTrainer creado\n")

    # 5. Extraer observaciones para preentrenamiento
    print("5. Extrayendo observaciones...")
    observations = trainer.collect_observations(
        data=data,
        lookback_window=lookback_window,
        n_samples=n_samples
    )
    print(f"   - Observaciones extraídas: {observations.shape}")
    print(f"   - Forma: (n_samples={observations.shape[0]}, lookback={observations.shape[1]}, features={observations.shape[2]})\n")

    # 6. Normalizar observaciones
    print("6. Normalizando observaciones...")
    observations_normalized = trainer.fit_transform_observations(observations)
    print(f"   - Observaciones normalizadas")
    print(f"   - Mean shape: {trainer.normalization_mean.shape}")
    print(f"   - Std shape: {trainer.normalization_std.shape}\n")

    # 7. Entrenar autoencoder
    print("7. Entrenando autoencoder...")
    loss_history = trainer.train(observations_normalized, n_epochs=n_epochs)

    print(f"   - Entrenamiento completado")
    print(f"   - Loss inicial: {loss_history[0]:.6f}")
    print(f"   - Loss final: {loss_history[-1]:.6f}")
    print(f"   - Mejora: {loss_history[0] - loss_history[-1]:.6f}\n")

    # 8. Guardar encoder preentrenado
    print("8. Guardando encoder preentrenado...")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = models_dir / "pretrained_encoder.pth"
    trainer.save_encoder(str(encoder_path))
    print(f"   - Encoder guardado en: {encoder_path}")
    print(f"   - Parámetros de normalización incluidos\n")

    # 9. Verificar que se puede cargar
    print("9. Verificando carga del encoder...")
    loaded_encoder = AutoencoderTrainer.load_encoder(str(encoder_path), device="cpu")
    print(f"   - Encoder cargado correctamente")
    print(f"   - Dimensión latente: {loaded_encoder.latent_dim}")
    print(f"   - Dimensión de entrada: {loaded_encoder.input_dim}\n")

    # 10. Verificar metadatos
    print("10. Verificando metadatos del checkpoint...")
    metadata = AutoencoderTrainer.load_checkpoint_metadata(str(encoder_path))
    print(f"   - Metadatos cargados correctamente")
    print(f"   - Normalization mean: {metadata['normalization_mean'] is not None}")
    print(f"   - Normalization std: {metadata['normalization_std'] is not None}\n")

    print("=== Ejemplo completado ===")
    print("\nNotas:")
    print("- Este es un preentrenamiento experimental y educativo.")
    print("- El entrenamiento es corto (10 epochs) para demostración.")
    print("- No se afirma que el encoder preentrenado mejore el rendimiento RL.")
    print("- Para resultados de producción, se requiere más entrenamiento y tuning.")
    print("- Este ejemplo sirve como base para experimentar con encoders preentrenados.")
    print("\nUso del encoder preentrenado:")
    print(f"- El encoder guardado en {encoder_path} puede cargarse en LatentDQNAgent")
    print(f"- Usar: agent.load_pretrained_encoder('{encoder_path}')")


if __name__ == "__main__":
    main()
