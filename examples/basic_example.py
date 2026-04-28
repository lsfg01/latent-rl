"""Ejemplo básico de uso de la librería latent-rl.

Este ejemplo demuestra el flujo completo sin usar clases abstractas directamente.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import os

from latent_rl.data import CSVDataLoader, DataPreprocessor
from latent_rl.envs import FinancialEnv
from latent_rl.representations import MLPLatentEncoder
from latent_rl.evaluation import FinancialMetrics


def create_sample_data():
    """Crea datos OHLCV de ejemplo."""
    np.random.seed(42)

    n_samples = 100
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")

    # Simular precios con random walk
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = base_price * np.cumprod(1 + returns)

    data = pd.DataFrame({
        "Date": dates,
        "Open": prices * (1 + np.random.normal(0, 0.005, n_samples)),
        "High": prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        "Close": prices,
        "Volume": np.random.randint(1000, 10000, n_samples)
    })

    return data


def main():
    """Función principal del ejemplo."""
    print("=== Ejemplo Básico de Latent RL ===\n")

    # 1. Crear datos de ejemplo
    print("1. Creando datos de ejemplo...")
    data = create_sample_data()
    print(f"   Datos creados: {len(data)} filas")
    print(f"   Columnas: {list(data.columns)}\n")

    # 2. Guardar datos en CSV temporal
    print("2. Guardando datos en CSV temporal...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
        data.to_csv(f, index=False)

    try:
        # 3. Cargar datos
        print("3. Cargando datos desde CSV...")
        loader = CSVDataLoader(temp_path)
        loaded_data = loader.load_ohlcv()
        print(f"   Datos cargados: {len(loaded_data)} filas\n")

        # 4. Preprocesar datos
        print("4. Preprocesando datos...")
        preprocessor = DataPreprocessor()

        # Limpiar NaN
        clean_data = preprocessor.clean_nan(loaded_data, method="forward_fill")
        print(f"   Datos limpios: {len(clean_data)} filas")

        # Normalizar
        normalized_data = preprocessor.normalize(clean_data, method="minmax")
        print(f"   Datos normalizados\n")

        # 5. Crear entorno financiero
        print("5. Creando entorno financiero...")
        env = FinancialEnv(
            data=normalized_data,
            initial_balance=10000.0,
            transaction_cost=0.001,
            lookback_window=10
        )
        print(f"   Espacio de observaciones: {env.observation_space.shape}")
        print(f"   Espacio de acciones: {env.action_space.n}\n")

        # 6. Crear encoder latente
        print("6. Creando encoder latente...")
        encoder = MLPLatentEncoder(
            input_dim=env.n_features,
            latent_dim=8,
            hidden_dims=[16, 12],
            activation="relu"
        )
        print(f"   Encoder creado: {encoder.input_dim} -> {encoder.latent_dim}\n")

        # 7. Simular episodio simple
        print("7. Simulando episodio simple...")
        obs, info = env.reset()
        print(f"   Estado inicial:")
        print(f"   - Balance: {info['balance']:.2f}")
        print(f"   - Posición: {info['position']}")
        print(f"   - Precio actual: {info['current_price']:.2f}\n")

        # Ejecutar algunos pasos
        print("   Ejecutando pasos...")
        total_reward = 0.0
        steps = 0

        for step in range(20):
            # Acción aleatoria (en un caso real, usaríamos una política entrenada)
            action = np.random.choice([0, 1, 2])  # 0=hold, 1=buy, 2=sell

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if step % 5 == 0:
                print(f"   Paso {step}: Acción={action}, Recompensa={reward:.4f}, "
                      f"Balance={info['balance']:.2f}, Posición={info['position']}")

            if terminated or truncated:
                break

        print(f"\n   Episodio finalizado después de {steps} pasos")
        print(f"   Recompensa total: {total_reward:.4f}")
        print(f"   Balance final: {info['balance']:.2f}")
        print(f"   Profit realizado: {info['realized_profit']:.2f}\n")

        # 8. Probar encoder con observaciones
        print("8. Probando encoder latente...")
        # Crear batch de observaciones
        batch_size = 5
        obs_batch = torch.randn(batch_size, env.n_features)

        # Codificar
        with torch.no_grad():
            latent_repr = encoder.encode(obs_batch)
            reconstructed = encoder.decode(latent_repr)

        print(f"   Entrada: {obs_batch.shape}")
        print(f"   Representación latente: {latent_repr.shape}")
        print(f"   Reconstrucción: {reconstructed.shape}")
        print(f"   Pérdida de reconstrucción: {encoder.reconstruction_loss(obs_batch).item():.6f}\n")

        # 9. Calcular métricas financieras
        print("9. Calculando métricas financieras...")
        # Simular retornos del episodio
        simulated_returns = pd.Series(np.random.normal(0.001, 0.02, 50))

        sharpe = FinancialMetrics.sharpe_ratio(simulated_returns)
        sortino = FinancialMetrics.sortino_ratio(simulated_returns)
        max_dd = FinancialMetrics.max_drawdown(simulated_returns)
        total_ret = FinancialMetrics.total_return(simulated_returns)
        win_rate = FinancialMetrics.win_rate(simulated_returns)

        print(f"   Sharpe Ratio: {sharpe:.4f}")
        print(f"   Sortino Ratio: {sortino:.4f}")
        print(f"   Max Drawdown: {max_dd:.4f}")
        print(f"   Retorno Total: {total_ret:.4f}")
        print(f"   Win Rate: {win_rate:.2%}\n")

        print("=== Ejemplo completado exitosamente ===")

    finally:
        # Limpiar archivo temporal
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    main()