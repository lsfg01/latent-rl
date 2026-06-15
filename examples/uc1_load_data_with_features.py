"""
Caso de uso 1: Carga de datos con cache y features tecnicas.

Demuestra:
  - DataCache: evita re-descargas (cache en disco CSV.gz)
  - FeatureEngineer: 8 indicadores tecnicos disponibles
  - TickerConfig: fechas y parametros distintos por activo
  - Sin internet: usa datos sinteticos para la demo

Ejecutar:
    python examples/uc1_load_data_with_features.py
"""

import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

# Permitir importar desde la raiz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latent_rl.data.cache import DataCache
from latent_rl.data.features import FeatureEngineer, AVAILABLE_FEATURES


# ── Datos sinteticos (sin internet) ──────────────────────────────────────────

def make_ohlcv(n=500, seed=1, trend=0.0003):
    rng = np.random.default_rng(seed)
    price = 100.0 * np.cumprod(1 + rng.normal(trend, 0.015, n))
    noise = lambda s: rng.normal(0, s, n)
    return pd.DataFrame({
        "Open":   price * (1 + noise(0.003)),
        "High":   price * (1 + np.abs(noise(0.007))),
        "Low":    price * (1 - np.abs(noise(0.007))),
        "Close":  price,
        "Volume": rng.integers(10_000, 100_000, n).astype(float),
    })


def mock_download(*args, **kwargs):
    """Simula yf.download sin internet."""
    return make_ohlcv()


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("UC1: Carga de datos con cache y features tecnicas")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── 1. DataCache: primera descarga ────────────────────────────────────
        print("\n[1] DataCache — primera descarga (simula Yahoo Finance):")
        cache = DataCache(tmpdir)

        with patch("yfinance.download", side_effect=mock_download):
            df_spy = cache.get_or_download("SPY", "2020-01-01", "2024-01-01", interval="1d")

        print(f"     SPY descargado: {len(df_spy)} filas  |  columnas: {list(df_spy.columns)}")

        # ── 2. DataCache: segunda lectura (desde disco) ───────────────────────
        print("\n[2] DataCache — segunda lectura (cache hit, sin red):")
        with patch("yfinance.download", side_effect=Exception("No debe llamarse")):
            df_spy2 = cache.get_or_download("SPY", "2020-01-01", "2024-01-01", interval="1d")

        print(f"     Cargado desde disco: {len(df_spy2)} filas (identico)")

        # ── 3. FeatureEngineer — todos los indicadores ────────────────────────
        print(f"\n[3] FeatureEngineer — {len(AVAILABLE_FEATURES)} features disponibles:")
        for f in AVAILABLE_FEATURES:
            print(f"     - {f}")

        fe = FeatureEngineer()

        # Subconjunto de features para trading RL
        selected = ["log_return", "rsi_14", "atr_pct", "market_regime", "ma_ratio"]
        df_features = fe.transform(df_spy, selected)

        print(f"\n     DataFrame resultante: {df_features.shape[0]} filas x {df_features.shape[1]} cols")
        print(f"     Columnas: {list(df_features.columns)}")
        print(f"\n     Estadisticas de features (ultimas 10 filas):")
        stats = df_features[selected].tail(10).describe().round(4)
        print(stats.to_string())

        # ── 4. Verificar posicion de close (columna 3) ───────────────────────
        print(f"\n[4] Columna 3 = '{df_features.columns[3]}' (FinancialEnv la necesita aqui)")
        assert df_features.columns[3] == "close", "Close debe ser columna 3"
        print("     OK — compatible con FinancialEnv")

        # ── 5. Sin NaN ───────────────────────────────────────────────────────
        nan_count = df_features.isnull().sum().sum()
        print(f"\n[5] NaN en el DataFrame: {nan_count} (debe ser 0)")
        assert nan_count == 0, "Hay NaN!"
        print("     OK — sin valores faltantes")

        # ── 6. market_regime — distribucion ──────────────────────────────────
        print("\n[6] market_regime — distribucion de senales:")
        counts = df_features["market_regime"].value_counts().sort_index()
        for val, cnt in counts.items():
            label = {1.0: "alcista (+1)", -1.0: "bajista (-1)", 0.0: "transicion (0)"}[val]
            print(f"     {label}: {cnt} ({100*cnt/len(df_features):.1f}%)")

    print("\n" + "=" * 60)
    print("UC1 completado correctamente.")
    print("=" * 60)


if __name__ == "__main__":
    main()
