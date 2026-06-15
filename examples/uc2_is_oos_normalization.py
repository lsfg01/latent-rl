"""
Caso de uso 2: Normalizacion IS/OOS sin data leakage.

Demuestra:
  - FeatureNormalizer: z-score fit en IS, transform en OOS
  - Sin leakage: los parametros OOS no afectan la normalizacion
  - OHLCV intactos: FinancialEnv necesita precios reales
  - normalize_is_oos(): funcion de utilidad del pipeline

Ejecutar:
    python examples/uc2_is_oos_normalization.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latent_rl.data.features import FeatureEngineer
from latent_rl.data.normalizer import FeatureNormalizer


# ── Datos sinteticos ──────────────────────────────────────────────────────────

def make_ohlcv(n=600, seed=42):
    rng = np.random.default_rng(seed)
    # Simular un mercado con tendencia alcista en IS y bajista en OOS
    trend_is  = rng.normal(0.0005, 0.015, int(n * 0.7))
    trend_oos = rng.normal(-0.0003, 0.018, n - int(n * 0.7))
    ret = np.concatenate([trend_is, trend_oos])
    price = 100.0 * np.cumprod(1 + ret)

    noise = lambda s: rng.normal(0, s, n)
    return pd.DataFrame({
        "open":   price * (1 + noise(0.003)),
        "high":   price * (1 + np.abs(noise(0.007))),
        "low":    price * (1 - np.abs(noise(0.007))),
        "close":  price,
        "volume": rng.integers(5_000, 50_000, n).astype(float),
    })


def main():
    print("=" * 60)
    print("UC2: Normalizacion IS/OOS sin data leakage")
    print("=" * 60)

    # ── 1. Preparar datos con features ────────────────────────────────────────
    print("\n[1] Generando datos con features tecnicas...")
    df = make_ohlcv(600)
    fe = FeatureEngineer()
    data = fe.transform(df, ["log_return", "rsi_14", "atr_pct", "volume_ratio"])

    split = int(0.7 * len(data))
    data_is  = data.iloc[:split].reset_index(drop=True)
    data_oos = data.iloc[split:].reset_index(drop=True)

    print(f"     Total: {len(data)} filas  |  IS: {len(data_is)}  |  OOS: {len(data_oos)}")

    # ── 2. Estadisticas ANTES de normalizar ────────────────────────────────────
    feature_cols = ["log_return", "rsi_14", "atr_pct", "volume_ratio"]

    print("\n[2] Estadisticas ANTES de normalizar:")
    print(f"{'Feature':<18} {'IS mean':>10} {'IS std':>10} {'OOS mean':>10} {'OOS std':>10}")
    print("-" * 58)
    for col in feature_cols:
        print(
            f"{col:<18} "
            f"{data_is[col].mean():>10.4f} "
            f"{data_is[col].std():>10.4f} "
            f"{data_oos[col].mean():>10.4f} "
            f"{data_oos[col].std():>10.4f}"
        )

    # ── 3. Fit en IS, transform en ambos ─────────────────────────────────────
    print("\n[3] Aplicando FeatureNormalizer (fit en IS)...")
    norm = FeatureNormalizer()
    data_is_norm  = norm.fit_transform(data_is)
    data_oos_norm = norm.transform(data_oos)

    print(f"     Parametros IS ajustados:")
    for col in feature_cols:
        print(f"       {col}: mean={norm.mean_[col]:+.4f}, std={norm.std_[col]:.4f}")

    # ── 4. Estadisticas DESPUES de normalizar ─────────────────────────────────
    print("\n[4] Estadisticas DESPUES de normalizar:")
    print(f"{'Feature':<18} {'IS mean':>10} {'IS std':>10} {'OOS mean':>10} {'OOS std':>10}")
    print("-" * 58)
    for col in feature_cols:
        print(
            f"{col:<18} "
            f"{data_is_norm[col].mean():>10.4e} "
            f"{data_is_norm[col].std():>10.4f} "
            f"{data_oos_norm[col].mean():>10.4f} "
            f"{data_oos_norm[col].std():>10.4f}"
        )

    # ── 5. Verificar ausencia de leakage ─────────────────────────────────────
    print("\n[5] Verificacion de ausencia de leakage:")
    for col in feature_cols:
        is_mean_norm = abs(data_is_norm[col].mean())
        is_std_norm  = abs(data_is_norm[col].std() - 1.0)
        assert is_mean_norm < 1e-5, f"IS mean != 0 para {col}"
        assert is_std_norm  < 1e-4, f"IS std != 1 para {col}"
        print(f"     {col}: IS mean ~0 ({is_mean_norm:.2e}), IS std ~1 ({data_is_norm[col].std():.4f})  [OK]")

    # ── 6. OHLCV sin modificar ────────────────────────────────────────────────
    print("\n[6] Verificacion de OHLCV intactos:")
    for col in ["open", "high", "low", "close", "volume"]:
        assert data_is_norm[col].equals(data_is[col]), f"{col} fue modificado!"
        print(f"     {col}: sin cambios  [OK]")

    # ── 7. Resumen clave ──────────────────────────────────────────────────────
    print("\n[7] Resumen — por que importa:")
    print("     - IS: media=0, std=1 por construccion del fit")
    print("     - OOS: media != 0 (distribucion distinta a IS, normal en finanzas)")
    print("     - OHLCV: precios reales intactos (FinancialEnv los necesita)")
    print("     - El encoder ve features en escala comparable [0, 3 sigmas aprox]")
    print("     - Sin leakage: estadisticos OOS no influyeron en el normalizador")

    print("\n" + "=" * 60)
    print("UC2 completado correctamente.")
    print("=" * 60)


if __name__ == "__main__":
    main()
