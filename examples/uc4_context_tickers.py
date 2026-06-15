"""
Caso de uso 4: Features de activos correlacionados (context tickers).

Demuestra:
  - TickerConfig.context_tickers: activos correlacionados como features extra
  - load_context_features(): carga log_return de activos de contexto
  - Alineacion posicional (mismo calendario de trading)
  - El encoder latente ve informacion de mercado mas amplia

Justificacion:
  Si AAPL sube mientras SPY baja, es informacion valiosa sobre el sector.
  El log_return del S&P500 (SPY) como feature del Bitcoin (BTC-USD) captura
  la correlacion riesgo-on/riesgo-off entre activos.

Ejecutar:
    python examples/uc4_context_tickers.py
"""

import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latent_rl.data.features import FeatureEngineer
from latent_rl.experiments.utils import load_context_features
from latent_rl.experiments.config import ExperimentConfig, TickerConfig


# ── Datos sinteticos correlacionados ─────────────────────────────────────────

def make_correlated_ohlcv(n=300, seed=99, correlation=0.6):
    """Genera un activo con retornos parcialmente correlacionados al base."""
    rng = np.random.default_rng(seed)
    base_ret  = rng.normal(0.0002, 0.012, n)
    idio_ret  = rng.normal(0.0001, 0.010, n)
    combined  = correlation * base_ret + (1 - correlation) * idio_ret
    price = 100.0 * np.cumprod(1 + combined)
    noise = lambda s: rng.normal(0, s, n)
    return pd.DataFrame({
        "Open":   price * (1 + noise(0.003)),
        "High":   price * (1 + np.abs(noise(0.006))),
        "Low":    price * (1 - np.abs(noise(0.006))),
        "Close":  price,
        "Volume": rng.integers(5_000, 50_000, n).astype(float),
    })


SPY_MOCK = make_correlated_ohlcv(300, seed=1,  correlation=0.8)
QQQ_MOCK = make_correlated_ohlcv(300, seed=2,  correlation=0.7)
BTC_MOCK = make_correlated_ohlcv(300, seed=10, correlation=0.3)

_MOCKS = {"SPY": SPY_MOCK, "QQQ": QQQ_MOCK, "BTC-USD": BTC_MOCK}

def mock_download(ticker=None, *args, **kwargs):
    data = _MOCKS.get(ticker, SPY_MOCK)
    return data.copy()


def main():
    print("=" * 60)
    print("UC4: Features de activos correlacionados (context tickers)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── 1. Sin context tickers (baseline) ─────────────────────────────────
        print("\n[1] Carga base: solo BTC-USD sin contexto")
        fe = FeatureEngineer()
        df_btc_raw = BTC_MOCK.copy()
        df_btc_raw.columns = ["open", "high", "low", "close", "volume"]
        df_btc = fe.transform(df_btc_raw, ["log_return", "rsi_14"])
        print(f"     Columnas SIN contexto: {list(df_btc.columns)}")
        print(f"     Shape: {df_btc.shape}")

        # ── 2. Con context tickers ────────────────────────────────────────────
        print("\n[2] Carga con context tickers [SPY, QQQ]")

        with patch("yfinance.download", side_effect=mock_download):
            ctx_df = load_context_features(
                context_tickers=["SPY", "QQQ"],
                start="2020-01-01",
                end="2024-01-01",
                n_obs=None,
                interval="1d",
                cache_dir=tmpdir,
                primary_len=len(df_btc),
            )

        df_btc_ctx = pd.concat([df_btc.reset_index(drop=True), ctx_df], axis=1)
        print(f"     Columnas CON contexto:  {list(df_btc_ctx.columns)}")
        print(f"     Shape: {df_btc_ctx.shape}  ({df_btc_ctx.shape[1] - df_btc.shape[1]} features extra)")

        # ── 3. Correlacion con el activo principal ────────────────────────────
        print("\n[3] Correlacion de los context features con log_return de BTC-USD:")
        for ctx_col in ["spy_log_return", "qqq_log_return"]:
            corr = df_btc_ctx["log_return"].corr(df_btc_ctx[ctx_col])
            print(f"     BTC log_return vs {ctx_col}: r={corr:+.3f}")

        # ── 4. TickerConfig con context_tickers ───────────────────────────────
        print("\n[4] ExperimentConfig con context_tickers en TickerConfig:")
        cfg = ExperimentConfig(
            tickers=["BTC-USD"],
            ticker_configs=[
                TickerConfig(
                    ticker="BTC-USD",
                    start_date="2020-01-01",
                    context_tickers=["SPY", "QQQ"],
                )
            ],
            features=["log_return", "rsi_14"],
        )
        tc = cfg.get_ticker_config("BTC-USD")
        print(f"     Ticker principal: BTC-USD")
        print(f"     Context tickers:  {tc.context_tickers}")
        print(f"     Features base:    {cfg.features}")
        print(f"     Features totales: {cfg.features} + {[t.lower().replace('-','_')+'_log_return' for t in tc.context_tickers]}")

        # ── 5. Stats de los context features ─────────────────────────────────
        print("\n[5] Estadisticas de features de contexto:")
        print(f"{'Feature':<22} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
        print("-" * 56)
        for col in ["log_return", "spy_log_return", "qqq_log_return"]:
            s = df_btc_ctx[col]
            print(f"{col:<22} {s.mean():>+8.4f} {s.std():>8.4f} {s.min():>+8.4f} {s.max():>+8.4f}")

        # ── 6. Alineacion posicional ──────────────────────────────────────────
        print("\n[6] Alineacion posicional:")
        print(f"     Longitud BTC-USD: {len(df_btc)} filas")
        print(f"     Longitud ctx_df:  {len(ctx_df)} filas")
        print(f"     Longitud merged:  {len(df_btc_ctx)} filas")
        print("     Nota: para activos del mismo mercado las fechas coinciden.")
        print("     Para cross-market (acciones vs cripto): se usa la cola (tail)")
        print("     del activo de contexto para alinear con el activo principal.")

    print("\n" + "=" * 60)
    print("UC4 completado correctamente.")
    print("=" * 60)


if __name__ == "__main__":
    main()
