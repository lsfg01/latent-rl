"""
Verificación del pipeline de datos mejorado.
Usa datos sintéticos + mock de Yahoo Finance para no necesitar internet.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

pytestmark = pytest.mark.integration


def ok(_msg):
    """Conserva las anotaciones del antiguo verificador sin producir salida."""


# ── Datos sintéticos ──────────────────────────────────────────────────────────

def make_ohlcv(n=300, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    noise = lambda s: rng.normal(0, s, n)
    return pd.DataFrame({
        "open":   price * (1 + noise(0.003)),
        "high":   price * (1 + np.abs(noise(0.007))),
        "low":    price * (1 - np.abs(noise(0.007))),
        "close":  price,
        "volume": rng.integers(5_000, 50_000, n).astype(float),
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_feature_engineer():
    from latent_rl.data.features import FeatureEngineer, AVAILABLE_FEATURES

    df = make_ohlcv(200)
    fe = FeatureEngineer()

    result = fe.transform(df, AVAILABLE_FEATURES)
    expected_cols = ["open", "high", "low", "close", "volume"] + AVAILABLE_FEATURES
    assert list(result.columns) == expected_cols, f"Columnas: {list(result.columns)}"
    ok(f"Columnas correctas: {list(result.columns)}")

    assert not result.isnull().any().any(), "Hay NaN en el resultado"
    ok("Sin NaN")

    assert len(result) == len(df), "Longitud incorrecta"
    ok(f"Longitud conservada: {len(result)} filas")

    # Verificar que columna 3 sigue siendo close (para FinancialEnv)
    assert result.columns[3] == "close", "Close debe ser columna 3"
    ok("Close en posición 3 (FinancialEnv compatible)")


def test_feature_engineer_empty():
    from latent_rl.data.features import FeatureEngineer

    df = make_ohlcv(100)
    fe = FeatureEngineer()
    result = fe.transform(df, [])
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    ok("Lista vacía -> solo OHLCV")


def test_feature_engineer_subset():
    from latent_rl.data.features import FeatureEngineer

    df = make_ohlcv(150)
    fe = FeatureEngineer()
    result = fe.transform(df, ["log_return", "rsi_14"])
    assert "log_return" in result.columns
    assert "rsi_14" in result.columns
    assert "volume_ratio" not in result.columns
    ok("Subconjunto correcto")

    # RSI debe estar en [0, 100]
    assert result["rsi_14"].between(0, 100).all(), "RSI fuera de [0, 100]"
    ok("RSI en rango [0, 100]")

    # log_return primera fila = 0.0
    assert result["log_return"].iloc[0] == 0.0, "Primera fila log_return != 0"
    ok("Primera fila log_return = 0")


def test_datacache():
    from latent_rl.data.cache import DataCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DataCache(tmpdir)

        # Simular descarga mockeando yf.download
        df_mock = make_ohlcv(150)
        df_mock.columns = ["open", "high", "low", "close", "volume"]

        def mock_download(*args, **kwargs):
            # yfinance devuelve columnas capitalizadas
            result = df_mock.copy()
            result.columns = ["Open", "High", "Low", "Close", "Volume"]
            return result

        with patch("yfinance.download", side_effect=mock_download):
            df1 = cache.get_or_download("TEST", "2020-01-01", "2023-12-31")

        ok(f"Primera carga (descarga): {len(df1)} filas")

        # Segunda carga debe leer del disco sin llamar a yfinance
        with patch("yfinance.download", side_effect=Exception("No debe llamarse")):
            df2 = cache.get_or_download("TEST", "2020-01-01", "2023-12-31")

        ok(f"Segunda carga (caché hit): {len(df2)} filas")
        assert len(df1) == len(df2), "Caché devuelve datos distintos"
        ok("Datos de caché idénticos a los originales")

        # Verificar que el archivo existe
        cache_files = list(Path(tmpdir).glob("*.csv.gz"))
        assert len(cache_files) == 1, f"Esperado 1 archivo, encontrado {len(cache_files)}"
        ok(f"Archivo de caché creado: {cache_files[0].name}")

        # Probar clear
        removed = cache.clear("TEST")
        assert removed == 1
        ok("DataCache.clear() elimina archivos correctamente")


def test_config_new_fields():
    from latent_rl.experiments.config import ExperimentConfig, TickerConfig

    # Configuración básica con valores por defecto
    cfg = ExperimentConfig()
    assert cfg.cache_dir == ".data_cache"
    assert cfg.interval == "1d"
    assert cfg.features == []
    ok(f"Valores por defecto OK — features={cfg.features}")

    # TickerConfig overrides
    cfg2 = ExperimentConfig(
        tickers=["SPY", "BTC-USD"],
        start_date="2020-01-01",
        end_date="2024-01-01",
        ticker_configs=[
            TickerConfig("SPY",     start_date="2015-01-01"),
            TickerConfig("BTC-USD", start_date="2019-01-01", n_obs=500, interval="1d"),
        ]
    )
    spy_params = cfg2.resolve_ticker_params("SPY")
    btc_params = cfg2.resolve_ticker_params("BTC-USD")

    assert spy_params["start"] == "2015-01-01"
    assert spy_params["end"]   == "2024-01-01"  # hereda global
    assert btc_params["start"] == "2019-01-01"
    assert btc_params["n_obs"] == 500
    ok("resolve_ticker_params() funciona correctamente")

    # Ticker sin TickerConfig usa globales
    cfg3 = ExperimentConfig(
        tickers=["AAPL"],
        start_date="2018-01-01",
        end_date="2023-01-01",
    )
    aapl_params = cfg3.resolve_ticker_params("AAPL")
    assert aapl_params["start"] == "2018-01-01"
    assert aapl_params["end"]   == "2023-01-01"
    ok("Ticker sin TickerConfig hereda fechas globales")

    # Validación de features inválidos
    with pytest.raises(ValueError):
        ExperimentConfig(tickers=["SPY"], features=["feature_inexistente"])

    # Validación de TickerConfig con ticker no en tickers
    with pytest.raises(ValueError):
        ExperimentConfig(
            tickers=["SPY"],
            ticker_configs=[TickerConfig("AAPL")]
        )


def test_load_yfinance_data_with_features():
    from latent_rl.experiments.config import ExperimentConfig
    from latent_rl.experiments.utils import load_ticker_with_config

    df_mock = make_ohlcv(200)

    def mock_download(*args, **kwargs):
        result = df_mock.copy()
        result.columns = ["Open", "High", "Low", "Close", "Volume"]
        return result

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("yfinance.download", side_effect=mock_download):
            cfg = ExperimentConfig(
                tickers=["SPY"],
                start_date="2020-01-01",
                end_date="2023-12-31",
                cache_dir=tmpdir,
                features=["log_return", "volume_ratio"],
            )
            df = load_ticker_with_config("SPY", cfg)

    assert "log_return"   in df.columns
    assert "volume_ratio" in df.columns
    assert df.columns[3]  == "close"  # posición 3 sigue siendo close
    assert not df.isnull().any().any()
    ok(f"Columnas: {list(df.columns)} — sin NaN")


def test_pipeline_with_features():
    from latent_rl.experiments.config import ExperimentConfig
    from latent_rl.experiments.runner import run_single_seed

    df_raw = make_ohlcv(300)

    # Aplicar features manualmente para simular lo que hace load_ticker_with_config
    from latent_rl.data.features import FeatureEngineer
    fe = FeatureEngineer()
    data = fe.transform(df_raw, ["log_return", "volume_ratio"])

    split = int(0.7 * len(data))
    data_is  = data.iloc[:split].reset_index(drop=True)
    data_oos = data.iloc[split:].reset_index(drop=True)

    assert data.shape[1] == 7, f"Esperado 7 cols, got {data.shape[1]}"
    ok(f"n_features = {data.shape[1]} (5 OHLCV + 2 features)")

    cfg = ExperimentConfig(
        seeds=[0],
        n_training_episodes=2,
        n_eval_episodes=1,
        max_steps_per_episode=40,
        lookback_window=10,
        dqn_hidden_dim=32,
        dqn_buffer_capacity=200,
        dqn_batch_size=16,
        latent_dim=8,
        encoder_hidden_dims=[8, 16],
        encoder_type="tcn",
        tcn_dilations=[1, 2, 4],
        tcn_channels=16,
        latent_q_hidden_dim=32,
        latent_buffer_capacity=200,
        latent_batch_size=16,
        pretrain_n_samples=30,
        pretrain_n_epochs=2,
        features=["log_return", "volume_ratio"],  # para que cfg sea consistente
        run_arms=["A", "B", "C"],
    )

    results = run_single_seed(seed=0, data_is=data_is, data_oos=data_oos, cfg=cfg)

    for name in ["RandomAgent", "BuyAndHoldAgent", "A", "B", "C"]:
        assert name in results
        ok(f"{name}: IS={results[name]['is']['total_return']:.4f}  OOS={results[name]['oos']['total_return']:.4f}")


# ── Runner ────────────────────────────────────────────────────────────────────

