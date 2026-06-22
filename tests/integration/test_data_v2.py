"""
Verificacion de las mejoras de datos v2:
  - FeatureNormalizer IS/OOS sin leakage
  - Walk-Forward splits
  - Nuevos campos en ExperimentConfig / TickerConfig
  - Context ticker features (mocked)
  - Pipeline completo con normalizacion
"""

import tempfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

pytestmark = pytest.mark.integration


def ok(_msg):
    """Conserva las anotaciones del antiguo verificador sin producir salida."""


# Datos sinteticos compartidos
def make_ohlcv(n=400, seed=42) -> pd.DataFrame:
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

def test_normalizer_no_leakage():
    from latent_rl.data.features import FeatureEngineer
    from latent_rl.data.normalizer import FeatureNormalizer

    df = make_ohlcv(300)
    fe = FeatureEngineer()
    data = fe.transform(df, ["log_return", "rsi_14", "volume_ratio"])

    split = int(0.7 * len(data))
    is_raw  = data.iloc[:split].reset_index(drop=True)
    oos_raw = data.iloc[split:].reset_index(drop=True)

    norm = FeatureNormalizer()
    is_norm  = norm.fit_transform(is_raw)
    oos_norm = norm.transform(oos_raw)

    # IS normalizado: media ~0, std ~1 en features tecnicas
    for col in ["log_return", "rsi_14", "volume_ratio"]:
        is_mean = is_norm[col].mean()
        is_std  = is_norm[col].std()
        assert abs(is_mean) < 1e-6, f"{col}: IS mean={is_mean:.6f} (esperado ~0)"
        assert abs(is_std - 1.0) < 1e-4, f"{col}: IS std={is_std:.4f} (esperado ~1)"
        ok(f"IS {col}: mean={is_mean:.4e}, std={is_std:.4f}")

    # OOS normalizado con params IS: NO es cero-centrado necesariamente
    # (distribuciones IS y OOS difieren en series temporales financieras)
    ok("OOS transformado con params IS (sin leakage)")

    # OHLCV no se toca
    pd.testing.assert_frame_equal(is_norm[["open", "high", "low", "close", "volume"]],
                                   is_raw[["open", "high", "low", "close", "volume"]])
    ok("Columnas OHLCV no normalizadas (intactas)")

    # Sin NaN en ningun split
    assert not is_norm.isnull().any().any()
    assert not oos_norm.isnull().any().any()
    ok("Sin NaN en IS ni en OOS normalizados")


def test_walk_forward_splits():
    from latent_rl.experiments.utils import walk_forward_splits

    data = make_ohlcv(300)
    splits = walk_forward_splits(data, n_windows=5, is_ratio=0.6)

    assert len(splits) == 5, f"Esperadas 5 ventanas, got {len(splits)}"
    ok(f"Numero de ventanas: {len(splits)}")

    # El ratio fija el ancla inicial; después el IS se expande con cada OOS.
    assert len(splits[0][0]) == int(len(data) * 0.6)
    for current, following in zip(splits, splits[1:]):
        wf_is, wf_oos = current
        next_is, _ = following
        assert len(next_is) == len(wf_is) + len(wf_oos)

    # Sin solapamiento: los indices de ventanas consecutivas no se solapan
    # (comprobado indirectamente: ventana i empieza donde acaba i-1)
    splits2 = walk_forward_splits(data, n_windows=3, is_ratio=0.7)
    total_oos = sum(len(oos_) for _, oos_ in splits2)
    assert total_oos == len(data) - int(len(data) * 0.7)

    # Error con datos insuficientes
    tiny = make_ohlcv(15)
    with pytest.raises(ValueError):
        walk_forward_splits(tiny, n_windows=5)


def test_normalize_is_oos_util():
    from latent_rl.data.features import FeatureEngineer
    from latent_rl.experiments.utils import normalize_is_oos

    df = make_ohlcv(200)
    fe = FeatureEngineer()
    data = fe.transform(df, ["log_return", "atr_pct"])

    split = int(0.7 * len(data))
    data_is  = data.iloc[:split].reset_index(drop=True)
    data_oos = data.iloc[split:].reset_index(drop=True)

    is_norm, oos_norm, norm = normalize_is_oos(data_is, data_oos)

    assert norm.fitted, "Normalizador debe estar ajustado"
    assert "log_return" in norm.mean_
    assert "atr_pct"    in norm.mean_
    ok(f"Normalizador ajustado — features: {list(norm.mean_.keys())}")

    # IS centrado
    assert abs(is_norm["log_return"].mean()) < 1e-6
    ok("IS log_return centrado en 0")

    # OHLCV intactos
    assert is_norm["close"].equals(data_is["close"])
    ok("Close intacto tras normalize_is_oos")


def test_config_new_fields():
    from latent_rl.experiments.config import ExperimentConfig, TickerConfig

    # Valores por defecto
    cfg = ExperimentConfig(tickers=["SPY"])
    assert cfg.normalize_features is True
    assert cfg.wf_enabled is False
    assert cfg.wf_n_windows == 5
    assert cfg.wf_is_ratio == 0.6
    ok("Valores por defecto OK")

    # context_tickers en TickerConfig
    tc = TickerConfig("AAPL", context_tickers=["SPY", "QQQ"])
    assert tc.context_tickers == ["SPY", "QQQ"]
    ok("context_tickers en TickerConfig OK")

    # WF con validacion
    cfg_wf = ExperimentConfig(
        tickers=["SPY"],
        wf_enabled=True,
        wf_n_windows=4,
        wf_is_ratio=0.65,
    )
    assert cfg_wf.wf_enabled
    assert cfg_wf.wf_n_windows == 4
    ok("ExperimentConfig con WF habilitado OK")

    # WF invalido: menos de 2 ventanas
    with pytest.raises(ValueError):
        ExperimentConfig(tickers=["SPY"], wf_enabled=True, wf_n_windows=1)

    # WF invalido: is_ratio fuera de rango
    with pytest.raises(ValueError):
        ExperimentConfig(tickers=["SPY"], wf_enabled=True, wf_n_windows=3, wf_is_ratio=1.1)

    # normalize_features=False desactiva normalizacion
    cfg_no_norm = ExperimentConfig(tickers=["SPY"], normalize_features=False)
    assert not cfg_no_norm.normalize_features
    ok("normalize_features=False OK")


def test_context_features_mocked():
    from latent_rl.experiments.utils import load_context_features

    df_mock_spy = make_ohlcv(300, seed=1)

    def mock_download(*args, **kwargs):
        result = df_mock_spy.copy()
        result.columns = ["Open", "High", "Low", "Close", "Volume"]
        return result

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("yfinance.download", side_effect=mock_download):
            ctx_df = load_context_features(
                context_tickers=["SPY", "QQQ"],
                start="2020-01-01",
                end="2024-01-01",
                n_obs=None,
                interval="1d",
                cache_dir=tmpdir,
                primary_len=300,
            )

    assert "ctx_SPY_log_return" in ctx_df.columns, f"Columnas: {list(ctx_df.columns)}"
    assert "ctx_QQQ_log_return" in ctx_df.columns
    assert len(ctx_df) == 300
    assert not ctx_df.isnull().any().any()
    ok(f"Columnas de contexto: {list(ctx_df.columns)}")
    ok(f"Longitud alineada: {len(ctx_df)} filas")
    ok("Sin NaN en features de contexto")

    # Lista vacia devuelve DataFrame vacio
    ctx_empty = load_context_features([], "2020-01-01", "2024-01-01", None, "1d", tmpdir, 100)
    assert ctx_empty.empty
    ok("Lista vacia devuelve DataFrame vacio")


def test_pipeline_with_normalization():
    from latent_rl.experiments.config import ExperimentConfig
    from latent_rl.experiments.runner import run_single_seed
    from latent_rl.experiments.utils import normalize_is_oos, split_data
    from latent_rl.data.features import FeatureEngineer

    df_raw = make_ohlcv(300)
    fe = FeatureEngineer()
    data = fe.transform(df_raw, ["log_return", "volume_ratio"])

    data_is, data_oos = split_data(data, train_ratio=0.7)
    data_is_norm, data_oos_norm, norm = normalize_is_oos(data_is, data_oos)

    # Comprobar que el normalizador no afecta la columna close
    assert data_is_norm["close"].equals(data_is["close"])
    ok("Close no alterado por normalizacion")

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
        features=["log_return", "volume_ratio"],
        normalize_features=True,
        run_arms=["A", "B", "C"],
    )

    results = run_single_seed(
        seed=0, data_is=data_is_norm, data_oos=data_oos_norm, cfg=cfg
    )

    expected_agents = [
        "RandomAgent", "BuyAndHoldAgent", "A", "B", "C",
    ]
    for name in expected_agents:
        assert name in results
        ok(f"{name}: IS={results[name]['is']['total_return']:.4f}  "
           f"OOS={results[name]['oos']['total_return']:.4f}")


def test_walk_forward_full():
    from latent_rl.experiments.utils import walk_forward_splits, normalize_is_oos
    from latent_rl.experiments.runner import run_single_seed
    from latent_rl.experiments.config import ExperimentConfig
    from latent_rl.data.features import FeatureEngineer

    df_raw = make_ohlcv(400)
    fe = FeatureEngineer()
    data = fe.transform(df_raw, ["log_return"])

    splits = walk_forward_splits(data, n_windows=3, is_ratio=0.6)
    assert len(splits) == 3
    ok(f"Generadas {len(splits)} ventanas WF")

    cfg = ExperimentConfig(
        seeds=[0],
        n_training_episodes=2,
        n_eval_episodes=1,
        max_steps_per_episode=40,
        lookback_window=5,
        dqn_hidden_dim=16,
        dqn_buffer_capacity=100,
        dqn_batch_size=8,
        latent_dim=4,
        encoder_hidden_dims=[4, 8],
        encoder_type="tcn",
        tcn_dilations=[1, 2],
        tcn_channels=8,
        latent_q_hidden_dim=16,
        latent_buffer_capacity=100,
        latent_batch_size=8,
        pretrain_n_samples=20,
        pretrain_n_epochs=2,
        features=["log_return"],
        run_arms=["A"],
    )

    wf_results = []
    for wf_idx, (wf_is, wf_oos) in enumerate(splits):
        wf_is_n, wf_oos_n, _ = normalize_is_oos(wf_is, wf_oos)
        res = run_single_seed(seed=0, data_is=wf_is_n, data_oos=wf_oos_n, cfg=cfg)
        wf_results.append(res)
        ok(f"Ventana {wf_idx+1}: completada OK")

    assert len(wf_results) == 3
    ok(f"Walk-Forward completado: {len(wf_results)} ventanas procesadas")


# ── Runner ────────────────────────────────────────────────────────────────────

