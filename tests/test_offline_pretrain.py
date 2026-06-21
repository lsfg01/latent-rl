"""Tests para PretrainConfig y el módulo offline de preentrenamiento."""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from latent_rl.pretraining.config import PretrainConfig


class TestPretrainConfig:
    """Tests para PretrainConfig (validación de configuración)."""

    def _minimal(self, **overrides):
        base = dict(
            universe=["AAPL", "MSFT", "GOOGL"],
            eval_tickers=["SPY"],
            relatives={"SPY": []},
            features=["log_return", "rsi_14"],
        )
        base.update(overrides)
        return PretrainConfig(**base)

    def test_valid_config(self):
        cfg = self._minimal()
        assert cfg.universe == ["AAPL", "MSFT", "GOOGL"]

    def test_empty_universe_raises(self):
        with pytest.raises(ValueError, match="universe"):
            self._minimal(universe=[])

    def test_eval_ticker_in_universe_raises(self):
        """Un eval_ticker en universe viola el principio anti-fuga."""
        with pytest.raises(ValueError, match="anti-fuga"):
            self._minimal(universe=["AAPL", "SPY", "MSFT"])

    def test_relative_in_universe_raises(self):
        """Un pariente del eval_ticker en universe también lo viola."""
        with pytest.raises(ValueError, match="anti-fuga"):
            # IVV es pariente de SPY
            self._minimal(
                universe=["AAPL", "IVV", "MSFT"],
                relatives={"SPY": ["IVV"]},
            )

    def test_val_ratio_validation(self):
        with pytest.raises(ValueError, match="val_ratio"):
            self._minimal(val_ratio=0.0)
        with pytest.raises(ValueError, match="val_ratio"):
            self._minimal(val_ratio=1.0)

    def test_k_must_be_positive(self):
        with pytest.raises(ValueError, match="k debe"):
            self._minimal(k=0)

    def test_lambda_must_be_non_negative(self):
        with pytest.raises(ValueError, match="lambda_forecast"):
            self._minimal(lambda_forecast=-0.1)

    def test_unknown_feature_raises(self):
        with pytest.raises(ValueError, match="no reconocidos"):
            self._minimal(features=["log_return", "foobar_feature"])

    def test_missing_log_return_raises(self):
        """log_return es requerido para forecasting targets."""
        with pytest.raises(ValueError, match="log_return"):
            self._minimal(features=["rsi_14", "atr_pct"])

    def test_excluded_symbols(self):
        cfg = self._minimal(
            relatives={"SPY": ["IVV", "VOO"]},
        )
        excluded = cfg.excluded_symbols()
        assert "SPY" in excluded
        assert "IVV" in excluded
        assert "VOO" in excluded


class TestOfflineWindowGeneration:
    """Verifica que las ventanas nunca cruzan activos."""

    def test_windows_do_not_cross_assets(self):
        """Si concatenamos ventanas por activo, no hay cruces."""
        from latent_rl.pretraining.encoder_pretrainer import EncoderPretrainer
        from latent_rl.representations import TCNLatentEncoder

        L, F, k = 5, 2, 2
        features = ["log_return", "rsi_14"]

        enc = TCNLatentEncoder(L, F, 8, kernel_size=3, dilations=[1])
        p = EncoderPretrainer(enc, k=k)

        rng = np.random.default_rng(42)
        n = 50
        df_a = pd.DataFrame({
            "log_return": rng.normal(0, 0.01, n),
            "rsi_14": rng.normal(50, 10, n),
        })
        df_b = pd.DataFrame({
            "log_return": rng.normal(0, 0.02, n),
            "rsi_14": rng.normal(60, 5, n),
        })

        Xa, Ya = p.make_windows(df_a, features, L)
        Xb, Yb = p.make_windows(df_b, features, L)
        X_concat = np.concatenate([Xa, Xb], axis=0)

        # El número de ventanas = sum por activo (sin cruce)
        expected_per_asset = n - L - k + 1
        assert len(Xa) == expected_per_asset
        assert len(Xb) == expected_per_asset
        assert len(X_concat) == 2 * expected_per_asset

    def test_artifact_has_required_keys(self, tmp_path):
        """El artefacto guardado contiene arch_config, norm_stats y provenance."""
        from latent_rl.representations import TCNLatentEncoder
        from latent_rl.representations.artifact import save_encoder_artifact, load_encoder_artifact

        encoder = TCNLatentEncoder(10, 3, 8)
        norm_stats = {"mean": {"log_return": 0.0}, "std": {"log_return": 1.0}}
        provenance = {"universe": ["AAPL"], "trained_at": "2024-01-01"}

        path = tmp_path / "test_enc.pt"
        save_encoder_artifact(encoder, norm_stats, provenance, path)

        enc_loaded, ns, prov = load_encoder_artifact(path)
        assert ns == norm_stats
        assert prov == provenance
        assert "encoder_type" in enc_loaded.get_arch_config()
