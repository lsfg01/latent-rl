"""
Tests para el modulo de experimentacion multi-ticker y el loader del dashboard.

Cubre:
- ExperimentConfig: validaciones multi-ticker
- load_tickers_data: descarga bulk con mock
- aggregate_ticker_results: agregacion cross-ticker
- _detect_tickers / get_available_tickers: deteccion de estructura de resultados
- load_all_dashboard_data: deteccion automatica de formato
"""

from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from latent_rl.experiments import ExperimentConfig, aggregate_ticker_results
from latent_rl.experiments.utils import load_tickers_data
from dashboard.data.loader import (
    _detect_tickers,
    get_available_tickers,
    load_all_dashboard_data,
    load_agent_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _minimal_ohlcv() -> pd.DataFrame:
    """DataFrame OHLCV minimo para usar como mock de datos descargados."""
    return pd.DataFrame({
        "open":   [100.0, 101.0, 102.0],
        "high":   [102.0, 103.0, 104.0],
        "low":    [99.0,  100.0, 101.0],
        "close":  [101.0, 102.0, 103.0],
        "volume": [1_000_000.0, 1_100_000.0, 900_000.0],
    })


def _ivl_record(latent_agent: str = "LatentDQN", ivl: float = 0.3,
                ticker: str = "SPY") -> dict:
    """Registro IVL minimo para tests de agregacion."""
    return {
        "ticker":           ticker,
        "direct_agent":     "DQNAgent",
        "latent_agent":     latent_agent,
        "ivl":              ivl,
        "delta_sharpe":     0.10,
        "delta_mdd":        0.05,
        "delta_seed_std":   0.02,
        "delta_is_oos_gap": 0.03,
        "interpretation":   "latent_advantage",
    }


def _summary_csv_content() -> str:
    """Contenido minimo valido de agent_summary.csv."""
    return (
        "agent_name,mean_return_is,std_return_is,mean_sharpe_is,mean_mdd_is,"
        "mean_equity_is,mean_return_oos,std_return_oos,mean_sharpe_oos,"
        "mean_mdd_oos,mean_equity_oos,mean_n_trades,seed_std_return_is\n"
        "DQNAgent,0.05,0.01,0.8,-0.02,10500,0.03,0.01,0.6,-0.03,10300,5,0.01\n"
    )


# ---------------------------------------------------------------------------
# ExperimentConfig — validaciones multi-ticker
# ---------------------------------------------------------------------------

class TestExperimentConfigTickers:

    def test_single_ticker_valid(self):
        cfg = ExperimentConfig(tickers=["SPY"])
        assert cfg.tickers == ["SPY"]

    def test_multiple_tickers_valid(self):
        cfg = ExperimentConfig(tickers=["SPY", "AAPL", "BTC-USD"])
        assert len(cfg.tickers) == 3

    def test_empty_tickers_raises(self):
        with pytest.raises(ValueError, match="tickers no puede estar"):
            ExperimentConfig(tickers=[])

    def test_duplicate_tickers_raises(self):
        with pytest.raises(ValueError, match="duplicados"):
            ExperimentConfig(tickers=["SPY", "SPY"])

    def test_empty_string_ticker_raises(self):
        with pytest.raises(ValueError):
            ExperimentConfig(tickers=["SPY", ""])

    def test_whitespace_only_ticker_raises(self):
        with pytest.raises(ValueError):
            ExperimentConfig(tickers=["   "])

    def test_default_is_single_spy(self):
        cfg = ExperimentConfig()
        assert cfg.tickers == ["SPY"]

    def test_ivl_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sumar 1.0"):
            ExperimentConfig(ivl_weights={"sharpe": 0.5, "mdd": 0.5,
                                           "seed_std": 0.5, "is_oos_gap": 0.5})

    def test_train_ratio_must_be_in_open_interval(self):
        with pytest.raises(ValueError, match="train_ratio"):
            ExperimentConfig(train_ratio=0.0)
        with pytest.raises(ValueError, match="train_ratio"):
            ExperimentConfig(train_ratio=1.0)


# ---------------------------------------------------------------------------
# load_tickers_data
# ---------------------------------------------------------------------------

class TestLoadTickersData:

    def test_returns_dict_keyed_by_ticker(self):
        mock_df = _minimal_ohlcv()
        with patch("latent_rl.experiments.utils.load_yfinance_data",
                   return_value=mock_df) as mock_load:
            result = load_tickers_data(["SPY", "AAPL"], "2020-01-01", "2023-12-31")

        assert set(result.keys()) == {"SPY", "AAPL"}
        assert mock_load.call_count == 2

    def test_each_df_has_ohlcv_columns(self):
        mock_df = _minimal_ohlcv()
        with patch("latent_rl.experiments.utils.load_yfinance_data",
                   return_value=mock_df):
            result = load_tickers_data(["SPY"], "2020-01-01", "2023-12-31")

        assert list(result["SPY"].columns) == ["open", "high", "low", "close", "volume"]

    def test_raises_with_ticker_name_on_failure(self):
        def fail_on_aapl(ticker, **kwargs):
            if ticker == "AAPL":
                raise ValueError("No data for AAPL")
            return _minimal_ohlcv()

        with patch("latent_rl.experiments.utils.load_yfinance_data",
                   side_effect=fail_on_aapl):
            with pytest.raises(ValueError, match="AAPL"):
                load_tickers_data(["SPY", "AAPL"], "2020-01-01", "2023-12-31")

    def test_n_obs_passed_to_each_ticker(self):
        mock_df = _minimal_ohlcv()
        with patch("latent_rl.experiments.utils.load_yfinance_data",
                   return_value=mock_df) as mock_load:
            load_tickers_data(["SPY"], "2020-01-01", "2023-12-31", n_obs=100)

        _, kwargs = mock_load.call_args
        assert kwargs.get("n_obs") == 100


# ---------------------------------------------------------------------------
# aggregate_ticker_results
# ---------------------------------------------------------------------------

class TestAggregateTickerResults:

    def test_single_ticker_returns_dataframe(self):
        results = {"SPY": {"ivl_records": [_ivl_record("LatentDQN", 0.3)]}}
        cfg = ExperimentConfig(tickers=["SPY"])
        df = aggregate_ticker_results(results, cfg)
        assert not df.empty
        assert "ticker" in df.columns
        assert "SPY" in df["ticker"].values

    def test_multi_ticker_includes_mean_and_std_rows(self):
        results = {
            "SPY":  {"ivl_records": [_ivl_record("LatentDQN", 0.3, ticker="SPY")]},
            "AAPL": {"ivl_records": [_ivl_record("LatentDQN", 0.1, ticker="AAPL")]},
        }
        cfg = ExperimentConfig(tickers=["SPY", "AAPL"])
        df = aggregate_ticker_results(results, cfg)
        assert "MEAN" in df["ticker"].values
        assert "STD" in df["ticker"].values

    def test_mean_row_is_average_of_tickers(self):
        results = {
            "SPY":  {"ivl_records": [_ivl_record("LatentDQN", 0.4, ticker="SPY")]},
            "AAPL": {"ivl_records": [_ivl_record("LatentDQN", 0.2, ticker="AAPL")]},
        }
        cfg = ExperimentConfig(tickers=["SPY", "AAPL"])
        df = aggregate_ticker_results(results, cfg)
        mean_row = df[(df["ticker"] == "MEAN") & (df["latent_agent"] == "LatentDQN")]
        assert abs(mean_row.iloc[0]["ivl"] - 0.3) < 1e-9

    def test_empty_ivl_records_returns_empty_dataframe(self):
        results = {"SPY": {"ivl_records": []}}
        cfg = ExperimentConfig(tickers=["SPY"])
        df = aggregate_ticker_results(results, cfg)
        assert df.empty

    def test_missing_ivl_records_key_handled(self):
        results = {"SPY": {}}  # sin clave ivl_records
        cfg = ExperimentConfig(tickers=["SPY"])
        df = aggregate_ticker_results(results, cfg)
        assert df.empty

    def test_multiple_latent_agents(self):
        results = {
            "SPY": {
                "ivl_records": [
                    _ivl_record("LatentDQN (no pretrained)", 0.3),
                    _ivl_record("LatentDQN (pretrained)",    0.1),
                ]
            }
        }
        cfg = ExperimentConfig(tickers=["SPY"])
        df = aggregate_ticker_results(results, cfg)
        assert len(df["latent_agent"].unique()) == 2


# ---------------------------------------------------------------------------
# _detect_tickers / get_available_tickers
# ---------------------------------------------------------------------------

class TestDetectTickers:

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert _detect_tickers(tmp_path) == []

    def test_nonexistent_directory_returns_empty_list(self, tmp_path):
        assert _detect_tickers(tmp_path / "nonexistent") == []

    def test_subdir_without_csv_is_ignored(self, tmp_path):
        (tmp_path / "SPY").mkdir()
        assert _detect_tickers(tmp_path) == []

    def test_subdir_with_csv_is_detected(self, tmp_path):
        spy = tmp_path / "SPY"
        spy.mkdir()
        (spy / "agent_summary.csv").write_text(_summary_csv_content())
        assert _detect_tickers(tmp_path) == ["SPY"]

    def test_multiple_tickers_sorted_alphabetically(self, tmp_path):
        for ticker in ["SPY", "AAPL", "BTC-USD"]:
            d = tmp_path / ticker
            d.mkdir()
            (d / "agent_summary.csv").write_text(_summary_csv_content())
        result = _detect_tickers(tmp_path)
        assert result == sorted(["AAPL", "BTC-USD", "SPY"])

    def test_flat_csv_at_root_is_not_detected_as_ticker(self, tmp_path):
        (tmp_path / "agent_summary.csv").write_text(_summary_csv_content())
        assert _detect_tickers(tmp_path) == []

    def test_get_available_tickers_delegates_to_detect(self, tmp_path):
        spy = tmp_path / "SPY"
        spy.mkdir()
        (spy / "agent_summary.csv").write_text(_summary_csv_content())
        with patch("dashboard.data.loader.get_results_path", return_value=tmp_path):
            result = get_available_tickers()
        assert result == ["SPY"]


# ---------------------------------------------------------------------------
# load_all_dashboard_data — deteccion automatica
# ---------------------------------------------------------------------------

class TestLoadAllDashboardData:

    def test_loads_flat_format_when_no_tickers(self, tmp_path):
        (tmp_path / "agent_summary.csv").write_text(_summary_csv_content())
        with patch("dashboard.data.loader.get_results_path", return_value=tmp_path):
            data = load_all_dashboard_data(ticker=None)
        df, err = data["agent_summary"]
        assert df is not None
        assert err is None

    def test_loads_ticker_format_when_ticker_specified(self, tmp_path):
        spy = tmp_path / "SPY"
        spy.mkdir()
        (spy / "agent_summary.csv").write_text(_summary_csv_content())
        with patch("dashboard.data.loader.get_results_path", return_value=tmp_path):
            data = load_all_dashboard_data(ticker="SPY")
        df, err = data["agent_summary"]
        assert df is not None
        assert err is None

    def test_auto_detects_ticker_dir_when_no_flat_file(self, tmp_path):
        spy = tmp_path / "SPY"
        spy.mkdir()
        (spy / "agent_summary.csv").write_text(_summary_csv_content())
        # No flat file at root
        with patch("dashboard.data.loader.get_results_path", return_value=tmp_path):
            data = load_all_dashboard_data(ticker=None)
        df, err = data["agent_summary"]
        assert df is not None

    def test_prefers_flat_format_when_both_exist(self, tmp_path):
        # Flat file at root
        (tmp_path / "agent_summary.csv").write_text(_summary_csv_content())
        # Also ticker subdir
        spy = tmp_path / "SPY"
        spy.mkdir()
        (spy / "agent_summary.csv").write_text(
            _summary_csv_content().replace("DQNAgent", "LatentDQN")
        )
        with patch("dashboard.data.loader.get_results_path", return_value=tmp_path):
            data = load_all_dashboard_data(ticker=None)
        df, err = data["agent_summary"]
        # Should have loaded the flat file (DQNAgent), not the ticker one (LatentDQN)
        assert "DQNAgent" in df["agent_name"].values

    def test_returns_error_tuple_when_missing(self, tmp_path):
        with patch("dashboard.data.loader.get_results_path", return_value=tmp_path):
            data = load_all_dashboard_data(ticker=None)
        df, err = data["agent_summary"]
        assert df is None
        assert err is not None
        assert isinstance(err, str)

    def test_always_has_all_expected_keys(self, tmp_path):
        with patch("dashboard.data.loader.get_results_path", return_value=tmp_path):
            data = load_all_dashboard_data(ticker=None)
        expected_keys = {
            "agent_summary", "agent_seed_metrics",
            "ivl_results", "latent_index", "ticker_comparison",
        }
        assert expected_keys == set(data.keys())
