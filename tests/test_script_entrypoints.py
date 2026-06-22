"""Compatibilidad de rutas para los entrypoints operativos."""

from pathlib import Path

import pandas as pd

from scripts.experiments.a_hidden_sweep import build_config as build_sweep_config
from scripts.experiments.a_vs_d import build_config as build_a_vs_d_config
from scripts.experiments.abcd_robust_h64 import build_config as build_robust_config
from scripts.experiments.latent_abcd import build_config as build_latent_config
from scripts.experiments.pretrain_encoder import build_config as build_pretrain_config
from scripts.experiments.real_data_baseline import config as real_data_config
from scripts.utilities.compute_ivl import (
    _rebuild_aggregated,
    build_config as build_ivl_config,
)


def test_real_campaign_outputs_live_below_results():
    assert Path(build_latent_config().results_dir) == Path("results/latent_abcd")
    assert Path(build_a_vs_d_config().results_dir) == Path("results/a_vs_d")
    assert Path(build_robust_config().results_dir) == Path("results/abcd_robust_h64")
    assert Path(build_sweep_config(64).results_dir) == Path(
        "results/a_hidden_sweep/h64"
    )
    assert Path(real_data_config.results_dir) == Path("results/real_data_baseline")


def test_smoke_outputs_cannot_overwrite_real_campaigns():
    assert Path(build_latent_config(smoke=True).results_dir) == Path(
        "results/smoke/latent_abcd"
    )
    assert Path(build_a_vs_d_config(smoke=True).results_dir) == Path(
        "results/smoke/a_vs_d"
    )
    assert Path(build_robust_config(smoke=True).results_dir) == Path(
        "results/smoke/abcd_robust_h64"
    )
    assert Path(build_sweep_config(64, smoke=True).results_dir) == Path(
        "results/smoke/a_hidden_sweep/h64"
    )


def test_smoke_encoder_artifact_is_isolated():
    real = build_pretrain_config()
    smoke = build_pretrain_config(smoke=True)
    assert Path(real.output_path) == Path("models/encoders/tcn_heavy.pt")
    assert Path(smoke.output_path) == Path("models/smoke/encoders/tcn_heavy.pt")
    assert Path(build_latent_config(smoke=True).heavy_encoder_path) == Path(
        smoke.output_path
    )
    assert Path(build_a_vs_d_config(smoke=True).heavy_encoder_path) == Path(
        smoke.output_path
    )
    assert Path(build_robust_config(smoke=True).heavy_encoder_path) == Path(
        smoke.output_path
    )


def test_ivl_utility_accepts_campaign_paths_and_arm_names():
    cfg = build_ivl_config(
        results_dir="results/a_vs_d",
        tickers=["SPY", "TSLA"],
        direct_agent="A",
        latent_agents=["D"],
    )
    assert Path(cfg.results_dir) == Path("results/a_vs_d")
    assert cfg.tickers == ["SPY", "TSLA"]
    assert cfg.direct_agent == "A"
    assert cfg.latent_agents == ["D"]


def test_ivl_utility_rebuilds_all_metrics_required_by_current_runner():
    metric_columns = {
        "mean_return_is": 0.1,
        "std_return_is": 0.01,
        "mean_sharpe_is": 1.0,
        "mean_mdd_is": -0.2,
        "mean_equity_is": 11_000.0,
        "mean_return_oos": 0.05,
        "std_return_oos": 0.02,
        "mean_sharpe_oos": 0.7,
        "mean_mdd_oos": -0.25,
        "mean_equity_oos": 10_500.0,
        "mean_n_trades": 4.0,
        "seed_std_return_is": 0.03,
        "seed_std_sharpe_oos": 0.04,
    }
    summary = pd.DataFrame([{"agent_name": "A", **metric_columns}]).set_index(
        "agent_name"
    )
    aggregated = _rebuild_aggregated(summary)
    assert aggregated["A"]["seed_std_sharpe_oos"] == 0.04
    assert set(metric_columns).issubset(aggregated["A"])
