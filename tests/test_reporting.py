"""Tests del loader y exportador reproducible de figuras."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go

from latent_rl.reporting.memory_figures import (
    create_ivl_heatmap_figure,
    create_sharpe_gap_figure,
    create_sharpe_oos_figure,
    create_trades_figure,
    export_memory_figures,
)
from latent_rl.reporting.results_loader import load_aggregated_results


def _write_results_fixture(root):
    root.mkdir()
    (root / "experiment_config.json").write_text(
        json.dumps({"tickers": ["SPY", "TSLA"], "seeds": [0, 1]}),
        encoding="utf-8",
    )
    comparison = []
    for ticker_index, ticker in enumerate(("SPY", "TSLA")):
        ticker_dir = root / ticker
        ticker_dir.mkdir()
        summary = pd.DataFrame(
            [
                {
                    "agent_name": agent,
                    "mean_sharpe_is": 1.0 + index,
                    "mean_sharpe_oos": 0.5 + index,
                    "mean_mdd_oos": -0.1 * (index + 1),
                    "mean_return_oos": 0.05 * (index + 1),
                    "mean_n_trades": 2.0 + index,
                    "mean_equity_oos": 10_000 + index,
                }
                for index, agent in enumerate(("A", "B", "C", "D"))
            ]
        )
        summary.to_csv(ticker_dir / "agent_summary.csv", index=False)
        seeds = pd.DataFrame(
            [
                {
                    "agent_name": agent,
                    "seed": seed,
                    "split": split,
                    "sharpe": 0.2 + seed,
                    "total_return": 0.1 + seed,
                }
                for agent in ("A", "B", "C", "D")
                for seed in (0, 1)
                for split in ("is", "oos")
            ]
        )
        seeds.to_csv(ticker_dir / "agent_seed_metrics.csv", index=False)
        ivl = pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "latent_agent": agent,
                    "ivl": value + ticker_index,
                    "delta_sharpe": value,
                    "delta_mdd": -value,
                    "delta_seed_std": -value,
                    "delta_is_oos_gap": -value,
                }
                for agent, value in zip(("B", "C", "D"), (0.1, 0.3, -0.2))
            ]
        )
        ivl.to_csv(ticker_dir / "ivl_results.csv", index=False)
        comparison.extend(ivl.to_dict("records"))
        pd.DataFrame(
            [
                {
                    "agent_name": agent,
                    "seed": 0,
                    "best_val_episode": 5,
                    "best_val_score": 1.0,
                    "best_val_sharpe": 1.1,
                    "best_val_mdd": -0.2,
                }
                for agent in ("A", "B", "C", "D")
            ]
        ).to_csv(ticker_dir / "validation_metrics.csv", index=False)
    pd.DataFrame(comparison).to_csv(root / "ticker_comparison.csv", index=False)


def test_aggregated_loader_detects_tickers_and_derives_columns(tmp_path):
    results_dir = tmp_path / "run"
    _write_results_fixture(results_dir)

    loaded = load_aggregated_results(results_dir)

    assert loaded.summary_all["ticker"].drop_duplicates().tolist() == ["SPY", "TSLA"]
    assert {"sharpe_gap", "abs_mdd_oos", "agent_display", "is_latent"}.issubset(
        loaded.summary_all.columns
    )
    assert loaded.summary_all.loc[
        loaded.summary_all["agent_name"] == "C", "agent_display"
    ].iloc[0] == "C · LatentDQN-IS-Frozen"
    assert set(loaded.seed_all["split"]) == {"oos"}
    assert len(loaded.validation_all) == 8


def test_required_figure_helpers_return_plotly_figures(tmp_path):
    results_dir = tmp_path / "run"
    _write_results_fixture(results_dir)
    loaded = load_aggregated_results(results_dir)

    figures = [
        create_sharpe_oos_figure(loaded.summary_all),
        create_sharpe_gap_figure(loaded.summary_all),
        create_ivl_heatmap_figure(loaded.ivl_all),
        create_trades_figure(loaded.summary_all),
    ]

    assert all(isinstance(figure, go.Figure) for figure in figures)
    assert figures[2].data[0].x[-1] == "MEAN"


def test_exporter_generates_required_figures_and_manifest(tmp_path):
    results_dir = tmp_path / "run"
    out_dir = tmp_path / "artifacts"
    _write_results_fixture(results_dir)

    manifest = export_memory_figures(results_dir, out_dir, formats=["html"])

    assert [item["id"] for item in manifest["figures"]] == [
        "fig_01",
        "fig_02",
        "fig_03",
        "fig_04",
    ]
    assert (out_dir / "figure_manifest.json").exists()
    assert all(
        (out_dir / filename).exists()
        for item in manifest["figures"]
        for filename in item["files"]
    )
