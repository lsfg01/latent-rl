"""Carga agregada y de solo lectura de una run experimental existente."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .agent_metadata import agent_display_name


@dataclass(frozen=True)
class AggregatedResults:
    """DataFrames y configuracion consolidados de una run."""

    results_dir: Path
    summary_all: pd.DataFrame
    seed_all: pd.DataFrame
    ivl_all: pd.DataFrame
    validation_all: pd.DataFrame
    ticker_comparison: pd.DataFrame
    config: dict[str, Any]


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_config(results_dir: Path) -> dict[str, Any]:
    path = results_dir / "experiment_config.json"
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"No se pudo leer {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} debe contener un objeto JSON")
    return loaded


def _ticker_directories(results_dir: Path, config: dict[str, Any]) -> list[Path]:
    detected = {
        path.name: path
        for path in results_dir.iterdir()
        if path.is_dir() and (path / "agent_summary.csv").exists()
    }
    configured = [str(ticker) for ticker in config.get("tickers", [])]
    ordered_names = [name for name in configured if name in detected]
    ordered_names.extend(sorted(name for name in detected if name not in ordered_names))
    return [detected[name] for name in ordered_names]


def _concat_ticker_csvs(ticker_dirs: list[Path], filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker_dir in ticker_dirs:
        path = ticker_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "ticker" not in frame.columns:
            frame.insert(0, "ticker", ticker_dir.name)
        else:
            frame["ticker"] = frame["ticker"].fillna(ticker_dir.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _add_agent_display(frame: pd.DataFrame, column: str = "agent_name") -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame
    result = frame.copy()
    result[f"{column}_display" if column != "agent_name" else "agent_display"] = (
        result[column].map(agent_display_name)
    )
    return result


def _derive_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    summary = _add_agent_display(summary)
    if summary.empty:
        return summary
    if {"mean_sharpe_is", "mean_sharpe_oos"}.issubset(summary.columns):
        summary["sharpe_gap"] = (
            summary["mean_sharpe_is"] - summary["mean_sharpe_oos"]
        ).abs()
    if "mean_mdd_oos" in summary.columns:
        summary["abs_mdd_oos"] = summary["mean_mdd_oos"].abs()
    if "agent_name" in summary.columns:
        summary["is_latent"] = summary["agent_name"].isin(["B", "C", "D"])
    return summary


def load_aggregated_results(results_dir: str | Path) -> AggregatedResults:
    """Carga una run existente sin modificar ni recalcular sus resultados."""
    root = Path(results_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"No existe el directorio de resultados: {root}")

    config = _read_config(root)
    ticker_dirs = _ticker_directories(root, config)
    if not ticker_dirs:
        raise FileNotFoundError(
            f"No se encontraron subdirectorios con agent_summary.csv en {root}"
        )

    summary = _derive_summary_columns(
        _concat_ticker_csvs(ticker_dirs, "agent_summary.csv")
    )
    seeds = _add_agent_display(
        _concat_ticker_csvs(ticker_dirs, "agent_seed_metrics.csv")
    )
    if not seeds.empty and "split" in seeds.columns:
        oos = seeds[seeds["split"].astype(str).str.lower() == "oos"]
        if not oos.empty:
            seeds = oos.reset_index(drop=True)

    ivl = _add_agent_display(
        _concat_ticker_csvs(ticker_dirs, "ivl_results.csv"), "latent_agent"
    )
    validation = _add_agent_display(
        _concat_ticker_csvs(ticker_dirs, "validation_metrics.csv")
    )
    comparison = _add_agent_display(
        _read_optional_csv(root / "ticker_comparison.csv"), "latent_agent"
    )

    return AggregatedResults(
        results_dir=root,
        summary_all=summary,
        seed_all=seeds,
        ivl_all=ivl,
        validation_all=validation,
        ticker_comparison=comparison,
        config=config,
    )
