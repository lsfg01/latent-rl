"""Figuras internas reproducibles generadas desde CSV finales."""

from __future__ import annotations

import json
import importlib.util
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .agent_metadata import AGENT_COLORS, AGENT_ORDER, agent_display_name
from .results_loader import AggregatedResults, load_aggregated_results


PLOT_TEMPLATE = "plotly_white"
LATENT_AGENTS = ["B", "C", "D"]


@dataclass(frozen=True)
class FigureArtifact:
    """Figura y trazabilidad minima para el manifest."""

    id: str
    title: str
    stem: str
    figure: go.Figure
    source_csv: tuple[str, ...]
    columns: tuple[str, ...]


def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": "#718096", "size": 14},
    )
    return _style_figure(fig, title)


def _style_figure(fig: go.Figure, title: str, *, height: int = 700) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title={"text": title, "x": 0.01, "xanchor": "left"},
        font={"family": "Arial, sans-serif", "color": "#1a202c"},
        margin={"l": 72, "r": 36, "t": 82, "b": 72},
        height=height,
        legend={"orientation": "h", "y": -0.18, "x": 0},
    )
    return fig


def _ordered_agents(frame: pd.DataFrame, allowed: Iterable[str]) -> list[str]:
    present = set(frame.get("agent_name", pd.Series(dtype=str)).dropna().astype(str))
    allowed_set = set(allowed)
    ordered = [agent for agent in AGENT_ORDER if agent in present and agent in allowed_set]
    ordered.extend(sorted((present & allowed_set) - set(ordered)))
    return ordered


def _grouped_bar(
    frame: pd.DataFrame,
    *,
    metric: str,
    agents: list[str],
    title: str,
    yaxis_title: str,
) -> go.Figure:
    required = {"ticker", "agent_name", metric}
    if frame.empty or not required.issubset(frame.columns):
        return _empty_figure(title, f"Faltan datos para {metric}")

    fig = go.Figure()
    for agent in agents:
        subset = frame[frame["agent_name"] == agent]
        values = subset[metric].astype(float)
        fig.add_trace(
            go.Bar(
                name=agent_display_name(agent),
                x=subset["ticker"],
                y=values,
                marker_color=AGENT_COLORS.get(agent, "#718096"),
                text=[f"{value:.2f}" for value in values],
                textposition="outside",
                cliponaxis=False,
                customdata=[[agent]] * len(subset),
                hovertemplate=(
                    "Ticker=%{x}<br>Agente=%{fullData.name}<br>"
                    f"{yaxis_title}=%{{y:.4f}}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=0, line_color="#718096", line_width=1)
    fig.update_layout(barmode="group", xaxis_title="Ticker", yaxis_title=yaxis_title)
    return _style_figure(fig, title)


def create_sharpe_oos_figure(
    summary_all: pd.DataFrame, *, include_benchmarks: bool = False
) -> go.Figure:
    """F1: Sharpe OOS por brazo y ticker."""
    allowed = ["A", "B", "C", "D"]
    if include_benchmarks:
        allowed.append("BuyAndHoldAgent")
    agents = _ordered_agents(summary_all, allowed)
    return _grouped_bar(
        summary_all,
        metric="mean_sharpe_oos",
        agents=agents,
        title="Sharpe OOS por agente y ticker",
        yaxis_title="Sharpe OOS",
    )


def create_sharpe_gap_figure(
    summary_all: pd.DataFrame, *, include_benchmarks: bool = False
) -> go.Figure:
    """F2: gap absoluto de Sharpe entre IS y OOS."""
    frame = summary_all.copy()
    if "sharpe_gap" not in frame.columns and {
        "mean_sharpe_is",
        "mean_sharpe_oos",
    }.issubset(frame.columns):
        frame["sharpe_gap"] = (
            frame["mean_sharpe_is"] - frame["mean_sharpe_oos"]
        ).abs()
    allowed = ["A", "B", "C", "D"]
    if include_benchmarks:
        allowed.extend(["BuyAndHoldAgent", "RandomAgent"])
    agents = _ordered_agents(frame, allowed)
    return _grouped_bar(
        frame,
        metric="sharpe_gap",
        agents=agents,
        title="Gap absoluto de Sharpe IS/OOS (menor es mejor)",
        yaxis_title="|Sharpe IS - Sharpe OOS|",
    )


def create_ivl_heatmap_figure(ivl_all: pd.DataFrame) -> go.Figure:
    """F3: IVL de B/C/D por ticker, con media calculada sin duplicados."""
    title = "IVL por agente latente y ticker"
    required = {"ticker", "latent_agent", "ivl"}
    if ivl_all.empty or not required.issubset(ivl_all.columns):
        return _empty_figure(title, "No hay datos IVL por ticker")

    detail = ivl_all[
        (~ivl_all["ticker"].astype(str).str.upper().isin(["MEAN", "STD"]))
        & (ivl_all["latent_agent"].isin(LATENT_AGENTS))
    ].drop_duplicates(subset=["ticker", "latent_agent"], keep="last")
    pivot = detail.pivot(index="latent_agent", columns="ticker", values="ivl")
    pivot = pivot.reindex([agent for agent in LATENT_AGENTS if agent in pivot.index])
    if pivot.empty:
        return _empty_figure(title, "No hay filas IVL para B/C/D")
    pivot["MEAN"] = pivot.mean(axis=1, skipna=True)

    z = pivot.to_numpy(dtype=float)
    finite = np.abs(z[np.isfinite(z)])
    limit = float(finite.max()) if finite.size else 1.0
    limit = limit if limit > 0 else 1.0
    text = np.where(np.isnan(z), "", np.vectorize(lambda value: f"{value:+.3f}")(z))
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=pivot.columns.tolist(),
            y=[agent_display_name(agent) for agent in pivot.index],
            zmin=-limit,
            zmax=limit,
            zmid=0,
            colorscale="RdBu",
            text=text,
            texttemplate="%{text}",
            hovertemplate="Ticker=%{x}<br>Agente=%{y}<br>IVL=%{z:.4f}<extra></extra>",
            colorbar={"title": "IVL"},
        )
    )
    fig.update_layout(xaxis_title="Ticker", yaxis_title="Agente latente")
    return _style_figure(fig, title, height=560)


def create_trades_figure(summary_all: pd.DataFrame) -> go.Figure:
    """F4: numero medio de operaciones de A/B/C/D."""
    agents = _ordered_agents(summary_all, ["A", "B", "C", "D"])
    return _grouped_bar(
        summary_all,
        metric="mean_n_trades",
        agents=agents,
        title="Número medio de trades por agente y ticker",
        yaxis_title="Nº medio de trades",
    )


def create_risk_return_scatter_figure(summary_all: pd.DataFrame) -> go.Figure:
    """F5: retorno OOS frente al drawdown OOS absoluto."""
    title = "Riesgo y rentabilidad fuera de muestra"
    frame = summary_all.copy()
    if "abs_mdd_oos" not in frame.columns and "mean_mdd_oos" in frame.columns:
        frame["abs_mdd_oos"] = frame["mean_mdd_oos"].abs()
    required = {"ticker", "agent_name", "abs_mdd_oos", "mean_return_oos"}
    if frame.empty or not required.issubset(frame.columns):
        return _empty_figure(title, "Faltan métricas OOS de riesgo/retorno")
    frame = frame[frame["agent_name"].isin(["A", "B", "C", "D"])]

    fig = go.Figure()
    for agent in _ordered_agents(frame, ["A", "B", "C", "D"]):
        subset = frame[frame["agent_name"] == agent]
        fig.add_trace(
            go.Scatter(
                name=agent_display_name(agent),
                x=subset["abs_mdd_oos"],
                y=subset["mean_return_oos"],
                text=subset["ticker"],
                mode="markers+text",
                textposition="top center",
                marker={"size": 13, "color": AGENT_COLORS.get(agent)},
                hovertemplate=(
                    "Ticker=%{text}<br>Drawdown abs.=%{x:.4f}<br>"
                    "Retorno OOS=%{y:.4f}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=0, line_color="#cbd5e0", line_width=1)
    fig.update_layout(
        xaxis_title="|Max Drawdown OOS|",
        yaxis_title="Retorno OOS",
    )
    return _style_figure(fig, title)


def create_seed_stability_figure(seed_all: pd.DataFrame) -> go.Figure:
    """F6: distribucion del Sharpe OOS entre semillas."""
    title = "Estabilidad del Sharpe OOS entre semillas"
    required = {"ticker", "agent_name", "sharpe"}
    if seed_all.empty or not required.issubset(seed_all.columns):
        return _empty_figure(title, "No hay métricas Sharpe por semilla")
    frame = seed_all[seed_all["agent_name"].isin(["A", "B", "C", "D"])]
    fig = go.Figure()
    for agent in _ordered_agents(frame, ["A", "B", "C", "D"]):
        subset = frame[frame["agent_name"] == agent]
        fig.add_trace(
            go.Box(
                name=agent_display_name(agent),
                x=subset["ticker"],
                y=subset["sharpe"],
                marker_color=AGENT_COLORS.get(agent),
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
            )
        )
    fig.add_hline(y=0, line_color="#cbd5e0", line_width=1)
    fig.update_layout(
        boxmode="group", xaxis_title="Ticker", yaxis_title="Sharpe OOS"
    )
    return _style_figure(fig, title)


def build_memory_figures(
    results: AggregatedResults, *, include_optional: bool = False
) -> list[FigureArtifact]:
    """Construye las figuras sin escribir archivos."""
    figures = [
        FigureArtifact(
            "fig_01",
            "Sharpe OOS por agente y ticker",
            "fig_01_sharpe_oos_by_agent_ticker",
            create_sharpe_oos_figure(
                results.summary_all, include_benchmarks=include_optional
            ),
            ("agent_summary.csv",),
            ("ticker", "agent_name", "mean_sharpe_oos"),
        ),
        FigureArtifact(
            "fig_02",
            "Gap Sharpe IS/OOS por agente y ticker",
            "fig_02_sharpe_is_oos_gap_by_agent_ticker",
            create_sharpe_gap_figure(
                results.summary_all, include_benchmarks=include_optional
            ),
            ("agent_summary.csv",),
            ("ticker", "agent_name", "mean_sharpe_is", "mean_sharpe_oos"),
        ),
        FigureArtifact(
            "fig_03",
            "IVL por agente latente y ticker",
            "fig_03_ivl_heatmap_latent_ticker",
            create_ivl_heatmap_figure(results.ivl_all),
            ("ivl_results.csv",),
            ("ticker", "latent_agent", "ivl"),
        ),
        FigureArtifact(
            "fig_04",
            "Número medio de trades por agente y ticker",
            "fig_04_trades_by_agent_ticker",
            create_trades_figure(results.summary_all),
            ("agent_summary.csv",),
            ("ticker", "agent_name", "mean_n_trades"),
        ),
    ]
    if include_optional:
        figures.extend(
            [
                FigureArtifact(
                    "fig_05",
                    "Scatter OOS riesgo-rentabilidad",
                    "fig_05_oos_risk_return_scatter",
                    create_risk_return_scatter_figure(results.summary_all),
                    ("agent_summary.csv",),
                    ("ticker", "agent_name", "mean_mdd_oos", "mean_return_oos"),
                ),
                FigureArtifact(
                    "fig_06",
                    "Estabilidad por semillas",
                    "fig_06_seed_stability_sharpe_oos",
                    create_seed_stability_figure(results.seed_all),
                    ("agent_seed_metrics.csv",),
                    ("ticker", "agent_name", "seed", "sharpe", "split"),
                ),
            ]
        )
    return figures


def _write_figure(
    artifact: FigureArtifact,
    out_dir: Path,
    formats: tuple[str, ...],
    *,
    width: int,
    height: int,
    scale: float,
    static_export_available: bool,
) -> list[str]:
    files: list[str] = []
    html_path = out_dir / f"{artifact.stem}.html"
    for file_format in formats:
        if file_format == "html":
            artifact.figure.write_html(html_path, include_plotlyjs="cdn", full_html=True)
            if html_path.name not in files:
                files.append(html_path.name)
            continue

        if not static_export_available:
            artifact.figure.write_html(html_path, include_plotlyjs="cdn", full_html=True)
            if html_path.name not in files:
                files.append(html_path.name)
            continue

        path = out_dir / f"{artifact.stem}.{file_format}"
        try:
            artifact.figure.write_image(
                path, format=file_format, width=width, height=height, scale=scale
            )
            files.append(path.name)
        except Exception as exc:  # Plotly usa errores distintos segun Kaleido/version.
            warnings.warn(
                f"No se pudo exportar {path.name} ({exc}). "
                f"Se genera {html_path.name} como fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            artifact.figure.write_html(html_path, include_plotlyjs="cdn", full_html=True)
            if html_path.name not in files:
                files.append(html_path.name)
    return files


def export_memory_figures(
    results_dir: str | Path,
    out_dir: str | Path,
    *,
    formats: Iterable[str] = ("png", "svg"),
    width: int = 1200,
    height: int = 700,
    scale: float = 2,
    include_optional: bool = False,
) -> dict:
    """Genera artefactos y devuelve el contenido del manifest."""
    normalized_formats = tuple(dict.fromkeys(str(fmt).lower() for fmt in formats))
    unsupported = set(normalized_formats) - {"png", "svg", "html"}
    if unsupported:
        raise ValueError(f"Formatos no soportados: {sorted(unsupported)}")
    if not normalized_formats:
        raise ValueError("Debe indicarse al menos un formato")
    if width <= 0 or height <= 0 or scale <= 0:
        raise ValueError("width, height y scale deben ser positivos")

    results = load_aggregated_results(results_dir)
    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = build_memory_figures(results, include_optional=include_optional)
    static_requested = any(fmt in {"png", "svg"} for fmt in normalized_formats)
    static_export_available = importlib.util.find_spec("kaleido") is not None
    if static_requested and not static_export_available:
        warnings.warn(
            "Kaleido no esta disponible: PNG/SVG se sustituyen por HTML interactivo.",
            RuntimeWarning,
            stacklevel=2,
        )

    manifest_figures = []
    for artifact in artifacts:
        files = _write_figure(
            artifact,
            destination,
            normalized_formats,
            width=width,
            height=height,
            scale=scale,
            static_export_available=static_export_available,
        )
        manifest_figures.append(
            {
                "id": artifact.id,
                "title": artifact.title,
                "files": files,
                "source_csv": list(artifact.source_csv),
                "columns": list(artifact.columns),
            }
        )

    manifest = {
        "results_dir": str(results.results_dir),
        "out_dir": str(destination),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "figures": manifest_figures,
    }
    (destination / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
