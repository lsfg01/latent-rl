"""
Generacion de graficos Plotly para el dashboard.

Todos los graficos usan el template 'plotly_white' y margenes compactos
para encajar en el diseno academico del dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional, Tuple

from latent_rl.reporting.agent_metadata import (
    AGENT_COLORS as AGENT_COLOR_MAP,
    agent_display_name,
)

# Paleta coherente con el CSS
ACCENT      = "#3b6fd4"
POSITIVE    = "#276749"
NEGATIVE    = "#c53030"
NEUTRAL     = "#718096"
TEMPLATE    = "plotly_white"
MARGIN      = dict(l=40, r=20, t=48, b=36)
TITLE_FONT  = dict(size=13, color="#1a202c")
AXIS_FONT   = dict(size=11, color="#718096")

AGENT_COLORS = [
    "#3b6fd4", "#276749", "#c53030", "#b7791f", "#553c9a",
]

AGENT_METRICS = {
    "mean_sharpe_oos": ("Sharpe OOS", "Sharpe OOS por Agente"),
    "mean_return_oos": ("Retorno OOS", "Retorno OOS por Agente"),
    "mean_mdd_oos": ("Max Drawdown OOS", "Max Drawdown OOS por Agente"),
    "sharpe_gap": ("|Sharpe IS - Sharpe OOS|", "Gap Sharpe IS/OOS por Agente"),
    "mean_n_trades": ("Nº medio de trades", "Operativa Media por Agente"),
    "mean_equity_oos": ("Equity final OOS", "Equity Final OOS por Agente"),
}

SEED_METRICS = {
    "sharpe": "Sharpe OOS",
    "total_return": "Retorno total OOS",
}


# ── helpers internos ──────────────────────────────────────────────────────────

def _first_existing_col(df: pd.DataFrame, candidates: list, required: bool = True):
    """
    Devuelve el primer nombre de candidatos que exista en df.columns.

    Compatible con el formato antiguo (mean_return) y el nuevo IS/OOS
    (mean_return_oos, mean_return_is).
    """
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"No se encontro ninguna de estas columnas: {candidates}. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )
    return None


def _detect_agent_col(df: pd.DataFrame) -> str:
    return _first_existing_col(df, ["agent_name", "agent", "Agent"])


def _filter_split(df: pd.DataFrame, preferred: str = "oos") -> tuple:
    """
    Filtra a las filas del split preferido cuando existe la columna 'split'.
    Si no existe o el filtro queda vacio, devuelve el DataFrame original.

    Returns:
        (df_filtered, label_suffix)
    """
    if "split" not in df.columns:
        return df, ""
    filtered = df[df["split"] == preferred]
    if filtered.empty:
        return df, ""
    return filtered, f" (OOS)"


def _base_layout(fig: go.Figure, title: str, **kwargs) -> go.Figure:
    """Aplica configuracion base a cualquier figura."""
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text=title, font=TITLE_FONT, x=0, xanchor="left", pad=dict(l=4)),
        margin=MARGIN,
        font=AXIS_FONT,
        plot_bgcolor="white",
        paper_bgcolor="white",
        **kwargs,
    )
    return fig


def create_empty_figure(message: str = "No hay datos disponibles") -> go.Figure:
    """Figura vacia con mensaje centrado."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color="#718096"),
    )
    fig.update_layout(
        template=TEMPLATE,
        margin=MARGIN,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    )
    return fig


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> Tuple[bool, str]:
    if df is None:
        return False, "DataFrame es None"
    if df.empty:
        return False, "DataFrame esta vacio"
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return False, f"Faltan columnas: {missing}"
    return True, ""


# ── figuras principales ───────────────────────────────────────────────────────

def create_agent_metric_bar(
    df: pd.DataFrame, metric: str = "mean_sharpe_oos"
) -> go.Figure:
    """Barras por agente para una metrica seleccionada."""
    if df is None or df.empty:
        return create_empty_figure("No hay datos de agentes disponibles")
    frame = df.copy()
    if metric == "sharpe_gap" and {
        "mean_sharpe_is",
        "mean_sharpe_oos",
    }.issubset(frame.columns):
        frame[metric] = (frame["mean_sharpe_is"] - frame["mean_sharpe_oos"]).abs()
    if metric not in frame.columns:
        return create_empty_figure(f"La metrica {metric} no esta disponible")
    try:
        agent_col = _detect_agent_col(frame)
    except KeyError as exc:
        return create_empty_figure(str(exc))

    label, title = AGENT_METRICS.get(metric, (metric, f"{metric} por Agente"))
    agents = frame[agent_col].astype(str).tolist()
    values = frame[metric].astype(float).tolist()
    fig = go.Figure(
        go.Bar(
            y=[agent_display_name(agent) for agent in agents],
            x=values,
            orientation="h",
            marker_color=[AGENT_COLOR_MAP.get(agent, ACCENT) for agent in agents],
            text=[f"{value:+.3f}" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"Agente=%{{y}}<br>{label}=%{{x:.4f}}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#cbd5e0", line_width=1)
    _base_layout(
        fig,
        title,
        xaxis_title=label,
        yaxis_title="",
        height=350,
        bargap=0.3,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def create_agent_return_bar(df: pd.DataFrame) -> go.Figure:
    """
    Barras horizontales del retorno medio por agente.

    Detecta automaticamente las columnas disponibles y prioriza OOS.
    Compatible con el formato antiguo (mean_return) y el nuevo IS/OOS.
    """
    if df is None or df.empty:
        return create_empty_figure("No hay datos de agentes disponibles")

    try:
        agent_col  = _first_existing_col(df, ["agent_name", "agent", "Agent"])
        return_col = _first_existing_col(df, [
            "mean_return_oos", "mean_return", "mean_return_is",
            "return_mean", "total_return_mean", "mean_total_return",
        ])
    except KeyError as exc:
        return create_empty_figure(str(exc))

    std_col = _first_existing_col(df, [
        "std_return_oos", "seed_std_return_oos", "std_return", "seed_std_return",
        "std_return_is", "seed_std_return_is",
    ], required=False)

    x_label = "Retorno OOS" if "oos" in return_col else "Retorno Medio"
    title   = "Retorno por Agente (OOS)" if "oos" in return_col else "Retorno Medio por Agente"

    agents  = df[agent_col].tolist()
    values  = df[return_col].tolist()
    colors  = [POSITIVE if v >= 0 else NEGATIVE for v in values]

    error_x = None
    if std_col:
        error_x = dict(type="data", array=df[std_col].tolist(), visible=True,
                       color="#a0aec0", thickness=1.5, width=4)

    fig = go.Figure(go.Bar(
        y=[agent_display_name(agent) for agent in agents],
        x=values,
        orientation="h",
        marker_color=colors,
        error_x=error_x,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=11),
    ))

    fig.add_vline(x=0, line_color="#cbd5e0", line_width=1)
    _base_layout(fig, title, xaxis_title=x_label, yaxis_title="",
                 height=320, bargap=0.3,
                 yaxis=dict(autorange="reversed"))
    return fig


def create_seed_return_boxplot(df: pd.DataFrame) -> go.Figure:
    """
    Boxplot de retornos por semilla para cada agente.

    Filtra a OOS cuando existe la columna 'split'.
    """
    is_valid, error_msg = validate_dataframe(df, ["seed", "total_return"])
    if not is_valid:
        return create_empty_figure(f"Sin datos: {error_msg}")

    try:
        agent_col = _detect_agent_col(df)
    except KeyError as exc:
        return create_empty_figure(str(exc))

    df_plot, split_label = _filter_split(df)
    agents = df_plot[agent_col].unique()

    fig = go.Figure()
    for i, agent in enumerate(agents):
        data = df_plot[df_plot[agent_col] == agent]["total_return"]
        fig.add_trace(go.Box(
            y=data,
            name=agent,
            boxpoints="all",
            jitter=0.4,
            pointpos=-1.6,
            marker=dict(size=5, color=AGENT_COLORS[i % len(AGENT_COLORS)]),
            line=dict(color=AGENT_COLORS[i % len(AGENT_COLORS)]),
        ))

    _base_layout(fig, f"Distribucion de Retornos por Semilla{split_label}",
                 yaxis_title="Retorno Total", xaxis_title="",
                 height=320, showlegend=False)
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e0", line_width=1)
    return fig


def create_seed_metric_boxplot(
    df: pd.DataFrame, metric: str = "sharpe"
) -> go.Figure:
    """Boxplot OOS por semilla para Sharpe o retorno total."""
    label = SEED_METRICS.get(metric)
    if label is None:
        return create_empty_figure(f"Metrica por semilla no soportada: {metric}")
    is_valid, error_msg = validate_dataframe(df, ["seed", metric])
    if not is_valid:
        return create_empty_figure(f"Sin datos: {error_msg}")
    try:
        agent_col = _detect_agent_col(df)
    except KeyError as exc:
        return create_empty_figure(str(exc))
    frame, _ = _filter_split(df)
    fig = go.Figure()
    for i, agent in enumerate(frame[agent_col].unique()):
        values = frame.loc[frame[agent_col] == agent, metric]
        color = AGENT_COLOR_MAP.get(str(agent), AGENT_COLORS[i % len(AGENT_COLORS)])
        fig.add_trace(
            go.Box(
                y=values,
                name=agent_display_name(agent),
                boxpoints="all",
                jitter=0.4,
                pointpos=-1.6,
                marker=dict(size=5, color=color),
                line=dict(color=color),
            )
        )
    _base_layout(
        fig,
        f"Distribucion de {label} por Semilla",
        yaxis_title=label,
        xaxis_title="",
        height=320,
        showlegend=False,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e0", line_width=1)
    return fig


def create_seed_return_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Scatter de retorno por semilla para cada agente.

    Filtra a OOS cuando existe la columna 'split'.
    """
    is_valid, error_msg = validate_dataframe(df, ["seed", "total_return"])
    if not is_valid:
        return create_empty_figure(f"Sin datos: {error_msg}")

    try:
        agent_col = _detect_agent_col(df)
    except KeyError as exc:
        return create_empty_figure(str(exc))

    df_plot, split_label = _filter_split(df)
    fig = go.Figure()

    for i, agent in enumerate(df_plot[agent_col].unique()):
        d = df_plot[df_plot[agent_col] == agent]
        fig.add_trace(go.Scatter(
            x=d["seed"],
            y=d["total_return"],
            mode="markers+lines",
            name=agent,
            marker=dict(size=7, color=AGENT_COLORS[i % len(AGENT_COLORS)]),
            line=dict(width=1.5, color=AGENT_COLORS[i % len(AGENT_COLORS)]),
        ))

    _base_layout(fig, f"Retorno por Semilla{split_label}",
                 xaxis_title="Semilla", yaxis_title="Retorno Total",
                 height=320, hovermode="x unified",
                 legend=dict(font=dict(size=10), orientation="h",
                             y=-0.22, x=0, xanchor="left"))
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e0", line_width=1)
    return fig


def create_seed_metric_scatter(
    df: pd.DataFrame, metric: str = "sharpe"
) -> go.Figure:
    """Evolucion OOS de una metrica por semilla y agente."""
    label = SEED_METRICS.get(metric)
    if label is None:
        return create_empty_figure(f"Metrica por semilla no soportada: {metric}")
    is_valid, error_msg = validate_dataframe(df, ["seed", metric])
    if not is_valid:
        return create_empty_figure(f"Sin datos: {error_msg}")
    try:
        agent_col = _detect_agent_col(df)
    except KeyError as exc:
        return create_empty_figure(str(exc))
    frame, _ = _filter_split(df)
    fig = go.Figure()
    for i, agent in enumerate(frame[agent_col].unique()):
        subset = frame[frame[agent_col] == agent].sort_values("seed")
        color = AGENT_COLOR_MAP.get(str(agent), AGENT_COLORS[i % len(AGENT_COLORS)])
        fig.add_trace(
            go.Scatter(
                x=subset["seed"],
                y=subset[metric],
                mode="markers+lines",
                name=agent_display_name(agent),
                marker=dict(size=7, color=color),
                line=dict(width=1.5, color=color),
            )
        )
    _base_layout(
        fig,
        f"{label} por Semilla",
        xaxis_title="Semilla",
        yaxis_title=label,
        height=320,
        hovermode="x unified",
        legend=dict(font=dict(size=10), orientation="h", y=-0.22, x=0),
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e0", line_width=1)
    return fig


def create_ivl_delta_bar(df: pd.DataFrame) -> go.Figure:
    """Componentes IVL de uno o varios agentes latentes."""
    is_valid, error_msg = validate_dataframe(
        df, ["delta_sharpe", "delta_mdd", "delta_seed_std", "delta_is_oos_gap"])
    if not is_valid:
        return create_empty_figure(f"Sin datos IVL: {error_msg}")

    labels = ["Δ Sharpe", "Δ MDD", "Δ Seed Std", "Δ Gap IS/OOS"]
    columns = ["delta_sharpe", "delta_mdd", "delta_seed_std", "delta_is_oos_gap"]
    fig = go.Figure()
    agent_col = "latent_agent" if "latent_agent" in df.columns else None
    for index, (_, row) in enumerate(df.iterrows()):
        agent = str(row[agent_col]) if agent_col else f"Latente {index + 1}"
        values = [float(row[column]) for column in columns]
        fig.add_trace(
            go.Bar(
                name=agent_display_name(agent),
                x=labels,
                y=values,
                marker_color=AGENT_COLOR_MAP.get(agent, AGENT_COLORS[index % len(AGENT_COLORS)]),
                text=[f"{value:+.3f}" for value in values],
                textposition="outside",
                cliponaxis=False,
            )
        )

    fig.add_hline(y=0, line_color="#cbd5e0", line_width=1)
    _base_layout(fig, "Componentes del IVL",
                 yaxis_title="Valor del delta", height=330, barmode="group",
                 legend=dict(font=dict(size=10), orientation="h", y=-0.28, x=0))
    return fig


def create_price_vs_latent_plot(df: pd.DataFrame) -> go.Figure:
    """Precio Close vs Indice PCA en dos ejes."""
    is_valid, error_msg = validate_dataframe(df, ["sample", "close", "pca_index"])
    if not is_valid:
        return create_empty_figure(f"Sin datos: {error_msg}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["sample"], y=df["close"],
        name="Close", line=dict(color=ACCENT, width=1.5), yaxis="y"))
    fig.add_trace(go.Scatter(
        x=df["sample"], y=df["pca_index"],
        name="PCA Index", line=dict(color="#b7791f", width=1.5), yaxis="y2"))

    _base_layout(fig, "Precio Close vs Indice PCA",
                 xaxis_title="Muestra", height=300,
                 yaxis=dict(title=dict(text="Close", font=dict(color=ACCENT)),
                            tickfont=dict(color=ACCENT)),
                 yaxis2=dict(title=dict(text="PCA", font=dict(color="#b7791f")),
                             tickfont=dict(color="#b7791f"),
                             overlaying="y", side="right"),
                 legend=dict(font=dict(size=10), orientation="h",
                             y=-0.22, x=0))
    return fig


def create_latent_indices_plot(df: pd.DataFrame) -> go.Figure:
    """Comparacion de indices latentes normalizados."""
    is_valid, error_msg = validate_dataframe(
        df, ["sample", "norm_index", "first_component_index", "pca_index"])
    if not is_valid:
        return create_empty_figure(f"Sin datos: {error_msg}")

    def norm(s):
        std = s.std()
        return (s - s.mean()) / std if std > 1e-8 else s

    fig = go.Figure()
    for col, label, color in [
        ("norm_index", "Norm", ACCENT),
        ("first_component_index", "First Comp.", "#276749"),
        ("pca_index", "PCA", "#b7791f"),
    ]:
        fig.add_trace(go.Scatter(
            x=df["sample"], y=norm(df[col]),
            name=label, line=dict(color=color, width=1.5), opacity=0.85))

    _base_layout(fig, "Indices Latentes (normalizados)",
                 xaxis_title="Muestra", yaxis_title="Valor norm.",
                 height=300,
                 legend=dict(font=dict(size=10), orientation="h",
                             y=-0.22, x=0))
    return fig


def create_pca_vs_next_return_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter Indice PCA vs Retorno siguiente."""
    is_valid, error_msg = validate_dataframe(df, ["pca_index", "next_return"])
    if not is_valid:
        return create_empty_figure(f"Sin datos: {error_msg}")

    df_c = df[["pca_index", "next_return"]].dropna()
    if df_c.empty:
        return create_empty_figure("Sin datos validos despues de eliminar NaN")

    fig = go.Figure(go.Scatter(
        x=df_c["pca_index"], y=df_c["next_return"],
        mode="markers",
        marker=dict(size=6, color=df_c["next_return"],
                    colorscale="RdYlGn", showscale=True,
                    colorbar=dict(title=dict(text="Retorno sig.", font=dict(size=10)),
                                  thickness=10)),
    ))

    if len(df_c) > 1:
        z = np.polyfit(df_c["pca_index"], df_c["next_return"], 1)
        x_t = np.linspace(df_c["pca_index"].min(), df_c["pca_index"].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_t, y=np.poly1d(z)(x_t),
            mode="lines", name="Tendencia",
            line=dict(color="#c53030", dash="dash", width=1.5)))

    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e0", line_width=1)
    _base_layout(fig, "PCA vs Retorno siguiente",
                 xaxis_title="Indice PCA", yaxis_title="Retorno sig.",
                 height=300, showlegend=False)
    return fig


def create_cross_ticker_ivl_chart(
    df: pd.DataFrame, view: str = "bars"
) -> go.Figure:
    """
    Barras agrupadas del IVL por ticker y agente latente.

    Muestra como varia el IVL entre distintos activos, lo que permite
    evaluar la consistencia de la ventaja latente entre mercados.

    Args:
        df: DataFrame con columnas ``ticker``, ``latent_agent``, ``ivl``.
            Puede contener filas de resumen (ticker == "MEAN" o "STD")
            que se excluyen automaticamente.

    Returns:
        Figura Plotly con barras agrupadas por ticker.
    """
    if df is None or df.empty:
        return create_empty_figure("Sin datos cross-ticker disponibles")

    detail = df[~df["ticker"].astype(str).str.upper().isin(["MEAN", "STD"])].copy()
    if detail.empty:
        return create_empty_figure("Sin datos de tickers individuales")

    detail = detail.drop_duplicates(subset=["ticker", "latent_agent"], keep="last")
    mean_rows = (
        detail.groupby("latent_agent", as_index=False)["ivl"].mean().assign(ticker="MEAN")
    )
    plot_df = pd.concat([detail, mean_rows], ignore_index=True)
    latent_agents = plot_df["latent_agent"].unique()

    if view == "heatmap":
        pivot = plot_df.pivot(index="latent_agent", columns="ticker", values="ivl")
        column_order = detail["ticker"].drop_duplicates().tolist() + ["MEAN"]
        pivot = pivot.reindex(columns=column_order)
        z = pivot.to_numpy(dtype=float)
        finite = np.abs(z[np.isfinite(z)])
        limit = float(finite.max()) if finite.size else 1.0
        limit = limit if limit > 0 else 1.0
        text = np.where(np.isnan(z), "", np.vectorize(lambda value: f"{value:+.3f}")(z))
        fig = go.Figure(go.Heatmap(
            z=z,
            x=pivot.columns,
            y=[agent_display_name(agent) for agent in pivot.index],
            zmin=-limit,
            zmax=limit,
            zmid=0,
            colorscale="RdBu",
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="IVL"),
        ))
        _base_layout(fig, "IVL Cross-Ticker", xaxis_title="Ticker",
                     yaxis_title="Agente latente", height=360)
        return fig

    fig = go.Figure()
    for i, latent_agent in enumerate(latent_agents):
        sub = plot_df[plot_df["latent_agent"] == latent_agent]
        color = AGENT_COLOR_MAP.get(str(latent_agent), AGENT_COLORS[i % len(AGENT_COLORS)])
        ivl_vals = sub["ivl"].tolist()
        fig.add_trace(go.Bar(
            name=agent_display_name(latent_agent),
            x=sub["ticker"].tolist(),
            y=ivl_vals,
            marker_color=color,
            text=[f"{v:+.3f}" for v in ivl_vals],
            textposition="outside",
            cliponaxis=False,
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="#718096", line_width=1.5)
    _base_layout(fig, "IVL por Ticker y Agente Latente",
                 xaxis_title="Ticker", yaxis_title="IVL",
                 height=360,
                 barmode="group",
                 legend=dict(font=dict(size=10), orientation="h", y=-0.28, x=0))
    return fig
