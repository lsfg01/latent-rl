"""
Dashboard de Ventaja Latente

Aplicacion Dash interactiva para visualizar comparaciones de agentes RL
y el Indice de Ventaja Latente (IVL).

Soporta experimentos con uno o varios tickers:
- Un ticker:        resultados en results/{ticker}/
- Varios tickers:   resultados en results/{ticker}/ + cross-ticker en results/

La logica de datos y calculos no se modifica en este fichero.
Solo se define el layout y los callbacks de actualizacion.
"""

import argparse
import sys
from pathlib import Path

import dash
from dash import dcc, html, dash_table, Input, Output, State
import pandas as pd
import subprocess

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# parse_known_args: ignora argumentos internos de Dash/Werkzeug
_cli_parser = argparse.ArgumentParser(add_help=False)
_cli_parser.add_argument("--results-dir", default=None,
                         help="Directorio de resultados (default: results/)")
_cli_args, _ = _cli_parser.parse_known_args()

if _cli_args.results_dir:
    # Debe importarse antes de cualquier otro import de dashboard.data
    from dashboard.data.loader import set_results_dir
    set_results_dir(_cli_args.results_dir)

from dashboard.data import (
    load_all_dashboard_data,
    get_available_tickers,
    load_ticker_comparison,
)
from dashboard.components import (
    create_kpi_card,
    create_ivl_card,
    create_delta_card,
    create_agent_metric_bar,
    create_seed_metric_boxplot,
    create_seed_metric_scatter,
    create_ivl_delta_bar,
    create_cross_ticker_ivl_chart,
)
from latent_rl.reporting.agent_metadata import (
    AGENT_DESCRIPTIONS,
    AGENT_DISPLAY,
    agent_display_name,
)

# ---------------------------------------------------------------------------
# Constantes de estilo (complementan el CSS — solo para casos puntuales)
# ---------------------------------------------------------------------------
COLORS = {
    "bg":       "#f4f6fb",
    "card":     "#ffffff",
    "accent":   "#3b6fd4",
    "text":     "#1a202c",
    "muted":    "#718096",
    "border":   "#e2e8f0",
    "positive": "#276749",
    "negative": "#c53030",
}

TABLE_HEADER_STYLE = {
    "backgroundColor": "#ebf0fb",
    "color": COLORS["text"],
    "fontWeight": "700",
    "fontSize": "12px",
    "textAlign": "center",
    "borderBottom": f"2px solid {COLORS['accent']}",
    "padding": "10px 8px",
}

TABLE_CELL_STYLE = {
    "textAlign": "center",
    "fontSize": "12px",
    "padding": "8px",
    "color": COLORS["text"],
    "borderBottom": f"1px solid {COLORS['border']}",
}

TABLE_CELL_COND = [
    {"if": {"row_index": "odd"}, "backgroundColor": "#f7fafc"},
    {"if": {"row_index": "even"}, "backgroundColor": COLORS["card"]},
]


# ---------------------------------------------------------------------------
# Helpers de serializacion
# ---------------------------------------------------------------------------

def dataframe_to_records(df):
    if df is None:
        return None
    return df.copy().to_dict("records")


def records_to_dataframe(records):
    if records is None:
        return None
    return pd.DataFrame(records)


def get_first_existing_column(df: pd.DataFrame, candidates: list, required: bool = True):
    """
    Devuelve el primer nombre de columna de candidates que exista en df.

    Compatible con el formato antiguo (mean_return) y el nuevo IS/OOS
    (mean_return_oos, mean_return_is, ...).

    Raises:
        KeyError: Si required=True y ninguna columna de candidates existe.
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


# ---------------------------------------------------------------------------
# Ejecucion de scripts de generacion de datos
# ---------------------------------------------------------------------------

def run_dashboard_data_scripts():
    """
    Ejecuta los scripts de generacion de datos del dashboard en orden.

    Returns:
        tuple: (success: bool, log: str, error: str or None)
    """
    scripts = [
        "examples/compare_agents_multiseed_experiment.py",
        "scripts/utilities/compute_ivl.py",
    ]

    log_lines = []
    all_success = True

    for script in scripts:
        script_path = PROJECT_ROOT / script
        log_lines.append(f"\n{'='*50}")
        log_lines.append(f"Ejecutando: {script}")
        log_lines.append(f"{'='*50}\n")

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=600,
            )

            if result.stdout:
                log_lines.append(result.stdout)
            if result.stderr:
                log_lines.append(f"\n[STDERR]\n{result.stderr}")

            if result.returncode != 0:
                all_success = False
                log_lines.append(f"\nError: {script} fallo con codigo {result.returncode}")
                break
            else:
                log_lines.append(f"\nCompletado: {script}")

        except subprocess.TimeoutExpired:
            all_success = False
            log_lines.append(f"\nError: {script} excedio el tiempo limite (10 min)")
            break
        except Exception as exc:
            all_success = False
            log_lines.append(f"\nError inesperado al ejecutar {script}: {exc}")
            break

    full_log = "\n".join(log_lines)
    truncated_log = full_log[-4000:] if len(full_log) > 4000 else full_log
    return all_success, truncated_log, None if all_success else "Error en la ejecucion de scripts"


# ---------------------------------------------------------------------------
# Carga inicial de datos
# ---------------------------------------------------------------------------
print("Cargando datos del dashboard...")

available_tickers = get_available_tickers()
initial_ticker = available_tickers[0] if available_tickers else None

raw_dashboard_data = load_all_dashboard_data(ticker=initial_ticker)

agent_summary_df,      agent_summary_error      = raw_dashboard_data["agent_summary"]
agent_seed_metrics_df, agent_seed_metrics_error  = raw_dashboard_data["agent_seed_metrics"]
ivl_results_df,        ivl_results_error         = raw_dashboard_data["ivl_results"]
ticker_comparison_df,  ticker_comparison_error    = raw_dashboard_data["ticker_comparison"]
validation_metrics_df, validation_metrics_error   = raw_dashboard_data["validation_metrics"]
experiment_config,     experiment_config_error     = raw_dashboard_data["experiment_config"]

dashboard_data_serializable = {
    "agent_summary":      dataframe_to_records(agent_summary_df),
    "agent_seed_metrics": dataframe_to_records(agent_seed_metrics_df),
    "ivl_results":        dataframe_to_records(ivl_results_df),
    "ticker_comparison":  dataframe_to_records(ticker_comparison_df),
    "validation_metrics": dataframe_to_records(validation_metrics_df),
    "experiment_config":  experiment_config,
}

missing_files = {k: v for k, v in {
    "agent_summary.csv":      agent_summary_error,
    "agent_seed_metrics.csv": agent_seed_metrics_error,
    "ivl_results.csv":        ivl_results_error,
}.items() if v}

# ---------------------------------------------------------------------------
# Deteccion de columnas — compatible con formato antiguo y nuevo IS/OOS
# ---------------------------------------------------------------------------
if agent_summary_df is not None and not agent_summary_df.empty:
    AGENT_COL = get_first_existing_column(
        agent_summary_df, ["agent_name", "agent", "Agent"])
    RETURN_COL = get_first_existing_column(
        agent_summary_df,
        ["mean_return_oos", "mean_return", "mean_return_is",
         "return_mean", "total_return_mean", "mean_total_return"])
    STD_RETURN_COL = get_first_existing_column(
        agent_summary_df,
        ["std_return_oos", "seed_std_return_oos", "std_return", "seed_std_return",
         "std_return_is", "seed_std_return_is"],
        required=False)
    TRADES_COL = get_first_existing_column(
        agent_summary_df,
        ["mean_n_trades", "mean_trades", "n_trades_mean", "trades_mean"],
        required=False)
    EQUITY_COL = get_first_existing_column(
        agent_summary_df,
        ["mean_final_equity_oos", "mean_final_equity", "final_equity_mean",
         "mean_equity_oos", "mean_equity"],
        required=False)
else:
    AGENT_COL      = "agent_name"
    RETURN_COL     = "mean_return"
    STD_RETURN_COL = None
    TRADES_COL     = None
    EQUITY_COL     = None

# ---------------------------------------------------------------------------
# KPIs iniciales
# ---------------------------------------------------------------------------
kpi_data = {}
if agent_summary_df is not None and not agent_summary_df.empty:
    kpi_data["num_agents"]  = len(agent_summary_df)
    best_metric_col = (
        "mean_sharpe_oos" if "mean_sharpe_oos" in agent_summary_df.columns
        else RETURN_COL
    )
    kpi_data["best_agent"] = agent_display_name(agent_summary_df.loc[
        agent_summary_df[best_metric_col].idxmax(), AGENT_COL])
    kpi_data["best_metric"] = agent_summary_df[best_metric_col].max()
    kpi_data["best_metric_label"] = (
        "Sharpe OOS" if best_metric_col == "mean_sharpe_oos" else "Retorno"
    )
else:
    kpi_data["num_agents"]  = 0
    kpi_data["best_agent"]  = "N/A"
    kpi_data["best_metric"] = 0.0
    kpi_data["best_metric_label"] = "Sharpe OOS"

if agent_seed_metrics_df is not None and not agent_seed_metrics_df.empty:
    kpi_data["num_seeds"] = agent_seed_metrics_df["seed"].nunique()
else:
    kpi_data["num_seeds"] = 0

if ivl_results_df is not None and not ivl_results_df.empty:
    best_ivl_row = ivl_results_df.loc[ivl_results_df["ivl"].idxmax()]
    kpi_data["ivl_value"] = best_ivl_row["ivl"]
    kpi_data["ivl_agent"] = agent_display_name(best_ivl_row.get("latent_agent"))
else:
    kpi_data["ivl_value"]          = 0.0
    kpi_data["ivl_agent"]          = "N/A"


# ---------------------------------------------------------------------------
# Componentes de layout
# ---------------------------------------------------------------------------

def _missing_files_alert():
    """Banda de avisos de archivos faltantes, solo si hay alguno."""
    if not missing_files:
        return html.Div()
    alerts = [
        html.Div([
            html.Span(f"Archivo faltante: {fname}", className="alert-title"),
            html.Span(f"  {msg}", className="alert-message"),
        ], className="alert alert-warning")
        for fname, msg in missing_files.items()
    ]
    return html.Div(alerts, style={"marginBottom": "8px"})


def _ticker_selector_row():
    """
    Fila del selector de ticker.

    Visible cuando hay mas de un ticker disponible; oculta pero presente
    en el DOM cuando hay cero o un ticker (para que los callbacks funcionen).
    """
    style = (
        {"display": "flex"} if len(available_tickers) >= 1
        else {"display": "none"}
    )
    options = [{"label": t, "value": t} for t in available_tickers]
    value   = available_tickers[0] if available_tickers else None

    return html.Div([
        html.Span("Ticker:", className="ticker-selector-label"),
        dcc.Dropdown(
            id="ticker-selector",
            options=options,
            value=value,
            clearable=False,
            style={"minWidth": "180px", "fontSize": "14px"},
        ),
    ], className="ticker-selector-row", style=style)


def _agent_experiment_details():
    """Banner informativo compartido por cualquier run A/B/C/D."""
    return html.Details([
        html.Summary("¿Qué compara este experimento?"),
        html.P(
            "A es el baseline directo; B, C y D son variantes LatentDQN que "
            "difieren en cómo se obtiene y entrena su representación."
        ),
        html.Ul([
            html.Li([html.Strong(f"{AGENT_DISPLAY[agent]}: "),
                     AGENT_DESCRIPTIONS[agent]])
            for agent in ("A", "B", "C", "D")
        ]),
    ], className="experiment-details")


def _metric_selector(selector_id, options, value, label="Métrica:"):
    return html.Div([
        html.Span(label, className="ticker-selector-label"),
        dcc.Dropdown(
            id=selector_id,
            options=[{"label": option_label, "value": option_value}
                     for option_label, option_value in options],
            value=value,
            clearable=False,
            style={"minWidth": "240px", "fontSize": "14px"},
        ),
    ], className="inline-selector")


def _control_panel():
    """Panel de control compacto en una sola fila."""
    return html.Div([
        html.Div("Control de Experimento", className="section-title"),
        html.Div([
            html.P(
                "Los scripts pueden tardar varios minutos. "
                "El experimento descarga datos de Yahoo Finance y entrena agentes.",
                className="control-warning",
            ),
            html.Div([
                html.Button(
                    "Generar datos",
                    id="generate-btn",
                    n_clicks=0,
                    className="control-button control-button-primary",
                ),
                html.Button(
                    "Recargar CSV",
                    id="reload-btn",
                    n_clicks=0,
                    className="control-button control-button-secondary",
                ),
                html.Span(
                    "Listo",
                    id="run-status-text",
                    className="run-status",
                    **{"data-status": "ready"},
                ),
            ], className="control-buttons"),
            html.Div(id="run-result-message", className="run-result"),
            html.Div(id="run-log", className="run-log"),
        ], className="control-content"),
    ], className="section control-section")


def _kpi_row():
    """Fila de 4 tarjetas KPI."""
    return html.Div([
        create_kpi_card("Agentes comparados", str(kpi_data["num_agents"])),
        create_kpi_card("Semillas", str(kpi_data["num_seeds"])),
        html.Div(
            create_kpi_card(
                "Mejor agente (OOS)" if RETURN_COL and "oos" in RETURN_COL else "Mejor agente",
                kpi_data["best_agent"],
                f"{kpi_data['best_metric_label']}: {kpi_data['best_metric']:+.4f}",
            ),
            id="best-agent-kpi-container",
        ),
        html.Div(
            create_kpi_card(
                "Mejor latente por IVL",
                kpi_data["ivl_agent"],
                f"IVL: {kpi_data['ivl_value']:+.4f}",
            ),
            id="ivl-kpi-container",
        ),
    ], className="kpi-grid", style={"marginBottom": "16px"})


def _section(title, *children, class_extra=""):
    return html.Div([
        html.Div(title, className="section-title"),
        *children,
    ], className=f"section {class_extra}".strip())


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------

def create_layout():
    show_cross_ticker = len(available_tickers) > 1

    return html.Div([
        dcc.Store(id="dashboard-data", data=dashboard_data_serializable),

        # ── Header ──────────────────────────────────────────────────────────
        html.Div([
            html.H1("Dashboard de Ventaja Latente", className="header-title"),
            html.Div(
                "Comparacion exploratoria de agentes financieros con y sin representaciones latentes",
                className="subtitle",
            ),
        ], className="header"),

        # ── Contenido principal ─────────────────────────────────────────────
        html.Div([

            # Control + alertas
            _control_panel(),
            _missing_files_alert(),
            _agent_experiment_details(),

            # KPIs
            _kpi_row(),

            # Selector de ticker (visible solo si >1 ticker)
            _ticker_selector_row(),

            # ── Seccion IVL ─────────────────────────────────────────────────
            _section("Indice de Ventaja Latente (IVL)",

                _metric_selector(
                    "latent-agent-selector",
                    [
                        ("Todos", "ALL"),
                        (AGENT_DISPLAY["B"], "B"),
                        (AGENT_DISPLAY["C"], "C"),
                        (AGENT_DISPLAY["D"], "D"),
                    ],
                    "ALL",
                    label="Agente latente:",
                ),

                # Valor + badge (layout horizontal)
                html.Div([
                    html.Div([
                        html.Div(id="ivl-overview-card"),
                    ], style={"minWidth": "180px"}),

                    html.Div([
                        html.Div("Componentes", className="section-title",
                                 style={"fontSize": "0.85em", "marginBottom": "10px"}),
                        html.Div(id="ivl-delta-cards"),
                    ], style={"flex": "1"}),
                ], className="ivl-card"),

                html.Div(id="ivl-summary-table", className="table-container"),

                # Grafica de deltas
                html.Div(id="ivl-delta-chart", className="graph-container"),
            ),

            # ── Comparacion cross-ticker (solo si hay >1 ticker) ─────────────
            html.Div([
                html.Div("Comparacion Cross-Ticker", className="section-title"),
                html.P(
                    "IVL calculado de forma independiente para cada activo. "
                    "Permite evaluar si la ventaja latente es consistente entre mercados.",
                    className="cross-ticker-note",
                ),
                dcc.RadioItems(
                    id="cross-ticker-view-selector",
                    options=[
                        {"label": " Barras", "value": "bars"},
                        {"label": " Heatmap", "value": "heatmap"},
                    ],
                    value="bars",
                    inline=True,
                    className="view-selector",
                ),
                html.Div(id="cross-ticker-chart", className="graph-container"),
                html.Div(id="cross-ticker-summary-table", className="table-container"),
            ], className="section",
               style={} if show_cross_ticker else {"display": "none"}),

            # ── Comparacion de agentes ───────────────────────────────────────
            _section("Comparacion de Agentes",
                _metric_selector(
                    "agent-metric-selector",
                    [
                        ("Sharpe OOS", "mean_sharpe_oos"),
                        ("Retorno OOS", "mean_return_oos"),
                        ("Max Drawdown OOS", "mean_mdd_oos"),
                        ("Gap Sharpe IS/OOS", "sharpe_gap"),
                        ("Nº trades medio", "mean_n_trades"),
                        ("Equity final OOS", "mean_equity_oos"),
                    ],
                    "mean_sharpe_oos",
                ),
                html.Div([
                    html.Div(id="agent-comparison-chart", className="graph-container"),
                    html.Div(id="agent-summary-table",    className="table-container"),
                ], className="chart-grid"),
            ),

            # ── Metricas por semilla ─────────────────────────────────────────
            _section("Metricas por Semilla",
                _metric_selector(
                    "seed-metric-selector",
                    [
                        ("Sharpe OOS", "sharpe"),
                        ("Retorno total OOS", "total_return"),
                    ],
                    "sharpe",
                ),
                html.Div([
                    html.Div(id="seed-boxplot-chart", className="graph-container"),
                    html.Div(id="seed-scatter-chart", className="graph-container"),
                ], className="chart-grid"),
            ),

            html.Details([
                html.Summary("Validación interna"),
                html.P(
                    "Trazabilidad del checkpoint seleccionado en el tramo de validación.",
                    className="cross-ticker-note",
                ),
                html.Div(id="validation-summary-table", className="table-container"),
            ], className="section collapsible-section"),

            # ── Notas metodologicas ──────────────────────────────────────────
            _section("Notas Metodologicas",
                html.Div([
                    html.Div(id="methodology-config"),
                    html.H3("Glosario de columnas"),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Columna", style={"textAlign": "left", "paddingRight": "24px"}),
                            html.Th("Descripcion",  style={"textAlign": "left"}),
                        ])),
                        html.Tbody([
                            html.Tr([html.Td(html.Strong("Agente"),                  style={"paddingRight": "24px", "verticalAlign": "top"}), html.Td("Nombre del agente evaluado (RandomAgent, BuyAndHold, DQNAgent, LatentDQNAgent).")]),
                            html.Tr([html.Td(html.Strong("Retorno medio IS"),        style={"paddingRight": "24px", "verticalAlign": "top"}), html.Td("Media del retorno total obtenido durante la fase In-Sample (entrenamiento), promediado sobre todas las semillas.")]),
                            html.Tr([html.Td(html.Strong("Desv. retorno IS"),        style={"paddingRight": "24px", "verticalAlign": "top"}), html.Td("Desviacion tipica del retorno IS entre semillas. Mide la variabilidad del agente en entrenamiento.")]),
                            html.Tr([html.Td(html.Strong("Sharpe IS"),               style={"paddingRight": "24px", "verticalAlign": "top"}), html.Td("Ratio de Sharpe medio en IS: retorno ajustado por la volatilidad. Valores mas altos indican mejor relacion rentabilidad/riesgo.")]),
                            html.Tr([html.Td(html.Strong("Max Drawdown IS"),         style={"paddingRight": "24px", "verticalAlign": "top"}), html.Td("Caida maxima media desde el pico de equity hasta el valle en IS. Valor negativo; cuanto mas proximo a 0, mejor.")]),
                            html.Tr([html.Td(html.Strong("Equity media IS"),         style={"paddingRight": "24px", "verticalAlign": "top"}), html.Td("Valor medio del portfolio al final de cada episodio IS (balance inicial + ganancias/perdidas).")]),
                            html.Tr([html.Td(html.Strong("Retorno medio OOS"),       style={"paddingRight": "24px", "verticalAlign": "top"}), html.Td("Media del retorno total en la fase Out-of-Sample (datos no vistos durante el entrenamiento).")]),
                            html.Tr([html.Td(html.Strong("Desv. retorno OOS"),       style={"paddingRight": "24px", "verticalAlign": "top"}), html.Td("Desviacion tipica del retorno OOS entre semillas.")]),
                            html.Tr([html.Td(html.Strong("Sharpe OOS"),              style={"paddingRight": "24px", "verticalAlign": "top"}), html.Td("Ratio de Sharpe medio en OOS. Es el indicador principal de generalizacion del agente.")]),
                            html.Tr([html.Td(html.Strong("Max Drawdown OOS"),        style={"paddingRight": "24px", "verticalAlign": "top"}), html.Td("Caida maxima media en OOS. Refleja el riesgo real del agente sobre datos no vistos.")]),
                            html.Tr([html.Td(html.Strong("Equity media OOS"),        style={"paddingRight": "24px", "verticalAlign": "top"}), html.Td("Valor medio del portfolio al final de cada episodio OOS.")]),
                            html.Tr([html.Td(html.Strong("Nº trades medio"),         style={"paddingRight": "24px", "verticalAlign": "top"}), html.Td("Numero medio de operaciones (compra/venta) realizadas por episodio. Afecta a los costes de transaccion.")]),
                            html.Tr([html.Td(html.Strong("Desv. entre semillas IS"), style={"paddingRight": "24px", "verticalAlign": "top"}), html.Td("Desviacion tipica del retorno IS calculada entre distintas semillas. Mide la estabilidad del agente ante distintas inicializaciones.")]),
                        ]),
                    ], style={"borderCollapse": "collapse", "width": "100%", "marginBottom": "24px",
                              "fontSize": "13px", "lineHeight": "1.6"}),

                    html.H3("Limitaciones de interpretación", style={"marginTop": "8px"}),
                    html.Ul([
                        html.Li("Dashboard exploratorio — no prueba superioridad estadistica."),
                        html.Li("El IVL es una metrica agregada; analizar cada componente por separado."),
                        html.Li("El split IS/OOS es temporal: IS precede siempre a OOS."),
                        html.Li("Resultados pasados no garantizan resultados futuros."),
                    ]),
                ], className="notes"),
            ),

            # ── Footer ──────────────────────────────────────────────────────
            html.Div(
                "Dashboard de Ventaja Latente — TFM sobre RL con representaciones latentes",
                className="footer",
            ),

        ], className="dashboard-container"),
    ])


# ---------------------------------------------------------------------------
# App Dash
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    title="Dashboard de Ventaja Latente",
    suppress_callback_exceptions=True,
    assets_folder="assets",
)

app.layout = create_layout()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("dashboard-data",    "data"),
    Output("run-result-message","children"),
    Output("run-log",           "children"),
    Output("run-status-text",   "children"),
    Output("run-status-text",   "data-status"),
    Output("generate-btn",      "disabled"),
    Input("generate-btn",       "n_clicks"),
    Input("reload-btn",         "n_clicks"),
    Input("ticker-selector",    "value"),
    State("dashboard-data",     "data"),
    prevent_initial_call=True,
)
def update_dashboard_data(generate_clicks, reload_clicks, selected_ticker, current_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_data, html.Div(), html.Div(), "Listo", "ready", False

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "generate-btn":
        success, log, error = run_dashboard_data_scripts()
        if success:
            try:
                # Volver a detectar tickers despues de generar
                new_tickers  = get_available_tickers()
                load_ticker  = new_tickers[0] if new_tickers else None
                raw = load_all_dashboard_data(ticker=load_ticker)
                new_data = {
                    "agent_summary":      dataframe_to_records(raw["agent_summary"][0]),
                    "agent_seed_metrics": dataframe_to_records(raw["agent_seed_metrics"][0]),
                    "ivl_results":        dataframe_to_records(raw["ivl_results"][0]),
                    "ticker_comparison":  dataframe_to_records(raw["ticker_comparison"][0]),
                    "validation_metrics": dataframe_to_records(raw["validation_metrics"][0]),
                    "experiment_config":  raw["experiment_config"][0],
                }
                return (new_data,
                        html.Div("Datos generados y cargados correctamente",
                                 className="alert alert-success"),
                        html.Div(log, className="run-log"),
                        "Completado", "completed", False)
            except Exception as exc:
                return (current_data,
                        html.Div(f"Scripts completados, pero error al cargar datos: {exc}",
                                 className="alert alert-warning"),
                        html.Div(log, className="run-log"),
                        "Error al cargar", "error", False)
        else:
            return (current_data,
                    html.Div(f"Error: {error}", className="alert alert-danger"),
                    html.Div(log, className="run-log"),
                    "Error", "error", False)

    elif button_id in ("reload-btn", "ticker-selector"):
        # Recargar datos para el ticker seleccionado (o auto-detectar si es None)
        try:
            raw = load_all_dashboard_data(ticker=selected_ticker)
            new_data = {
                "agent_summary":      dataframe_to_records(raw["agent_summary"][0]),
                "agent_seed_metrics": dataframe_to_records(raw["agent_seed_metrics"][0]),
                "ivl_results":        dataframe_to_records(raw["ivl_results"][0]),
                "ticker_comparison":  dataframe_to_records(raw["ticker_comparison"][0]),
                "validation_metrics": dataframe_to_records(raw["validation_metrics"][0]),
                "experiment_config":  raw["experiment_config"][0],
            }
            if button_id == "ticker-selector":
                label = f"Datos cargados para ticker: {selected_ticker or 'auto'}"
            else:
                label = "Datos recargados desde los CSV existentes"
            return (new_data,
                    html.Div(label, className="alert alert-success"),
                    html.Div("", className="run-log"),
                    "Recargado", "completed", False)
        except Exception as exc:
            return (current_data,
                    html.Div(f"Error al recargar: {exc}", className="alert alert-danger"),
                    html.Div(str(exc), className="run-log"),
                    "Error", "error", False)

    return current_data, html.Div(), html.Div(), "Listo", "ready", False


@app.callback(
    Output("ivl-kpi-container", "children"),
    Output("ivl-overview-card", "children"),
    Output("ivl-summary-table", "children"),
    Output("ivl-delta-cards",   "children"),
    Output("ivl-delta-chart",   "children"),
    Input("dashboard-data",     "data"),
    Input("latent-agent-selector", "value"),
)
def update_ivl_section(data, selected_latent="ALL"):
    try:
        ivl_df = records_to_dataframe((data or {}).get("ivl_results"))
        if ivl_df is None or ivl_df.empty:
            empty = html.Div("No hay datos de IVL disponibles",
                             style={"color": "#718096", "fontSize": "0.9em"})
            return empty, empty, empty, empty, empty

        if "latent_agent" not in ivl_df.columns:
            ivl_df["latent_agent"] = "LatentDQNAgent"
        best_row = ivl_df.loc[ivl_df["ivl"].idxmax()]
        selected_df = (
            ivl_df if selected_latent in (None, "ALL")
            else ivl_df[ivl_df["latent_agent"] == selected_latent]
        )
        if selected_df.empty:
            selected_df = ivl_df
        detail_row = (
            best_row if selected_latent in (None, "ALL") else selected_df.iloc[0]
        )

        best_agent = agent_display_name(best_row["latent_agent"])
        ivl_kpi = create_kpi_card(
            "Mejor latente por IVL",
            best_agent,
            f"IVL: {float(best_row['ivl']):+.4f}",
        )
        overview = html.Div([
            create_ivl_card(
                float(detail_row["ivl"]),
                str(detail_row.get("interpretation", "neutral")),
            ),
            html.Div(
                ("Mejor del ticker: " if selected_latent in (None, "ALL")
                 else "Agente seleccionado: ")
                + agent_display_name(detail_row["latent_agent"]),
                className="ivl-selection-note",
            ),
        ])

        interpretation_labels = {
            "latent_advantage": "Ventaja latente",
            "direct_advantage": "Ventaja directa",
            "neutral": "Neutral",
        }
        table_df = ivl_df[[
            "latent_agent", "ivl", "interpretation", "delta_sharpe",
            "delta_mdd", "delta_seed_std", "delta_is_oos_gap",
        ]].copy()
        table_df["latent_agent"] = table_df["latent_agent"].map(agent_display_name)
        table_df["interpretation"] = table_df["interpretation"].map(
            lambda value: interpretation_labels.get(str(value), str(value))
        )
        numeric_columns = [
            "ivl", "delta_sharpe", "delta_mdd", "delta_seed_std",
            "delta_is_oos_gap",
        ]
        table_df[numeric_columns] = table_df[numeric_columns].round(4)
        table_df.columns = [
            "Agente latente", "IVL", "Interpretación", "ΔSharpe", "ΔMDD",
            "ΔSeedStd", "ΔGap IS/OOS",
        ]
        table = dash_table.DataTable(
            data=table_df.to_dict("records"),
            columns=[{"name": column, "id": column} for column in table_df.columns],
            style_table={"overflowX": "auto", "fontSize": "12px"},
            style_header=TABLE_HEADER_STYLE,
            style_cell=TABLE_CELL_STYLE,
            style_data_conditional=TABLE_CELL_COND,
        )

        delta_cards = html.Div([
            create_delta_card("Delta Sharpe",    detail_row["delta_sharpe"],    is_better_positive=True),
            create_delta_card("Delta MDD",       detail_row["delta_mdd"],       is_better_positive=False),
            create_delta_card("Delta Seed Std",  detail_row["delta_seed_std"],  is_better_positive=False),
            create_delta_card("Delta IS/OOS Gap",detail_row["delta_is_oos_gap"],is_better_positive=False),
        ], className="delta-grid")

        delta_chart = dcc.Graph(
            figure=create_ivl_delta_bar(selected_df),
            config={"displayModeBar": False},
        )
        return ivl_kpi, overview, table, delta_cards, delta_chart
    except Exception as exc:
        err = html.Div(f"Error IVL: {exc}", style={"color": "#c53030"})
        return err, err, err, err, err


@app.callback(
    Output("cross-ticker-chart", "children"),
    Output("cross-ticker-summary-table", "children"),
    Input("dashboard-data",      "data"),
    Input("cross-ticker-view-selector", "value"),
)
def update_cross_ticker(data, view="bars"):
    try:
        ct_df = records_to_dataframe((data or {}).get("ticker_comparison"))
        if ct_df is None or ct_df.empty:
            empty = html.Div(
                "Sin datos cross-ticker. Ejecuta el experimento con varios tickers.",
                style={"color": "#718096", "fontSize": "0.9em"},
            )
            return empty, html.Div()
        chart = dcc.Graph(
            figure=create_cross_ticker_ivl_chart(ct_df, view=view),
            config={"displayModeBar": False},
        )
        detail = ct_df[
            ~ct_df["ticker"].astype(str).str.upper().isin(["MEAN", "STD"])
        ].drop_duplicates(subset=["ticker", "latent_agent"], keep="last")
        stats = detail.groupby("latent_agent")["ivl"].agg(["mean", "std"]).reset_index()
        stats["Agente latente"] = stats["latent_agent"].map(agent_display_name)
        stats["IVL medio"] = stats["mean"].round(4)
        stats["IVL std."] = stats["std"].fillna(0).round(4)
        stats = stats[["Agente latente", "IVL medio", "IVL std."]]
        table = dash_table.DataTable(
            data=stats.to_dict("records"),
            columns=[{"name": column, "id": column} for column in stats.columns],
            style_table={"overflowX": "auto", "fontSize": "12px"},
            style_header=TABLE_HEADER_STYLE,
            style_cell=TABLE_CELL_STYLE,
            style_data_conditional=TABLE_CELL_COND,
        )
        return chart, table
    except Exception as exc:
        err = html.Div(f"Error cross-ticker: {exc}", style={"color": "#c53030"})
        return err, err


@app.callback(
    Output("best-agent-kpi-container", "children"),
    Output("agent-comparison-chart", "children"),
    Output("agent-summary-table",    "children"),
    Input("dashboard-data",          "data"),
    Input("agent-metric-selector",   "value"),
)
def update_agent_comparison(data, selected_metric="mean_sharpe_oos"):
    try:
        summary_records = data.get("agent_summary")
        summary_df = records_to_dataframe(summary_records)

        if summary_df is None or summary_df.empty:
            msg = html.Div("No hay datos de agentes disponibles",
                           style={"color": "#718096"})
            return msg, msg, msg

        best_col = (
            "mean_sharpe_oos" if "mean_sharpe_oos" in summary_df.columns
            else selected_metric
        )
        best_row = summary_df.loc[summary_df[best_col].idxmax()]
        best_label = "Sharpe OOS" if best_col == "mean_sharpe_oos" else best_col
        best_kpi = create_kpi_card(
            "Mejor agente (OOS)",
            agent_display_name(best_row.get("agent_name", best_row.get("agent"))),
            f"{best_label}: {float(best_row[best_col]):+.4f}",
        )

        chart = dcc.Graph(
            figure=create_agent_metric_bar(summary_df, selected_metric),
            config={"displayModeBar": False},
        )

        # Columnas legibles: excluir columnas muy largas
        COL_LABELS = {
            "agent_name":        "Agente",
            "mean_return_is":    "Retorno medio IS",
            "std_return_is":     "Desv. retorno IS",
            "mean_sharpe_is":    "Sharpe IS",
            "mean_mdd_is":       "Max Drawdown IS",
            "mean_equity_is":    "Equity media IS",
            "mean_return_oos":   "Retorno medio OOS",
            "std_return_oos":    "Desv. retorno OOS",
            "mean_sharpe_oos":   "Sharpe OOS",
            "mean_mdd_oos":      "Max Drawdown OOS",
            "mean_equity_oos":   "Equity media OOS",
            "mean_n_trades":     "Nº trades medio",
            "seed_std_return_is":"Desv. entre semillas IS",
        }
        table_df = summary_df.copy()
        if "agent_name" in table_df.columns:
            table_df.insert(
                0, "agent_display", table_df["agent_name"].map(agent_display_name)
            )
        if {"mean_sharpe_is", "mean_sharpe_oos"}.issubset(table_df.columns):
            table_df["sharpe_gap"] = (
                table_df["mean_sharpe_is"] - table_df["mean_sharpe_oos"]
            ).abs()
        COL_LABELS["agent_display"] = "Agente"
        COL_LABELS["sharpe_gap"] = "Gap Sharpe IS/OOS"
        display_cols = [
            c for c in table_df.columns
            if c not in ("seed", "agent_name")
        ]
        table = dash_table.DataTable(
            data=table_df[display_cols].round(4).to_dict("records"),
            columns=[{"name": COL_LABELS.get(c, c), "id": c} for c in display_cols],
            style_table={"overflowX": "auto", "fontSize": "12px"},
            style_header=TABLE_HEADER_STYLE,
            style_cell=TABLE_CELL_STYLE,
            style_data_conditional=TABLE_CELL_COND,
            page_size=10,
        )
        return best_kpi, chart, table
    except Exception as exc:
        err = html.Div(f"Error comparacion: {exc}", style={"color": "#c53030"})
        return err, err, err


@app.callback(
    Output("seed-boxplot-chart", "children"),
    Output("seed-scatter-chart", "children"),
    Input("dashboard-data",      "data"),
    Input("seed-metric-selector", "value"),
)
def update_seed_metrics(data, selected_metric="sharpe"):
    try:
        seed_df = records_to_dataframe(data.get("agent_seed_metrics"))
        if seed_df is None or seed_df.empty:
            msg = html.Div("No hay datos por semilla disponibles",
                           style={"color": "#718096"})
            return msg, msg

        boxplot = dcc.Graph(figure=create_seed_metric_boxplot(seed_df, selected_metric),
                            config={"displayModeBar": False})
        scatter = dcc.Graph(figure=create_seed_metric_scatter(seed_df, selected_metric),
                            config={"displayModeBar": False})
        return boxplot, scatter
    except Exception as exc:
        err = html.Div(f"Error metricas por semilla: {exc}", style={"color": "#c53030"})
        return err, err


@app.callback(
    Output("validation-summary-table", "children"),
    Input("dashboard-data", "data"),
)
def update_validation_summary(data):
    """Resume la seleccion de checkpoints sin convertirla en narrativa principal."""
    try:
        validation_df = records_to_dataframe((data or {}).get("validation_metrics"))
        if validation_df is None or validation_df.empty:
            return html.Div(
                "Esta run no contiene validation_metrics.csv.",
                style={"color": "#718096", "fontSize": "0.9em"},
            )
        required = [
            "agent_name", "best_val_episode", "best_val_score",
            "best_val_sharpe", "best_val_mdd",
        ]
        missing = [column for column in required if column not in validation_df.columns]
        if missing:
            return html.Div(
                f"Validación disponible, pero faltan columnas: {missing}",
                style={"color": "#718096", "fontSize": "0.9em"},
            )
        summary = validation_df.groupby("agent_name", as_index=False)[required[1:]].mean()
        summary.insert(0, "Agente", summary["agent_name"].map(agent_display_name))
        summary = summary.drop(columns="agent_name").rename(columns={
            "best_val_episode": "Episodio seleccionado (media)",
            "best_val_score": "Best val score (media)",
            "best_val_sharpe": "Best val Sharpe (media)",
            "best_val_mdd": "Best val MDD (media)",
        })
        summary = summary.round(4)
        return dash_table.DataTable(
            data=summary.to_dict("records"),
            columns=[{"name": column, "id": column} for column in summary.columns],
            style_table={"overflowX": "auto", "fontSize": "12px"},
            style_header=TABLE_HEADER_STYLE,
            style_cell=TABLE_CELL_STYLE,
            style_data_conditional=TABLE_CELL_COND,
        )
    except Exception as exc:
        return html.Div(f"Error de validación: {exc}", style={"color": "#c53030"})


@app.callback(
    Output("methodology-config", "children"),
    Input("dashboard-data", "data"),
)
def update_methodology_notes(data):
    """Expone la configuracion efectiva en vez de asumir valores fijos."""
    config = (data or {}).get("experiment_config") or {}
    if not config:
        return html.Div([
            html.H3("Configuración de la run"),
            html.P(
                "No hay experiment_config.json; las métricas mostradas se leen "
                "directamente de los CSV disponibles."
            ),
        ])

    labels = [
        ("tickers", "Tickers"),
        ("seeds", "Semillas"),
        ("n_training_episodes", "Episodios de entrenamiento"),
        ("max_steps_per_episode", "Máximo de pasos por episodio"),
        ("reward_mode", "Modo de recompensa"),
        ("random_start_train", "Inicio aleatorio en entrenamiento"),
        ("use_internal_validation", "Validación interna"),
        ("internal_val_ratio", "Ratio de validación interna"),
        ("dqn_hidden_dim", "Dimensión oculta DirectDQN"),
        ("latent_q_hidden_dim", "Dimensión oculta Q latente"),
        ("transaction_cost", "Coste de transacción"),
    ]

    def render_value(value):
        if isinstance(value, bool):
            return "Sí" if value else "No"
        if isinstance(value, list):
            return ", ".join(map(str, value))
        return "—" if value is None else str(value)

    rows = [
        html.Tr([
            html.Td(html.Strong(label), style={"paddingRight": "24px"}),
            html.Td(render_value(config.get(key))),
        ])
        for key, label in labels
        if key in config
    ]
    return html.Div([
        html.H3("Configuración de la run"),
        html.Table(html.Tbody(rows), className="config-table"),
        html.P(
            "Sharpe OOS es la vista inicial porque facilita la comparación "
            "riesgo-retorno entre activos; debe interpretarse junto con retorno, "
            "drawdown, estabilidad entre semillas, gap IS/OOS e IVL.",
            className="methodology-note",
        ),
    ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Dashboard de Ventaja Latente")
    print("=" * 60)

    from dashboard.data.loader import get_results_path
    print(f"\nDirectorio de resultados: {get_results_path()}")
    if available_tickers:
        print(f"Tickers disponibles: {available_tickers}")
    print("\nhttp://127.0.0.1:8050")
    print("Ctrl+C para detener\n")

    if missing_files:
        print("Archivos faltantes:")
        for fname, err in missing_files.items():
            print(f"  - {fname}")
        print()

    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=8050)
