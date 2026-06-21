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
    create_ivl_kpi_card,
    create_delta_card,
    create_agent_return_bar,
    create_seed_return_boxplot,
    create_seed_return_scatter,
    create_ivl_delta_bar,
    create_cross_ticker_ivl_chart,
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
        "examples/compute_ivl_from_results.py",
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

dashboard_data_serializable = {
    "agent_summary":      dataframe_to_records(agent_summary_df),
    "agent_seed_metrics": dataframe_to_records(agent_seed_metrics_df),
    "ivl_results":        dataframe_to_records(ivl_results_df),
    "ticker_comparison":  dataframe_to_records(ticker_comparison_df),
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
    kpi_data["best_agent"]  = agent_summary_df.loc[
        agent_summary_df[RETURN_COL].idxmax(), AGENT_COL]
    kpi_data["best_return"] = agent_summary_df[RETURN_COL].max()
else:
    kpi_data["num_agents"]  = 0
    kpi_data["best_agent"]  = "N/A"
    kpi_data["best_return"] = 0.0

if agent_seed_metrics_df is not None and not agent_seed_metrics_df.empty:
    kpi_data["num_seeds"] = agent_seed_metrics_df["seed"].nunique()
else:
    kpi_data["num_seeds"] = 0

if ivl_results_df is not None and not ivl_results_df.empty:
    kpi_data["ivl_value"]          = ivl_results_df.iloc[0]["ivl"]
    kpi_data["ivl_interpretation"] = ivl_results_df.iloc[0]["interpretation"]
else:
    kpi_data["ivl_value"]          = 0.0
    kpi_data["ivl_interpretation"] = "N/A"


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
        create_kpi_card(
            "Mejor agente (OOS)" if RETURN_COL and "oos" in RETURN_COL else "Mejor agente",
            kpi_data["best_agent"],
            f"Retorno: {kpi_data['best_return']:+.4f}",
        ),
        create_ivl_kpi_card(kpi_data["ivl_value"], kpi_data["ivl_interpretation"]),
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

            # KPIs
            _kpi_row(),

            # Selector de ticker (visible solo si >1 ticker)
            _ticker_selector_row(),

            # ── Seccion IVL ─────────────────────────────────────────────────
            _section("Indice de Ventaja Latente (IVL)",

                # Valor + badge (layout horizontal)
                html.Div([
                    html.Div([
                        create_ivl_card(
                            kpi_data["ivl_value"],
                            kpi_data["ivl_interpretation"],
                        ),
                    ], style={"minWidth": "180px"}),

                    html.Div([
                        html.Div("Componentes", className="section-title",
                                 style={"fontSize": "0.85em", "marginBottom": "10px"}),
                        html.Div(id="ivl-delta-cards"),
                    ], style={"flex": "1"}),
                ], className="ivl-card"),

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
                html.Div(id="cross-ticker-chart", className="graph-container"),
            ], className="section",
               style={} if show_cross_ticker else {"display": "none"}),

            # ── Comparacion de agentes ───────────────────────────────────────
            _section("Comparacion de Agentes",
                html.Div([
                    html.Div(id="agent-comparison-chart", className="graph-container"),
                    html.Div(id="agent-summary-table",    className="table-container"),
                ], className="chart-grid"),
            ),

            # ── Metricas por semilla ─────────────────────────────────────────
            _section("Metricas por Semilla",
                html.Div([
                    html.Div(id="seed-boxplot-chart", className="graph-container"),
                    html.Div(id="seed-scatter-chart", className="graph-container"),
                ], className="chart-grid"),
            ),

            # ── Notas metodologicas ──────────────────────────────────────────
            _section("Notas Metodologicas",
                html.Div([
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

                    html.H3("Limitaciones del experimento", style={"marginTop": "8px"}),
                    html.Ul([
                        html.Li("Dashboard exploratorio — no prueba superioridad estadistica."),
                        html.Li("Entrenamiento corto (10 episodios) con datos reales de Yahoo Finance."),
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
    Output("ivl-delta-cards", "children"),
    Output("ivl-delta-chart",  "children"),
    Input("dashboard-data",    "data"),
)
def update_ivl_section(data):
    try:
        ivl_df = records_to_dataframe(data.get("ivl_results"))
        if ivl_df is None or ivl_df.empty:
            empty = html.Div("No hay datos de IVL disponibles",
                             style={"color": "#718096", "fontSize": "0.9em"})
            return empty, empty

        row = ivl_df.iloc[0]
        delta_cards = html.Div([
            create_delta_card("Delta Sharpe",    row["delta_sharpe"],    is_better_positive=True),
            create_delta_card("Delta MDD",       row["delta_mdd"],       is_better_positive=False),
            create_delta_card("Delta Seed Std",  row["delta_seed_std"],  is_better_positive=False),
            create_delta_card("Delta IS/OOS Gap",row["delta_is_oos_gap"],is_better_positive=False),
        ], className="delta-grid")

        delta_chart = dcc.Graph(
            figure=create_ivl_delta_bar(ivl_df),
            config={"displayModeBar": False},
        )
        return delta_cards, delta_chart
    except Exception as exc:
        err = html.Div(f"Error IVL: {exc}", style={"color": "#c53030"})
        return err, err


@app.callback(
    Output("cross-ticker-chart", "children"),
    Input("dashboard-data",      "data"),
)
def update_cross_ticker(data):
    try:
        ct_df = records_to_dataframe(data.get("ticker_comparison"))
        if ct_df is None or ct_df.empty:
            return html.Div(
                "Sin datos cross-ticker. Ejecuta el experimento con varios tickers.",
                style={"color": "#718096", "fontSize": "0.9em"},
            )
        chart = dcc.Graph(
            figure=create_cross_ticker_ivl_chart(ct_df),
            config={"displayModeBar": False},
        )
        return chart
    except Exception as exc:
        return html.Div(f"Error cross-ticker: {exc}", style={"color": "#c53030"})


@app.callback(
    Output("agent-comparison-chart", "children"),
    Output("agent-summary-table",    "children"),
    Input("dashboard-data",          "data"),
)
def update_agent_comparison(data):
    try:
        summary_records = data.get("agent_summary")
        summary_df = records_to_dataframe(summary_records)

        if summary_df is None or summary_df.empty:
            msg = html.Div("No hay datos de agentes disponibles",
                           style={"color": "#718096"})
            return msg, msg

        chart = dcc.Graph(
            figure=create_agent_return_bar(summary_df),
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
        display_cols = [c for c in summary_df.columns if c not in ("seed",)]
        table = dash_table.DataTable(
            data=summary_records or [],
            columns=[{"name": COL_LABELS.get(c, c), "id": c} for c in display_cols],
            style_table={"overflowX": "auto", "fontSize": "12px"},
            style_header=TABLE_HEADER_STYLE,
            style_cell=TABLE_CELL_STYLE,
            style_data_conditional=TABLE_CELL_COND,
            page_size=10,
        )
        return chart, table
    except Exception as exc:
        err = html.Div(f"Error comparacion: {exc}", style={"color": "#c53030"})
        return err, err


@app.callback(
    Output("seed-boxplot-chart", "children"),
    Output("seed-scatter-chart", "children"),
    Input("dashboard-data",      "data"),
)
def update_seed_metrics(data):
    try:
        seed_df = records_to_dataframe(data.get("agent_seed_metrics"))
        if seed_df is None or seed_df.empty:
            msg = html.Div("No hay datos por semilla disponibles",
                           style={"color": "#718096"})
            return msg, msg

        boxplot = dcc.Graph(figure=create_seed_return_boxplot(seed_df),
                            config={"displayModeBar": False})
        scatter = dcc.Graph(figure=create_seed_return_scatter(seed_df),
                            config={"displayModeBar": False})
        return boxplot, scatter
    except Exception as exc:
        err = html.Div(f"Error metricas por semilla: {exc}", style={"color": "#c53030"})
        return err, err


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

    app.run(debug=True, host="127.0.0.1", port=8050)
