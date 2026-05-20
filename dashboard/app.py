"""
Dashboard de Ventaja Latente

Aplicación Dash interactiva para visualizar comparaciones de agentes
y el Índice de Ventaja Latente (IVL).
"""

import sys
from pathlib import Path

import dash
from dash import dcc, html, dash_table, Input, Output, State
import pandas as pd
import subprocess

# Configurar PROJECT_ROOT y sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Imports absolutos
from dashboard.data import load_all_dashboard_data
from dashboard.components import (
    create_kpi_card,
    create_ivl_card,
    create_delta_card,
    create_agent_return_bar,
    create_seed_return_boxplot,
    create_seed_return_scatter,
    create_ivl_delta_bar,
    create_price_vs_latent_plot,
    create_latent_indices_plot,
    create_pca_vs_next_return_scatter
)


def dataframe_to_records(df):
    """
    Convierte un DataFrame a records JSON serializables.

    Args:
        df: DataFrame a convertir

    Returns:
        Lista de diccionarios o None si df es None
    """
    if df is None:
        return None
    return df.copy().to_dict("records")


def records_to_dataframe(records):
    """
    Reconstruye un DataFrame desde records JSON serializables.

    Args:
        records: Lista de diccionarios o None

    Returns:
        DataFrame o None si records es None
    """
    if records is None:
        return None
    return pd.DataFrame(records)


def get_first_existing_column(df: pd.DataFrame, candidates: list, required: bool = True):
    """
    Devuelve el primer nombre de columna de `candidates` que exista en `df`.

    Permite que el dashboard sea compatible con el formato antiguo (mean_return)
    y con el nuevo formato IS/OOS (mean_return_oos, mean_return_is, …).

    Args:
        df: DataFrame donde buscar
        candidates: Lista de nombres candidatos, ordenados por preferencia
        required: Si True, lanza KeyError cuando ningún candidato existe

    Returns:
        Nombre de la primera columna encontrada, o None si required=False y
        ninguna columna existe.

    Raises:
        KeyError: Si required=True y ningún candidato se encuentra en df.
    """
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"No se encontró ninguna de estas columnas: {candidates}. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )
    return None


def run_dashboard_data_scripts():
    """
    Ejecuta los scripts de generación de datos del dashboard.

    Returns:
        tuple: (success: bool, log: str, error: str or None)
    """
    scripts = [
        "examples/latent_index_example.py",
        "examples/compare_agents_multiseed_experiment.py",
        "examples/compute_ivl_from_results.py",
    ]

    log_lines = []
    all_success = True

    for script in scripts:
        script_path = PROJECT_ROOT / script
        log_lines.append(f"\n{'='*60}")
        log_lines.append(f"Ejecutando: {script}")
        log_lines.append(f"{'='*60}\n")

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=600  # 10 minutos por script
            )

            # Añadir stdout
            if result.stdout:
                log_lines.append(result.stdout)

            # Añadir stderr si hay
            if result.stderr:
                log_lines.append(f"\n[STDERR]\n{result.stderr}")

            # Verificar resultado
            if result.returncode != 0:
                all_success = False
                log_lines.append(f"\n❌ Error: {script} falló con código {result.returncode}")
                break
            else:
                log_lines.append(f"\n✅ Completado: {script}")

        except subprocess.TimeoutExpired:
            all_success = False
            log_lines.append(f"\n❌ Error: {script} excedió el tiempo límite (10 minutos)")
            break
        except Exception as e:
            all_success = False
            log_lines.append(f"\n❌ Error inesperado al ejecutar {script}: {str(e)}")
            break

    # Truncar log a los últimos 4000 caracteres
    full_log = "\n".join(log_lines)
    truncated_log = full_log[-4000:] if len(full_log) > 4000 else full_log

    return all_success, truncated_log, None if all_success else "Error en la ejecución de scripts"


# Inicializar la aplicación Dash con carpeta assets
app = dash.Dash(
    __name__,
    title="Dashboard de Ventaja Latente",
    suppress_callback_exceptions=True,
    assets_folder="assets"
)

# Cargar datos
print("Cargando datos del dashboard...")
raw_dashboard_data = load_all_dashboard_data()

# Extraer DataFrames y errores
agent_summary_df, agent_summary_error = raw_dashboard_data['agent_summary']
agent_seed_metrics_df, agent_seed_metrics_error = raw_dashboard_data['agent_seed_metrics']
ivl_results_df, ivl_results_error = raw_dashboard_data['ivl_results']
latent_index_df, latent_index_error = raw_dashboard_data['latent_index']

# Convertir DataFrames a records JSON serializables para dcc.Store
dashboard_data_serializable = {
    'agent_summary': dataframe_to_records(agent_summary_df),
    'agent_seed_metrics': dataframe_to_records(agent_seed_metrics_df),
    'ivl_results': dataframe_to_records(ivl_results_df),
    'latent_index': dataframe_to_records(latent_index_df)
}

# Verificar archivos faltantes
missing_files = {
    'agent_summary.csv': agent_summary_error,
    'agent_seed_metrics.csv': agent_seed_metrics_error,
    'ivl_results.csv': ivl_results_error,
    'latent_index.csv': latent_index_error
}

# Filtrar solo archivos faltantes
missing_files = {k: v for k, v in missing_files.items() if v}

# ---------------------------------------------------------------------------
# Detectar columnas disponibles — compatible con formato antiguo y nuevo IS/OOS
# ---------------------------------------------------------------------------
if agent_summary_df is not None and not agent_summary_df.empty:
    AGENT_COL = get_first_existing_column(
        agent_summary_df,
        ["agent_name", "agent", "Agent"],
    )
    RETURN_COL = get_first_existing_column(
        agent_summary_df,
        ["mean_return_oos", "mean_return", "mean_return_is",
         "return_mean", "total_return_mean", "mean_total_return"],
    )
    STD_RETURN_COL = get_first_existing_column(
        agent_summary_df,
        ["std_return_oos", "seed_std_return_oos", "std_return", "seed_std_return",
         "std_return_is", "seed_std_return_is"],
        required=False,
    )
    TRADES_COL = get_first_existing_column(
        agent_summary_df,
        ["mean_n_trades", "mean_trades", "n_trades_mean", "num_trades_mean", "trades_mean"],
        required=False,
    )
    EQUITY_COL = get_first_existing_column(
        agent_summary_df,
        ["mean_final_equity_oos", "mean_final_equity", "final_equity_mean",
         "mean_equity_oos", "mean_equity"],
        required=False,
    )
else:
    AGENT_COL = "agent_name"
    RETURN_COL = "mean_return"
    STD_RETURN_COL = None
    TRADES_COL = None
    EQUITY_COL = None

# ---------------------------------------------------------------------------
# Calcular KPIs si hay datos
# ---------------------------------------------------------------------------
kpi_data = {}
if agent_summary_df is not None and not agent_summary_df.empty:
    kpi_data['num_agents'] = len(agent_summary_df)
    kpi_data['best_agent'] = agent_summary_df.loc[
        agent_summary_df[RETURN_COL].idxmax(), AGENT_COL
    ]
    kpi_data['best_return'] = agent_summary_df[RETURN_COL].max()
else:
    kpi_data['num_agents'] = 0
    kpi_data['best_agent'] = 'N/A'
    kpi_data['best_return'] = 0.0

if agent_seed_metrics_df is not None and not agent_seed_metrics_df.empty:
    kpi_data['num_seeds'] = agent_seed_metrics_df['seed'].nunique()
else:
    kpi_data['num_seeds'] = 0

if ivl_results_df is not None and not ivl_results_df.empty:
    kpi_data['ivl_value'] = ivl_results_df.iloc[0]['ivl']
    kpi_data['ivl_interpretation'] = ivl_results_df.iloc[0]['interpretation']
else:
    kpi_data['ivl_value'] = 0.0
    kpi_data['ivl_interpretation'] = 'N/A'


def create_missing_files_alert():
    """Crea una alerta con los archivos faltantes."""
    if not missing_files:
        return html.Div()

    alerts = []
    for filename, error_msg in missing_files.items():
        alerts.append(
            html.Div([
                html.Div(f"⚠️ {filename}", className="alert-title"),
                html.Div(error_msg, className="alert-message")
            ], className="alert alert-warning")
        )

    return html.Div(alerts)


def create_layout():
    """Crea el layout del dashboard."""
    return html.Div([
        dcc.Store(id='dashboard-data', data=dashboard_data_serializable),

        # Header
        html.Div([
            html.H1("Dashboard de Ventaja Latente", className="header-title"),
            html.Div(
                "Comparación exploratoria de agentes financieros con y sin representaciones latentes",
                className="subtitle"
            )
        ], className="header"),

        # Control de Experimento
        html.Div([
            html.H2("Control de Experimento", className="section-title"),

            html.Div([
                html.P(
                    "⚠️ La generación de datos puede tardar varios minutos. "
                    "El botón se deshabilitará durante la ejecución.",
                    className="control-warning"
                ),

                html.Div([
                    html.Button(
                        "▶ Generar datos del dashboard",
                        id="generate-btn",
                        n_clicks=0,
                        className="control-button control-button-primary"
                    ),
                    html.Button(
                        "🔄 Recargar datos",
                        id="reload-btn",
                        n_clicks=0,
                        className="control-button control-button-secondary"
                    )
                ], className="control-buttons"),

                html.Div(
                    id="run-status-text",
                    className="run-status",
                    children="Listo",
                    **{"data-status": "ready"}
                ),

                html.Div(id="run-result-message", className="run-result"),

                html.Div(id="run-log", className="run-log")
            ], className="control-content")
        ], className="section control-section"),

        # Alerta de archivos faltantes
        create_missing_files_alert(),

        # Contenedor principal
        html.Div([

            # KPI Cards
            html.Div([
                html.H2("Resumen", className="section-title"),
                html.Div([
                    create_kpi_card(
                        "Número de Agentes",
                        str(kpi_data['num_agents'])
                    ),
                    create_kpi_card(
                        "Número de Semillas",
                        str(kpi_data['num_seeds'])
                    ),
                    create_kpi_card(
                        "Mejor Agente",
                        kpi_data['best_agent'],
                        f"Retorno: {kpi_data['best_return']:.4f}"
                    ),
                    create_kpi_card(
                        "IVL Principal",
                        f"{kpi_data['ivl_value']:+.4f}",
                        kpi_data['ivl_interpretation'].replace('_', ' ').title()
                    )
                ], className="kpi-grid")
            ], className="section"),

            # IVL Section
            html.Div([
                html.H2("Índice de Ventaja Latente (IVL)", className="section-title"),

                # IVL Card grande
                create_ivl_card(
                    kpi_data['ivl_value'],
                    kpi_data['ivl_interpretation']
                ),

                # Delta cards
                html.Div([
                    html.H3("Componentes del IVL", style={'margin-bottom': '15px'}),
                    html.Div(id="ivl-delta-cards")
                ], className="delta-grid"),

                # Gráfico de deltas
                html.Div(id="ivl-delta-chart", className="graph-container"),

                # Interpretación
                html.Div([
                    html.H3("Interpretación", style={'margin-bottom': '10px'}),
                    html.P(
                        "El IVL indica ventaja agregada bajo las métricas y pesos definidos; "
                        "no prueba superioridad general.",
                        style={'color': '#666', 'font-style': 'italic'}
                    )
                ], style={'margin-top': '20px'})

            ], className="section"),

            # Comparación de Agentes
            html.Div([
                html.H2("Comparación de Agentes", className="section-title"),

                html.Div(id="agent-comparison-chart", className="graph-container"),

                html.Div(id="agent-summary-table", className="table-container")

            ], className="section"),

            # Métricas por Semilla
            html.Div([
                html.H2("Métricas por Semilla", className="section-title"),

                html.Div(id="seed-boxplot-chart", className="graph-container"),

                html.Div(id="seed-scatter-chart", className="graph-container")

            ], className="section"),

            # Espacio Latente Auxiliar
            html.Div([
                html.H2("Espacio Latente (Análisis Auxiliar)", className="section-title"),

                html.Div([
                    html.P(
                        "⚠️ Análisis auxiliar del espacio latente; no corresponde al IVL principal.",
                        className="auxiliary-note"
                    )
                ]),

                html.Div(id="price-vs-latent-chart", className="graph-container"),

                html.Div(id="latent-indices-chart", className="graph-container"),

                html.Div(id="pca-vs-return-chart", className="graph-container")

            ], className="section"),

            # Notas Metodológicas
            html.Div([
                html.H2("Notas Metodológicas", className="section-title"),
                html.Div([
                    html.H3("⚠️ Importante"),
                    html.Ul([
                        html.Li("Este dashboard es exploratorio y no prueba superioridad estadística."),
                        html.Li("Los resultados requieren validación con datos reales de mercado."),
                        html.Li("Es necesario implementar splits temporales In-Sample / Out-of-Sample apropiados."),
                        html.Li("Se requieren métricas financieras adicionales para una evaluación completa."),
                        html.Li("El IVL es una métrica agregada que no sustituye el análisis individual de cada componente."),
                        html.Li("La correlación observada no implica causalidad."),
                        html.Li("Resultados pasados no garantizan resultados futuros.")
                    ])
                ], className="notes")
            ], className="section"),

            # Footer
            html.Div([
                html.P("Dashboard de Ventaja Latente - Generado automáticamente")
            ], className="footer")

        ], className="dashboard-container")
    ])


# Crear layout
app.layout = create_layout()


# Callback para generar/recargar datos
@app.callback(
    Output('dashboard-data', 'data'),
    Output('run-result-message', 'children'),
    Output('run-log', 'children'),
    Output('run-status-text', 'children'),
    Output('run-status-text', 'data-status'),
    Output('generate-btn', 'disabled'),
    Input('generate-btn', 'n_clicks'),
    Input('reload-btn', 'n_clicks'),
    State('dashboard-data', 'data'),
    prevent_initial_call=True
)
def update_dashboard_data(generate_clicks, reload_clicks, current_data):
    """
    Genera o recarga los datos del dashboard.

    Args:
        generate_clicks: Número de clicks en el botón de generar
        reload_clicks: Número de clicks en el botón de recargar
        current_data: Datos actuales del dashboard

    Returns:
        tuple: (dashboard_data, result_message, log, status_text, data_status, generate_disabled)
    """
    ctx = dash.callback_context

    if not ctx.triggered:
        return current_data, html.Div(), html.Div(), "Listo", 'ready', False

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Caso 1: Generar datos (ejecutar scripts)
    if button_id == 'generate-btn':
        # Ejecutar scripts
        success, log, error = run_dashboard_data_scripts()

        if success:
            # Recargar datos
            try:
                raw_data = load_all_dashboard_data()
                new_data = {
                    'agent_summary': dataframe_to_records(raw_data['agent_summary'][0]),
                    'agent_seed_metrics': dataframe_to_records(raw_data['agent_seed_metrics'][0]),
                    'ivl_results': dataframe_to_records(raw_data['ivl_results'][0]),
                    'latent_index': dataframe_to_records(raw_data['latent_index'][0])
                }

                return (
                    new_data,
                    html.Div("✅ Datos generados y cargados correctamente", className="alert alert-success"),
                    html.Div(log, className="run-log"),
                    "Completado",
                    'completed',
                    False  # Habilitar botón
                )
            except Exception as e:
                return (
                    current_data,  # Mantener datos anteriores
                    html.Div(f"⚠️ Scripts completados pero error al cargar datos: {str(e)}", className="alert alert-warning"),
                    html.Div(log, className="run-log"),
                    "Error al cargar datos",
                    'error',
                    False
                )
        else:
            return (
                current_data,  # Mantener datos anteriores
                html.Div(f"❌ {error}", className="alert alert-danger"),
                html.Div(log, className="run-log"),
                "Error",
                'error',
                False
            )

    # Caso 2: Recargar datos (solo CSVs existentes)
    elif button_id == 'reload-btn':
        try:
            raw_data = load_all_dashboard_data()
            new_data = {
                'agent_summary': dataframe_to_records(raw_data['agent_summary'][0]),
                'agent_seed_metrics': dataframe_to_records(raw_data['agent_seed_metrics'][0]),
                'ivl_results': dataframe_to_records(raw_data['ivl_results'][0]),
                'latent_index': dataframe_to_records(raw_data['latent_index'][0])
            }

            return (
                new_data,
                html.Div("✅ Datos recargados correctamente", className="alert alert-success"),
                html.Div("Datos recargados desde CSVs existentes", className="run-log"),
                "Recargado",
                'completed',
                False
            )
        except Exception as e:
            return (
                current_data,
                html.Div(f"❌ Error al recargar datos: {str(e)}", className="alert alert-danger"),
                html.Div(str(e), className="run-log"),
                "Error",
                'error',
                False
            )

    return current_data, html.Div(), html.Div(), "Listo", 'ready', False


# Callbacks para actualizar gráficos y componentes
@app.callback(
    dash.dependencies.Output('ivl-delta-cards', 'children'),
    dash.dependencies.Output('ivl-delta-chart', 'children'),
    dash.dependencies.Input('dashboard-data', 'data')
)
def update_ivl_section(data):
    """Actualiza la sección del IVL."""
    try:
        ivl_records = data.get('ivl_results')
        ivl_df = records_to_dataframe(ivl_records)

        if ivl_df is None or ivl_df.empty:
            return html.Div("No hay datos de IVL disponibles"), html.Div("No hay datos disponibles")

        row = ivl_df.iloc[0]

        # Crear delta cards
        delta_cards = html.Div([
            create_delta_card("Delta Sharpe", row['delta_sharpe'], is_better_positive=True),
            create_delta_card("Delta MDD", row['delta_mdd'], is_better_positive=False),
            create_delta_card("Delta Seed Std", row['delta_seed_std'], is_better_positive=False),
            create_delta_card("Delta IS/OOS Gap", row['delta_is_oos_gap'], is_better_positive=False)
        ], className="delta-grid")

        # Crear gráfico de deltas
        delta_chart = dcc.Graph(figure=create_ivl_delta_bar(ivl_df))

        return delta_cards, delta_chart
    except Exception as e:
        return html.Div(f"Error al cargar IVL: {str(e)}"), html.Div("Error al cargar datos")


@app.callback(
    dash.dependencies.Output('agent-comparison-chart', 'children'),
    dash.dependencies.Output('agent-summary-table', 'children'),
    dash.dependencies.Input('dashboard-data', 'data')
)
def update_agent_comparison(data):
    """Actualiza la comparación de agentes."""
    try:
        summary_records = data.get('agent_summary')
        summary_df = records_to_dataframe(summary_records)

        if summary_df is None or summary_df.empty:
            return html.Div("No hay datos de agentes disponibles"), html.Div("No hay datos disponibles")

        # Crear gráfico
        chart = dcc.Graph(figure=create_agent_return_bar(summary_df))

        # Crear tabla
        table = dash_table.DataTable(
            data=summary_records if summary_records else [],
            columns=[{"name": i, "id": i} for i in summary_df.columns] if summary_df is not None else [],
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': 'rgb(102, 126, 234)',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_cell={
                'textAlign': 'center',
                'padding': '10px'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )

        return chart, table
    except Exception as e:
        return html.Div(f"Error al cargar comparación: {str(e)}"), html.Div("Error al cargar datos")


@app.callback(
    dash.dependencies.Output('seed-boxplot-chart', 'children'),
    dash.dependencies.Output('seed-scatter-chart', 'children'),
    dash.dependencies.Input('dashboard-data', 'data')
)
def update_seed_metrics(data):
    """Actualiza las métricas por semilla."""
    try:
        seed_records = data.get('agent_seed_metrics')
        seed_df = records_to_dataframe(seed_records)

        if seed_df is None or seed_df.empty:
            return html.Div("No hay datos por semilla disponibles"), html.Div("No hay datos disponibles")

        # Crear gráficos
        boxplot = dcc.Graph(figure=create_seed_return_boxplot(seed_df))
        scatter = dcc.Graph(figure=create_seed_return_scatter(seed_df))

        return boxplot, scatter
    except Exception as e:
        return html.Div(f"Error al cargar métricas por semilla: {str(e)}"), html.Div("Error al cargar datos")


@app.callback(
    dash.dependencies.Output('price-vs-latent-chart', 'children'),
    dash.dependencies.Output('latent-indices-chart', 'children'),
    dash.dependencies.Output('pca-vs-return-chart', 'children'),
    dash.dependencies.Input('dashboard-data', 'data')
)
def update_latent_space(data):
    """Actualiza la sección del espacio latente."""
    try:
        latent_records = data.get('latent_index')
        latent_df = records_to_dataframe(latent_records)

        if latent_df is None or latent_df.empty:
            return (
                html.Div("No hay datos de índice latente disponibles"),
                html.Div("No hay datos disponibles"),
                html.Div("No hay datos disponibles")
            )

        # Crear gráficos
        price_chart = dcc.Graph(figure=create_price_vs_latent_plot(latent_df))
        indices_chart = dcc.Graph(figure=create_latent_indices_plot(latent_df))
        pca_chart = dcc.Graph(figure=create_pca_vs_next_return_scatter(latent_df))

        return price_chart, indices_chart, pca_chart
    except Exception as e:
        return (
            html.Div(f"Error al cargar espacio latente: {str(e)}"),
            html.Div("Error al cargar datos"),
            html.Div("Error al cargar datos")
        )


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Dashboard de Ventaja Latente")
    print("="*60)
    print(f"\nAbriendo dashboard en: http://127.0.0.1:8050")
    print("\nPresiona Ctrl+C para detener el servidor\n")

    if missing_files:
        print("⚠️ Archivos faltantes:")
        for filename, error in missing_files.items():
            print(f"  - {filename}")
            print(f"    {error}")
        print()

    app.run(debug=True, host='127.0.0.1', port=8050)
