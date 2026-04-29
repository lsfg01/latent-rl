"""
Generador de Dashboard HTML para Índice Latente

Este script genera un dashboard HTML estático con visualizaciones del índice latente
calculado por LatentIndexCalculator.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import webbrowser
import warnings

warnings.filterwarnings('ignore')

# Configuración de matplotlib para mejor calidad
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


def load_data(csv_path: Path) -> pd.DataFrame:
    """Carga los datos del CSV de índice latente."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo {csv_path}\n"
            f"Primero ejecuta: python examples/latent_index_example.py"
        )
    return pd.read_csv(csv_path)


def calculate_kpis(df: pd.DataFrame) -> dict:
    """Calcula los KPIs del dashboard."""
    close = df['close']
    pca_index = df['pca_index']

    # Calcular next_return
    next_return = close.pct_change().shift(-1)

    # Calcular correlación entre pca_index y next_return
    correlation = pca_index.corr(next_return)

    kpis = {
        'num_samples': len(df),
        'close_initial': close.iloc[0],
        'close_final': close.iloc[-1],
        'total_return': (close.iloc[-1] / close.iloc[0] - 1) * 100,
        'pca_index_mean': pca_index.mean(),
        'pca_index_std': pca_index.std(),
        'correlation': correlation
    }

    return kpis


def generate_price_and_pca_chart(df: pd.DataFrame, output_path: Path):
    """Genera gráfico de precio e índice PCA."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Eje izquierdo: precio close
    color1 = '#1f77b4'
    ax1.set_xlabel('Muestra')
    ax1.set_ylabel('Precio Close', color=color1)
    ax1.plot(df['sample'], df['close'], color=color1, label='Close', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Eje derecho: pca_index
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'
    ax2.set_ylabel('Índice PCA', color=color2)
    ax2.plot(df['sample'], df['pca_index'], color=color2, label='PCA Index', linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Precio Close vs Índice PCA')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def generate_latent_indices_comparison(df: pd.DataFrame, output_path: Path):
    """Genera gráfico comparativo de índices latentes normalizados."""
    # Normalizar todos los índices
    norm_index_normalized = (df['norm_index'] - df['norm_index'].mean()) / df['norm_index'].std()
    first_component_normalized = (df['first_component_index'] - df['first_component_index'].mean()) / df['first_component_index'].std()
    pca_index_normalized = (df['pca_index'] - df['pca_index'].mean()) / df['pca_index'].std()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['sample'], norm_index_normalized, label='Norm Index (normalizado)',
            color='#1f77b4', linewidth=1.5, alpha=0.7)
    ax.plot(df['sample'], first_component_normalized, label='First Component Index (normalizado)',
            color='#ff7f0e', linewidth=1.5, alpha=0.7)
    ax.plot(df['sample'], pca_index_normalized, label='PCA Index (normalizado)',
            color='#2ca02c', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Muestra')
    ax.set_ylabel('Valor Normalizado')
    ax.set_title('Comparación de Índices Latentes (Normalizados)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def generate_pca_index_distribution(df: pd.DataFrame, output_path: Path):
    """Genera histograma del índice PCA."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df['pca_index'], bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Índice PCA')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución del Índice PCA')
    ax.grid(True, alpha=0.3, axis='y')

    # Añadir línea de media
    mean = df['pca_index'].mean()
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Media: {mean:.3f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def generate_pca_index_vs_next_return(df: pd.DataFrame, output_path: Path):
    """Genera scatter plot entre índice PCA y retorno siguiente."""
    close = df['close']
    pca_index = df['pca_index']

    # Calcular next_return
    next_return = close.pct_change().shift(-1)

    # Eliminar NaN
    valid_data = pd.DataFrame({
        'pca_index': pca_index,
        'next_return': next_return
    }).dropna()

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(valid_data['pca_index'], valid_data['next_return'],
                         c=valid_data['next_return'], cmap='RdYlGn', alpha=0.6, s=50)

    ax.set_xlabel('Índice PCA')
    ax.set_ylabel('Retorno Siguiente')
    ax.set_title('Índice PCA vs Retorno Siguiente')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Añadir línea de tendencia
    if len(valid_data) > 1:
        z = np.polyfit(valid_data['pca_index'], valid_data['next_return'], 1)
        p = np.poly1d(z)
        ax.plot(valid_data['pca_index'], p(valid_data['pca_index']),
                "r--", alpha=0.8, linewidth=2, label='Tendencia')
        ax.legend()

    plt.colorbar(scatter, ax=ax, label='Retorno Siguiente')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def generate_rolling_correlation(df: pd.DataFrame, output_path: Path, window: int = 20):
    """Genera gráfico de correlación rolling entre índice PCA y retorno siguiente."""
    close = df['close']
    pca_index = df['pca_index']

    # Calcular next_return
    next_return = close.pct_change().shift(-1)

    # Calcular correlación rolling
    rolling_corr = pca_index.rolling(window=window).corr(next_return)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['sample'], rolling_corr, color='#1f77b4', linewidth=1.5)
    ax.set_xlabel('Muestra')
    ax.set_ylabel('Correlación Rolling')
    ax.set_title(f'Correlación Rolling entre Índice PCA y Retorno Siguiente (ventana={window})')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def generate_html_dashboard(df: pd.DataFrame, kpis: dict, output_path: Path, assets_path: Path):
    """Genera el HTML del dashboard."""

    # Formatear KPIs
    kpis_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Número de Muestras</div>
            <div class="kpi-value">{kpis['num_samples']:,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Close Inicial</div>
            <div class="kpi-value">${kpis['close_initial']:.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Close Final</div>
            <div class="kpi-value">${kpis['close_final']:.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Retorno Total</div>
            <div class="kpi-value {kpis['total_return'] >= 0 and 'positive' or 'negative'}">
                {kpis['total_return']:+.2f}%
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Media PCA Index</div>
            <div class="kpi-value">{kpis['pca_index_mean']:.4f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Desv. Est. PCA Index</div>
            <div class="kpi-value">{kpis['pca_index_std']:.4f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Correlación PCA vs Retorno</div>
            <div class="kpi-value {kpis['correlation'] >= 0 and 'positive' or 'negative'}">
                {kpis['correlation']:+.4f}
            </div>
        </div>
    </div>
    """

    # Generar tabla de primeras 10 filas
    table_rows = ""
    for _, row in df.head(10).iterrows():
        table_rows += f"""
        <tr>
            <td>{int(row['sample'])}</td>
            <td>${row['close']:.2f}</td>
            <td>{row['norm_index']:.4f}</td>
            <td>{row['first_component_index']:.4f}</td>
            <td>{row['pca_index']:.4f}</td>
        </tr>
        """

    table_html = f"""
    <div class="table-container">
        <h3>Primeras 10 Filas de Datos</h3>
        <table>
            <thead>
                <tr>
                    <th>Muestra</th>
                    <th>Close</th>
                    <th>Norm Index</th>
                    <th>First Component</th>
                    <th>PCA Index</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    """

    # HTML completo
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard de Índice Latente</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}

        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
            font-style: italic;
        }}

        .content {{
            padding: 30px;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .kpi-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
        }}

        .kpi-label {{
            font-size: 0.9em;
            color: #555;
            margin-bottom: 8px;
            font-weight: 600;
        }}

        .kpi-value {{
            font-size: 1.8em;
            font-weight: 700;
            color: #333;
        }}

        .kpi-value.positive {{
            color: #2e7d32;
        }}

        .kpi-value.negative {{
            color: #c62828;
        }}

        .charts-section {{
            margin-bottom: 30px;
        }}

        .chart-container {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}

        .chart-container h3 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 8px;
        }}

        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}

        .table-container {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}

        .table-container h3 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 8px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 4px;
            overflow: hidden;
        }}

        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        th, td {{
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid #e0e0e0;
        }}

        th {{
            font-weight: 600;
            font-size: 0.9em;
        }}

        tbody tr:hover {{
            background: #f5f5f5;
        }}

        .notes {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin-top: 30px;
            border-radius: 4px;
        }}

        .notes h3 {{
            color: #856404;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}

        .notes ul {{
            margin-left: 20px;
            color: #856404;
        }}

        .notes li {{
            margin-bottom: 8px;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #e0e0e0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Dashboard de Índice Latente</h1>
            <div class="subtitle">Análisis Experimental de Representaciones Latentes</div>
        </div>

        <div class="content">
            {kpis_html}

            <div class="charts-section">
                <div class="chart-container">
                    <h3>Precio Close vs Índice PCA</h3>
                    <img src="assets/price_and_pca_index.png" alt="Precio Close vs Índice PCA">
                </div>

                <div class="chart-container">
                    <h3>Comparación de Índices Latentes (Normalizados)</h3>
                    <img src="assets/latent_indices_comparison.png" alt="Comparación de Índices Latentes">
                </div>

                <div class="chart-container">
                    <h3>Distribución del Índice PCA</h3>
                    <img src="assets/pca_index_distribution.png" alt="Distribución del Índice PCA">
                </div>

                <div class="chart-container">
                    <h3>Índice PCA vs Retorno Siguiente</h3>
                    <img src="assets/pca_index_vs_next_return.png" alt="Índice PCA vs Retorno Siguiente">
                </div>

                <div class="chart-container">
                    <h3>Correlación Rolling entre Índice PCA y Retorno Siguiente</h3>
                    <img src="assets/rolling_correlation.png" alt="Correlación Rolling">
                </div>
            </div>

            {table_html}

            <div class="notes">
                <h3>⚠️ Notas Importantes</h3>
                <ul>
                    <li>El índice latente presentado es <strong>experimental</strong> y requiere validación adicional.</li>
                    <li>La correlación observada <strong>no implica causalidad</strong> entre el índice latente y los retornos futuros.</li>
                    <li>Se requiere validación con datos reales y splits temporales apropiados antes de cualquier uso práctico.</li>
                    <li>Los resultados pueden variar significativamente dependiendo del período de tiempo y las condiciones del mercado.</li>
                </ul>
            </div>
        </div>

        <div class="footer">
            Generado automáticamente por LatentIndexCalculator Dashboard
        </div>
    </div>
</body>
</html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    """Función principal para generar el dashboard."""
    # Definir rutas
    project_root = Path(__file__).parent.parent
    csv_path = project_root / 'results' / 'latent_index_example.csv'
    dashboard_dir = project_root / 'reports' / 'dashboard'
    assets_dir = dashboard_dir / 'assets'
    html_path = dashboard_dir / 'latent_index_dashboard.html'

    try:
        # Cargar datos
        print("Cargando datos...")
        df = load_data(csv_path)

        # Calcular KPIs
        print("Calculando KPIs...")
        kpis = calculate_kpis(df)

        # Crear directorios
        print("Creando directorios...")
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)

        # Generar gráficos
        print("Generando gráficos...")
        generate_price_and_pca_chart(df, assets_dir / 'price_and_pca_index.png')
        print("  ✓ price_and_pca_index.png")

        generate_latent_indices_comparison(df, assets_dir / 'latent_indices_comparison.png')
        print("  ✓ latent_indices_comparison.png")

        generate_pca_index_distribution(df, assets_dir / 'pca_index_distribution.png')
        print("  ✓ pca_index_distribution.png")

        generate_pca_index_vs_next_return(df, assets_dir / 'pca_index_vs_next_return.png')
        print("  ✓ pca_index_vs_next_return.png")

        generate_rolling_correlation(df, assets_dir / 'rolling_correlation.png')
        print("  ✓ rolling_correlation.png")

        # Generar HTML
        print("Generando HTML...")
        generate_html_dashboard(df, kpis, html_path, assets_dir)

        # Abrir en navegador
        print(f"Abriendo dashboard en navegador...")
        webbrowser.open(f'file:///{html_path.as_posix()}')

        print(f"\n✓ Dashboard generado exitosamente!")
        print(f"  HTML: {html_path}")
        print(f"  Assets: {assets_dir}")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()