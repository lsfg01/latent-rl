"""
Generación de gráficos Plotly para el dashboard.

Este módulo proporciona funciones para crear todos los gráficos
interactivos del dashboard usando Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional, Tuple


def create_empty_figure(message: str = "No hay datos disponibles") -> go.Figure:
    """
    Crea una figura vacía con un mensaje centrado.

    Args:
        message: Mensaje a mostrar en la figura

    Returns:
        Figura Plotly vacía con anotación
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="gray")
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> Tuple[bool, str]:
    """
    Valida que un DataFrame tenga las columnas requeridas.

    Args:
        df: DataFrame a validar
        required_columns: Lista de columnas requeridas

    Returns:
        Tuple con (es_valido, mensaje_error)
    """
    if df is None:
        return False, "DataFrame es None"

    if df.empty:
        return False, "DataFrame está vacío"

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Faltan columnas: {missing_columns}"

    return True, ""


def create_agent_return_bar(
    df: pd.DataFrame,
    title: str = "Retorno Medio por Agente"
) -> go.Figure:
    """
    Crea un gráfico de barras con el retorno medio por agente.

    Args:
        df: DataFrame con columnas agent_name, mean_return, std_return
        title: Título del gráfico

    Returns:
        Figura Plotly con el gráfico de barras
    """
    is_valid, error_msg = validate_dataframe(df, ['agent_name', 'mean_return', 'std_return'])
    if not is_valid:
        return create_empty_figure(f"No hay datos disponibles: {error_msg}")

    fig = go.Figure()

    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            name=row['agent_name'],
            x=[row['mean_return']],
            y=[row['agent_name']],
            orientation='h',
            error_x=dict(
                type='data',
                array=[row['std_return']],
                visible=True
            ),
            text=[f"{row['mean_return']:.4f}"],
            textposition='outside'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Retorno Medio",
        yaxis_title="Agente",
        barmode='group',
        height=400,
        margin=dict(l=150, r=50, t=50, b=50)
    )

    return fig


def create_seed_return_boxplot(
    df: pd.DataFrame,
    title: str = "Distribución de Retornos por Semilla"
) -> go.Figure:
    """
    Crea un boxplot de retornos por semilla para cada agente.

    Args:
        df: DataFrame con columnas agent_name, seed, total_return
        title: Título del gráfico

    Returns:
        Figura Plotly con el boxplot
    """
    is_valid, error_msg = validate_dataframe(df, ['agent_name', 'seed', 'total_return'])
    if not is_valid:
        return create_empty_figure(f"No hay datos disponibles: {error_msg}")

    fig = go.Figure()

    for agent in df['agent_name'].unique():
        agent_data = df[df['agent_name'] == agent]['total_return']
        fig.add_trace(go.Box(
            y=agent_data,
            name=agent,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))

    fig.update_layout(
        title=title,
        yaxis_title="Retorno Total",
        xaxis_title="Agente",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


def create_seed_return_scatter(
    df: pd.DataFrame,
    title: str = "Retorno Total por Semilla"
) -> go.Figure:
    """
    Crea un scatter plot de retornos por semilla.

    Args:
        df: DataFrame con columnas agent_name, seed, total_return
        title: Título del gráfico

    Returns:
        Figura Plotly con el scatter plot
    """
    is_valid, error_msg = validate_dataframe(df, ['agent_name', 'seed', 'total_return'])
    if not is_valid:
        return create_empty_figure(f"No hay datos disponibles: {error_msg}")

    fig = go.Figure()

    for agent in df['agent_name'].unique():
        agent_data = df[df['agent_name'] == agent]
        fig.add_trace(go.Scatter(
            x=agent_data['seed'],
            y=agent_data['total_return'],
            mode='markers+lines',
            name=agent,
            marker=dict(size=8),
            line=dict(width=1)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Semilla",
        yaxis_title="Retorno Total",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified'
    )

    return fig


def create_ivl_delta_bar(
    df: pd.DataFrame,
    title: str = "Componentes del IVL"
) -> go.Figure:
    """
    Crea un gráfico de barras con los componentes del IVL.

    Args:
        df: DataFrame con columnas delta_sharpe, delta_mdd, delta_seed_std, delta_is_oos_gap
        title: Título del gráfico

    Returns:
        Figura Plotly con el gráfico de barras
    """
    is_valid, error_msg = validate_dataframe(df, ['delta_sharpe', 'delta_mdd', 'delta_seed_std', 'delta_is_oos_gap'])
    if not is_valid:
        return create_empty_figure(f"No hay datos disponibles: {error_msg}")

    # Usar la primera fila (asumimos una comparación)
    row = df.iloc[0]

    deltas = {
        'Delta Sharpe': row['delta_sharpe'],
        'Delta MDD': row['delta_mdd'],
        'Delta Seed Std': row['delta_seed_std'],
        'Delta IS/OOS Gap': row['delta_is_oos_gap']
    }

    # Determinar colores (verde para valores positivos en Sharpe, negativos en otros)
    colors = []
    for name, value in deltas.items():
        if name == 'Delta Sharpe':
            colors.append('green' if value > 0 else 'red')
        else:
            colors.append('green' if value < 0 else 'red')

    fig = go.Figure(data=[
        go.Bar(
            x=list(deltas.keys()),
            y=list(deltas.values()),
            marker_color=colors,
            text=[f"{v:+.4f}" for v in deltas.values()],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title=title,
        yaxis_title="Valor del Delta",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Añadir línea en cero
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    return fig


def create_price_vs_latent_plot(
    df: pd.DataFrame,
    title: str = "Precio Close vs Índice PCA"
) -> go.Figure:
    """
    Crea un gráfico con precio close e índice PCA en dos ejes.

    Args:
        df: DataFrame con columnas sample, close, pca_index
        title: Título del gráfico

    Returns:
        Figura Plotly con el gráfico de dos ejes
    """
    is_valid, error_msg = validate_dataframe(df, ['sample', 'close', 'pca_index'])
    if not is_valid:
        return create_empty_figure(f"No hay datos disponibles: {error_msg}")

    fig = go.Figure()

    # Eje izquierdo: precio close
    fig.add_trace(go.Scatter(
        x=df['sample'],
        y=df['close'],
        name='Close',
        line=dict(color='blue', width=1.5),
        yaxis='y'
    ))

    # Eje derecho: pca_index
    fig.add_trace(go.Scatter(
        x=df['sample'],
        y=df['pca_index'],
        name='PCA Index',
        line=dict(color='orange', width=1.5),
        yaxis='y2'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Muestra',
        yaxis=dict(
            title=dict(text='Precio Close', font=dict(color='blue')),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title=dict(text='Índice PCA', font=dict(color='orange')),
            tickfont=dict(color='orange'),
            overlaying='y',
            side='right'
        ),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(x=0.01, y=0.99)
    )

    return fig


def create_latent_indices_plot(
    df: pd.DataFrame,
    title: str = "Comparación de Índices Latentes (Normalizados)"
) -> go.Figure:
    """
    Crea un gráfico comparativo de índices latentes normalizados.

    Args:
        df: DataFrame con columnas sample, norm_index, first_component_index, pca_index
        title: Título del gráfico

    Returns:
        Figura Plotly con el gráfico comparativo
    """
    is_valid, error_msg = validate_dataframe(df, ['sample', 'norm_index', 'first_component_index', 'pca_index'])
    if not is_valid:
        return create_empty_figure(f"No hay datos disponibles: {error_msg}")

    # Normalizar índices
    norm_index_norm = (df['norm_index'] - df['norm_index'].mean()) / df['norm_index'].std()
    first_component_norm = (df['first_component_index'] - df['first_component_index'].mean()) / df['first_component_index'].std()
    pca_index_norm = (df['pca_index'] - df['pca_index'].mean()) / df['pca_index'].std()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['sample'],
        y=norm_index_norm,
        name='Norm Index (normalizado)',
        line=dict(color='blue', width=1.5),
        opacity=0.7
    ))

    fig.add_trace(go.Scatter(
        x=df['sample'],
        y=first_component_norm,
        name='First Component Index (normalizado)',
        line=dict(color='orange', width=1.5),
        opacity=0.7
    ))

    fig.add_trace(go.Scatter(
        x=df['sample'],
        y=pca_index_norm,
        name='PCA Index (normalizado)',
        line=dict(color='green', width=1.5),
        opacity=0.7
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Muestra',
        yaxis_title='Valor Normalizado',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(x=0.01, y=0.99)
    )

    return fig


def create_pca_vs_next_return_scatter(
    df: pd.DataFrame,
    title: str = "Índice PCA vs Retorno Siguiente"
) -> go.Figure:
    """
    Crea un scatter plot entre índice PCA y retorno siguiente.

    Args:
        df: DataFrame con columnas pca_index, next_return
        title: Título del gráfico

    Returns:
        Figura Plotly con el scatter plot
    """
    is_valid, error_msg = validate_dataframe(df, ['pca_index', 'next_return'])
    if not is_valid:
        return create_empty_figure(f"No hay datos disponibles: {error_msg}")

    # Eliminar NaN
    df_clean = df[['pca_index', 'next_return']].dropna()

    if df_clean.empty:
        return create_empty_figure("No hay datos válidos después de eliminar NaN")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_clean['pca_index'],
        y=df_clean['next_return'],
        mode='markers',
        marker=dict(
            size=8,
            color=df_clean['next_return'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='Retorno Siguiente')
        ),
        name='PCA vs Retorno'
    ))

    # Añadir línea de tendencia si hay suficientes datos
    if len(df_clean) > 1:
        z = np.polyfit(df_clean['pca_index'], df_clean['next_return'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df_clean['pca_index'].min(), df_clean['pca_index'].max(), 100)
        y_trend = p(x_trend)

        fig.add_trace(go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            name='Tendencia',
            line=dict(color='red', dash='dash')
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Índice PCA',
        yaxis_title='Retorno Siguiente',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Añadir línea en cero
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    return fig