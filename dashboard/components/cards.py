"""
Componentes de tarjetas KPI para el dashboard.

Este módulo proporciona funciones para crear tarjetas de indicadores
clave de rendimiento (KPI) con estilos consistentes.
"""

import dash.html as html


def create_kpi_card(
    title: str,
    value: str,
    subtitle: str = None,
    className: str = "kpi-card"
) -> html.Div:
    """
    Crea una tarjeta KPI con título, valor y subtítulo opcional.

    Args:
        title: Título de la tarjeta
        value: Valor principal a mostrar
        subtitle: Subtítulo opcional
        className: Clase CSS adicional

    Returns:
        Componente html.Div con la tarjeta KPI
    """
    children = [
        html.Div(title, className="kpi-title"),
        html.Div(value, className="kpi-value")
    ]

    if subtitle:
        children.append(html.Div(subtitle, className="kpi-subtitle"))

    return html.Div(children, className=className)


def create_ivl_card(ivl_value: float, interpretation: str) -> html.Div:
    """
    Crea una tarjeta especial para mostrar el IVL.

    Args:
        ivl_value: Valor del IVL
        interpretation: Interpretación del IVL

    Returns:
        Componente html.Div con la tarjeta del IVL
    """
    # Determinar clase según el valor
    if ivl_value > 0:
        value_class = "ivl-value-positive"
        icon = "⬆️"
    elif ivl_value < 0:
        value_class = "ivl-value-negative"
        icon = "⬇️"
    else:
        value_class = "ivl-value-neutral"
        icon = "➡️"

    return html.Div([
        html.Div("Índice de Ventaja Latente (IVL)", className="ivl-title"),
        html.Div([
            html.Span(icon, className="ivl-icon"),
            html.Span(f"{ivl_value:+.4f}", className=f"ivl-value {value_class}")
        ], className="ivl-value-container"),
        html.Div(interpretation.replace("_", " ").title(), className="ivl-interpretation")
    ], className="ivl-card")


def create_delta_card(
    title: str,
    value: float,
    is_better_positive: bool = True
) -> html.Div:
    """
    Crea una tarjeta para mostrar un delta del IVL.

    Args:
        title: Título del delta (ej: "Delta Sharpe")
        value: Valor del delta
        is_better_positive: Si True, valores positivos son buenos (verde)

    Returns:
        Componente html.Div con la tarjeta del delta
    """
    # Determinar si el valor es "bueno"
    if is_better_positive:
        is_good = value > 0
    else:
        is_good = value < 0

    value_class = "delta-value-good" if is_good else "delta-value-bad"
    sign = "+" if value >= 0 else ""

    return html.Div([
        html.Div(title, className="delta-title"),
        html.Div(f"{sign}{value:.4f}", className=f"delta-value {value_class}")
    ], className="delta-card")