"""
Componentes de tarjetas KPI para el dashboard.
"""

import dash.html as html


def create_kpi_card(
    title: str,
    value: str,
    subtitle: str = None,
    className: str = "kpi-card"
) -> html.Div:
    """
    Crea una tarjeta KPI con titulo, valor y subtitulo opcional.

    Args:
        title: Etiqueta superior de la tarjeta
        value: Valor principal (numero o texto corto)
        subtitle: Texto secundario opcional
        className: Clase CSS de la tarjeta

    Returns:
        html.Div con la tarjeta KPI
    """
    children = [
        html.Div(title, className="kpi-title"),
        html.Div(value, className="kpi-value"),
    ]
    if subtitle:
        children.append(html.Div(subtitle, className="kpi-subtitle"))

    return html.Div(children, className=className)


def create_ivl_kpi_card(ivl_value: float, interpretation: str) -> html.Div:
    """
    Tarjeta KPI especifica para el IVL — se integra en el grid de 4 KPIs.

    Aplica color semantico: verde (positivo), rojo (negativo), gris (neutral).
    """
    if ivl_value > 0.01:
        card_cls  = "kpi-card kpi-card-positive"
        val_cls   = "kpi-value kpi-value-positive"
        badge_cls = "ivl-interpretation ivl-interpretation-positive"
        badge_txt = "Latent Advantage"
    elif ivl_value < -0.01:
        card_cls  = "kpi-card kpi-card-negative"
        val_cls   = "kpi-value kpi-value-negative"
        badge_cls = "ivl-interpretation ivl-interpretation-negative"
        badge_txt = "Direct Advantage"
    else:
        card_cls  = "kpi-card kpi-card-neutral"
        val_cls   = "kpi-value kpi-value-neutral"
        badge_cls = "ivl-interpretation ivl-interpretation-neutral"
        badge_txt = "Neutral"

    return html.Div([
        html.Div("IVL Principal", className="kpi-title"),
        html.Div(f"{ivl_value:+.4f}", className=val_cls),
        html.Div(badge_txt, className=badge_cls),
    ], className=card_cls)


def create_ivl_card(ivl_value: float, interpretation: str) -> html.Div:
    """
    Card destacada del IVL para la seccion de detalle.

    Muestra el valor grande con color semantico y la interpretacion como badge.
    Compatible con el layout horizontal de la nueva seccion IVL.
    """
    if ivl_value > 0.01:
        val_cls   = "ivl-value ivl-value-positive"
        badge_cls = "ivl-interpretation ivl-interpretation-positive"
        icon      = "+"
    elif ivl_value < -0.01:
        val_cls   = "ivl-value ivl-value-negative"
        badge_cls = "ivl-interpretation ivl-interpretation-negative"
        icon      = "-"
    else:
        val_cls   = "ivl-value ivl-value-neutral"
        badge_cls = "ivl-interpretation ivl-interpretation-neutral"
        icon      = "~"

    label = interpretation.replace("_", " ").title()

    return html.Div([
        html.Div("Indice de Ventaja Latente", className="ivl-title"),
        html.Div([
            html.Span(f"{ivl_value:+.4f}", className=val_cls),
        ], className="ivl-value-container"),
        html.Div(label, className=badge_cls),
    ])


def create_delta_card(
    title: str,
    value: float,
    is_better_positive: bool = True,
) -> html.Div:
    """
    Crea un badge de delta del IVL.

    Args:
        title: Nombre del componente (ej: "Delta Sharpe")
        value: Valor numerico del delta
        is_better_positive: Si True, positivo = bueno (verde)

    Returns:
        html.Div con el badge de delta
    """
    is_good = value > 0 if is_better_positive else value < 0
    val_cls = "delta-value delta-value-good" if is_good else "delta-value delta-value-bad"
    sign    = "+" if value >= 0 else ""

    return html.Div([
        html.Div(title, className="delta-title"),
        html.Div(f"{sign}{value:.4f}", className=val_cls),
    ], className="delta-card")
