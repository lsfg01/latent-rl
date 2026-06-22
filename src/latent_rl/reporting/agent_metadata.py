"""Nombres y descripciones publicas de los agentes experimentales."""

from __future__ import annotations

from typing import Any


AGENT_DISPLAY = {
    "A": "A · DirectDQN",
    "B": "B · LatentDQN-FT",
    "C": "C · LatentDQN-IS-Frozen",
    "D": "D · LatentDQN-Offline-Frozen",
    "RandomAgent": "Random",
    "BuyAndHoldAgent": "Buy & Hold",
    # Compatibilidad con resultados anteriores a los brazos A/B/C/D.
    "DQNAgent": "DirectDQN",
    "LatentDQNAgent": "LatentDQN",
}

AGENT_DESCRIPTIONS = {
    "A": (
        "DQN directo entrenado sobre ventanas de observaciones financieras "
        "normalizadas."
    ),
    "B": (
        "LatentDQN con encoder inicializado aleatoriamente y entrenado "
        "conjuntamente con la cabeza Q durante RL."
    ),
    "C": (
        "LatentDQN con encoder ligero preentrenado en el tramo in-sample "
        "mediante reconstrucción/forecast y congelado durante RL."
    ),
    "D": (
        "LatentDQN con encoder pesado preentrenado offline sobre un universo "
        "externo de activos y congelado durante RL."
    ),
}

AGENT_ORDER = ["A", "B", "C", "D", "BuyAndHoldAgent", "RandomAgent"]

AGENT_COLORS = {
    "A": "#3b6fd4",
    "B": "#2f855a",
    "C": "#b7791f",
    "D": "#805ad5",
    "BuyAndHoldAgent": "#4a5568",
    "RandomAgent": "#a0aec0",
    "DQNAgent": "#3b6fd4",
    "LatentDQNAgent": "#2f855a",
}


def agent_display_name(agent: Any) -> str:
    """Devuelve el nombre explicativo sin alterar la clave persistida."""
    if agent is None:
        return ""
    return AGENT_DISPLAY.get(str(agent), str(agent))
