"""
Loader de datos para el dashboard.

Este módulo proporciona funciones para cargar los CSVs generados por los
experimentos y manejar errores cuando faltan archivos.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple


def get_results_path() -> Path:
    """Obtiene la ruta al directorio de resultados."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "results"


def load_agent_summary() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Carga el resumen de agentes.

    Returns:
        Tuple con (DataFrame, error_message). Si hay error, DataFrame es None.
    """
    csv_path = get_results_path() / "agent_summary.csv"

    if not csv_path.exists():
        error_msg = (
            f"No se encontró {csv_path}\n"
            f"Ejecuta: python examples/compare_agents_multiseed_experiment.py"
        )
        return None, error_msg

    try:
        df = pd.read_csv(csv_path)
        return df, None
    except Exception as e:
        return None, f"Error al cargar agent_summary.csv: {str(e)}"


def load_agent_seed_metrics() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Carga las métricas por semilla de agentes.

    Returns:
        Tuple con (DataFrame, error_message). Si hay error, DataFrame es None.
    """
    csv_path = get_results_path() / "agent_seed_metrics.csv"

    if not csv_path.exists():
        error_msg = (
            f"No se encontró {csv_path}\n"
            f"Ejecuta: python examples/compare_agents_multiseed_experiment.py"
        )
        return None, error_msg

    try:
        df = pd.read_csv(csv_path)
        return df, None
    except Exception as e:
        return None, f"Error al cargar agent_seed_metrics.csv: {str(e)}"


def load_ivl_results() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Carga los resultados del IVL.

    Returns:
        Tuple con (DataFrame, error_message). Si hay error, DataFrame es None.
    """
    csv_path = get_results_path() / "ivl_results.csv"

    if not csv_path.exists():
        error_msg = (
            f"No se encontró {csv_path}\n"
            f"Ejecuta: python examples/latent_advantage_example.py "
            f"o el futuro exportador de resultados IVL"
        )
        return None, error_msg

    try:
        df = pd.read_csv(csv_path)
        return df, None
    except Exception as e:
        return None, f"Error al cargar ivl_results.csv: {str(e)}"


def load_latent_index() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Carga el índice latente.

    Returns:
        Tuple con (DataFrame, error_message). Si hay error, DataFrame es None.
    """
    csv_path = get_results_path() / "latent_index.csv"

    if not csv_path.exists():
        error_msg = (
            f"No se encontró {csv_path}\n"
            f"Ejecuta: python examples/latent_index_example.py"
        )
        return None, error_msg

    try:
        df = pd.read_csv(csv_path)
        return df, None
    except Exception as e:
        return None, f"Error al cargar latent_index.csv: {str(e)}"


def load_all_dashboard_data() -> Dict[str, Tuple[Optional[pd.DataFrame], Optional[str]]]:
    """
    Carga todos los datos necesarios para el dashboard.

    Returns:
        Diccionario con nombre del dataset -> (DataFrame, error_message)
    """
    return {
        "agent_summary": load_agent_summary(),
        "agent_seed_metrics": load_agent_seed_metrics(),
        "ivl_results": load_ivl_results(),
        "latent_index": load_latent_index()
    }


def check_missing_files() -> Dict[str, str]:
    """
    Verifica qué archivos faltan y devuelve mensajes de error.

    Returns:
        Diccionario con nombre del archivo -> mensaje de error (vacío si existe)
    """
    missing_files = {}

    all_loaders = {
        "agent_summary.csv": load_agent_summary,
        "agent_seed_metrics.csv": load_agent_seed_metrics,
        "ivl_results.csv": load_ivl_results,
        "latent_index.csv": load_latent_index
    }

    for filename, loader in all_loaders.items():
        df, error = loader()
        if error is not None:
            missing_files[filename] = error
        else:
            missing_files[filename] = ""

    return missing_files