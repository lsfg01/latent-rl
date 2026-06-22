"""
Loader de datos para el dashboard.

Soporta dos formatos de resultados:
- Formato plano (legacy): ``results/agent_summary.csv``
- Formato multi-ticker:   ``results/{ticker}/agent_summary.csv``

La detección es automática: si existen subdirectorios con datos validos
(contienen agent_summary.csv) se usa el formato multi-ticker; si solo
existe el archivo plano se usa el formato legacy.
"""

import json

import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_results_dir_override: Optional[Path] = None


def set_results_dir(path: str) -> None:
    """Sobreescribe el directorio de resultados (usado por la CLI del dashboard)."""
    global _results_dir_override
    _results_dir_override = Path(path).resolve()


def get_results_path() -> Path:
    """Obtiene la ruta al directorio de resultados."""
    if _results_dir_override is not None:
        return _results_dir_override
    project_root = Path(__file__).parent.parent.parent
    return project_root / "results"


# ---------------------------------------------------------------------------
# Deteccion de tickers
# ---------------------------------------------------------------------------

def _detect_tickers(results_dir: Optional[Path] = None) -> List[str]:
    """
    Detecta los tickers disponibles escaneando subdirectorios de results/.

    Un subdirectorio se considera un ticker valido si contiene
    ``agent_summary.csv``.

    Args:
        results_dir: Directorio base. Por defecto usa get_results_path().

    Returns:
        Lista de nombres de ticker ordenada alfabeticamente.
    """
    if results_dir is None:
        results_dir = get_results_path()
    tickers: List[str] = []
    if results_dir.exists():
        for subdir in sorted(results_dir.iterdir()):
            if subdir.is_dir() and (subdir / "agent_summary.csv").exists():
                tickers.append(subdir.name)
    return tickers


def get_available_tickers() -> List[str]:
    """
    Devuelve los tickers disponibles en el directorio de resultados.

    Returns:
        Lista de nombres de ticker. Vacia si solo existe el formato plano.
    """
    return _detect_tickers()


# ---------------------------------------------------------------------------
# Resolucion de directorio segun ticker
# ---------------------------------------------------------------------------

def _resolve_data_dir(ticker: Optional[str] = None) -> Path:
    """
    Resuelve el directorio de datos segun el ticker.

    - ``ticker=None``   → ``results/`` (formato plano)
    - ``ticker="SPY"``  → ``results/SPY/``
    """
    base = get_results_path()
    return base / ticker if ticker is not None else base


# ---------------------------------------------------------------------------
# Carga de archivos individuales
# ---------------------------------------------------------------------------

def load_agent_summary(
    ticker: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Carga el resumen de agentes.

    Args:
        ticker: Si se especifica, carga desde ``results/{ticker}/agent_summary.csv``.
                Si None, carga desde ``results/agent_summary.csv`` (formato plano).

    Returns:
        Tuple ``(DataFrame, error_message)``. DataFrame es None si hay error.
    """
    csv_path = _resolve_data_dir(ticker) / "agent_summary.csv"
    if not csv_path.exists():
        return None, (
            f"No se encontro {csv_path}\n"
            "Ejecuta: python examples/compare_agents_multiseed_experiment.py"
        )
    try:
        return pd.read_csv(csv_path), None
    except Exception as exc:
        return None, f"Error al cargar agent_summary.csv: {exc}"


def load_agent_seed_metrics(
    ticker: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Carga las metricas por semilla de agentes.

    Args:
        ticker: Si se especifica, carga desde ``results/{ticker}/agent_seed_metrics.csv``.

    Returns:
        Tuple ``(DataFrame, error_message)``. DataFrame es None si hay error.
    """
    csv_path = _resolve_data_dir(ticker) / "agent_seed_metrics.csv"
    if not csv_path.exists():
        return None, (
            f"No se encontro {csv_path}\n"
            "Ejecuta: python examples/compare_agents_multiseed_experiment.py"
        )
    try:
        return pd.read_csv(csv_path), None
    except Exception as exc:
        return None, f"Error al cargar agent_seed_metrics.csv: {exc}"


def load_ivl_results(
    ticker: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Carga los resultados del IVL.

    Args:
        ticker: Si se especifica, carga desde ``results/{ticker}/ivl_results.csv``.

    Returns:
        Tuple ``(DataFrame, error_message)``. DataFrame es None si hay error.
    """
    csv_path = _resolve_data_dir(ticker) / "ivl_results.csv"
    if not csv_path.exists():
        return None, (
            f"No se encontro {csv_path}\n"
            "Ejecuta: python -m scripts.utilities.compute_ivl"
        )
    try:
        return pd.read_csv(csv_path), None
    except Exception as exc:
        return None, f"Error al cargar ivl_results.csv: {exc}"


def load_latent_index() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Carga el indice latente (analisis auxiliar, no dependiente de ticker).

    Busca primero el nombre actual (``latent_index_data.csv``) y luego el
    nombre legacy (``latent_index.csv``).

    Returns:
        Tuple ``(DataFrame, error_message)``. DataFrame es None si hay error.
    """
    base = get_results_path()
    for fname in ("latent_index_data.csv", "latent_index.csv"):
        csv_path = base / fname
        if csv_path.exists():
            try:
                return pd.read_csv(csv_path), None
            except Exception as exc:
                return None, f"Error al cargar {fname}: {exc}"

    return None, (
        f"No se encontro latent_index_data.csv en {base}\n"
        "Ejecuta: python examples/latent_index_example.py"
    )


def load_ticker_comparison() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Carga el resumen cross-ticker generado por el pipeline multi-ticker.

    El archivo ``results/ticker_comparison.csv`` solo existe cuando el
    experimento se ejecuto con mas de un ticker.

    Returns:
        Tuple ``(DataFrame, error_message)``. DataFrame es None si no existe.
    """
    csv_path = get_results_path() / "ticker_comparison.csv"
    if not csv_path.exists():
        return None, (
            f"No se encontro {csv_path} "
            "(solo disponible en modo multi-ticker)"
        )
    try:
        return pd.read_csv(csv_path), None
    except Exception as exc:
        return None, f"Error al cargar ticker_comparison.csv: {exc}"


def load_validation_metrics(
    ticker: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Carga la trazabilidad de validacion interna cuando esta disponible."""
    csv_path = _resolve_data_dir(ticker) / "validation_metrics.csv"
    if not csv_path.exists():
        return None, f"No se encontro {csv_path} (archivo opcional)"
    try:
        return pd.read_csv(csv_path), None
    except Exception as exc:
        return None, f"Error al cargar validation_metrics.csv: {exc}"


def load_experiment_config() -> Tuple[Optional[dict[str, Any]], Optional[str]]:
    """Carga la configuracion estructurada de la run cuando existe."""
    json_path = get_results_path() / "experiment_config.json"
    if not json_path.exists():
        return None, f"No se encontro {json_path} (archivo opcional)"
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("la raiz JSON no es un objeto")
        return payload, None
    except Exception as exc:
        return None, f"Error al cargar experiment_config.json: {exc}"


# ---------------------------------------------------------------------------
# Carga agregada
# ---------------------------------------------------------------------------

def _effective_ticker(ticker: Optional[str]) -> Optional[str]:
    """
    Determina el ticker efectivo a usar:
    - Si se especifica explicitamente, lo usa tal cual.
    - Si es None y existen subdirectorios de tickers sin formato plano,
      selecciona el primero disponible.
    - Si es None y existe el formato plano, devuelve None (usa formato plano).
    """
    if ticker is not None:
        return ticker
    available = _detect_tickers()
    if available and not (get_results_path() / "agent_summary.csv").exists():
        return available[0]
    return None


def load_all_dashboard_data(
    ticker: Optional[str] = None,
) -> Dict[str, Tuple[Any, Optional[str]]]:
    """
    Carga todos los datos necesarios para el dashboard.

    Deteccion automatica de formato:

    - Si ``ticker`` se especifica → carga desde ``results/{ticker}/``.
    - Si ``ticker=None`` y existen subdirectorios de tickers pero no existe
      el formato plano → selecciona el primer ticker disponible.
    - Si ``ticker=None`` y existe el formato plano → carga desde ``results/``.

    Args:
        ticker: Ticker a cargar. None activa la deteccion automatica.

    Returns:
        Diccionario ``{dataset_name: (DataFrame, error_message)}``.
        Claves: ``"agent_summary"``, ``"agent_seed_metrics"``,
        ``"ivl_results"``, ``"validation_metrics"``, ``"experiment_config"``,
        ``"latent_index"`` y ``"ticker_comparison"``.
    """
    eff = _effective_ticker(ticker)
    return {
        "agent_summary":      load_agent_summary(eff),
        "agent_seed_metrics": load_agent_seed_metrics(eff),
        "ivl_results":        load_ivl_results(eff),
        "validation_metrics": load_validation_metrics(eff),
        "experiment_config":  load_experiment_config(),
        "latent_index":       load_latent_index(),
        "ticker_comparison":  load_ticker_comparison(),
    }


def check_missing_files(
    ticker: Optional[str] = None,
) -> Dict[str, str]:
    """
    Verifica que archivos faltan y devuelve mensajes de error.

    Args:
        ticker: Ticker a verificar. None usa deteccion automatica.

    Returns:
        Diccionario ``{filename: error_message}`` (cadena vacia si existe).
    """
    eff = _effective_ticker(ticker)
    missing: Dict[str, str] = {}
    for fname, result in [
        ("agent_summary.csv",      load_agent_summary(eff)),
        ("agent_seed_metrics.csv", load_agent_seed_metrics(eff)),
        ("ivl_results.csv",        load_ivl_results(eff)),
        ("latent_index_data.csv",  load_latent_index()),
    ]:
        _, error = result
        missing[fname] = error or ""
    return missing
