"""
Utilidades reutilizables para experimentos de comparación de agentes.

Este módulo centraliza las funciones de carga de datos, evaluación de agentes
y exportación de resultados que antes estaban incrustadas en el script de ejemplo.
Puede importarse desde cualquier script de experimentación o notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from latent_rl.envs import FinancialEnv
from latent_rl.evaluation.metrics import FinancialMetrics


# ---------------------------------------------------------------------------
# Carga y preparación de datos
# ---------------------------------------------------------------------------

def load_yfinance_data(
    ticker: str = "SPY",
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    n_obs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Descarga datos OHLCV reales desde Yahoo Finance y los normaliza.

    Args:
        ticker: Símbolo del activo (p. ej. "SPY", "AAPL", "BTC-USD").
        start: Fecha de inicio en formato "YYYY-MM-DD".
        end: Fecha de fin en formato "YYYY-MM-DD".
        n_obs: Si se especifica, recorta las últimas n_obs filas.
               Útil para controlar el tamaño del experimento sin cambiar fechas.

    Returns:
        DataFrame con columnas lowercase [open, high, low, close, volume],
        índice numérico reseteado y sin NaN.

    Raises:
        ValueError: Si no se descarga ningún dato para el ticker/periodo indicado.
    """
    print(f"  Descargando {ticker} ({start} -> {end})...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(
            f"No se obtuvieron datos para ticker='{ticker}' en el rango {start}-{end}."
        )

    # yfinance devuelve columnas capitalizadas (Open, High, …) o MultiIndex tuplas
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df = df.reset_index(drop=True)

    if n_obs is not None:
        df = df.tail(n_obs).reset_index(drop=True)

    print(f"  Datos descargados: {len(df)} filas")
    return df


def load_tickers_data(
    tickers: List[str],
    start: str,
    end: str,
    n_obs: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Descarga datos OHLCV para múltiples tickers de Yahoo Finance.

    Cada ticker se descarga de forma independiente. Si alguno falla, se
    propaga el error inmediatamente indicando qué ticker causó el problema.

    Args:
        tickers: Lista de símbolos (p. ej. ["SPY", "AAPL", "BTC-USD"]).
        start: Fecha de inicio en formato "YYYY-MM-DD".
        end: Fecha de fin en formato "YYYY-MM-DD".
        n_obs: Si se especifica, recorta las últimas n_obs filas por ticker.

    Returns:
        Diccionario ``{ticker: DataFrame}`` con columnas OHLCV lowercase.

    Raises:
        ValueError: Si algún ticker no devuelve datos.
    """
    result: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            result[ticker] = load_yfinance_data(
                ticker=ticker, start=start, end=end, n_obs=n_obs
            )
        except ValueError as exc:
            raise ValueError(f"Error al cargar ticker '{ticker}': {exc}") from exc
    return result


def split_data(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide los datos en In-Sample (IS) y Out-of-Sample (OOS).

    La división es estrictamente temporal: IS precede siempre a OOS.

    Args:
        data: DataFrame completo con datos OHLCV.
        train_ratio: Fracción de datos para IS (el resto va a OOS).

    Returns:
        Tupla (data_is, data_oos) con índice reseteado en ambos DataFrames.
    """
    split_idx = int(len(data) * train_ratio)
    data_is = data.iloc[:split_idx].reset_index(drop=True)
    data_oos = data.iloc[split_idx:].reset_index(drop=True)
    return data_is, data_oos


# ---------------------------------------------------------------------------
# Evaluación de agentes
# ---------------------------------------------------------------------------

def select_action_for_evaluation(agent: Any, obs: np.ndarray) -> int:
    """
    Selecciona una acción para evaluación, manejando diferentes firmas de select_action.

    Algunos agentes exponen `select_action(obs, training=False)` para usar política
    determinista; otros solo aceptan `select_action(obs)`.

    Args:
        agent: Agente a evaluar.
        obs: Observación actual.

    Returns:
        Acción seleccionada (int).
    """
    try:
        return agent.select_action(obs, training=False)
    except TypeError:
        return agent.select_action(obs)


def evaluate_agent(
    agent: Any,
    env: FinancialEnv,
    name: str,
    n_episodes: int = 3,
) -> Dict[str, float]:
    """
    Evalúa un agente en el entorno y recoge métricas financieras.

    Registra la equity paso a paso para derivar Sharpe y MDD a partir de la
    serie de retornos del episodio. Las métricas devueltas son la media sobre
    todos los episodios de evaluación.

    Args:
        agent: Agente a evaluar.
        env: Entorno financiero ya configurado con el split correcto (IS u OOS).
        name: Nombre del agente (solo para el campo "name" del dict resultado).
        n_episodes: Número de episodios de evaluación.

    Returns:
        Diccionario con métricas agregadas::

            {
                "name": str,
                "total_reward": float,
                "final_equity": float,
                "total_return": float,
                "realized_profit": float,
                "n_trades": float,
                "steps": float,
                "sharpe": float,
                "max_drawdown": float,
            }
    """
    metrics_list: List[Dict[str, float]] = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        equity_history = [env.initial_balance]

        if hasattr(agent, "reset"):
            agent.reset()
        if hasattr(agent, "set_epsilon"):
            agent.set_epsilon(0.0)

        while not done:
            action = select_action_for_evaluation(agent, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1
            equity_history.append(info["equity"])

        equity_arr = np.array(equity_history)
        step_returns = np.diff(equity_arr) / equity_arr[:-1]

        n = len(step_returns)
        sharpe = FinancialMetrics.sharpe_ratio(step_returns, periods_per_year=n) if n >= 2 else 0.0
        mdd = FinancialMetrics.max_drawdown(step_returns) if n >= 1 else 0.0

        metrics_list.append({
            "total_reward": episode_reward,
            "final_equity": info["equity"],
            "total_return": (info["equity"] - env.initial_balance) / env.initial_balance,
            "realized_profit": info["realized_profit"],
            "n_trades": info["n_trades"],
            "steps": steps,
            "sharpe": sharpe,
            "max_drawdown": mdd,
        })

    keys = ["total_reward", "final_equity", "total_return",
            "realized_profit", "n_trades", "steps", "sharpe", "max_drawdown"]
    return {
        "name": name,
        **{k: float(np.mean([m[k] for m in metrics_list])) for k in keys},
    }


# ---------------------------------------------------------------------------
# Agregación de resultados multi-semilla
# ---------------------------------------------------------------------------

def aggregate_results(
    all_results: List[Dict[str, Dict[str, Dict[str, float]]]],
) -> Dict[str, Dict[str, float]]:
    """
    Agrega resultados de múltiples semillas calculando la media por agente.

    Args:
        all_results: Lista de resultados por semilla.
            Estructura: ``seed_results[agent_name]["is" | "oos"][metric_name]``

    Returns:
        Diccionario ``aggregated[agent_name][metric]`` con medias y desviaciones
        sobre todas las semillas.
    """
    agent_names = list(all_results[0].keys())
    aggregated: Dict[str, Dict[str, float]] = {}

    for agent_name in agent_names:
        is_returns, oos_returns = [], []
        is_sharpes, oos_sharpes = [], []
        is_mdds, oos_mdds = [], []
        is_equities, oos_equities = [], []
        n_trades_list = []

        for seed_results in all_results:
            m_is = seed_results[agent_name]["is"]
            m_oos = seed_results[agent_name]["oos"]

            is_returns.append(m_is["total_return"])
            oos_returns.append(m_oos["total_return"])
            is_sharpes.append(m_is["sharpe"])
            oos_sharpes.append(m_oos["sharpe"])
            is_mdds.append(m_is["max_drawdown"])
            oos_mdds.append(m_oos["max_drawdown"])
            is_equities.append(m_is["final_equity"])
            oos_equities.append(m_oos["final_equity"])
            n_trades_list.append(m_is["n_trades"])

        aggregated[agent_name] = {
            "name": agent_name,
            # IS
            "mean_return_is":  float(np.mean(is_returns)),
            "std_return_is":   float(np.std(is_returns)),
            "mean_sharpe_is":  float(np.mean(is_sharpes)),
            "mean_mdd_is":     float(np.mean(is_mdds)),
            "mean_equity_is":  float(np.mean(is_equities)),
            # OOS
            "mean_return_oos": float(np.mean(oos_returns)),
            "std_return_oos":  float(np.std(oos_returns)),
            "mean_sharpe_oos": float(np.mean(oos_sharpes)),
            "mean_mdd_oos":    float(np.mean(oos_mdds)),
            "mean_equity_oos": float(np.mean(oos_equities)),
            # Cross-sectional
            "mean_n_trades":       float(np.mean(n_trades_list)),
            "seed_std_return_is":  float(np.std(is_returns)),
        }

    return aggregated


# ---------------------------------------------------------------------------
# Presentación y exportación
# ---------------------------------------------------------------------------

def print_summary_table(aggregated: Dict[str, Dict[str, float]]) -> None:
    """Imprime una tabla resumen con estadísticas IS y OOS por agente."""
    print("\n" + "=" * 140)
    print("TABLA RESUMEN DE AGENTES (MULTI-SEMILLA, IS vs OOS)")
    print("=" * 140)

    header = (
        f"{'Agente':<35} "
        f"{'IS Return':<12} {'IS Sharpe':<12} {'IS MDD':<12} "
        f"{'OOS Return':<12} {'OOS Sharpe':<12} {'OOS MDD':<12}"
    )
    print(header)
    print("-" * 140)

    for metrics in aggregated.values():
        print(
            f"{metrics['name']:<35} "
            f"{metrics['mean_return_is']:<12.4f} "
            f"{metrics['mean_sharpe_is']:<12.4f} "
            f"{metrics['mean_mdd_is']:<12.4f} "
            f"{metrics['mean_return_oos']:<12.4f} "
            f"{metrics['mean_sharpe_oos']:<12.4f} "
            f"{metrics['mean_mdd_oos']:<12.4f}"
        )

    print("=" * 140)


def print_ranking(aggregated: Dict[str, Dict[str, float]]) -> None:
    """Imprime el ranking de agentes ordenado por mean_return_oos."""
    print("\n" + "=" * 60)
    print("RANKING POR MEAN RETURN OOS")
    print("=" * 60)

    sorted_items = sorted(
        aggregated.items(), key=lambda x: x[1]["mean_return_oos"], reverse=True
    )
    for i, (_, metrics) in enumerate(sorted_items, 1):
        print(
            f"{i}. {metrics['name']:<35} "
            f"OOS={metrics['mean_return_oos']:.4f}  IS={metrics['mean_return_is']:.4f}"
        )

    print("=" * 60)


def export_dashboard_results(
    all_results: List[Dict[str, Dict[str, Dict[str, float]]]],
    aggregated: Dict[str, Dict[str, float]],
    results_dir: str | Path = "results",
) -> None:
    """
    Exporta los resultados en formato CSV para el dashboard y compute_ivl_from_results.py.

    Archivos generados:

    - ``agent_summary.csv``: una fila por agente con métricas IS/OOS agregadas.
    - ``agent_seed_metrics.csv``: una fila por (agente, semilla, split).

    Args:
        all_results: Lista de resultados por semilla.
        aggregated: Diccionario con métricas agregadas por agente.
        results_dir: Directorio donde guardar los CSVs (se crea si no existe).
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 120)
    print("EXPORTANDO RESULTADOS PARA EL DASHBOARD")
    print("=" * 120)

    # 1. agent_summary.csv
    summary_data = [
        {
            "agent_name": metrics["name"],
            "mean_return_is":  metrics["mean_return_is"],
            "std_return_is":   metrics["std_return_is"],
            "mean_sharpe_is":  metrics["mean_sharpe_is"],
            "mean_mdd_is":     metrics["mean_mdd_is"],
            "mean_equity_is":  metrics["mean_equity_is"],
            "mean_return_oos": metrics["mean_return_oos"],
            "std_return_oos":  metrics["std_return_oos"],
            "mean_sharpe_oos": metrics["mean_sharpe_oos"],
            "mean_mdd_oos":    metrics["mean_mdd_oos"],
            "mean_equity_oos": metrics["mean_equity_oos"],
            "mean_n_trades":       metrics["mean_n_trades"],
            "seed_std_return_is":  metrics["seed_std_return_is"],
        }
        for metrics in aggregated.values()
    ]
    summary_path = results_path / "agent_summary.csv"
    pd.DataFrame(summary_data).to_csv(summary_path, index=False)
    print(f"  agent_summary.csv: {len(summary_data)} agentes -> {summary_path}")

    # 2. agent_seed_metrics.csv
    seed_rows = []
    for seed_idx, seed_results in enumerate(all_results):
        for agent_name, splits in seed_results.items():
            for split_name, metrics in splits.items():
                seed_rows.append({
                    "agent_name":     agent_name,
                    "seed":           seed_idx,
                    "split":          split_name,
                    "total_reward":   metrics["total_reward"],
                    "total_return":   metrics["total_return"],
                    "final_equity":   metrics["final_equity"],
                    "realized_profit": metrics["realized_profit"],
                    "n_trades":       metrics["n_trades"],
                    "steps":          metrics["steps"],
                    "sharpe":         metrics["sharpe"],
                    "max_drawdown":   metrics["max_drawdown"],
                })

    seed_path = results_path / "agent_seed_metrics.csv"
    pd.DataFrame(seed_rows).to_csv(seed_path, index=False)
    print(f"  agent_seed_metrics.csv: {len(seed_rows)} filas -> {seed_path}")

    print("\n" + "=" * 120)
    print("RESULTADOS EXPORTADOS CORRECTAMENTE")
    print("=" * 120)
