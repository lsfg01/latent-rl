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
from latent_rl.data.cache import DataCache
from latent_rl.data.features import FeatureEngineer
from latent_rl.data.normalizer import FeatureNormalizer


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
            "mean_n_trades":          float(np.mean(n_trades_list)),
            "seed_std_return_is":     float(np.std(is_returns)),
            "seed_std_sharpe_oos":    float(np.std(oos_sharpes)),
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
    seeds: Optional[List[int]] = None,
) -> None:
    """
    Exporta los resultados para el dashboard y ``scripts.utilities.compute_ivl``.

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
            "seed_std_sharpe_oos": metrics["seed_std_sharpe_oos"],
        }
        for metrics in aggregated.values()
    ]
    summary_path = results_path / "agent_summary.csv"
    pd.DataFrame(summary_data).to_csv(summary_path, index=False)
    print(f"  agent_summary.csv: {len(summary_data)} agentes -> {summary_path}")

    # 2. agent_seed_metrics.csv
    seed_rows = []
    if seeds is not None and len(seeds) != len(all_results):
        raise ValueError("seeds debe tener la misma longitud que all_results")
    seed_values = seeds if seeds is not None else list(range(len(all_results)))
    for seed, seed_results in zip(seed_values, all_results):
        for agent_name, splits in seed_results.items():
            for split_name in ("is", "oos"):
                metrics = splits[split_name]
                seed_rows.append({
                    "agent_name":     agent_name,
                    "seed":           seed,
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


# ---------------------------------------------------------------------------
# Pipeline de datos avanzado (cache, features, normalizacion, walk-forward)
# ---------------------------------------------------------------------------

def load_ticker_with_config(
    ticker: str,
    cfg: "ExperimentConfig",
) -> pd.DataFrame:
    """
    Carga datos OHLCV para un ticker usando DataCache y FeatureEngineer.

    Usa los parametros resueltos del TickerConfig especifico del ticker (si
    existe), y aplica los features tecnicos configurados en ExperimentConfig.

    Args:
        ticker: Simbolo del activo.
        cfg: Configuracion del experimento.

    Returns:
        DataFrame con columnas [open, high, low, close, volume, *features].
    """
    from latent_rl.experiments.config import ExperimentConfig
    params = cfg.resolve_ticker_params(ticker)
    cache = DataCache(cfg.cache_dir)
    df = cache.get_or_download(
        ticker=ticker,
        start=params["start"],
        end=params["end"],
        interval=params["interval"],
    )
    if params["n_obs"] is not None:
        df = df.tail(params["n_obs"]).reset_index(drop=True)

    if cfg.features:
        fe = FeatureEngineer()
        df = fe.transform(df, cfg.features)

    tc = cfg.get_ticker_config(ticker)
    if tc.context_tickers:
        ctx = load_context_features(
            context_tickers=tc.context_tickers,
            start=params["start"],
            end=params["end"],
            n_obs=params["n_obs"],
            interval=params["interval"],
            cache_dir=cfg.cache_dir,
            primary_len=len(df),
        )
        df = pd.concat([df.reset_index(drop=True), ctx.reset_index(drop=True)], axis=1)

    return df.reset_index(drop=True)


def load_context_features(
    context_tickers: List[str],
    start: str,
    end: str,
    n_obs: Optional[int],
    interval: str,
    cache_dir: str,
    primary_len: int,
) -> pd.DataFrame:
    """
    Descarga y alinea log_return de los tickers de contexto con el primario.

    Cada ticker de contexto contribuye una columna "ctx_<ticker>_log_return".
    Si la longitud difiere del primario se recorta o rellena con 0.

    Args:
        context_tickers: Lista de tickers adicionales.
        start: Fecha de inicio.
        end: Fecha de fin.
        n_obs: Limite de filas (mismo que el primario).
        interval: Intervalo de velas.
        cache_dir: Directorio de cache.
        primary_len: Numero de filas del ticker primario.

    Returns:
        DataFrame con columnas ctx_<ticker>_log_return alineado al primario.
    """
    cache = DataCache(cache_dir)
    fe = FeatureEngineer()
    parts: Dict[str, pd.Series] = {}

    for ctx_ticker in context_tickers:
        try:
            df_ctx = cache.get_or_download(ctx_ticker, start, end, interval)
            if n_obs is not None:
                df_ctx = df_ctx.tail(n_obs).reset_index(drop=True)
            df_ctx = fe.transform(df_ctx, ["log_return"])
            lr = df_ctx["log_return"].reset_index(drop=True)

            if len(lr) > primary_len:
                lr = lr.iloc[-primary_len:].reset_index(drop=True)
            elif len(lr) < primary_len:
                pad = pd.Series([0.0] * (primary_len - len(lr)))
                lr = pd.concat([pad, lr]).reset_index(drop=True)

            parts[f"ctx_{ctx_ticker}_log_return"] = lr
        except Exception as exc:
            print(f"  [AVISO] No se pudo cargar contexto '{ctx_ticker}': {exc}")

    return pd.DataFrame(parts)


def normalize_is_oos(
    data_is: pd.DataFrame,
    data_oos: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, FeatureNormalizer]:
    """
    Aplica normalizacion z-score: fit en IS, transform en IS y OOS.

    Las columnas OHLCV se excluyen automaticamente de la normalizacion.
    No hay data leakage: el OOS nunca ve las estadisticas en fase de fit.

    Args:
        data_is: Datos In-Sample (se usan para calcular media/std).
        data_oos: Datos Out-of-Sample (solo se transforma, no se ajusta).

    Returns:
        Tupla (data_is_norm, data_oos_norm, normalizer).
    """
    normalizer = FeatureNormalizer()
    data_is_norm = normalizer.fit_transform(data_is)
    data_oos_norm = normalizer.transform(data_oos)
    return data_is_norm, data_oos_norm, normalizer


def normalize_train_val_oos(
    data_train: pd.DataFrame,
    data_val: pd.DataFrame,
    data_oos: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureNormalizer]:
    """Ajusta en IS_train y aplica las mismas estadisticas a val y OOS."""
    normalizer = FeatureNormalizer()
    train_norm = normalizer.fit_transform(data_train)
    val_norm = normalizer.transform(data_val)
    oos_norm = normalizer.transform(data_oos)
    return train_norm, val_norm, oos_norm, normalizer


def split_internal_validation(
    data_is: pd.DataFrame,
    val_ratio: float = 0.2,
    lookback: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide IS cronologicamente en train/val y valida que ambos sean evaluables."""
    split_idx = int(len(data_is) * (1.0 - val_ratio))
    data_train = data_is.iloc[:split_idx].copy().reset_index(drop=True)
    data_val = data_is.iloc[split_idx:].copy().reset_index(drop=True)
    minimum = lookback + 2
    if len(data_train) < minimum or len(data_val) < minimum:
        raise ValueError(
            "IS demasiado corto para validacion interna: "
            f"train={len(data_train)}, val={len(data_val)}, minimo={minimum}"
        )
    return data_train, data_val


def walk_forward_splits(
    data: pd.DataFrame,
    n_windows: int = 5,
    is_ratio: float = 0.6,
    lookback: int = 1,
    min_oos_steps: int = 1,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Genera N ventanas expansivas (ancladas) de (IS, OOS) para Walk-Forward Analysis.

    La estrategia es **expansiva/anclada**: el IS siempre comienza en la fila 0
    y crece en cada ventana, reproduciendo el despliegue real «entrena con todo
    el pasado, evalúa el periodo siguiente».

    Semántica de ``is_ratio`` (cambiada respecto a la versión anterior):
        Antes: fracción IS dentro de cada bloque disjunto e igual.
        Ahora: **fracción inicial de anclaje** — ``anchor_end = floor(n * is_ratio)``
        es el fin del IS de la primera ventana. El tramo restante se divide en
        ``n_windows`` bloques OOS contiguos e iguales.

    Ventana k (k = 0 … K−1):
        IS  = data[0 : oos_start]   (crece con k)
        OOS = data[oos_start : oos_end]  (contiguo, sin solapamiento)

    Args:
        data: DataFrame completo con todos los datos.
        n_windows: Número de ventanas (≥ 2).
        is_ratio: Fracción inicial de anclaje. La primera ventana tiene un IS
            de ``floor(n * is_ratio)`` filas; las siguientes lo heredan y amplían.
        lookback: Tamaño de ventana de observación del entorno (para validar que
            IS y OOS tienen filas suficientes). Por defecto 1.
        min_oos_steps: Mínimo de filas OOS evaluables por encima del lookback.

    Returns:
        Lista de ``n_windows`` tuplas ``(data_is, data_oos)`` con índice reseteado.

    Raises:
        ValueError: Si ``n_windows < 2``, o si el ancla o los bloques OOS son
            demasiado cortos dados ``lookback`` y ``min_oos_steps``.
    """
    n = len(data)
    K = n_windows

    if K < 2:
        raise ValueError(f"n_windows debe ser >= 2, got {K}")

    anchor_end = int(n * is_ratio)
    min_is_rows = lookback + 1
    if anchor_end < min_is_rows:
        raise ValueError(
            f"walk_forward_splits: is_ratio={is_ratio} deja anchor_end={anchor_end} filas, "
            f"pero se necesitan al menos {min_is_rows} (lookback={lookback}+1) para el IS inicial. "
            f"Aumenta is_ratio o usa más datos."
        )

    remaining = n - anchor_end
    b = remaining // K
    min_oos_rows = lookback + min_oos_steps
    if b < min_oos_rows:
        raise ValueError(
            f"walk_forward_splits: bloque OOS de {b} filas es menor que el mínimo requerido "
            f"{min_oos_rows} (lookback={lookback} + min_oos_steps={min_oos_steps}). "
            f"Reduce n_windows, aumenta is_ratio, o usa más datos."
        )

    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for k in range(K):
        oos_start = anchor_end + k * b
        oos_end   = anchor_end + (k + 1) * b if k < K - 1 else n
        wf_is  = data.iloc[0:oos_start].reset_index(drop=True)
        wf_oos = data.iloc[oos_start:oos_end].reset_index(drop=True)
        splits.append((wf_is, wf_oos))

    return splits
