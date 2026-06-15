"""
Ingeniería de features técnicos para datos OHLCV.

Transforma precios absolutos en representaciones scale-free y estacionarias
aptas para el encoder latente. Los primeros 5 features siempre son OHLCV
para que FinancialEnv pueda seguir usando el precio de cierre (columna 3).
Los indicadores calculados se añaden como columnas adicionales.

Features disponibles
--------------------
log_return      log(close_t / close_{t-1}) — estacionario, scale-free
high_low_range  (high - low) / close       — volatilidad intradía relativa
close_open_pct  (close - open) / open      — dirección y fuerza de la vela
volume_ratio    volume / rolling_mean(20)  — actividad relativa del mercado
rsi_14          RSI de 14 períodos         — momentum [0-100]
atr_pct         ATR(14) / close            — volatilidad relativa entre activos
market_regime   señal discreta -1/0/+1 según MA50 vs MA200
ma_ratio        (MA50 - MA200) / close     — versión continua del régimen
"""

import numpy as np
import pandas as pd
from typing import List

AVAILABLE_FEATURES = [
    "log_return",
    "high_low_range",
    "close_open_pct",
    "volume_ratio",
    "rsi_14",
    "atr_pct",
    "market_regime",
    "ma_ratio",
]


class FeatureEngineer:
    """
    Añade features técnicos a un DataFrame OHLCV.

    Siempre conserva las 5 columnas OHLCV originales como primeras columnas
    (el FinancialEnv las necesita para calcular equity). Los indicadores
    calculados se añaden a continuación.

    Usage::

        fe = FeatureEngineer()
        df_enriquecido = fe.transform(df_ohlcv, ["log_return", "rsi_14"])
        # df_enriquecido.columns = [open, high, low, close, volume,
        #                            log_return, rsi_14]
    """

    def transform(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Aplica los features seleccionados al DataFrame OHLCV.

        Args:
            df: DataFrame con columnas [open, high, low, close, volume].
            features: Lista de nombres de features a añadir.
                      Deben ser un subconjunto de AVAILABLE_FEATURES.
                      Si está vacía devuelve el DataFrame original sin cambios.

        Returns:
            DataFrame con las 5 columnas OHLCV + las columnas de features,
            sin NaN (las filas iniciales con NaN por ventanas se rellenan con 0).

        Raises:
            ValueError: Si algún feature no es reconocido o faltan columnas OHLCV.
        """
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame debe tener columnas OHLCV: faltan {missing}")

        unknown = [f for f in features if f not in AVAILABLE_FEATURES]
        if unknown:
            raise ValueError(
                f"Features no reconocidos: {unknown}. "
                f"Disponibles: {AVAILABLE_FEATURES}"
            )

        if not features:
            return df.copy()

        out = df[required].copy().reset_index(drop=True)

        dispatch = {
            "log_return":     self._log_return,
            "high_low_range": self._high_low_range,
            "close_open_pct": self._close_open_pct,
            "volume_ratio":   self._volume_ratio,
            "rsi_14":         self._rsi_14,
            "atr_pct":        self._atr_pct,
            "market_regime":  self._market_regime,
            "ma_ratio":       self._ma_ratio,
        }

        for name in features:
            series = dispatch[name](out)
            out[name] = series.fillna(0.0).values

        return out

    # ── Features individuales ─────────────────────────────────────────────────

    def _log_return(self, df: pd.DataFrame) -> pd.Series:
        """log(close_t / close_{t-1}). Primera fila = 0."""
        ret = np.log(df["close"] / df["close"].shift(1))
        return ret.fillna(0.0)

    def _high_low_range(self, df: pd.DataFrame) -> pd.Series:
        """(high - low) / close. Volatilidad intradía relativa."""
        return (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    def _close_open_pct(self, df: pd.DataFrame) -> pd.Series:
        """(close - open) / open. Dirección y fuerza de la vela."""
        return (df["close"] - df["open"]) / df["open"].replace(0, np.nan)

    def _volume_ratio(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """volume / rolling_mean(volume, window). Actividad relativa del mercado."""
        rolling_mean = df["volume"].rolling(window, min_periods=1).mean()
        return df["volume"] / rolling_mean.replace(0, np.nan)

    def _rsi_14(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """RSI de 14 períodos sobre close. Rango [0, 100]."""
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)  # Neutral al inicio

    def _atr_pct(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """ATR(14) / close. Volatilidad relativa comparables entre activos."""
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr = tr.ewm(com=window - 1, min_periods=window).mean()
        return (atr / df["close"].replace(0, np.nan)).fillna(0.0)

    def _market_regime(self, df: pd.DataFrame,
                       fast: int = 50, slow: int = 200) -> pd.Series:
        """
        Señal discreta de régimen de mercado:
          +1  MA50 > MA200  (tendencia alcista)
           0  cruce reciente (transición, últimos 5 días)
          -1  MA50 < MA200  (tendencia bajista)

        Requiere al menos `slow` filas para ser significativo.
        Las filas iniciales sin suficiente historia se rellenan con 0.
        """
        ma_fast = df["close"].rolling(fast,  min_periods=1).mean()
        ma_slow = df["close"].rolling(slow, min_periods=1).mean()
        diff = ma_fast - ma_slow

        # Detectar cruce en los últimos 5 días
        crossed = (diff * diff.shift(5)) < 0

        regime = pd.Series(np.where(diff > 0, 1.0, -1.0), index=df.index)
        regime[crossed] = 0.0
        return regime.fillna(0.0)

    def _ma_ratio(self, df: pd.DataFrame,
                  fast: int = 50, slow: int = 200) -> pd.Series:
        """
        (MA50 - MA200) / close — versión continua del régimen de mercado.
        Positivo en tendencia alcista, negativo en bajista, ~0 en lateral.
        Scale-free gracias a la división por close.
        """
        ma_fast = df["close"].rolling(fast,  min_periods=1).mean()
        ma_slow = df["close"].rolling(slow, min_periods=1).mean()
        return ((ma_fast - ma_slow) / df["close"].replace(0, np.nan)).fillna(0.0)
