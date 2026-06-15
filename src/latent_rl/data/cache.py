"""Caché local de datos OHLCV para evitar descargas repetidas de Yahoo Finance."""

import hashlib
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional


class DataCache:
    """
    Caché de datos OHLCV en disco usando CSV comprimido (gzip).

    Al solicitar un ticker+rango+intervalo ya descargado, lee directamente
    del disco sin tocar la red. Esto hace los experimentos reproducibles
    offline y evita el rate-limiting de Yahoo Finance.

    Estructura de archivos::

        cache_dir/
          SPY_2020-01-01_2024-01-01_1d.csv.gz
          BTC-USD_2018-01-01_2024-01-01_1d.csv.gz
          ...

    Usage::

        cache = DataCache(".data_cache")
        df = cache.get_or_download("SPY", "2020-01-01", "2024-01-01")
    """

    REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

    def __init__(self, cache_dir: str = ".data_cache"):
        self.cache_dir = Path(cache_dir)

    def get_or_download(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Devuelve datos OHLCV para el ticker y rango solicitados.

        Busca primero en caché local; si no existe o se fuerza refresco,
        descarga de Yahoo Finance y guarda en disco.

        Args:
            ticker: Símbolo del activo (p.ej. "SPY", "BTC-USD").
            start: Fecha de inicio "YYYY-MM-DD".
            end: Fecha de fin "YYYY-MM-DD".
            interval: Intervalo de velas ("1d", "1wk", "1h", etc.).
            force_refresh: Si True, ignora caché y vuelve a descargar.

        Returns:
            DataFrame con columnas [open, high, low, close, volume],
            índice numérico, sin NaN.

        Raises:
            ValueError: Si Yahoo Finance no devuelve datos.
        """
        cache_path = self._path(ticker, start, end, interval)

        if not force_refresh and cache_path.exists():
            return self._load(cache_path)

        df = self._download(ticker, start, end, interval)
        self._save(df, cache_path)
        return df

    def clear(self, ticker: Optional[str] = None) -> int:
        """
        Elimina entradas de caché.

        Args:
            ticker: Si se especifica, elimina solo los archivos de ese ticker.
                    Si None, limpia toda la caché.

        Returns:
            Número de archivos eliminados.
        """
        if not self.cache_dir.exists():
            return 0

        pattern = f"{ticker}_*.csv.gz" if ticker else "*.csv.gz"
        files = list(self.cache_dir.glob(pattern))
        for f in files:
            f.unlink()
        return len(files)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _path(self, ticker: str, start: str, end: str, interval: str) -> Path:
        safe_ticker = ticker.replace("/", "-").replace("^", "")
        filename = f"{safe_ticker}_{start}_{end}_{interval}.csv.gz"
        return self.cache_dir / filename

    def _download(self, ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
        print(f"  [cache] Descargando {ticker} ({start} -> {end}, {interval})...")
        df = yf.download(
            ticker, start=start, end=end, interval=interval,
            auto_adjust=True, progress=False,
        )

        if df is None or df.empty:
            raise ValueError(
                f"Yahoo Finance no devolvió datos para '{ticker}' "
                f"en {start}→{end} con intervalo {interval}."
            )

        # Normalizar columnas (yfinance puede devolver MultiIndex)
        df.columns = [
            c[0].lower() if isinstance(c, tuple) else c.lower()
            for c in df.columns
        ]

        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas {missing} en datos de '{ticker}'.")

        df = df[self.REQUIRED_COLS].dropna().reset_index(drop=True)
        print(f"  [cache] {len(df)} filas descargadas.")
        return df

    def _save(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, compression="gzip")

    def _load(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, compression="gzip")
        print(f"  [cache] Cargando desde disco ({path.name}) — {len(df)} filas.")
        return df
