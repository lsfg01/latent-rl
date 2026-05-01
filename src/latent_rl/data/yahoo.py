"""Carga de datos financieros desde Yahoo Finance."""

import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import Optional, Union


class YahooFinanceLoader:
    """Cargador de datos financieros desde Yahoo Finance."""

    def __init__(self):
        """Inicializa el cargador de Yahoo Finance."""
        pass

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
        auto_adjust: bool = True,
        progress: bool = False
    ) -> pd.DataFrame:
        """
        Descarga datos OHLCV desde Yahoo Finance.

        Args:
            symbol: Símbolo del ticker (ej: "AAPL", "MSFT")
            start: Fecha de inicio en formato "YYYY-MM-DD"
            end: Fecha de fin en formato "YYYY-MM-DD"
            interval: Intervalo de datos ("1d", "1h", "5m", etc.)
            auto_adjust: Si es True, ajusta precios automáticamente
            progress: Si es True, muestra barra de progreso

        Returns:
            DataFrame con datos OHLCV

        Raises:
            ValueError: Si el símbolo es inválido o no hay datos
        """
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                progress=progress
            )

            if df.empty:
                raise ValueError(f"No se encontraron datos para {symbol}")

            # Aplanar columnas MultiIndex si existen
            if isinstance(df.columns, pd.MultiIndex):
                # Aplanar MultiIndex: tomar el primer nivel si el segundo es vacío o el símbolo
                df.columns = [
                    col[0] if col[1] == '' or col[1] == symbol else col[0]
                    for col in df.columns
                ]

            # Resetear índice para tener fecha como columna
            # Si el índice tiene nombre Date o Datetime, se convertirá en columna
            df = df.reset_index()

            # Renombrar columnas al formato estándar
            column_mapping = {
                "Date": "date",
                "Datetime": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "Volume": "volume"
            }

            df = df.rename(columns=column_mapping)

            # Asegurar que las columnas requeridas existan
            required_columns = ["date", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

            # Seleccionar columnas en orden estándar
            standard_columns = ["date", "open", "high", "low", "close", "adjusted_close", "volume"]
            available_columns = [col for col in standard_columns if col in df.columns]

            df = df[available_columns]

            return df

        except Exception as e:
            raise ValueError(f"Error al descargar datos para {symbol}: {str(e)}")

    def save_csv(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        index: bool = False
    ) -> None:
        """
        Guarda un DataFrame como CSV compatible con CSVDataLoader.

        Args:
            df: DataFrame con datos OHLCV
            file_path: Ruta donde guardar el archivo CSV
            index: Si es True, guarda el índice

        Raises:
            ValueError: Si el DataFrame no tiene el formato correcto
        """
        file_path = Path(file_path)

        # Crear directorio si no existe
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Validar formato del DataFrame
        required_columns = ["date", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"DataFrame no tiene el formato correcto. Faltan columnas: {missing_columns}")

        # Guardar como CSV
        df.to_csv(file_path, index=index)

    def download_and_save(
        self,
        symbol: str,
        start: str,
        end: str,
        file_path: Union[str, Path],
        interval: str = "1d",
        auto_adjust: bool = True,
        progress: bool = False
    ) -> pd.DataFrame:
        """
        Descarga datos y los guarda como CSV en un solo paso.

        Args:
            symbol: Símbolo del ticker
            start: Fecha de inicio
            end: Fecha de fin
            file_path: Ruta donde guardar el CSV
            interval: Intervalo de datos
            auto_adjust: Si es True, ajusta precios automáticamente
            progress: Si es True, muestra barra de progreso

        Returns:
            DataFrame con los datos descargados
        """
        df = self.download(symbol, start, end, interval, auto_adjust, progress)
        self.save_csv(df, file_path)
        return df