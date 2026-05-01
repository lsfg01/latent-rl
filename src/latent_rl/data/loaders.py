"""Carga de datos financieros desde archivos CSV."""

import pandas as pd
from pathlib import Path
from typing import Optional, Union


class CSVDataLoader:
    """Cargador de datos financieros desde archivos CSV locales."""

    def __init__(self, file_path: Union[str, Path]):
        """
        Inicializa el cargador de datos.

        Args:
            file_path: Ruta al archivo CSV con datos OHLCV
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    def load(
        self,
        date_column: str = "Date",
        parse_dates: bool = True,
        index_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV.

        Args:
            date_column: Nombre de la columna de fechas
            parse_dates: Si es True, intenta parsear la columna de fechas
            index_col: Columna para usar como índice (por defecto date_column)

        Returns:
            DataFrame con los datos cargados
        """
        if parse_dates:
            df = pd.read_csv(
                self.file_path,
                parse_dates=[date_column],
                index_col=index_col if index_col else date_column
            )
        else:
            df = pd.read_csv(self.file_path)

        # Ordenar por fecha si hay índice temporal
        if df.index.name and pd.api.types.is_datetime64_any_dtype(df.index):
            df = df.sort_index()

        return df

    def load_ohlcv(
        self,
        date_column: str = "Date",
        required_columns: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Carga datos OHLCV con validación de columnas requeridas.

        Args:
            date_column: Nombre de la columna de fechas
            required_columns: Lista de columnas requeridas (por defecto OHLCV)

        Returns:
            DataFrame con datos OHLCV

        Raises:
            ValueError: Si faltan columnas requeridas
        """
        if required_columns is None:
            required_columns = ["Open", "High", "Low", "Close", "Volume"]

        df = self.load(date_column=date_column)

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

        return df[required_columns]