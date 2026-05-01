"""Preprocesado de datos financieros."""

import pandas as pd
import numpy as np
from typing import Optional, Union


class DataPreprocessor:
    """Preprocesador de datos financieros."""

    def __init__(self):
        """Inicializa el preprocesador."""
        self.scaler_params = {}

    def clean_nan(self, df: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
        """
        Limpia valores NaN del DataFrame.

        Args:
            df: DataFrame a limpiar
            method: Método de limpieza ('forward_fill', 'backward_fill', 'drop', 'mean')

        Returns:
            DataFrame limpio
        """
        df_clean = df.copy()

        if method == "forward_fill":
            df_clean = df_clean.ffill()
            df_clean = df_clean.bfill()
        elif method == "backward_fill":
            df_clean = df_clean.bfill()
        elif method == "drop":
            df_clean = df_clean.dropna()
        elif method == "mean":
            df_clean = df_clean.fillna(df_clean.mean())
        else:
            raise ValueError(f"Método no soportado: {method}")

        # Si quedan NaN después del método principal, usar forward_fill
        if df_clean.isna().any().any():
            df_clean = df_clean.ffill()
            df_clean = df_clean.bfill()

        return df_clean

    def normalize(
        self,
        df: pd.DataFrame,
        method: str = "minmax",
        columns: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Normaliza columnas numéricas del DataFrame.

        Args:
            df: DataFrame a normalizar
            method: Método de normalización ('minmax', 'zscore')
            columns: Columnas a normalizar (todas las numéricas si es None)

        Returns:
            DataFrame normalizado
        """
        df_norm = df.copy()

        if columns is None:
            columns = df_norm.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df_norm.columns:
                continue

            if method == "minmax":
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()

                if max_val - min_val > 1e-8:  # Evitar división por cero
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
                else:
                    df_norm[col] = 0.0

                self.scaler_params[col] = {"method": "minmax", "min": min_val, "max": max_val}

            elif method == "zscore":
                mean_val = df_norm[col].mean()
                std_val = df_norm[col].std()

                if std_val > 1e-8:  # Evitar división por cero
                    df_norm[col] = (df_norm[col] - mean_val) / std_val
                else:
                    df_norm[col] = 0.0

                self.scaler_params[col] = {"method": "zscore", "mean": mean_val, "std": std_val}

        return df_norm

    def calculate_returns(
        self,
        df: pd.DataFrame,
        price_column: str = "Close",
        periods: int = 1
    ) -> pd.DataFrame:
        """
        Calcula retornos logarítmicos para una columna de precios.

        Args:
            df: DataFrame con datos de precios
            price_column: Nombre de la columna de precios
            periods: Número de periodos para el cálculo de retornos

        Returns:
            DataFrame con columna de retornos añadida
        """
        df_ret = df.copy()

        if price_column not in df_ret.columns:
            raise ValueError(f"Columna {price_column} no encontrada en DataFrame")

        df_ret[f"{price_column}_return"] = np.log(df_ret[price_column] / df_ret[price_column].shift(periods))

        return df_ret

    def create_features(
        self,
        df: pd.DataFrame,
        price_column: str = "Close",
        window: int = 5
    ) -> pd.DataFrame:
        """
        Crea features básicos de media móvil y desviación estándar.

        Args:
            df: DataFrame con datos de precios
            price_column: Nombre de la columna de precios
            window: Ventana para cálculos

        Returns:
            DataFrame con features añadidas
        """
        df_feat = df.copy()

        if price_column not in df_feat.columns:
            raise ValueError(f"Columna {price_column} no encontrada en DataFrame")

        df_feat[f"{price_column}_ma_{window}"] = df_feat[price_column].rolling(window=window).mean()
        df_feat[f"{price_column}_std_{window}"] = df_feat[price_column].rolling(window=window).std()

        return df_feat