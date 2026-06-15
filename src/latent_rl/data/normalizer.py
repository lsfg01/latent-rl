"""
Normalización z-score de features técnicos con separación estricta IS/OOS.

Flujo correcto:
    normalizer = FeatureNormalizer(feature_cols)
    data_is  = normalizer.fit_transform(data_is)   # calcula media/std en IS
    data_oos = normalizer.transform(data_oos)       # aplica los mismos params a OOS

Las columnas OHLCV (open, high, low, close, volume) nunca se normalizan porque
FinancialEnv las necesita en escala real para calcular equity y ejecutar órdenes.
Solo se normalizan las columnas de features técnicos añadidas por FeatureEngineer.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


OHLCV_COLS = {"open", "high", "low", "close", "volume"}


class FeatureNormalizer:
    """
    Normalización z-score de features técnicos, fit en IS y transform en OOS.

    Atributos públicos tras fit:
        mean_  -- media por columna (dict col -> float)
        std_   -- desviación típica por columna (dict col -> float)
        fitted -- True después de llamar a fit() o fit_transform()

    Usage::

        norm = FeatureNormalizer()
        data_is  = norm.fit_transform(data_is)
        data_oos = norm.transform(data_oos)
    """

    def __init__(self, feature_cols: Optional[List[str]] = None):
        """
        Args:
            feature_cols: Columnas a normalizar. Si None, se normalizan
                          automáticamente todas las que no sean OHLCV.
        """
        self._explicit_cols = feature_cols
        self.mean_:  dict = {}
        self.std_:   dict = {}
        self.fitted: bool = False

    def fit(self, data: pd.DataFrame) -> "FeatureNormalizer":
        """
        Calcula media y std de los features usando solo los datos IS.

        Args:
            data: DataFrame IS con columnas OHLCV + features.

        Returns:
            self (permite encadenar fit().transform()).
        """
        cols = self._resolve_cols(data)
        for col in cols:
            self.mean_[col] = float(data[col].mean())
            self.std_[col]  = float(data[col].std())
        self.fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la normalización calculada en fit() al DataFrame dado.

        Puede ser IS o OOS; los parámetros usados son siempre los del IS.

        Args:
            data: DataFrame con las mismas columnas que el usado en fit().

        Returns:
            DataFrame con las columnas de features normalizadas a z-score.
            Las columnas OHLCV no se modifican.

        Raises:
            RuntimeError: Si se llama antes de fit().
        """
        if not self.fitted:
            raise RuntimeError("Llama a fit() antes de transform().")

        out = data.copy()
        for col, mean in self.mean_.items():
            if col not in out.columns:
                continue
            std = self.std_[col]
            if std < 1e-8:
                out[col] = 0.0
            else:
                out[col] = (out[col] - mean) / std
        return out

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Atajo: fit() + transform() sobre los mismos datos IS."""
        return self.fit(data).transform(data)

    def _resolve_cols(self, data: pd.DataFrame) -> List[str]:
        """Devuelve las columnas a normalizar, excluyendo OHLCV."""
        if self._explicit_cols is not None:
            return [c for c in self._explicit_cols if c in data.columns]
        return [c for c in data.columns if c not in OHLCV_COLS]
