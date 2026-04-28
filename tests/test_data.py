"""Tests para el módulo de datos."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

from latent_rl.data import CSVDataLoader, DataPreprocessor, YahooFinanceLoader


class TestCSVDataLoader:
    """Tests para CSVDataLoader."""

    def test_init_with_existing_file(self):
        """Test de inicialización con archivo existente."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Open,High,Low,Close,Volume\n")
            f.write("2023-01-01,100,105,95,102,1000\n")
            temp_path = f.name

        try:
            loader = CSVDataLoader(temp_path)
            assert loader.file_path == Path(temp_path)
        finally:
            os.unlink(temp_path)

    def test_init_with_nonexistent_file(self):
        """Test de inicialización con archivo inexistente."""
        with pytest.raises(FileNotFoundError):
            CSVDataLoader("nonexistent.csv")

    def test_load_basic(self):
        """Test de carga básica de CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Open,High,Low,Close,Volume\n")
            f.write("2023-01-01,100,105,95,102,1000\n")
            f.write("2023-01-02,102,107,100,105,1100\n")
            temp_path = f.name

        try:
            loader = CSVDataLoader(temp_path)
            df = loader.load()

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "Date" in df.columns or df.index.name == "Date"
        finally:
            os.unlink(temp_path)

    def test_load_ohlcv(self):
        """Test de carga con validación OHLCV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Open,High,Low,Close,Volume\n")
            f.write("2023-01-01,100,105,95,102,1000\n")
            temp_path = f.name

        try:
            loader = CSVDataLoader(temp_path)
            df = loader.load_ohlcv()

            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        finally:
            os.unlink(temp_path)

    def test_load_ohlcv_missing_columns(self):
        """Test de carga OHLCV con columnas faltantes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Open,High\n")
            f.write("2023-01-01,100,105\n")
            temp_path = f.name

        try:
            loader = CSVDataLoader(temp_path)
            with pytest.raises(ValueError, match="Faltan columnas requeridas"):
                loader.load_ohlcv()
        finally:
            os.unlink(temp_path)


class TestDataPreprocessor:
    """Tests para DataPreprocessor."""

    def test_init(self):
        """Test de inicialización."""
        preprocessor = DataPreprocessor()
        assert preprocessor.scaler_params == {}

    def test_clean_nan_forward_fill(self):
        """Test de limpieza NaN con forward fill."""
        df = pd.DataFrame({
            "A": [1.0, np.nan, 3.0],
            "B": [np.nan, 2.0, np.nan]
        })

        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_nan(df, method="forward_fill")

        assert not df_clean.isna().any().any()
        assert df_clean.iloc[0, 0] == 1.0
        assert df_clean.iloc[1, 0] == 1.0  # Forward filled

    def test_clean_nan_drop(self):
        """Test de limpieza NaN con drop."""
        df = pd.DataFrame({
            "A": [1.0, np.nan, 3.0],
            "B": [4.0, 5.0, 6.0]
        })

        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_nan(df, method="drop")

        assert len(df_clean) == 2  # Una fila eliminada
        assert not df_clean.isna().any().any()

    def test_normalize_minmax(self):
        """Test de normalización min-max."""
        df = pd.DataFrame({
            "A": [1.0, 2.0, 3.0],
            "B": [10.0, 20.0, 30.0]
        })

        preprocessor = DataPreprocessor()
        df_norm = preprocessor.normalize(df, method="minmax")

        assert df_norm["A"].min() >= 0.0
        assert df_norm["A"].max() <= 1.0
        assert df_norm["B"].min() >= 0.0
        assert df_norm["B"].max() <= 1.0

    def test_normalize_zscore(self):
        """Test de normalización z-score."""
        df = pd.DataFrame({
            "A": [1.0, 2.0, 3.0],
            "B": [10.0, 20.0, 30.0]
        })

        preprocessor = DataPreprocessor()
        df_norm = preprocessor.normalize(df, method="zscore")

        # Verificar que la media es aproximadamente 0 y std aproximadamente 1
        assert abs(df_norm["A"].mean()) < 1e-10
        assert abs(df_norm["A"].std() - 1.0) < 1e-10

    def test_normalize_zero_variance(self):
        """Test de normalización con varianza cero."""
        df = pd.DataFrame({
            "A": [5.0, 5.0, 5.0],  # Varianza cero
            "B": [1.0, 2.0, 3.0]
        })

        preprocessor = DataPreprocessor()
        df_norm = preprocessor.normalize(df, method="zscore")

        # Columna con varianza cero debe ser 0
        assert all(df_norm["A"] == 0.0)

    def test_calculate_returns(self):
        """Test de cálculo de retornos."""
        df = pd.DataFrame({
            "Close": [100.0, 102.0, 104.0]
        })

        preprocessor = DataPreprocessor()
        df_ret = preprocessor.calculate_returns(df, price_column="Close")

        assert "Close_return" in df_ret.columns
        # Verificar que el primer retorno es NaN (no hay valor anterior)
        assert pd.isna(df_ret["Close_return"].iloc[0])

    def test_create_features(self):
        """Test de creación de features."""
        df = pd.DataFrame({
            "Close": [100.0, 102.0, 104.0, 106.0, 108.0]
        })

        preprocessor = DataPreprocessor()
        df_feat = preprocessor.create_features(df, price_column="Close", window=3)

        assert "Close_ma_3" in df_feat.columns
        assert "Close_std_3" in df_feat.columns


class TestYahooFinanceLoader:
    """Tests para YahooFinanceLoader."""

    @pytest.fixture
    def sample_yahoo_data(self):
        """Datos de ejemplo simulados de Yahoo Finance."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        return pd.DataFrame({
            "Date": dates,
            "Open": [100.0, 102.0, 104.0, 106.0, 108.0],
            "High": [105.0, 107.0, 109.0, 111.0, 113.0],
            "Low": [95.0, 97.0, 99.0, 101.0, 103.0],
            "Close": [102.0, 104.0, 106.0, 108.0, 110.0],
            "Adj Close": [101.5, 103.5, 105.5, 107.5, 109.5],
            "Volume": [1000, 1100, 1200, 1300, 1400]
        })

    @pytest.fixture
    def loader(self):
        """Loader de ejemplo para tests."""
        return YahooFinanceLoader()

    @patch('yfinance.download')
    def test_download_success(self, mock_download, sample_yahoo_data):
        """Test de descarga exitosa con mock."""
        # Configurar mock para retornar datos de prueba
        mock_download.return_value = sample_yahoo_data

        loader = YahooFinanceLoader()
        df = loader.download("AAPL", "2023-01-01", "2023-01-05")

        # Verificar que se llamó a yfinance.download
        mock_download.assert_called_once()

        # Verificar formato del DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "date" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    @patch('yfinance.download')
    def test_download_empty_data(self, mock_download):
        """Test de descarga con datos vacíos."""
        # Configurar mock para retornar DataFrame vacío
        mock_download.return_value = pd.DataFrame()

        loader = YahooFinanceLoader()

        with pytest.raises(ValueError, match="No se encontraron datos"):
            loader.download("INVALID", "2023-01-01", "2023-01-05")

    @patch('yfinance.download')
    def test_download_error(self, mock_download):
        """Test de descarga con error."""
        # Configurar mock para lanzar excepción
        mock_download.side_effect = Exception("Network error")

        loader = YahooFinanceLoader()

        with pytest.raises(ValueError, match="Error al descargar datos"):
            loader.download("AAPL", "2023-01-01", "2023-01-05")

    @patch('yfinance.download')
    def test_download_with_interval(self, mock_download, sample_yahoo_data):
        """Test de descarga con intervalo personalizado."""
        mock_download.return_value = sample_yahoo_data

        loader = YahooFinanceLoader()
        df = loader.download("AAPL", "2023-01-01", "2023-01-05", interval="1h")

        # Verificar que se pasó el intervalo correcto
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["interval"] == "1h"

    def test_save_csv(self, loader, sample_yahoo_data):
        """Test de guardado CSV."""
        # Convertir datos al formato estándar
        df = sample_yahoo_data.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjusted_close",
            "Volume": "volume"
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            loader.save_csv(df, temp_path)

            # Verificar que el archivo existe
            assert os.path.exists(temp_path)

            # Verificar que se puede cargar con CSVDataLoader
            csv_loader = CSVDataLoader(temp_path)
            loaded_df = csv_loader.load(date_column="date")

            assert len(loaded_df) == len(df)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_csv_invalid_format(self, loader):
        """Test de guardado CSV con formato inválido."""
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="DataFrame no tiene el formato correcto"):
                loader.save_csv(df, temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_csv_creates_directory(self, loader, sample_yahoo_data):
        """Test de guardado CSV crea directorio si no existe."""
        df = sample_yahoo_data.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjusted_close",
            "Volume": "volume"
        })

        # Crear ruta en directorio que no existe
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "subdir" / "nested" / "data.csv"

            loader.save_csv(df, temp_path)

            # Verificar que el archivo existe
            assert temp_path.exists()

    @patch('yfinance.download')
    def test_download_and_save(self, mock_download, sample_yahoo_data):
        """Test de descarga y guardado en un solo paso."""
        mock_download.return_value = sample_yahoo_data

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            loader = YahooFinanceLoader()
            df = loader.download_and_save("AAPL", "2023-01-01", "2023-01-05", temp_path)

            # Verificar que se descargó y guardó
            assert len(df) == 5
            assert os.path.exists(temp_path)

            # Verificar compatibilidad con CSVDataLoader
            csv_loader = CSVDataLoader(temp_path)
            loaded_df = csv_loader.load(date_column="date")
            assert len(loaded_df) == 5

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch('yfinance.download')
    def test_csv_format_compatibility(self, mock_download, sample_yahoo_data):
        """Test de compatibilidad de formato con CSVDataLoader."""
        mock_download.return_value = sample_yahoo_data

        loader = YahooFinanceLoader()
        df = loader.download("AAPL", "2023-01-01", "2023-01-05")

        # Guardar como CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            loader.save_csv(df, temp_path)

            # Cargar con CSVDataLoader
            csv_loader = CSVDataLoader(temp_path)
            loaded_df = csv_loader.load(date_column="date")

            # Verificar que las columnas son compatibles
            assert "date" in loaded_df.columns or loaded_df.index.name == "date"
            assert len(loaded_df) == len(df)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch('yfinance.download')
    def test_download_auto_adjust(self, mock_download, sample_yahoo_data):
        """Test de descarga con auto_adjust."""
        mock_download.return_value = sample_yahoo_data

        loader = YahooFinanceLoader()
        df = loader.download("AAPL", "2023-01-01", "2023-01-05", auto_adjust=True)

        # Verificar que se pasó auto_adjust=True
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["auto_adjust"] is True

    @patch('yfinance.download')
    def test_download_progress(self, mock_download, sample_yahoo_data):
        """Test de descarga con progress."""
        mock_download.return_value = sample_yahoo_data

        loader = YahooFinanceLoader()
        df = loader.download("AAPL", "2023-01-01", "2023-01-05", progress=True)

        # Verificar que se pasó progress=True
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["progress"] is True

    @patch('yfinance.download')
    def test_download_multiindex_columns(self, mock_download):
        """Test de descarga con columnas MultiIndex de yfinance."""
        # Crear DataFrame con columnas MultiIndex (como yfinance a veces devuelve)
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        multiindex_df = pd.DataFrame(
            {
                ("Open", "AAPL"): [100.0, 102.0, 104.0, 106.0, 108.0],
                ("High", "AAPL"): [105.0, 107.0, 109.0, 111.0, 113.0],
                ("Low", "AAPL"): [95.0, 97.0, 99.0, 101.0, 103.0],
                ("Close", "AAPL"): [102.0, 104.0, 106.0, 108.0, 110.0],
                ("Adj Close", "AAPL"): [101.5, 103.5, 105.5, 107.5, 109.5],
                ("Volume", "AAPL"): [1000, 1100, 1200, 1300, 1400]
            },
            index=dates
        )
        multiindex_df.columns = pd.MultiIndex.from_tuples(multiindex_df.columns)
        multiindex_df.index.name = "Date"

        mock_download.return_value = multiindex_df

        loader = YahooFinanceLoader()
        df = loader.download("AAPL", "2023-01-01", "2023-01-05")

        # Verificar que se aplanaron las columnas correctamente
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "date" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

        # Verificar que no hay columnas MultiIndex
        assert not isinstance(df.columns, pd.MultiIndex)

        # Verificar que el CSV guardado tiene el mismo número de filas
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            loader.save_csv(df, temp_path)

            # Verificar compatibilidad con CSVDataLoader
            csv_loader = CSVDataLoader(temp_path)
            loaded_df = csv_loader.load(date_column="date")

            # El número de filas debe ser exactamente el mismo
            assert len(loaded_df) == len(df) == 5

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)