"""Tests para el módulo de evaluación."""

import pytest
import numpy as np
import pandas as pd

from latent_rl.evaluation import FinancialMetrics


class TestFinancialMetrics:
    """Tests para FinancialMetrics."""

    @pytest.fixture
    def sample_returns(self):
        """Retornos de ejemplo para tests."""
        return pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.005, -0.005])

    @pytest.fixture
    def sample_returns_array(self):
        """Retornos de ejemplo como array."""
        return np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.005, -0.005])

    def test_sharpe_ratio(self, sample_returns):
        """Test de cálculo de ratio de Sharpe."""
        sharpe = FinancialMetrics.sharpe_ratio(sample_returns)

        assert isinstance(sharpe, float)
        assert sharpe >= 0  # Con estos datos debería ser positivo

    def test_sharpe_ratio_with_risk_free_rate(self, sample_returns):
        """Test de Sharpe con tasa libre de riesgo."""
        sharpe = FinancialMetrics.sharpe_ratio(sample_returns, risk_free_rate=0.02)

        assert isinstance(sharpe, float)

    def test_sharpe_ratio_zero_variance(self):
        """Test de Sharpe con varianza cero."""
        constant_returns = pd.Series([0.01, 0.01, 0.01, 0.01])

        sharpe = FinancialMetrics.sharpe_ratio(constant_returns)

        # Con varianza cero, Sharpe debe ser 0 o inf
        assert sharpe == 0.0 or sharpe == np.inf

    def test_sharpe_ratio_numpy_array(self, sample_returns_array):
        """Test de Sharpe con array numpy."""
        sharpe = FinancialMetrics.sharpe_ratio(sample_returns_array)

        assert isinstance(sharpe, float)

    def test_sortino_ratio(self, sample_returns):
        """Test de cálculo de ratio de Sortino."""
        sortino = FinancialMetrics.sortino_ratio(sample_returns)

        assert isinstance(sortino, float)
        assert sortino >= 0

    def test_sortino_ratio_no_negative_returns(self):
        """Test de Sortino sin retornos negativos."""
        positive_returns = pd.Series([0.01, 0.02, 0.01, 0.03])

        sortino = FinancialMetrics.sortino_ratio(positive_returns)

        # Sin retornos negativos, Sortino debe ser inf
        assert sortino == np.inf

    def test_sortino_ratio_all_negative_returns(self):
        """Test de Sortino con todos retornos negativos."""
        negative_returns = pd.Series([-0.01, -0.02, -0.01, -0.03])

        sortino = FinancialMetrics.sortino_ratio(negative_returns)

        # Con todos negativos, Sortino debe ser 0
        assert sortino == 0.0

    def test_max_drawdown(self, sample_returns):
        """Test de cálculo de drawdown máximo."""
        max_dd = FinancialMetrics.max_drawdown(sample_returns)

        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown es negativo o cero

    def test_max_drawdown_positive_returns(self):
        """Test de drawdown con retornos positivos."""
        positive_returns = pd.Series([0.01, 0.02, 0.01, 0.03])

        max_dd = FinancialMetrics.max_drawdown(positive_returns)

        # Con todos positivos, drawdown debe ser 0
        assert max_dd == 0.0

    def test_max_drawdown_numpy_array(self, sample_returns_array):
        """Test de drawdown con array numpy."""
        max_dd = FinancialMetrics.max_drawdown(sample_returns_array)

        assert isinstance(max_dd, float)
        assert max_dd <= 0

    def test_max_drawdown_duration(self, sample_returns):
        """Test de duración máxima de drawdown."""
        duration = FinancialMetrics.max_drawdown_duration(sample_returns)

        assert isinstance(duration, int)
        assert duration >= 0

    def test_total_return(self, sample_returns):
        """Test de retorno total."""
        total = FinancialMetrics.total_return(sample_returns)

        assert isinstance(total, float)

    def test_annualized_return(self, sample_returns):
        """Test de retorno anualizado."""
        annualized = FinancialMetrics.annualized_return(sample_returns)

        assert isinstance(annualized, float)

    def test_volatility(self, sample_returns):
        """Test de volatilidad."""
        vol = FinancialMetrics.volatility(sample_returns)

        assert isinstance(vol, float)
        assert vol >= 0

    def test_calmar_ratio(self, sample_returns):
        """Test de ratio de Calmar."""
        calmar = FinancialMetrics.calmar_ratio(sample_returns)

        assert isinstance(calmar, float)

    def test_calmar_ratio_zero_drawdown(self):
        """Test de Calmar con drawdown cero."""
        positive_returns = pd.Series([0.01, 0.02, 0.01, 0.03])

        calmar = FinancialMetrics.calmar_ratio(positive_returns)

        # Con drawdown cero, Calmar debe ser inf o 0
        assert calmar == np.inf or calmar == 0.0

    def test_win_rate(self, sample_returns):
        """Test de ratio de victorias."""
        win_rate = FinancialMetrics.win_rate(sample_returns)

        assert isinstance(win_rate, float)
        assert 0 <= win_rate <= 1

    def test_win_rate_all_wins(self):
        """Test de win rate con todas victorias."""
        all_wins = pd.Series([0.01, 0.02, 0.01, 0.03])

        win_rate = FinancialMetrics.win_rate(all_wins)

        assert win_rate == 1.0

    def test_win_rate_all_losses(self):
        """Test de win rate con todas pérdidas."""
        all_losses = pd.Series([-0.01, -0.02, -0.01, -0.03])

        win_rate = FinancialMetrics.win_rate(all_losses)

        assert win_rate == 0.0

    def test_profit_factor(self, sample_returns):
        """Test de factor de beneficio."""
        pf = FinancialMetrics.profit_factor(sample_returns)

        assert isinstance(pf, float)
        assert pf >= 0

    def test_profit_factor_no_losses(self):
        """Test de profit factor sin pérdidas."""
        all_wins = pd.Series([0.01, 0.02, 0.01, 0.03])

        pf = FinancialMetrics.profit_factor(all_wins)

        # Sin pérdidas, profit factor debe ser inf
        assert pf == np.inf

    def test_profit_factor_no_wins(self):
        """Test de profit factor sin victorias."""
        all_losses = pd.Series([-0.01, -0.02, -0.01, -0.03])

        pf = FinancialMetrics.profit_factor(all_losses)

        # Sin victorias, profit factor debe ser 0
        assert pf == 0.0

    def test_empty_returns(self):
        """Test con retornos vacíos."""
        empty_returns = pd.Series([])

        sharpe = FinancialMetrics.sharpe_ratio(empty_returns)
        sortino = FinancialMetrics.sortino_ratio(empty_returns)
        max_dd = FinancialMetrics.max_drawdown(empty_returns)

        # Deben manejar casos vacíos sin error
        assert isinstance(sharpe, float)
        assert isinstance(sortino, float)
        assert isinstance(max_dd, float)

    def test_single_return(self):
        """Test con un solo retorno."""
        single_return = pd.Series([0.01])

        sharpe = FinancialMetrics.sharpe_ratio(single_return)
        total = FinancialMetrics.total_return(single_return)

        assert isinstance(sharpe, float)
        assert total == 0.01

    def test_metrics_consistency(self, sample_returns):
        """Test de consistencia entre métricas."""
        sharpe = FinancialMetrics.sharpe_ratio(sample_returns)
        sortino = FinancialMetrics.sortino_ratio(sample_returns)
        max_dd = FinancialMetrics.max_drawdown(sample_returns)
        total = FinancialMetrics.total_return(sample_returns)

        # Sortino debe ser >= Sharpe (misma media, menor denominador)
        assert sortino >= sharpe or np.isinf(sortino)

        # Total debe ser consistente con los datos
        assert isinstance(total, float)

    def test_different_periods_per_year(self, sample_returns):
        """Test con diferentes periodos por año."""
        sharpe_daily = FinancialMetrics.sharpe_ratio(sample_returns, periods_per_year=252)
        sharpe_monthly = FinancialMetrics.sharpe_ratio(sample_returns, periods_per_year=12)

        # Deben ser diferentes
        assert sharpe_daily != sharpe_monthly

    def test_negative_risk_free_rate(self, sample_returns):
        """Test con tasa libre de riesgo negativa."""
        sharpe = FinancialMetrics.sharpe_ratio(sample_returns, risk_free_rate=-0.01)

        assert isinstance(sharpe, float)