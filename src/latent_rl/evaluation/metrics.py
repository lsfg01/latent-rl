"""Métricas financieras para evaluación de estrategias."""

import numpy as np
import pandas as pd
from typing import Union, Optional


class FinancialMetrics:
    """Calculadora de métricas financieras."""

    @staticmethod
    def sharpe_ratio(
        returns: Union[pd.Series, np.ndarray],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calcula el ratio de Sharpe anualizado.

        Args:
            returns: Serie de retornos
            risk_free_rate: Tasa libre de riesgo (anual)
            periods_per_year: Periodos por año para anualización

        Returns:
            Ratio de Sharpe anualizado
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        if len(returns) < 2:
            return 0.0

        # Convertir tasa anual a periodical
        period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

        # Calcular retornos excedentes
        excess_returns = returns - period_rf

        # Calcular media y desviación estándar
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)

        # Controlar división por cero
        if std_return < 1e-8:
            return 0.0 if mean_return <= 0 else np.inf

        # Anualizar
        sharpe = (mean_return * periods_per_year) / (std_return * np.sqrt(periods_per_year))

        return sharpe

    @staticmethod
    def sortino_ratio(
        returns: Union[pd.Series, np.ndarray],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calcula el ratio de Sortino anualizado.

        Args:
            returns: Serie de retornos
            risk_free_rate: Tasa libre de riesgo (anual)
            periods_per_year: Periodos por año para anualización

        Returns:
            Ratio de Sortino anualizado
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        if len(returns) == 0:
            return 0.0

        # Convertir tasa anual a periodical
        period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

        # Calcular retornos excedentes
        excess_returns = returns - period_rf

        # Calcular media y desviación estándar downside
        mean_return = np.mean(excess_returns)
        downside_returns = excess_returns[excess_returns < 0]

        # Si todos los retornos son negativos, devolver 0.0
        if len(downside_returns) == len(excess_returns):
            return 0.0

        if len(downside_returns) == 0:
            # No hay retornos negativos, Sortino es infinito
            return np.inf if mean_return > 0 else 0.0

        downside_std = np.std(downside_returns, ddof=1)

        # Controlar división por cero
        if downside_std < 1e-8:
            return 0.0 if mean_return <= 0 else np.inf

        # Anualizar
        sortino = (mean_return * periods_per_year) / (downside_std * np.sqrt(periods_per_year))

        return sortino

    @staticmethod
    def max_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calcula el drawdown máximo.

        Args:
            returns: Serie de retornos

        Returns:
            Drawdown máximo como valor negativo o cero (ej: -0.25 = caída máxima del 25%).
            Para usar esta métrica en LatentAdvantageIndex, pasar abs(max_drawdown).
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        if len(returns) == 0:
            return 0.0

        # Calcular retornos acumulados
        cumulative = np.cumprod(1 + returns)

        # Calcular máximo running
        running_max = np.maximum.accumulate(cumulative)

        # Calcular drawdown
        drawdown = (cumulative - running_max) / running_max

        # Retornar el drawdown máximo (más negativo)
        return np.min(drawdown)

    @staticmethod
    def max_drawdown_duration(returns: Union[pd.Series, np.ndarray]) -> int:
        """
        Calcula la duración máxima del drawdown en periodos.

        Args:
            returns: Serie de retornos

        Returns:
            Duración máxima del drawdown en periodos
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        # Calcular retornos acumulados
        cumulative = np.cumprod(1 + returns)

        # Calcular máximo running
        running_max = np.maximum.accumulate(cumulative)

        # Identificar periodos en drawdown
        in_drawdown = cumulative < running_max

        # Calcular duraciones de drawdown
        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    @staticmethod
    def total_return(returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calcula el retorno total.

        Args:
            returns: Serie de retornos

        Returns:
            Retorno total
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        result = np.prod(1 + returns) - 1
        return round(result, 10)

    @staticmethod
    def annualized_return(
        returns: Union[pd.Series, np.ndarray],
        periods_per_year: int = 252
    ) -> float:
        """
        Calcula el retorno anualizado.

        Args:
            returns: Serie de retornos
            periods_per_year: Periodos por año para anualización

        Returns:
            Retorno anualizado
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        total_ret = FinancialMetrics.total_return(returns)
        n_periods = len(returns)

        if n_periods == 0:
            return 0.0

        annualized = (1 + total_ret) ** (periods_per_year / n_periods) - 1

        return annualized

    @staticmethod
    def volatility(
        returns: Union[pd.Series, np.ndarray],
        periods_per_year: int = 252
    ) -> float:
        """
        Calcula la volatilidad anualizada.

        Args:
            returns: Serie de retornos
            periods_per_year: Periodos por año para anualización

        Returns:
            Volatilidad anualizada
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        if len(returns) == 0:
            return 0.0

        std_return = np.std(returns, ddof=1)
        annualized_vol = std_return * np.sqrt(periods_per_year)

        return annualized_vol

    @staticmethod
    def calmar_ratio(
        returns: Union[pd.Series, np.ndarray],
        periods_per_year: int = 252
    ) -> float:
        """
        Calcula el ratio de Calmar (retorno anualizado / drawdown máximo absoluto).

        Args:
            returns: Serie de retornos
            periods_per_year: Periodos por año para anualización

        Returns:
            Ratio de Calmar
        """
        annualized_ret = FinancialMetrics.annualized_return(returns, periods_per_year)
        max_dd = FinancialMetrics.max_drawdown(returns)

        # Controlar división por cero
        if abs(max_dd) < 1e-8:
            return 0.0 if annualized_ret <= 0 else np.inf

        calmar = annualized_ret / abs(max_dd)

        return calmar

    @staticmethod
    def win_rate(returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calcula el ratio de victorias (porcentaje de retornos positivos).

        Args:
            returns: Serie de retornos

        Returns:
            Ratio de victorias (0-1)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        if len(returns) == 0:
            return 0.0

        wins = np.sum(returns > 0)
        total = len(returns)

        return wins / total

    @staticmethod
    def profit_factor(returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calcula el factor de beneficio (suma de ganancias / suma de pérdidas).

        Args:
            returns: Serie de retornos

        Returns:
            Factor de beneficio
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        gains = returns[returns > 0]
        losses = returns[returns < 0]

        total_gain = np.sum(gains)
        total_loss = abs(np.sum(losses))

        # Controlar división por cero
        if total_loss < 1e-8:
            return np.inf if total_gain > 0 else 0.0

        return total_gain / total_loss
