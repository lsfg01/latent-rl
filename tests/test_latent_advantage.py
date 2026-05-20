"""
Tests para LatentAdvantageIndex
"""

import pytest
import numpy as np

from latent_rl.evaluation.latent_advantage import LatentAdvantageIndex


class TestLatentAdvantageIndex:
    """Tests para el cálculo del Índice de Ventaja Latente."""

    def test_basic_ivl_calculation(self):
        """Test básico de cálculo de IVL."""
        ivl_calculator = LatentAdvantageIndex()

        direct_metrics = {
            "sharpe": 1.0,
            "max_drawdown": 0.2,
            "seed_std": 0.1,
            "perf_is": 0.15,
            "perf_oos": 0.12
        }

        latent_metrics = {
            "sharpe": 1.5,
            "max_drawdown": 0.15,
            "seed_std": 0.05,
            "perf_is": 0.18,
            "perf_oos": 0.16
        }

        result = ivl_calculator.compute(direct_metrics, latent_metrics)

        # Verificar que se calculan todos los componentes
        assert "ivl" in result
        assert "delta_sharpe" in result
        assert "delta_mdd" in result
        assert "delta_seed_std" in result
        assert "delta_is_oos_gap" in result
        assert "interpretation" in result

        # Verificar cálculo de deltas
        assert result["delta_sharpe"] == pytest.approx(0.5)  # 1.5 - 1.0
        assert result["delta_mdd"] == pytest.approx(-0.05)  # 0.15 - 0.2
        assert result["delta_seed_std"] == pytest.approx(-0.05)  # 0.05 - 0.1

        # Verificar cálculo de delta_is_oos_gap
        gap_latent = abs(0.18 - 0.16)  # 0.02
        gap_direct = abs(0.15 - 0.12)  # 0.03
        expected_gap = gap_latent - gap_direct  # -0.01
        assert result["delta_is_oos_gap"] == pytest.approx(expected_gap)

    def test_positive_ivl(self):
        """Test de IVL positivo (ventaja latente)."""
        ivl_calculator = LatentAdvantageIndex()

        direct_metrics = {
            "sharpe": 1.0,
            "max_drawdown": 0.3,
            "seed_std": 0.2,
            "perf_is": 0.10,
            "perf_oos": 0.05
        }

        latent_metrics = {
            "sharpe": 2.0,  # Mejor Sharpe
            "max_drawdown": 0.1,  # Menor drawdown
            "seed_std": 0.05,  # Menor variabilidad entre semillas
            "perf_is": 0.20,
            "perf_oos": 0.18  # Menor gap IS/OOS
        }

        result = ivl_calculator.compute(direct_metrics, latent_metrics)

        assert result["ivl"] > 0
        assert result["interpretation"] == "latent_advantage"

    def test_negative_ivl(self):
        """Test de IVL negativo (ventaja directa)."""
        ivl_calculator = LatentAdvantageIndex()

        direct_metrics = {
            "sharpe": 2.0,
            "max_drawdown": 0.1,
            "seed_std": 0.05,
            "perf_is": 0.20,
            "perf_oos": 0.18
        }

        latent_metrics = {
            "sharpe": 1.0,  # Peor Sharpe
            "max_drawdown": 0.3,  # Mayor drawdown
            "seed_std": 0.2,  # Mayor variabilidad entre semillas
            "perf_is": 0.10,
            "perf_oos": 0.05  # Mayor gap IS/OOS
        }

        result = ivl_calculator.compute(direct_metrics, latent_metrics)

        assert result["ivl"] < 0
        assert result["interpretation"] == "direct_advantage"

    def test_neutral_ivl(self):
        """Test de IVL neutral."""
        ivl_calculator = LatentAdvantageIndex()

        direct_metrics = {
            "sharpe": 1.5,
            "max_drawdown": 0.2,
            "seed_std": 0.1,
            "perf_is": 0.15,
            "perf_oos": 0.13
        }

        latent_metrics = {
            "sharpe": 1.5,
            "max_drawdown": 0.2,
            "seed_std": 0.1,
            "perf_is": 0.15,
            "perf_oos": 0.13
        }

        result = ivl_calculator.compute(direct_metrics, latent_metrics)

        assert abs(result["ivl"]) <= ivl_calculator.neutral_threshold
        assert result["interpretation"] == "neutral"

    def test_custom_weights(self):
        """Test de pesos personalizados."""
        custom_weights = {
            "sharpe": 0.5,
            "mdd": 0.3,
            "seed_std": 0.1,
            "is_oos_gap": 0.1
        }

        ivl_calculator = LatentAdvantageIndex(weights=custom_weights)

        direct_metrics = {
            "sharpe": 1.0,
            "max_drawdown": 0.2,
            "seed_std": 0.1,
            "perf_is": 0.15,
            "perf_oos": 0.12
        }

        latent_metrics = {
            "sharpe": 2.0,
            "max_drawdown": 0.15,
            "seed_std": 0.05,
            "perf_is": 0.18,
            "perf_oos": 0.16
        }

        result = ivl_calculator.compute(direct_metrics, latent_metrics)

        # Verificar que se usan los pesos personalizados
        expected_ivl = (
            0.5 * 1.0  # sharpe weight * delta_sharpe
            - 0.3 * (-0.05)  # mdd weight * delta_mdd
            - 0.1 * (-0.05)  # seed_std weight * delta_seed_std
            - 0.1 * (-0.01)  # is_oos_gap weight * delta_is_oos_gap
        )

        assert result["ivl"] == pytest.approx(expected_ivl)

    def test_weights_not_summing_to_one_raises_error(self):
        """Test que pesos que no suman 1 lanzan ValueError."""
        invalid_weights = {
            "sharpe": 0.5,
            "mdd": 0.3,
            "seed_std": 0.1,
            "is_oos_gap": 0.2  # Suma 1.1
        }

        with pytest.raises(ValueError, match="Los pesos deben sumar 1.0"):
            LatentAdvantageIndex(weights=invalid_weights)

    def test_negative_weights_raise_error(self):
        """Test que pesos negativos lanzan ValueError."""
        invalid_weights = {
            "sharpe": -0.25,
            "mdd": 0.25,
            "seed_std": 0.25,
            "is_oos_gap": 0.75
        }

        with pytest.raises(ValueError, match="es negativo"):
            LatentAdvantageIndex(weights=invalid_weights)

    def test_missing_metrics_raise_error(self):
        """Test que métricas faltantes lanzan ValueError."""
        ivl_calculator = LatentAdvantageIndex()

        incomplete_metrics = {
            "sharpe": 1.0,
            "max_drawdown": 0.2
            # Faltan seed_std, perf_is, perf_oos
        }

        complete_metrics = {
            "sharpe": 1.5,
            "max_drawdown": 0.15,
            "seed_std": 0.05,
            "perf_is": 0.18,
            "perf_oos": 0.16
        }

        with pytest.raises(ValueError, match="Faltan métricas requeridas"):
            ivl_calculator.compute(incomplete_metrics, complete_metrics)

    def test_interpretation_works_correctly(self):
        """Test que la interpretación funciona correctamente."""
        ivl_calculator = LatentAdvantageIndex()

        # Ventaja latente
        assert ivl_calculator.interpret(0.1) == "latent_advantage"
        assert ivl_calculator.interpret(1.0) == "latent_advantage"

        # Neutral
        assert ivl_calculator.interpret(0.0) == "neutral"
        assert ivl_calculator.interpret(1e-7) == "neutral"
        assert ivl_calculator.interpret(-1e-7) == "neutral"

        # Ventaja directa
        assert ivl_calculator.interpret(-0.1) == "direct_advantage"
        assert ivl_calculator.interpret(-1.0) == "direct_advantage"

    def test_delta_is_oos_gap_calculation(self):
        """Test que delta_is_oos_gap se calcula correctamente."""
        ivl_calculator = LatentAdvantageIndex()

        # Caso 1: Latente tiene menor gap IS/OOS (mejor)
        direct_metrics = {
            "sharpe": 1.0,
            "max_drawdown": 0.2,
            "seed_std": 0.1,
            "perf_is": 0.20,
            "perf_oos": 0.10  # Gap de 0.10
        }

        latent_metrics = {
            "sharpe": 1.5,
            "max_drawdown": 0.15,
            "seed_std": 0.05,
            "perf_is": 0.18,
            "perf_oos": 0.16  # Gap de 0.02
        }

        result = ivl_calculator.compute(direct_metrics, latent_metrics)

        # delta_is_oos_gap = 0.02 - 0.10 = -0.08
        # Un valor negativo es bueno (menor gap)
        assert result["delta_is_oos_gap"] == pytest.approx(-0.08)

        # Caso 2: Latente tiene mayor gap IS/OOS (peor)
        direct_metrics = {
            "sharpe": 1.0,
            "max_drawdown": 0.2,
            "seed_std": 0.1,
            "perf_is": 0.15,
            "perf_oos": 0.14  # Gap de 0.01
        }

        latent_metrics = {
            "sharpe": 1.5,
            "max_drawdown": 0.15,
            "seed_std": 0.05,
            "perf_is": 0.20,
            "perf_oos": 0.10  # Gap de 0.10
        }

        result = ivl_calculator.compute(direct_metrics, latent_metrics)

        # delta_is_oos_gap = 0.10 - 0.01 = 0.09
        # Un valor positivo es malo (mayor gap)
        assert result["delta_is_oos_gap"] == pytest.approx(0.09)

    def test_default_weights(self):
        """Test que los pesos por defecto son correctos."""
        ivl_calculator = LatentAdvantageIndex()

        expected_weights = {
            "sharpe": 0.25,
            "mdd": 0.25,
            "seed_std": 0.25,
            "is_oos_gap": 0.25
        }

        assert ivl_calculator.weights == expected_weights

    def test_neutral_threshold_custom(self):
        """Test de umbral neutral personalizado."""
        custom_threshold = 0.01
        ivl_calculator = LatentAdvantageIndex(neutral_threshold=custom_threshold)

        # Valores dentro del umbral
        assert ivl_calculator.interpret(0.005) == "neutral"
        assert ivl_calculator.interpret(-0.005) == "neutral"

        # Valores fuera del umbral
        assert ivl_calculator.interpret(0.02) == "latent_advantage"
        assert ivl_calculator.interpret(-0.02) == "direct_advantage"

    # ------------------------------------------------------------------
    # Convención de signo de max_drawdown
    # ------------------------------------------------------------------

    def test_negative_mdd_values_same_result_as_positive(self):
        """Pasar max_drawdown negativo (convención FinancialMetrics) da el mismo resultado que positivo."""
        ivl_calculator = LatentAdvantageIndex()

        # Métricas con magnitudes positivas (convención del ejemplo del TFM)
        direct_pos = {"sharpe": 1.0, "max_drawdown": 0.25, "seed_std": 0.1, "perf_is": 0.15, "perf_oos": 0.12}
        latent_pos = {"sharpe": 1.5, "max_drawdown": 0.15, "seed_std": 0.05, "perf_is": 0.18, "perf_oos": 0.16}

        # Métricas equivalentes con signo negativo (convención de FinancialMetrics.max_drawdown())
        direct_neg = {**direct_pos, "max_drawdown": -0.25}
        latent_neg = {**latent_pos, "max_drawdown": -0.15}

        result_pos = ivl_calculator.compute(direct_pos, latent_pos)
        result_neg = ivl_calculator.compute(direct_neg, latent_neg)

        assert result_pos["ivl"] == pytest.approx(result_neg["ivl"])
        assert result_pos["delta_mdd"] == pytest.approx(result_neg["delta_mdd"])
        assert result_pos["interpretation"] == result_neg["interpretation"]

    def test_mdd_delta_sign_is_correct(self):
        """delta_mdd negativo cuando el agente latente tiene menor drawdown (ventaja latente)."""
        ivl_calculator = LatentAdvantageIndex()

        direct = {"sharpe": 1.0, "max_drawdown": 0.30, "seed_std": 0.1, "perf_is": 0.15, "perf_oos": 0.14}
        latent = {"sharpe": 1.0, "max_drawdown": 0.10, "seed_std": 0.1, "perf_is": 0.15, "perf_oos": 0.14}

        result = ivl_calculator.compute(direct, latent)

        # Latente tiene menor MDD → delta_mdd negativo → - w2 * delta_mdd positivo → IVL sube
        assert result["delta_mdd"] < 0
        assert result["ivl"] > 0

    def test_mdd_delta_sign_is_correct_with_negative_convention(self):
        """Misma lógica con convención negativa de FinancialMetrics."""
        ivl_calculator = LatentAdvantageIndex()

        direct = {"sharpe": 1.0, "max_drawdown": -0.30, "seed_std": 0.1, "perf_is": 0.15, "perf_oos": 0.14}
        latent = {"sharpe": 1.0, "max_drawdown": -0.10, "seed_std": 0.1, "perf_is": 0.15, "perf_oos": 0.14}

        result = ivl_calculator.compute(direct, latent)

        # Comportamiento idéntico al test anterior
        assert result["delta_mdd"] < 0
        assert result["ivl"] > 0