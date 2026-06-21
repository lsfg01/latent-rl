"""Tests para LatentAdvantageIndex (v2 — métricas acotadas y OOS).

Componentes:
    ΔSharpe   = Sharpe_OOS(latente) − Sharpe_OOS(directo)
    ΔMDD      = |MDD_OOS(latente)| − |MDD_OOS(directo)|
    Δσ_seeds  = σ_seeds(Sharpe_OOS)(latente) − σ_seeds(Sharpe_OOS)(directo)
    ΔGap      = |Sharpe_IS − Sharpe_OOS|(latente) − |Sharpe_IS − Sharpe_OOS|(directo)

Fórmula: IVL = w1·ΔSharpe − w2·ΔMDD − w3·Δσ_seeds − w4·ΔGap
Pesos por defecto: [0.35, 0.25, 0.20, 0.20]

Métricas requeridas: sharpe_oos, mdd_oos, seed_std_sharpe_oos, sharpe_is
"""

import pytest
import numpy as np

from latent_rl.evaluation.latent_advantage import LatentAdvantageIndex


# ---------------------------------------------------------------------------
# Fixtures de métricas tipo
# ---------------------------------------------------------------------------

def _metrics(sharpe_oos=1.0, mdd_oos=-0.20, seed_std_sharpe_oos=0.10, sharpe_is=1.2):
    """Crea un dict de métricas válido con valores por defecto."""
    return {
        "sharpe_oos":          sharpe_oos,
        "mdd_oos":             mdd_oos,
        "seed_std_sharpe_oos": seed_std_sharpe_oos,
        "sharpe_is":           sharpe_is,
    }


# ---------------------------------------------------------------------------
# Tests de construcción y validación
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_default_weights(self):
        ivl = LatentAdvantageIndex()
        assert ivl.weights == {"sharpe": 0.35, "mdd": 0.25, "seed_std": 0.20, "is_oos_gap": 0.20}

    def test_custom_weights_accepted(self):
        w = {"sharpe": 0.50, "mdd": 0.30, "seed_std": 0.10, "is_oos_gap": 0.10}
        ivl = LatentAdvantageIndex(weights=w)
        assert ivl.weights == w

    def test_weights_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sumar 1.0"):
            LatentAdvantageIndex(weights={"sharpe": 0.5, "mdd": 0.5,
                                          "seed_std": 0.5, "is_oos_gap": 0.5})

    def test_negative_weights_raise(self):
        with pytest.raises(ValueError, match="es negativo"):
            LatentAdvantageIndex(weights={"sharpe": -0.25, "mdd": 0.25,
                                          "seed_std": 0.25, "is_oos_gap": 0.75})


# ---------------------------------------------------------------------------
# Tests de validación de métricas
# ---------------------------------------------------------------------------

class TestMetricValidation:

    def test_missing_metrics_raise(self):
        ivl = LatentAdvantageIndex()
        incomplete = {"sharpe_oos": 1.0, "mdd_oos": -0.2}
        with pytest.raises(ValueError, match="Faltan métricas requeridas"):
            ivl.compute(incomplete, _metrics())

    def test_all_required_keys_accepted(self):
        ivl = LatentAdvantageIndex()
        result = ivl.compute(_metrics(), _metrics())
        assert "ivl" in result


# ---------------------------------------------------------------------------
# Tests de cálculo de componentes
# ---------------------------------------------------------------------------

class TestComponentCalculations:

    def test_delta_sharpe_oos(self):
        """ΔSharpe = Sharpe_OOS(latente) − Sharpe_OOS(directo)."""
        ivl = LatentAdvantageIndex()
        direct  = _metrics(sharpe_oos=1.0)
        latent  = _metrics(sharpe_oos=1.5)
        result  = ivl.compute(direct, latent)
        assert result["delta_sharpe"] == pytest.approx(0.5)

    def test_delta_mdd_magnitude(self):
        """ΔMDD = |MDD_OOS(latente)| − |MDD_OOS(directo)|.
        Latente con menor drawdown → delta_mdd < 0 → beneficia IVL."""
        ivl = LatentAdvantageIndex()
        direct = _metrics(mdd_oos=-0.30)   # magnitud 0.30
        latent = _metrics(mdd_oos=-0.10)   # magnitud 0.10
        result = ivl.compute(direct, latent)
        assert result["delta_mdd"] == pytest.approx(-0.20)

    def test_delta_mdd_positive_convention_same_result(self):
        """abs() se aplica internamente → positivo y negativo dan el mismo resultado."""
        ivl = LatentAdvantageIndex()
        direct_neg = _metrics(mdd_oos=-0.30)
        direct_pos = _metrics(mdd_oos=+0.30)
        latent = _metrics(mdd_oos=-0.10)
        r_neg = ivl.compute(direct_neg, latent)
        r_pos = ivl.compute(direct_pos, latent)
        assert r_neg["delta_mdd"] == pytest.approx(r_pos["delta_mdd"])

    def test_delta_seed_std_sharpe_oos(self):
        """Δσ_seeds usa σ del Sharpe OOS, no el retorno IS crudo."""
        ivl = LatentAdvantageIndex()
        direct = _metrics(seed_std_sharpe_oos=0.30)
        latent = _metrics(seed_std_sharpe_oos=0.10)
        result = ivl.compute(direct, latent)
        assert result["delta_seed_std"] == pytest.approx(-0.20)

    def test_delta_gap_uses_sharpe_not_returns(self):
        """ΔGap = |Sharpe_IS(L)−Sharpe_OOS(L)| − |Sharpe_IS(D)−Sharpe_OOS(D)|."""
        ivl = LatentAdvantageIndex()
        # Directo: Sharpe_IS=2.0, Sharpe_OOS=1.0 → gap=1.0
        direct = _metrics(sharpe_is=2.0, sharpe_oos=1.0)
        # Latente: Sharpe_IS=1.2, Sharpe_OOS=1.1 → gap=0.1
        latent = _metrics(sharpe_is=1.2, sharpe_oos=1.1)
        result = ivl.compute(direct, latent)
        assert result["delta_is_oos_gap"] == pytest.approx(0.1 - 1.0)

    def test_output_keys_present(self):
        ivl = LatentAdvantageIndex()
        result = ivl.compute(_metrics(), _metrics())
        for key in ("ivl", "interpretation",
                    "delta_sharpe", "delta_mdd", "delta_seed_std", "delta_is_oos_gap",
                    "delta_sharpe_norm", "delta_mdd_norm",
                    "delta_seed_std_norm", "delta_is_oos_gap_norm"):
            assert key in result, f"Falta clave: {key}"


# ---------------------------------------------------------------------------
# Tests de signo del IVL
# ---------------------------------------------------------------------------

class TestIVLSigns:

    def test_positive_ivl_when_latent_dominates(self):
        """Latente claramente mejor → IVL > 0."""
        ivl = LatentAdvantageIndex()
        direct = _metrics(sharpe_oos=0.5, mdd_oos=-0.30, seed_std_sharpe_oos=0.3,
                          sharpe_is=1.5)
        latent = _metrics(sharpe_oos=1.5, mdd_oos=-0.10, seed_std_sharpe_oos=0.1,
                          sharpe_is=1.6)  # gap IS-OOS pequeño
        result = ivl.compute(direct, latent)
        assert result["ivl"] > 0
        assert result["interpretation"] == "latent_advantage"

    def test_negative_ivl_when_direct_dominates(self):
        """Directo claramente mejor → IVL < 0."""
        ivl = LatentAdvantageIndex()
        direct = _metrics(sharpe_oos=1.5, mdd_oos=-0.10, seed_std_sharpe_oos=0.1,
                          sharpe_is=1.6)
        latent = _metrics(sharpe_oos=0.5, mdd_oos=-0.30, seed_std_sharpe_oos=0.3,
                          sharpe_is=1.5)
        result = ivl.compute(direct, latent)
        assert result["ivl"] < 0
        assert result["interpretation"] == "direct_advantage"

    def test_neutral_ivl_when_equal(self):
        m = _metrics()
        ivl = LatentAdvantageIndex()
        result = ivl.compute(m, m)
        assert abs(result["ivl"]) <= ivl.neutral_threshold
        assert result["interpretation"] == "neutral"

    def test_better_sharpe_oos_increases_ivl(self):
        """Mayor Sharpe OOS del latente debe subir el IVL (ΔSharpe > 0)."""
        ivl = LatentAdvantageIndex()
        base = _metrics()
        better = _metrics(sharpe_oos=base["sharpe_oos"] + 1.0)
        result = ivl.compute(base, better)
        assert result["delta_sharpe"] > 0
        assert result["ivl"] > 0

    def test_lower_mdd_increases_ivl(self):
        """Menor |MDD_OOS| del latente debe subir el IVL (ΔMDD < 0 → − w·ΔMDD > 0)."""
        ivl = LatentAdvantageIndex()
        direct = _metrics(mdd_oos=-0.30)
        latent = _metrics(mdd_oos=-0.05)
        result = ivl.compute(direct, latent)
        assert result["delta_mdd"] < 0
        assert result["ivl"] > 0

    def test_lower_seed_std_increases_ivl(self):
        ivl = LatentAdvantageIndex()
        direct = _metrics(seed_std_sharpe_oos=0.5)
        latent = _metrics(seed_std_sharpe_oos=0.1)
        result = ivl.compute(direct, latent)
        assert result["delta_seed_std"] < 0
        assert result["ivl"] > 0

    def test_lower_gap_increases_ivl(self):
        ivl = LatentAdvantageIndex()
        # Directo: gap |2.0−0.5| = 1.5; Latente: gap |1.1−1.0| = 0.1
        direct = _metrics(sharpe_is=2.0, sharpe_oos=0.5)
        latent = _metrics(sharpe_is=1.1, sharpe_oos=1.0)
        result = ivl.compute(direct, latent)
        assert result["delta_is_oos_gap"] < 0
        assert result["ivl"] > 0


# ---------------------------------------------------------------------------
# Tests de normalización (scales)
# ---------------------------------------------------------------------------

class TestNormalization:

    def test_without_scales_norm_equals_raw(self):
        ivl = LatentAdvantageIndex()
        result = ivl.compute(_metrics(), _metrics(sharpe_oos=1.5))
        assert result["delta_sharpe_norm"] == pytest.approx(result["delta_sharpe"])

    def test_with_scales_normalizes_correctly(self):
        ivl = LatentAdvantageIndex()
        direct = _metrics(sharpe_oos=0.0, mdd_oos=0.0, seed_std_sharpe_oos=0.0, sharpe_is=0.0)
        latent = _metrics(sharpe_oos=2.0, mdd_oos=0.0, seed_std_sharpe_oos=0.0, sharpe_is=0.0)
        scales = {"delta_sharpe": 2.0, "delta_mdd": 1.0,
                  "delta_seed_std": 1.0, "delta_is_oos_gap": 1.0}
        result = ivl.compute(direct, latent, scales=scales)
        # delta_sharpe = 2.0; norm = 2.0 / 2.0 = 1.0
        assert result["delta_sharpe_norm"] == pytest.approx(1.0)

    def test_zero_scale_gives_zero_norm(self):
        """Si la escala < 1e-8 la componente normalizada debe ser 0."""
        ivl = LatentAdvantageIndex()
        direct = _metrics(sharpe_oos=1.0)
        latent = _metrics(sharpe_oos=1.5)
        scales = {"delta_sharpe": 0.0, "delta_mdd": 0.0,
                  "delta_seed_std": 0.0, "delta_is_oos_gap": 0.0}
        result = ivl.compute(direct, latent, scales=scales)
        assert result["delta_sharpe_norm"] == pytest.approx(0.0)
        assert result["ivl"] == pytest.approx(0.0)

    def test_normalization_changes_ivl_magnitude_but_not_sign(self):
        """La normalización cambia la magnitud pero no el signo del IVL."""
        ivl = LatentAdvantageIndex()
        direct = _metrics(sharpe_oos=0.5)
        latent = _metrics(sharpe_oos=1.5)
        result_no_scales = ivl.compute(direct, latent, scales=None)
        scales = {"delta_sharpe": 5.0, "delta_mdd": 1.0,
                  "delta_seed_std": 1.0, "delta_is_oos_gap": 1.0}
        result_scaled = ivl.compute(direct, latent, scales=scales)
        # Mismo signo
        assert np.sign(result_no_scales["ivl"]) == np.sign(result_scaled["ivl"])
        # Magnitud reducida (al dividir por escala > 1)
        assert abs(result_scaled["ivl"]) < abs(result_no_scales["ivl"])


# ---------------------------------------------------------------------------
# Tests de interpretación
# ---------------------------------------------------------------------------

class TestInterpretation:

    def test_positive_is_latent_advantage(self):
        ivl = LatentAdvantageIndex()
        assert ivl.interpret(0.1) == "latent_advantage"
        assert ivl.interpret(1.0) == "latent_advantage"

    def test_negative_is_direct_advantage(self):
        ivl = LatentAdvantageIndex()
        assert ivl.interpret(-0.1) == "direct_advantage"
        assert ivl.interpret(-1.0) == "direct_advantage"

    def test_near_zero_is_neutral(self):
        ivl = LatentAdvantageIndex()
        assert ivl.interpret(0.0) == "neutral"
        assert ivl.interpret(1e-7) == "neutral"
        assert ivl.interpret(-1e-7) == "neutral"

    def test_custom_threshold(self):
        ivl = LatentAdvantageIndex(neutral_threshold=0.01)
        assert ivl.interpret(0.005) == "neutral"
        assert ivl.interpret(-0.005) == "neutral"
        assert ivl.interpret(0.02) == "latent_advantage"
        assert ivl.interpret(-0.02) == "direct_advantage"


# ---------------------------------------------------------------------------
# Test de regresión: diagnóstico BTC con métricas del piloto A-D
# Antes (v1): IVL ≈ +1.92 (inflado por retornos IS en brecha).
# Después (v2): IVL debe ser negativo o cercano a cero.
# ---------------------------------------------------------------------------

class TestBTCRegressionDiagnosis:

    def test_btc_pilot_does_not_produce_inflated_ivl(self):
        """
        Caso BTC del piloto (smoke, 1 semilla). Con las métricas IS del antiguo
        IVL el BTC producía +1.92 por la brecha de retornos acumulados (BTC IS
        puede ser 73x). Con la v2 (Sharpe OOS + gap en Sharpe) el IVL debe ser
        claramente < 0 (directo domina en OOS).
        """
        ivl = LatentAdvantageIndex()

        # Valores representativos del piloto (1 semilla):
        # A: Sharpe_IS≈0.24, Sharpe_OOS≈1.75, MDD_OOS≈-0.14, σ_seeds(Sharpe_OOS)≈0
        # D: Sharpe_IS≈0.79, Sharpe_OOS≈0.71, MDD_OOS≈-0.27, σ_seeds(Sharpe_OOS)≈0
        direct = {
            "sharpe_oos":          1.75,
            "mdd_oos":             -0.14,
            "seed_std_sharpe_oos": 0.0,
            "sharpe_is":           0.24,
        }
        latent = {
            "sharpe_oos":          0.71,
            "mdd_oos":             -0.27,
            "seed_std_sharpe_oos": 0.0,
            "sharpe_is":           0.79,
        }
        result = ivl.compute(direct, latent)

        # D tiene menor Sharpe OOS y mayor MDD OOS → el IVL debe ser negativo
        assert result["ivl"] < 0, (
            f"IVL debería ser negativo (directo domina en OOS), obtenido {result['ivl']:.4f}"
        )
        assert result["interpretation"] == "direct_advantage"

    def test_btc_gap_uses_sharpe_not_raw_returns(self):
        """
        Verifica que ΔGap no usa retornos crudos (que en BTC IS podían ser 73x).
        Con retornos IS: gap_directo ≈ |−0.17 − 0.62| = 0.79 para A.
        Con Sharpe: gap_directo ≈ |0.24 − 1.75| = 1.51. Ambos son ~O(1).
        El test solo verifica que las componentes son acotadas.
        """
        ivl = LatentAdvantageIndex()
        direct = {"sharpe_oos": 1.75, "mdd_oos": -0.14,
                  "seed_std_sharpe_oos": 0.0, "sharpe_is": 0.24}
        latent = {"sharpe_oos": 0.71, "mdd_oos": -0.27,
                  "seed_std_sharpe_oos": 0.0, "sharpe_is": 0.79}
        result = ivl.compute(direct, latent)

        # Las componentes crudas deben ser O(1), no O(10) ni O(100)
        for comp in ("delta_sharpe", "delta_mdd", "delta_seed_std", "delta_is_oos_gap"):
            assert abs(result[comp]) < 10, (
                f"Componente {comp}={result[comp]:.2f} fuera de rango ~O(1)"
            )
