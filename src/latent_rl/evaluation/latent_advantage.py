"""
Índice de Ventaja Latente (IVL) — v2

Fórmula:
    IVL = w1·ΔSharpe - w2·ΔMDD - w3·Δσ_seeds - w4·ΔGap

Componentes (todas sobre métricas acotadas y OOS):

    ΔSharpe   = Sharpe_OOS(latente) − Sharpe_OOS(directo)
    ΔMDD      = |MDD_OOS(latente)|  − |MDD_OOS(directo)|   (magnitud)
    Δσ_seeds  = σ_seeds(Sharpe_OOS)(latente) − σ_seeds(Sharpe_OOS)(directo)
    ΔGap      = gap(latente) − gap(directo),  gap = |Sharpe_IS − Sharpe_OOS|

Signos: mayor Sharpe OOS del latente suma; menor drawdown, menor varianza entre
semillas y menor brecha IS/OOS del latente suman (van restando).

Pesos por defecto: w = [0.35, 0.25, 0.20, 0.20].

Normalización (Cambio 2):
    Antes de ponderar, cada Δ se divide por su escala (std agrupado sobre todas las
    comparaciones del experimento). Si la escala < 1e-8, la componente se fija a 0.
    La normalización se aplica pasando ``scales`` a ``compute()``. Sin ``scales``,
    el IVL usa los deltas crudos (útil para comparaciones aisladas o tests).

Métricas requeridas en el dict de entrada:
    - sharpe_oos          : Sharpe OOS medio entre semillas
    - mdd_oos             : MDD OOS medio (negativo o positivo; se aplica abs())
    - seed_std_sharpe_oos : desviación típica del Sharpe OOS entre semillas
    - sharpe_is           : Sharpe IS medio (solo para calcular la brecha)
"""

from typing import Dict, Optional


class LatentAdvantageIndex:
    """Calcula el Índice de Ventaja Latente (IVL) con métricas acotadas y OOS."""

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "sharpe":     0.35,
        "mdd":        0.25,
        "seed_std":   0.20,
        "is_oos_gap": 0.20,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        neutral_threshold: float = 1e-6,
    ):
        """
        Args:
            weights: Pesos de las cuatro componentes. Si None usa DEFAULT_WEIGHTS.
            neutral_threshold: Umbral ±ε para interpretar IVL como neutral.
        """
        if weights is None:
            weights = dict(self.DEFAULT_WEIGHTS)

        self._validate_weights(weights)
        self.weights = weights
        self.neutral_threshold = neutral_threshold

    # ------------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------------

    def _validate_weights(self, weights: Dict[str, float]) -> None:
        for key, value in weights.items():
            if value < 0:
                raise ValueError(f"El peso '{key}' es negativo: {value}")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Los pesos deben sumar 1.0, pero suman {total:.6f}"
            )

    def _validate_metrics(self, metrics: Dict[str, float], name: str) -> None:
        required = ["sharpe_oos", "mdd_oos", "seed_std_sharpe_oos", "sharpe_is"]
        missing = [k for k in required if k not in metrics]
        if missing:
            raise ValueError(
                f"Faltan métricas requeridas en {name}: {missing}"
            )

    # ------------------------------------------------------------------
    # Cálculo
    # ------------------------------------------------------------------

    def compute(
        self,
        direct_metrics: Dict[str, float],
        latent_metrics: Dict[str, float],
        scales: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Calcula el IVL.

        Args:
            direct_metrics: Métricas del agente directo.
            latent_metrics: Métricas del agente latente.
            scales: Escalas de normalización por componente (std agrupado a nivel
                experimento). Si None, el IVL se calcula sobre los deltas crudos.

        Returns:
            Diccionario con:
                ivl, interpretation,
                delta_sharpe / _mdd / _seed_std / _is_oos_gap  (crudos),
                delta_sharpe_norm / _mdd_norm / _seed_std_norm / _is_oos_gap_norm
                    (normalizados; iguales a crudos si scales=None).
        """
        self._validate_metrics(direct_metrics, "direct_metrics")
        self._validate_metrics(latent_metrics, "latent_metrics")

        # Deltas crudos
        delta_sharpe   = latent_metrics["sharpe_oos"]           - direct_metrics["sharpe_oos"]
        delta_mdd      = abs(latent_metrics["mdd_oos"])         - abs(direct_metrics["mdd_oos"])
        delta_seed_std = latent_metrics["seed_std_sharpe_oos"]  - direct_metrics["seed_std_sharpe_oos"]
        gap_l = abs(latent_metrics["sharpe_is"] - latent_metrics["sharpe_oos"])
        gap_d = abs(direct_metrics["sharpe_is"] - direct_metrics["sharpe_oos"])
        delta_is_oos_gap = gap_l - gap_d

        raw = {
            "delta_sharpe":     delta_sharpe,
            "delta_mdd":        delta_mdd,
            "delta_seed_std":   delta_seed_std,
            "delta_is_oos_gap": delta_is_oos_gap,
        }

        # Normalización
        if scales is not None:
            def _norm(key: str, val: float) -> float:
                s = scales.get(key, 0.0)
                return val / s if s > 1e-8 else 0.0
            normalized = {k: _norm(k, v) for k, v in raw.items()}
        else:
            normalized = dict(raw)

        # IVL ponderado sobre componentes normalizadas (o crudas si sin scales)
        ivl = (
            self.weights["sharpe"]      * normalized["delta_sharpe"]
            - self.weights["mdd"]       * normalized["delta_mdd"]
            - self.weights["seed_std"]  * normalized["delta_seed_std"]
            - self.weights["is_oos_gap"]* normalized["delta_is_oos_gap"]
        )

        return {
            "ivl":               ivl,
            "interpretation":    self.interpret(ivl),
            # Componentes crudas
            "delta_sharpe":      delta_sharpe,
            "delta_mdd":         delta_mdd,
            "delta_seed_std":    delta_seed_std,
            "delta_is_oos_gap":  delta_is_oos_gap,
            # Componentes normalizadas
            "delta_sharpe_norm":     normalized["delta_sharpe"],
            "delta_mdd_norm":        normalized["delta_mdd"],
            "delta_seed_std_norm":   normalized["delta_seed_std"],
            "delta_is_oos_gap_norm": normalized["delta_is_oos_gap"],
        }

    def interpret(self, ivl: float) -> str:
        if ivl > self.neutral_threshold:
            return "latent_advantage"
        elif ivl < -self.neutral_threshold:
            return "direct_advantage"
        else:
            return "neutral"
