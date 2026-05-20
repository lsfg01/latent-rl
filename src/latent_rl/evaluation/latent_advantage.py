"""
Índice de Ventaja Latente (IVL)

El IVL es una métrica agregada que compara el rendimiento de un agente
que utiliza representaciones latentes frente a un agente directo.

IVL = w1 * delta_sharpe - w2 * delta_mdd - w3 * delta_seed_std - w4 * delta_is_oos_gap

donde:
- delta_sharpe    = sharpe_latent - sharpe_direct           (mayor es mejor → positivo = ventaja latente)
- delta_mdd       = |mdd_latent| - |mdd_direct|             (MAGNITUD del drawdown, valor positivo)
- delta_seed_std  = seed_std_latent - seed_std_direct        (menor std es mejor → negativo = ventaja latente)
- delta_is_oos_gap = |perf_is_latent - perf_oos_latent| - |perf_is_direct - perf_oos_direct|

Convención de max_drawdown:
    El campo "max_drawdown" en las métricas debe ser la MAGNITUD del drawdown
    (valor positivo, ej: 0.25 para una caída máxima del 25%).
    Si se usa FinancialMetrics.max_drawdown(), que devuelve un valor negativo,
    hay que pasar abs(max_drawdown) antes de construir el diccionario de métricas.
    El IVL aplica abs() internamente para tolerar ambas convenciones.
"""

from typing import Dict, Optional


class LatentAdvantageIndex:
    """
    Calcula el Índice de Ventaja Latente (IVL).

    El IVL es una métrica agregada que mide la ventaja de utilizar
    representaciones latentes frente a un enfoque directo.

    Un IVL positivo indica ventaja latente, negativo indica ventaja directa,
    y valores cercanos a cero indican neutralidad.

    Attributes:
        weights: Diccionario con pesos para cada componente del IVL.
        neutral_threshold: Umbral para considerar el IVL como neutral.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        neutral_threshold: float = 1e-6
    ):
        """
        Inicializa el calculador del IVL.

        Args:
            weights: Diccionario con pesos para cada componente.
                     Por defecto: {"sharpe": 0.25, "mdd": 0.25, "seed_std": 0.25, "is_oos_gap": 0.25}
            neutral_threshold: Umbral para considerar el IVL como neutral.

        Raises:
            ValueError: Si los pesos son inválidos (negativos o no suman 1).
        """
        if weights is None:
            weights = {
                "sharpe": 0.25,
                "mdd": 0.25,
                "seed_std": 0.25,
                "is_oos_gap": 0.25
            }

        self._validate_weights(weights)
        self.weights = weights
        self.neutral_threshold = neutral_threshold

    def _validate_weights(self, weights: Dict[str, float]) -> None:
        """
        Valida que los pesos sean correctos.

        Args:
            weights: Diccionario con pesos a validar.

        Raises:
            ValueError: Si algún peso es negativo o si no suman 1.
        """
        # Verificar que todos los pesos sean >= 0
        for key, value in weights.items():
            if value < 0:
                raise ValueError(f"El peso '{key}' es negativo: {value}")

        # Verificar que sumen 1 (con tolerancia)
        total = sum(weights.values())
        tolerance = 1e-6

        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"Los pesos deben sumar 1.0, pero suman {total:.6f}"
            )

    def _validate_metrics(self, metrics: Dict[str, float], name: str) -> None:
        """
        Valida que las métricas requeridas estén presentes.

        Args:
            metrics: Diccionario con métricas a validar.
            name: Nombre del conjunto de métricas (para mensajes de error).

        Raises:
            ValueError: Si falta alguna métrica requerida.
        """
        required_metrics = ["sharpe", "max_drawdown", "seed_std", "perf_is", "perf_oos"]

        missing = [m for m in required_metrics if m not in metrics]

        if missing:
            raise ValueError(
                f"Faltan métricas requeridas en {name}: {missing}"
            )

    def compute(
        self,
        direct_metrics: Dict[str, float],
        latent_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calcula el Índice de Ventaja Latente.

        Args:
            direct_metrics: Métricas del agente directo.
            latent_metrics: Métricas del agente latente.

        Returns:
            Diccionario con:
                - ivl: Valor del IVL
                - delta_sharpe: Diferencia de Sharpe
                - delta_mdd: Diferencia de Max Drawdown
                - delta_seed_std: Diferencia de desviación entre semillas
                - delta_is_oos_gap: Diferencia de gap IS/OOS
                - interpretation: Interpretación del resultado

        Raises:
            ValueError: Si faltan métricas requeridas.
        """
        # Validar métricas
        self._validate_metrics(direct_metrics, "direct_metrics")
        self._validate_metrics(latent_metrics, "latent_metrics")

        # Calcular deltas
        delta_sharpe = latent_metrics["sharpe"] - direct_metrics["sharpe"]
        # abs() para tolerar tanto valores positivos (magnitud) como negativos (convención de FinancialMetrics)
        delta_mdd = abs(latent_metrics["max_drawdown"]) - abs(direct_metrics["max_drawdown"])
        delta_seed_std = latent_metrics["seed_std"] - direct_metrics["seed_std"]

        # Calcular delta_is_oos_gap
        gap_latent = abs(latent_metrics["perf_is"] - latent_metrics["perf_oos"])
        gap_direct = abs(direct_metrics["perf_is"] - direct_metrics["perf_oos"])
        delta_is_oos_gap = gap_latent - gap_direct

        # Calcular IVL
        ivl = (
            self.weights["sharpe"] * delta_sharpe
            - self.weights["mdd"] * delta_mdd
            - self.weights["seed_std"] * delta_seed_std
            - self.weights["is_oos_gap"] * delta_is_oos_gap
        )

        # Interpretar resultado
        interpretation = self.interpret(ivl)

        return {
            "ivl": ivl,
            "delta_sharpe": delta_sharpe,
            "delta_mdd": delta_mdd,
            "delta_seed_std": delta_seed_std,
            "delta_is_oos_gap": delta_is_oos_gap,
            "interpretation": interpretation
        }

    def interpret(self, ivl: float) -> str:
        """
        Interpreta el valor del IVL.

        Args:
            ivl: Valor del IVL a interpretar.

        Returns:
            "latent_advantage" si ivl > neutral_threshold
            "neutral" si abs(ivl) <= neutral_threshold
            "direct_advantage" si ivl < -neutral_threshold
        """
        if ivl > self.neutral_threshold:
            return "latent_advantage"
        elif ivl < -self.neutral_threshold:
            return "direct_advantage"
        else:
            return "neutral"