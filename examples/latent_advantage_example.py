"""
Ejemplo de uso del Indice de Ventaja Latente (IVL)

Este ejemplo demuestra como calcular el IVL comparando un agente
que utiliza representaciones latentes frente a un agente directo.

El IVL es una metrica agregada que combina multiples dimensiones de rendimiento:
- Sharpe ratio
- Maximum drawdown
- Estabilidad entre semillas
- Gap entre rendimiento in-sample y out-of-sample

AVISO: Las metricas de este script son sinteticas y estan hardcodeadas con fines
pedagogicos. No reflejan resultados reales de ningun agente entrenado.
Para calcular el IVL de un experimento real, usa ``scripts.utilities.compute_ivl``.
despues de ejecutar compare_agents_multiseed_experiment.py.
"""

import pandas as pd
from pathlib import Path

from latent_rl.evaluation.latent_advantage import LatentAdvantageIndex


def print_metrics(name: str, metrics: dict) -> None:
    """Imprime metricas de forma formateada."""
    print(f"\n{'='*60}")
    print(f"Metricas {name}")
    print(f"{'='*60}")
    print(f"Sharpe Ratio:          {metrics['sharpe']:.4f}")
    print(f"Max Drawdown:          {metrics['max_drawdown']:.4f}")
    print(f"Seed Std Dev:          {metrics['seed_std']:.4f}")
    print(f"Performance IS:        {metrics['perf_is']:.4f}")
    print(f"Performance OOS:       {metrics['perf_oos']:.4f}")


def print_deltas(result: dict) -> None:
    """Imprime los deltas calculados."""
    print(f"\n{'='*60}")
    print("Deltas Calculados")
    print(f"{'='*60}")
    print(f"Delta Sharpe:          {result['delta_sharpe']:+.4f}")
    print(f"Delta Max Drawdown:    {result['delta_mdd']:+.4f}")
    print(f"Delta Seed Std:        {result['delta_seed_std']:+.4f}")
    print(f"Delta IS/OOS Gap:      {result['delta_is_oos_gap']:+.4f}")


def print_ivl_result(result: dict) -> None:
    """Imprime el resultado del IVL."""
    print(f"\n{'='*60}")
    print("Indice de Ventaja Latente (IVL)")
    print(f"{'='*60}")
    print(f"IVL:                    {result['ivl']:+.6f}")
    print(f"Interpretacion:         {result['interpretation']}")

    # Interpretacion en texto
    interpretation_map = {
        "latent_advantage": "Ventaja Latente - El enfoque latente supera al directo",
        "direct_advantage": "Ventaja Directa - El enfoque directo supera al latente",
        "neutral": "Neutral - No hay diferencia significativa"
    }

    print(f"\n{interpretation_map[result['interpretation']]}")


def main():
    """Ejecuta el ejemplo de calculo del IVL."""

    print("="*60)
    print("Ejemplo del Indice de Ventaja Latente (IVL)")
    print("="*60)

    # Crear calculador de IVL con pesos por defecto
    ivl_calculator = LatentAdvantageIndex()

    # Definir metricas sinteticas para un agente directo
    # Este seria un agente que usa directamente los precios/returns
    direct_metrics = {
        "sharpe": 1.2,           # Sharpe ratio moderado
        "max_drawdown": 0.25,    # Drawdown del 25%
        "seed_std": 0.15,        # Alta variabilidad entre semillas
        "perf_is": 0.18,         # Rendimiento in-sample del 18%
        "perf_oos": 0.10         # Rendimiento out-of-sample del 10% (gap grande)
    }

    # Definir metricas sinteticas para un agente latente
    # Este seria un agente que usa representaciones latentes
    latent_metrics = {
        "sharpe": 1.8,           # Mejor Sharpe ratio
        "max_drawdown": 0.15,    # Menor drawdown (15%)
        "seed_std": 0.08,        # Menor variabilidad entre semillas
        "perf_is": 0.20,         # Rendimiento in-sample del 20%
        "perf_oos": 0.17         # Rendimiento out-of-sample del 17% (gap menor)
    }

    # Imprimir metricas
    print_metrics("Agente Directo", direct_metrics)
    print_metrics("Agente Latente", latent_metrics)

    # Calcular IVL
    result = ivl_calculator.compute(direct_metrics, latent_metrics)

    # Imprimir resultados
    print_deltas(result)
    print_ivl_result(result)

    # Guardar resultados en CSV para el dashboard
    print(f"\n{'='*60}")
    print("Guardando resultados para el dashboard...")
    print(f"{'='*60}")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    ivl_results_df = pd.DataFrame([{
        "agent_direct": "Direct_Agent",
        "agent_latent": "Latent_Agent",
        "ivl": result["ivl"],
        "delta_sharpe": result["delta_sharpe"],
        "delta_mdd": result["delta_mdd"],
        "delta_seed_std": result["delta_seed_std"],
        "delta_is_oos_gap": result["delta_is_oos_gap"],
        "interpretation": result["interpretation"]
    }])

    ivl_results_path = results_dir / "ivl_results.csv"
    ivl_results_df.to_csv(ivl_results_path, index=False)
    print(f"Resultados del IVL guardados en: {ivl_results_path}")

    # Ejemplo con pesos personalizados
    print(f"\n{'='*60}")
    print("Ejemplo con Pesos Personalizados")
    print(f"{'='*60}")

    custom_weights = {
        "sharpe": 0.4,       # Mayor peso al Sharpe
        "mdd": 0.3,          # Peso moderado al drawdown
        "seed_std": 0.2,     # Menor peso a la estabilidad
        "is_oos_gap": 0.1    # Menor peso al gap IS/OOS
    }

    ivl_calculator_custom = LatentAdvantageIndex(weights=custom_weights)
    result_custom = ivl_calculator_custom.compute(direct_metrics, latent_metrics)

    print(f"\nPesos personalizados: {custom_weights}")
    print(f"IVL con pesos personalizados: {result_custom['ivl']:+.6f}")
    print(f"Interpretacion: {result_custom['interpretation']}")

    # Ejemplo donde el agente directo es mejor
    print(f"\n{'='*60}")
    print("Ejemplo donde el Agente Directo es Mejor")
    print(f"{'='*60}")

    direct_better_metrics = {
        "sharpe": 2.0,
        "max_drawdown": 0.10,
        "seed_std": 0.05,
        "perf_is": 0.25,
        "perf_oos": 0.22
    }

    latent_worse_metrics = {
        "sharpe": 1.0,
        "max_drawdown": 0.30,
        "seed_std": 0.20,
        "perf_is": 0.15,
        "perf_oos": 0.05
    }

    result_worse = ivl_calculator.compute(direct_better_metrics, latent_worse_metrics)
    print_ivl_result(result_worse)

    # Notas importantes
    print(f"\n{'='*60}")
    print("Notas Importantes")
    print(f"{'='*60}")
    print("""
1. El IVL NO sustituye a las metricas individuales:
   - Debe analizarse cada componente por separado
   - El IVL es una medida agregada de apoyo

2. El IVL sirve como medida agregada de apoyo:
   - Proporciona una vision holistica del rendimiento
   - Facilita comparaciones entre diferentes enfoques

3. Debe calcularse sobre comparaciones homogeneas:
   - Mismos periodos de tiempo
   - Mismas condiciones de mercado
   - Mismos hiperparametros (excepto la representacion)

4. La validacion formal requiere:
   - Datos reales de mercado
   - Multiples semillas aleatorias
   - Splits temporales In-Sample / Out-of-Sample apropiados
   - Validacion cruzada temporal

5. Interpretacion del IVL:
   - IVL > 0: Ventaja del enfoque latente
   - IVL ~= 0: No hay diferencia significativa
   - IVL < 0: Ventaja del enfoque directo

6. Limitaciones:
   - Los pesos pueden ajustarse segun el caso de uso
   - La correlacion no implica causalidad
   - Resultados pasados no garantizan resultados futuros
    """)


if __name__ == "__main__":
    main()
