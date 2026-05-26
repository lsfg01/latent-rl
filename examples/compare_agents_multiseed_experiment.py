"""Script experimental de comparación multi-semilla de agentes para el TFM.

Compara de forma homogénea los siguientes agentes con múltiples semillas:
    1. RandomAgent
    2. BuyAndHoldAgent
    3. DQNAgent
    4. LatentDQNAgent sin preentrenamiento
    5. LatentDQNAgent con encoder preentrenado

Utiliza datos reales descargados con yfinance.
El entrenamiento se realiza sobre datos IS (In-Sample) y la evaluación sobre
IS y OOS (Out-of-Sample). Los datos se descargan una sola vez; las semillas
controlan únicamente la inicialización de los agentes.

Toda la lógica reutilizable reside en:
    latent_rl.experiments.utils  (datos, evaluación, exportación)
    latent_rl.experiments.runner (entrenamiento, run_single_seed)
"""

from latent_rl.experiments.utils import (
    load_yfinance_data,
    split_data,
    aggregate_results,
    print_summary_table,
    print_ranking,
    export_dashboard_results,
)
from latent_rl.experiments.runner import run_single_seed


def main() -> None:
    """Función principal del experimento de comparación multi-semilla."""

    # ------------------------------------------------------------------ #
    # Parámetros del experimento — ajustar aquí para cambiar el activo,   #
    # el periodo o el tamaño del experimento.                              #
    # ------------------------------------------------------------------ #
    TICKER             = "SPY"
    START_DATE         = "2020-01-01"
    END_DATE           = "2023-12-31"
    N_OBS              = None   # None = todos los datos; int = últimas N filas
    TRAIN_RATIO        = 0.7    # Fracción IS
    SEEDS              = [0, 1, 2, 3, 4]
    N_TRAINING_EPISODES = 10
    N_EVAL_EPISODES    = 3
    # ------------------------------------------------------------------ #

    print("=" * 120)
    print("EXPERIMENTO DE COMPARACION MULTI-SEMILLA DE AGENTES (IS/OOS)")
    print("=" * 120)

    print("\n1. CONFIGURACION DEL EXPERIMENTO")
    print("-" * 120)
    print(f"Fuente de datos: Yahoo Finance  |  Ticker: {TICKER}")
    print(f"Periodo: {START_DATE} -> {END_DATE}  |  N_OBS: {'todos' if N_OBS is None else N_OBS}")
    print(f"Split IS/OOS: {int(TRAIN_RATIO * 100)}% / {int((1 - TRAIN_RATIO) * 100)}%")
    print(f"Semillas (solo agentes): {SEEDS}")
    print("Entorno: FinancialEnv (lookback_window=10, initial_balance=$10,000, tc=0.1%)")

    # --- Cargar y dividir datos UNA SOLA VEZ para todas las semillas --- #
    print("\n2. CARGA Y DIVISION DE DATOS")
    print("=" * 120)
    data = load_yfinance_data(ticker=TICKER, start=START_DATE, end=END_DATE, n_obs=N_OBS)
    data_is, data_oos = split_data(data, train_ratio=TRAIN_RATIO)
    print(f"  IS:  {len(data_is)} filas  |  OOS: {len(data_oos)} filas")

    # --- Experimento multi-semilla --------------------------------------- #
    print("\n3. EJECUTANDO EXPERIMENTO MULTI-SEMILLA")
    print("=" * 120)

    all_results = []

    for seed in SEEDS:
        seed_results = run_single_seed(
            seed=seed,
            data_is=data_is,
            data_oos=data_oos,
            n_training_episodes=N_TRAINING_EPISODES,
            n_eval_episodes=N_EVAL_EPISODES,
        )
        all_results.append(seed_results)

        print(f"\nResumen semilla {seed}:")
        for agent_name, splits in seed_results.items():
            m_is  = splits["is"]
            m_oos = splits["oos"]
            print(
                f"  {agent_name}: "
                f"IS Return={m_is['total_return']:.4f} (Sharpe={m_is['sharpe']:.3f}) | "
                f"OOS Return={m_oos['total_return']:.4f} (Sharpe={m_oos['sharpe']:.3f})"
            )

    # --- Agregacion y resultados ----------------------------------------- #
    print("\n4. AGREGANDO RESULTADOS")
    print("=" * 120)
    aggregated = aggregate_results(all_results)

    print("\n5. RESULTADOS")
    print("=" * 120)
    print_summary_table(aggregated)
    print_ranking(aggregated)

    export_dashboard_results(all_results, aggregated)

    print("\n" + "=" * 120)
    print("NOTAS FINALES")
    print("=" * 120)
    print(f"- Datos reales de {TICKER} descargados desde Yahoo Finance.")
    print("- El split IS/OOS es temporal: IS precede siempre a OOS.")
    print("- Los datos son identicos para todas las semillas; solo varia la init de los agentes.")
    print("- El preentrenamiento del encoder usa solo datos IS (sin data leakage).")
    print("- Para calcular el IVL, ejecutar compute_ivl_from_results.py.")

    print("\n" + "=" * 120)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 120)


if __name__ == "__main__":
    main()
