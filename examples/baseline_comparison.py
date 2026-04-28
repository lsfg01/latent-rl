"""Comparación de baseline: RandomAgent vs BuyAndHoldAgent.

Este ejemplo compara el rendimiento de dos agentes baseline:
- RandomAgent: Selección aleatoria de acciones
- BuyAndHoldAgent: Estrategia buy-and-hold

Ambos agentes se ejecutan en el mismo entorno con la misma semilla
para permitir una comparación justa.
"""

import numpy as np
import pandas as pd
from latent_rl.envs import FinancialEnv
from latent_rl.agents import RandomAgent, BuyAndHoldAgent


def run_episode(env, agent, seed=None):
    """
    Ejecuta un episodio completo con un agente.

    Args:
        env: Entorno financiero
        agent: Agente a ejecutar
        seed: Semilla para reproducibilidad

    Returns:
        dict: Métricas del episodio
    """
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    step_count = 0
    action_count = {"buy": 0, "hold": 0, "sell": 0}

    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

        # Contar acciones
        if action == 1:
            action_count["buy"] += 1
        elif action == 0:
            action_count["hold"] += 1
        elif action == 2:
            action_count["sell"] += 1

    # Calcular retorno total
    final_equity = info['equity']
    initial_balance = env.initial_balance
    total_return = (final_equity - initial_balance) / initial_balance

    return {
        "total_reward": total_reward,
        "final_balance": info['cash'],  # Cash disponible
        "final_equity": final_equity,  # Equity total
        "total_return": total_return,
        "steps": step_count,
        "n_trades": info['n_trades'],
        "action_count": action_count,
        "realized_profit": info['realized_profit']  # Profit/pérdida realizada
    }


def main():
    """Función principal del ejemplo comparativo."""
    print("=== Comparación de Baselines ===")
    print("RandomAgent vs BuyAndHoldAgent\n")

    # 1. Crear datos sintéticos para el ejemplo
    print("1. Creando datos sintéticos...")
    np.random.seed(42)
    n_steps = 200

    # Crear tendencia alcista para que buy-and-hold tenga ventaja
    trend = np.linspace(0, 20, n_steps)
    noise = np.random.randn(n_steps) * 2
    prices = 100 + trend + noise

    data = pd.DataFrame({
        "open": prices + np.random.randn(n_steps),
        "high": prices + np.random.randn(n_steps) + 2,
        "low": prices + np.random.randn(n_steps) - 2,
        "close": prices,
        "volume": np.random.randint(1000, 10000, n_steps)
    })

    print(f"   - Datos creados: {len(data)} filas")
    print(f"   - Precio inicial: {prices[0]:.2f}")
    print(f"   - Precio final: {prices[-1]:.2f}")
    print(f"   - Retorno del activo: {(prices[-1] - prices[0]) / prices[0]:.2%}\n")

    # 2. Crear entorno financiero
    print("2. Creando entorno financiero...")
    env = FinancialEnv(data, lookback_window=10, initial_balance=10000)
    print(f"   - Balance inicial: ${env.initial_balance:,.2f}")
    print(f"   - Coste de transacción: {env.transaction_cost:.1%}\n")

    # 3. Crear agentes
    print("3. Creando agentes...")
    random_agent = RandomAgent(env.action_space, seed=42)
    buy_and_hold_agent = BuyAndHoldAgent(env.action_space)
    print("   - RandomAgent creado")
    print("   - BuyAndHoldAgent creado\n")

    # 4. Ejecutar episodio único con cada agente
    print("4. Ejecutando episodio único...")

    # RandomAgent
    print("\n   --- RandomAgent ---")
    random_metrics = run_episode(env, random_agent, seed=42)
    print(f"   - Pasos: {random_metrics['steps']}")
    print(f"   - Recompensa total: {random_metrics['total_reward']:.4f}")
    print(f"   - Cash final: ${random_metrics['final_balance']:,.2f}")
    print(f"   - Equity final: ${random_metrics['final_equity']:,.2f}")
    print(f"   - Retorno total: {random_metrics['total_return']:.2%}")
    print(f"   - Profit realizado: ${random_metrics['realized_profit']:.2f}")
    print(f"   - Número de trades: {random_metrics['n_trades']}")
    print(f"   - Acciones: Buy={random_metrics['action_count']['buy']}, "
          f"Hold={random_metrics['action_count']['hold']}, "
          f"Sell={random_metrics['action_count']['sell']}")

    # BuyAndHoldAgent
    print("\n   --- BuyAndHoldAgent ---")
    buy_and_hold_metrics = run_episode(env, buy_and_hold_agent, seed=42)
    print(f"   - Pasos: {buy_and_hold_metrics['steps']}")
    print(f"   - Recompensa total: {buy_and_hold_metrics['total_reward']:.4f}")
    print(f"   - Cash final: ${buy_and_hold_metrics['final_balance']:,.2f}")
    print(f"   - Equity final: ${buy_and_hold_metrics['final_equity']:,.2f}")
    print(f"   - Retorno total: {buy_and_hold_metrics['total_return']:.2%}")
    print(f"   - Profit realizado: ${buy_and_hold_metrics['realized_profit']:.2f}")
    print(f"   - Número de trades: {buy_and_hold_metrics['n_trades']}")
    print(f"   - Acciones: Buy={buy_and_hold_metrics['action_count']['buy']}, "
          f"Hold={buy_and_hold_metrics['action_count']['hold']}, "
          f"Sell={buy_and_hold_metrics['action_count']['sell']}")

    # 5. Comparación de resultados
    print("\n5. Comparación de resultados:")
    print(f"   {'Métrica':<25} {'RandomAgent':<15} {'BuyAndHold':<15} {'Diferencia':<15}")
    print("-" * 70)

    metrics_to_compare = [
        ("Recompensa Total", "total_reward", "{:.4f}"),
        ("Equity Final ($)", "final_equity", "{:,.2f}"),
        ("Retorno Total (%)", "total_return", "{:.2%}"),
        ("Profit Realizado ($)", "realized_profit", "{:.2f}"),
        ("Número de Trades", "n_trades", "{}"),
    ]

    for label, key, format_str in metrics_to_compare:
        random_val = random_metrics[key]
        buy_hold_val = buy_and_hold_metrics[key]
        diff = buy_hold_val - random_val

        if key == "total_return":
            random_str = format_str.format(random_val)
            buy_hold_str = format_str.format(buy_hold_val)
            diff_str = format_str.format(diff)
        else:
            random_str = format_str.format(random_val)
            buy_hold_str = format_str.format(buy_hold_val)
            diff_str = format_str.format(diff)

        print(f"   {label:<25} {random_str:<15} {buy_hold_str:<15} {diff_str:<15}")

    # 6. Ejecutar múltiples episodios para análisis estadístico
    print("\n6. Análisis estadístico (10 episodios)...")
    n_episodes = 10

    random_rewards = []
    buy_and_hold_rewards = []

    for episode in range(n_episodes):
        # RandomAgent
        random_agent.reset()  # Resetear agente antes del episodio
        random_metrics = run_episode(env, random_agent, seed=42)  # Misma semilla para consistencia
        random_rewards.append(random_metrics['total_return'])

        # BuyAndHoldAgent
        buy_and_hold_agent.reset()  # Resetear agente antes del episodio
        buy_and_hold_metrics = run_episode(env, buy_and_hold_agent, seed=42)  # Misma semilla para consistencia
        buy_and_hold_rewards.append(buy_and_hold_metrics['total_return'])

    print("\n   --- RandomAgent ---")
    print(f"   - Media: {np.mean(random_rewards):.2%}")
    print(f"   - Desviación estándar: {np.std(random_rewards):.2%}")
    print(f"   - Mejor episodio: {np.max(random_rewards):.2%}")
    print(f"   - Peor episodio: {np.min(random_rewards):.2%}")

    print("\n   --- BuyAndHoldAgent ---")
    print(f"   - Media: {np.mean(buy_and_hold_rewards):.2%}")
    print(f"   - Desviación estándar: {np.std(buy_and_hold_rewards):.2%}")
    print(f"   - Mejor episodio: {np.max(buy_and_hold_rewards):.2%}")
    print(f"   - Peor episodio: {np.min(buy_and_hold_rewards):.2%}")

    # 7. Conclusión
    print("\n7. Conclusión:")
    random_mean = np.mean(random_rewards)
    buy_hold_mean = np.mean(buy_and_hold_rewards)

    if buy_hold_mean > random_mean:
        difference_pp = (buy_hold_mean - random_mean) * 100
        print(f"   BuyAndHoldAgent supera a RandomAgent por {difference_pp:.2f} puntos porcentuales de retorno total.")
    else:
        difference_pp = (random_mean - buy_hold_mean) * 100
        print(f"   RandomAgent supera a BuyAndHoldAgent por {difference_pp:.2f} puntos porcentuales de retorno total.")

    # 8. Cerrar entorno
    env.close()

    print("\n=== Comparación completada ===")
    print("\nNotas:")
    print("- Equity final y Retorno total son las métricas principales de rendimiento.")
    print("- Profit realizado solo incluye trades cerrados, no P&L no realizado.")
    print("- BuyAndHoldAgent mantiene posición abierta, por lo que su profit realizado es 0.")
    print("\nEstos baseline sirven como referencia para comparar con")
    print("futuros agentes de aprendizaje por refuerzo (DQN, PPO, etc.).")


if __name__ == "__main__":
    main()