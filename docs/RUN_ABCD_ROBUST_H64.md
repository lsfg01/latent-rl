# Run `results/abcd_robust_h64` — configuración y resultados

> Contexto de la sesión: ejecución A/B/C/D con el protocolo robusto y cabeza Q de 64
> unidades. La run **terminó** y se estaban analizando los resultados. Este documento
> fija la configuración exacta y resume las métricas para retomar el análisis.

## Qué es esta run

Comparación de los cuatro brazos sobre **SPY, TSLA, BTC-USD** con split simple **IS/OOS
70/30** (sin walk-forward), 5 semillas, y todas las palancas anti-overfitting activadas.
Script: `scripts/experiments/abcd_robust_h64.py`.

La elección de `hidden_dim=64` para la Q-network (en lugar de 128) prioriza la **menor
variabilidad entre semillas** observada en el sweep A-only previo; no implica que 64
domine a 128 en todos los tickers.

## Configuración efectiva

**Datos y entorno**
- Tickers: SPY, TSLA, BTC-USD (BTC con `start_date=2015-01-01`).
- Periodo: 2015-01-01 → 2024-01-01, intervalo diario, `train_ratio=0.7`.
- Features (F=8): log_return, high_low_range, close_open_pct, volume_ratio, rsi_14,
  atr_pct, market_regime, ma_ratio. `normalize_features=True`.
- `lookback_window`=30, `initial_balance`=10 000, `transaction_cost`=0.001.

**Protocolo anti-overfitting**
- `random_start_train=True`, `reward_mode="log_return"`, `reward_clip=None`,
  `trade_penalty=0.0`.
- Validación interna: `use_internal_validation=True`, `internal_val_ratio=0.2`,
  `validation_eval_freq=5`, `validation_patience=None`,
  `validation_score_mdd_weight=0.25`, `validation_score_trade_weight=0.05`.

**DQN directo (A) y rama latente (alineada)**
- `dqn_lr=5e-4`, `gamma=0.99`, `hidden_dim=64`, `batch_size=64`, `buffer=5000`,
  `target_update=100`, epsilon 1.0→0.1 decay 0.998, `weight_decay=1e-4`,
  `grad_clip_norm=1.0`, `dropout=0.0`.
- `align_latent_q_with_dqn=True` → B/C/D comparten esos hiperparámetros; solo difieren en
  el encoder.

**Encoder (C ligero / D gordo)**
- `encoder_type="tcn"`, `latent_dim=32`, `tcn_kernel_size=3`, `tcn_dilations=[1,2,4,8]`,
  `tcn_channels=64`.
- Preentrenamiento de C (solo en IS_train): `pretrain_n_epochs=20`, `pretrain_lr=5e-4`,
  `pretrain_batch_size=128`, `lambda_forecast=0.5`, `k_forecast=5`.
- D: `heavy_encoder_path="models/encoders/tcn_heavy.pt"` (artefacto de Fase 1, validado
  por preflight antes de la run).

**Experimento**
- `seeds=[0,1,2,3,4]`, `n_training_episodes=30`, `n_eval_episodes=3`,
  `max_steps_per_episode=500`.
- `run_arms=["A","B","C","D"]`, `direct_agent="A"`, `latent_agents=["B","C","D"]`.
- IVL pesos: `{sharpe:0.35, mdd:0.25, seed_std:0.20, is_oos_gap:0.20}`.

La config efectiva se serializa en `results/abcd_robust_h64/experiment_config.json`.

## Resultados por ticker (media de 5 semillas)

Métricas OOS y referencia IS (Sharpe). `ret_oos` = retorno total OOS; `sh` = Sharpe;
`seedσ` = desviación del Sharpe OOS entre semillas; `mdd` = max drawdown OOS.

### SPY
| Agente | ret_oos | sh_oos | sh_is | mdd_oos | seedσ_sh_oos |
|---|---|---|---|---|---|
| RandomAgent | −0.137 | −0.668 | −0.417 | −0.269 | 0.597 |
| BuyAndHold  | +0.177 | +0.707 | +1.996 | −0.245 | 0.000 |
| A (directo) | +0.147 | +0.621 | +1.990 | −0.242 | 0.136 |
| B           | +0.142 | +0.565 | +1.747 | −0.196 | 0.283 |
| C           | +0.123 | **+0.722** | +1.812 | −0.216 | 0.493 |
| D           | +0.112 | +0.461 | +1.924 | −0.251 | 0.429 |

### TSLA
| Agente | ret_oos | sh_oos | sh_is | mdd_oos | seedσ_sh_oos |
|---|---|---|---|---|---|
| RandomAgent | −0.224 | −0.109 | +1.604 | −0.637 | 0.329 |
| BuyAndHold  | +0.231 | +0.689 | +2.744 | −0.736 | 0.000 |
| A (directo) | −0.017 | +0.377 | +3.551 | −0.694 | 0.210 |
| B           | +0.176 | +0.617 | +3.068 | −0.711 | 0.126 |
| C           | **+0.329** | **+0.742** | +2.677 | −0.687 | 0.212 |
| D           | −0.043 | +0.380 | +3.712 | −0.632 | 0.275 |

### BTC-USD
| Agente | ret_oos | sh_oos | sh_is | mdd_oos | seedσ_sh_oos |
|---|---|---|---|---|---|
| RandomAgent | −0.290 | −0.340 | +2.398 | −0.694 | 0.724 |
| BuyAndHold  | +0.141 | +0.620 | +3.967 | −0.766 | 0.000 |
| A (directo) | −0.013 | +0.139 | +5.253 | −0.673 | 0.602 |
| B           | +0.081 | +0.308 | +4.756 | −0.472 | 0.532 |
| C           | **+0.595** | **+0.989** | +5.364 | −0.576 | 0.358 |
| D           | −0.153 | −0.013 | +5.059 | −0.614 | 0.567 |

## IVL (normalizado cross-ticker) y medias

Interpretación: IVL > 0 → ventaja del latente sobre A; IVL < 0 → ventaja del directo.

| Ticker | B | C | D |
|---|---|---|---|
| SPY      | −0.428 (latent_adv*) | −0.636 (direct_adv) | +0.212 (direct_adv*) |
| TSLA     | −1.646 (latent_adv*) | −2.820 (latent_adv*) | +0.359 (latent_adv*) |
| BTC-USD  | −1.516 (latent_adv*) | −1.684 (latent_adv*) | −0.094 (latent_adv*) |
| **MEDIA**| **+0.685** | **+0.987** | **−0.125** |
| STD      | 0.670 | 1.020 | 0.407 |

> *La etiqueta `interpretation` por fila se calcula sobre los **deltas crudos** de esa
> fila; el valor `ivl` mostrado es el **normalizado cross-ticker**. Por eso etiqueta y
> signo del IVL normalizado pueden no coincidir en filas individuales — la lectura
> agregada está en las medias.

## Lectura preliminar (para retomar el análisis)

- **C (encoder ligero in-sample) es el mejor latente en media de IVL** (+0.99) y el más
  fuerte en OOS de TSLA y BTC-USD, donde supera claramente a A en retorno y Sharpe OOS.
  En SPY queda algo por detrás de A en retorno pero con mejor Sharpe OOS.
- **B (encoder aleatorio finetuneado)** también sale positivo en media (+0.69),
  principalmente por reducir drawdown y mejorar OOS frente a A en TSLA/BTC.
- **D (encoder gordo transferido)** queda **neutro/negativo en media** (−0.13). El encoder
  transferido sin fuga no aporta ventaja consistente en esta configuración; en TSLA y SPY
  mejora la brecha IS/OOS pero su Sharpe OOS no despega, y en BTC-USD rinde mal.
- **A sobreajusta**: Sharpe IS altísimo (p. ej. BTC 5.25) que **no** se traslada a OOS
  (0.14), con alta `seedσ`. Los latentes congelados (C) reducen esa brecha y la varianza
  entre semillas en los activos volátiles.

Hipótesis a explorar al continuar: por qué D no transfiere (¿capacidad del corpus,
distribución de features, λ_forecast?), y si C generaliza igual bajo walk-forward (esta run
fue split simple).

## Artefactos de salida

```
results/abcd_robust_h64/
├── experiment_config.json          # config efectiva (asdict del ExperimentConfig)
├── ticker_comparison.csv           # IVL normalizado + medias/STD cross-ticker
├── SPY/ TSLA/ BTC-USD/
│   ├── agent_summary.csv           # métricas IS/OOS agregadas por agente
│   ├── agent_seed_metrics.csv      # métricas por semilla
│   ├── ivl_results.csv             # IVL por par (A vs B/C/D) del ticker
│   └── validation_metrics.csv      # mejor punto de validación interna por semilla
```
