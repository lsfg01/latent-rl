# Protocolo de Entrenamiento y Evaluación (brazos A–D, agente RL, walk-forward, IVL)

> Contexto de la sesión: commit `encoder robusto` (`e9f6a22`). Este documento agrupa los
> cambios del **protocolo experimental**: los cuatro brazos comparativos, el agente RL
> latente (incluido el modo de latentes precomputados), el endurecimiento del entorno y
> del DQN, la validación interna temporal, la optimización de encoders congelados, el
> análisis **Walk-Forward** y el rediseño del **IVL (v2)**.

---

## 1. Los cuatro brazos (A / B / C / D)

El experimento compara un agente "directo" contra tres agentes "latentes" que difieren
solo en **de dónde viene el encoder** y en si está **congelado**:

| Brazo | Agente | Encoder | Entrenamiento del encoder |
|---|---|---|---|
| **A** | `DQNAgent` directo | — (opera sobre `(L, F)` aplanado) | n/a |
| **B** | `LatentDQNAgent` | aleatorio | se **finetunea** junto al Q durante el RL |
| **C** | `LatentDQNAgent` | ligero, preentrenado **en el IS del pliegue** | recon+forecast en IS, luego **congelado** |
| **D** | `LatentDQNAgent` | "gordo" offline (artefacto `tcn_heavy.pt`) | preentrenado fuera (Fase 1), **congelado** |

Selección por `cfg.run_arms ⊆ {A,B,C,D}`. Cada semilla añade además dos referencias
fijas: `RandomAgent` y `BuyAndHoldAgent`.

Lectura de la comparación:
- **A vs B** aísla el efecto de *tener* un cuello de botella latente (encoder aleatorio
  finetuneado).
- **A vs C** mide la ventaja de preentrenar el encoder con el propio activo (in-sample).
- **A vs D** mide la ventaja de un encoder transferido desde un corpus disjunto
  (sin fuga de datos).

Funciones: `_train_arm_a/b/c/d` en `experiments/runner.py`. El brazo D se **omite con
gracia** si `heavy_encoder_path` no está o el artefacto es incompatible.

---

## 2. Agente RL latente: `LatentDQNAgent`

Reescritura del agente para soportar la familia de encoders y la precomputación.

- **Encoder por factory**: se construye con `encoder_type` + kwargs (mlp/tcn/gru). El
  espacio de observación nativo es `(L, F)`.
- **`freeze_encoder`**: si está activo, el optimizador entrena **solo la Q-network**; el
  encoder se evalúa en `no_grad`. Es el modo de los brazos C y D.
- **`precomputed_latents`** (nuevo): el agente **no construye encoder**; espera
  observaciones que ya son vectores `(latent_dim,)`. `select_action` y `update` usan la
  observación directamente como latente. Es la base de la optimización de §5.
- **`load_artifact_encoder(path, cfg)`**: inyecta un encoder desde un artefacto v2
  (con validación de compatibilidad). Usado por el brazo D.
- **`load_pretrained_encoder(path)`**: ruta legada para checkpoints del
  `AutoencoderTrainer`.

La **Q-network** (`LatentQNetwork`) es idéntica en todos los modos: `latent_dim → q_hidden
→ n_actions`, con dropout opcional. Objetivo DQN estándar con `target_network`,
`SmoothL1Loss` y actualización periódica.

### Endurecimiento (regularización) de A y latentes

Tanto `DQNAgent` como `LatentDQNAgent` ganaron palancas de regularización, expuestas en
`ExperimentConfig`:

- `weight_decay` (L2 en Adam),
- `grad_clip_norm` (recorte de norma del gradiente sobre los parámetros entrenables),
- `dropout` en la Q-network.

`align_latent_q_with_dqn=True` copia los hiperparámetros del DQN directo a la rama latente
(lr, gamma, hidden, batch, buffer, target_update, epsilon, weight_decay, grad_clip,
dropout) para que A y B/C/D **solo difieran en el encoder**, no en el optimizador.

---

## 3. Entorno: recompensas y arranque aleatorio (`FinancialEnv`)

Cambios para reducir el sobreajuste y dar una señal de recompensa más estable:

- **`reward_mode`**:
  - `"equity_delta_initial"` (heredado): variación de equity respecto al capital inicial.
  - `"log_return"` (usado en la run robusta): log-retorno del equity paso a paso —
    señal aditiva en el tiempo y menos sensible a la escala.
- **`reward_clip`**: recorta la recompensa a `[-clip, clip]` (None = sin recorte).
- **`trade_penalty`**: penalización fija aplicada **solo** a operaciones ejecutadas
  (desincentiva el over-trading).
- **Arranque aleatorio** (`reset(options={"random_start": True, "max_steps": ...})`): el
  episodio empieza en un paso aleatorio válido en lugar de siempre al inicio. Diversifica
  los estados iniciales de entrenamiento y combate el sobreajuste a la trayectoria fija.
  Se activa en entrenamiento con `cfg.random_start_train`.

---

## 4. Validación interna temporal (selección de modelo sin tocar OOS)

Para no seleccionar el modelo mirando el OOS, se añadió un **split de validación interno
al IS** (`use_internal_validation`):

- `split_internal_validation(IS, val_ratio, lookback)` parte el IS en `IS_train` / `IS_val`
  cronológicamente (sin barajar).
- La normalización z-score se ajusta en `IS_train` y se aplica a `IS_val` y OOS
  (`normalize_train_val_oos`) → **sin leakage**.
- Durante el entrenamiento, cada `validation_eval_freq` episodios se evalúa en `IS_val` y
  se calcula un **score de validación**:
  `score = Sharpe − w_mdd·|MDD| − w_trade·(trades/steps)`.
- Se guarda el **mejor checkpoint** según ese score (con `validation_patience` opcional
  para early stopping). Al terminar, el agente se restaura a ese mejor estado. Las métricas
  del mejor punto se exportan en `validation_metrics.csv`.

El estado del agente se captura/restaura en memoria (`_capture_agent_state` /
`_restore_agent_state`: encoder, q_network, target_network, optimizer, epsilon,
update_step), evitando E/S a disco durante la validación.

---

## 5. Optimización: precomputar latentes de encoders congelados

Para brazos C y D el encoder **no cambia** durante el RL, así que `encoder(obs)` produce
el mismo latente en cada época. La optimización (`FrozenEncoderContext` +
`LatentObservationWrapper`) lo aprovecha **sin alterar los resultados numéricos**:

1. `prepare_arm_c_encoder` entrena el encoder ligero de C una vez por pliegue y lo congela;
   el de D ya viene congelado del artefacto.
2. `precompute_latent_series(data, L, features, encoder, device)` recorre toda la serie
   replicando **exactamente** la lógica de `FinancialEnv._get_observation()` (ventana
   deslizante + zero-padding inicial) y devuelve `(T, latent_dim)`.
3. `LatentObservationWrapper` envuelve el entorno y, en `reset`/`step`, devuelve el latente
   precomputado en `env.current_step` en vez de la observación cruda. El espacio de
   observación pasa de `(L, F)` a `(latent_dim,)`.
4. El agente se crea en modo `precomputed_latents=True` (sin encoder interno) → el TCN/GRU
   **no se ejecuta** en cada `step`/`select_action` del bucle de RL.

Esto reduce el coste por experimento (latentes calculados una vez por ticker/pliegue,
reutilizados en todas las semillas) manteniendo equivalencia matemática. Si no hay
`frozen_ctx`, los brazos C/D caen a la ruta original (encoder dentro del agente).

---

## 6. Walk-Forward Analysis

Alternativa al split simple IS/OOS, activada con `wf_enabled`. Implementada en
`walk_forward_splits` (`experiments/utils.py`) con estrategia **expansiva / anclada**:

- El IS **siempre empieza en la fila 0** y crece en cada ventana ("entrena con todo el
  pasado, evalúa el periodo siguiente"). `wf_mode="expanding"` (sliding reservado a futuro).
- **Semántica de `wf_is_ratio`** (cambiada respecto a la versión anterior): ahora es la
  *fracción inicial de anclaje*. `anchor_end = floor(n · is_ratio)` es el fin del IS de la
  primera ventana; el resto se parte en `wf_n_windows` bloques OOS contiguos e iguales.
- Ventana `k`: `IS = data[0:oos_start]`, `OOS = data[oos_start:oos_end]`, sin solapamiento
  de OOS.
- Valida que el ancla y cada bloque OOS tengan filas suficientes dados `lookback` y
  `min_oos_steps`; si no, error explicativo.

Flujo por ventana (`_run_single_ticker` con `wf_enabled`):
1. Prepara splits train/val/eval sin fuga.
2. Construye `frozen_ctx` por ventana (el encoder de **C se reentrena** porque el IS crece;
   el de **D no**, su anti-fuga ya está garantizada por el corpus disjunto).
3. Ejecuta todas las semillas, agrega entre semillas → `window_agg`.
4. Calcula el IVL crudo de la ventana.

Agregación entre ventanas: `_aggregate_wf_windows` promedia las métricas (ya agregadas por
semilla) entre ventanas. `_export_wf_window_metrics` escribe `wf_window_metrics.csv` con una
fila por `(ventana, agente)`, incluyendo el IVL de cada ventana.

> Nota: la run robusta `results/abcd_robust_h64` se ejecutó con **split simple IS/OOS 70/30**
> (`wf_enabled=False`, `window=-1` en los CSV de IVL). El walk-forward queda disponible y
> probado (`tests/test_walkforward.py`) para corridas posteriores.

---

## 7. IVL v2 (Índice de Ventaja Latente)

`LatentAdvantageIndex` (`evaluation/latent_advantage.py`). Mide si un agente latente supera
al directo combinando cuatro componentes, **todas sobre OOS** y acotadas:

```
IVL = w_sharpe·ΔSharpe − w_mdd·ΔMDD − w_seed_std·Δσ_seeds − w_gap·ΔGap
```

| Componente | Definición | Signo |
|---|---|---|
| ΔSharpe | `Sharpe_OOS(lat) − Sharpe_OOS(dir)` | + (más Sharpe suma) |
| ΔMDD | `|MDD_OOS(lat)| − |MDD_OOS(dir)|` | − (menos drawdown suma) |
| Δσ_seeds | `σ_seeds(Sharpe_OOS)(lat) − …(dir)` | − (menos varianza entre semillas suma) |
| ΔGap | `gap(lat) − gap(dir)`, `gap = |Sharpe_IS − Sharpe_OOS|` | − (menos brecha IS/OOS suma) |

Pesos por defecto: `{sharpe:0.35, mdd:0.25, seed_std:0.20, is_oos_gap:0.20}` (deben sumar 1;
validado en `ExperimentConfig`).

Cambios clave de la v2:

1. **Métricas OOS y acotadas**: usa Sharpe/MDD OOS (no IS) y la dispersión entre semillas,
   alineando el índice con el rendimiento fuera de muestra y la **robustez**.
2. **Normalización por componente a nivel de experimento**: antes de ponderar, cada Δ se
   divide por su **escala** = `std agrupado de esa componente sobre todas las comparaciones
   (ticker × ventana)`. Si la escala < 1e-8, esa componente se fija a 0. Así las cuatro
   componentes son comparables aunque vivan en rangos distintos.
   - `compute(direct, latent, scales=None)`: sin `scales` usa deltas crudos (tests y
     comparaciones aisladas); con `scales` aplica la normalización.
   - En el pipeline, `compute_ivl` recoge **deltas crudos** por ticker; `_finalize_ivl_records`
     calcula las escalas cross-experimento y el IVL final una vez se tienen todos los pares;
     `run_experiment` lo distribuye de vuelta a cada ticker y exporta `ivl_results.csv`.
3. **Interpretación**: `latent_advantage` / `direct_advantage` / `neutral` según `±ε`.

Los CSV de IVL incluyen tanto los deltas crudos como los normalizados, el IVL final, la
interpretación y el contexto OOS (Sharpe/return IS y OOS de ambos agentes) para el dashboard.

---

## 8. Orquestación

- `run_single_seed`: una semilla → construye entornos IS/OOS/train/val, evalúa referencias,
  entrena cada brazo solicitado (usando `frozen_ctx` para C/D si está), devuelve métricas
  IS/OOS (+ validación).
- `_run_single_ticker`: prepara datos y splits, construye `frozen_ctx`, recorre semillas,
  agrega, exporta resultados del dashboard y recoge IVL crudo. Soporta walk-forward.
- `run_experiment`: carga el encoder gordo de D una vez para todos los tickers, recorre
  tickers, **normaliza el IVL cross-ticker**, exporta por ticker y, si hay >1 ticker,
  exporta la comparación cross-ticker.

## Tests relevantes añadidos/ampliados

`tests/test_walkforward.py`, `tests/test_precompute_latents.py`,
`tests/test_encoder_pretrainer.py`, `tests/test_offline_pretrain.py`,
`tests/test_representations.py`, `tests/test_envs.py`, `tests/test_latent_dqn_agent.py`,
`tests/test_latent_advantage.py`.
