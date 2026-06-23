# Protocolo experimental de entrenamiento y evaluación

## 1. Pregunta experimental

El experimento estudia si una representación latente del estado de mercado mejora la
generalización de un agente DQN financiero frente a usar directamente la ventana de
variables observadas.

La comparación no busca únicamente el mayor retorno histórico. Una representación se
considera útil si mantiene un rendimiento ajustado por riesgo competitivo fuera de
muestra, limita el drawdown, reduce la sensibilidad a la semilla y evita una degradación
excesiva entre entrenamiento y evaluación.

La campaña de referencia es `results/abcd_robust_h64`. Utiliza SPY, TSLA y BTC-USD para
contrastar tres perfiles de volatilidad y dinámica de mercado. Sus resultados son
evidencia descriptiva sobre estos activos y este periodo; no demuestran superioridad
universal ni rentabilidad futura.

---

## 2. Brazos A/B/C/D: qué hipótesis aísla cada comparación

| Brazo | Representación utilizada | Estado del encoder durante RL | Pregunta que responde |
|---|---|---|---|
| **A · DirectDQN** | Ventana financiera completa | No usa encoder | ¿Qué consigue el DQN sin cuello de botella latente? |
| **B · LatentDQN-FT** | Encoder inicializado al azar | Encoder y política se entrenan conjuntamente | ¿Comprimir y aprender la representación desde la recompensa RL aporta algo por sí solo? |
| **C · LatentDQN-IS-Frozen** | Encoder preentrenado con `IS_train` del activo | Congelado durante RL | ¿El preentrenamiento específico del activo mejora la representación sin adaptar el encoder a la recompensa? |
| **D · LatentDQN-Offline-Frozen** | Encoder preentrenado sobre un universo externo | Congelado durante RL | ¿Una representación transferida y no específica del activo generaliza mejor? |

Lectura recomendada:

- **A frente a B** combina dos efectos: compresión latente y aprendizaje conjunto. No
  permite atribuir una mejora exclusivamente al cuello de botella.
- **B frente a C** cambia tanto el origen de los pesos como el hecho de congelar el
  encoder. Una diferencia no debe interpretarse como efecto puro del preentrenamiento.
- **C frente a D** es la comparación más informativa sobre especialización frente a
  transferencia, porque ambos encoders permanecen congelados durante RL.
- **Random** y **Buy & Hold** son referencias de contexto, no brazos equivalentes para
  inferir el efecto causal de una representación.

Para que la comparación sea razonablemente justa, A y las cabezas Q de B/C/D comparten
tasa de aprendizaje, descuento, capacidad, replay buffer, exploración y regularización.
Así se reduce el riesgo de atribuir al encoder una ventaja producida por una política con
más capacidad o un optimizador distinto.

En la campaña de referencia, B, C y D también comparten tipo y capacidad de encoder. La
diferencia entre C y D es el corpus y presupuesto de preentrenamiento, no el tamaño de la
arquitectura.

---

## 3. Partición temporal y control de fuga de información

### Campaña principal

| Decisión | Valor | Motivo experimental |
|---|---:|---|
| Frecuencia | Diaria | Reduce ruido microestructural y mantiene un histórico suficientemente largo. |
| Periodo | 2015-01-01 a 2024-01-01 | Incluye regímenes alcistas, bajistas y episodios de alta volatilidad. |
| Split principal | 70% IS / 30% OOS | Reserva una fracción OOS material sin dejar demasiado poco histórico para aprender. |
| Validación interna | 20% del IS | Permite seleccionar checkpoint sin consultar OOS. |
| Orden temporal | Sin barajar | Preserva causalidad y reproduce el uso real de pasado para evaluar futuro. |

Con estas proporciones, aproximadamente el 56% del histórico total se usa para
entrenamiento efectivo, el 14% para validación y el 30% para evaluación OOS. Las
fronteras exactas dependen del número de observaciones válidas después de construir
indicadores y ventanas.

La normalización se ajusta solo con el tramo de entrenamiento y se reutiliza en
validación y OOS. Ajustarla con todo el histórico revelaría indirectamente la media y la
escala del futuro. Esta normalización por activo se aplica a todos los brazos, incluido D,
para que la política compare entradas con la misma escala y sin usar estadísticas OOS.

### Qué controla y qué no controla este diseño

El split temporal evita seleccionar el agente con métricas OOS y previene el leakage de
normalización. Sin embargo, una única frontera 70/30 puede favorecer o perjudicar un
régimen concreto. Por eso el resultado principal debe acompañarse de una comprobación
walk-forward cuando el presupuesto computacional lo permita.

---

## 4. Estado de mercado: ventana y variables

Cada observación resume **30 sesiones** mediante 8 variables, por lo que el estado bruto
tiene 240 valores antes de cualquier compresión latente. Treinta sesiones representan
aproximadamente un mes y medio bursátil: suficiente para patrones recientes sin exigir
al agente memorizar periodos muy largos.

| Variable | Información que aporta | Riesgo o redundancia |
|---|---|---|
| `log_return` | Dirección y magnitud relativa del movimiento diario | Muy ruidoso; también es el objetivo auxiliar de forecasting. |
| `high_low_range` | Amplitud intradía relativa | Puede solaparse con ATR, pero responde más al día actual. |
| `close_open_pct` | Dirección y fuerza de la vela | Sensible a gaps y eventos puntuales. |
| `volume_ratio` | Actividad respecto a su media de 20 días | El volumen no es comparable en escala absoluta entre activos. |
| `rsi_14` | Momentum de corto plazo | Indicador suavizado; puede reaccionar tarde. |
| `atr_pct` | Volatilidad relativa de 14 días | Resume riesgo reciente, no anticipa shocks. |
| `market_regime` | Tendencia discreta según MA50/MA200 | Pierde matices y necesita bastante historia. |
| `ma_ratio` | Intensidad continua del régimen MA50/MA200 | Está correlacionado con `market_regime`. |

La combinación mezcla retorno, volatilidad, momentum, volumen y régimen. La presencia
de variables relacionadas es deliberada para comparar si el encoder aprende a condensar
información redundante. Como análisis de sensibilidad, conviene repetir con subconjuntos
por familia antes de concluir que las ocho variables son necesarias.

### Sensibilidad de `lookback_window`

- Un valor **menor** enfatiza reacción rápida, reduce dimensionalidad y puede perder
  contexto de régimen.
- Un valor **mayor** incorpora más historia, pero aumenta la capacidad necesaria y el
  riesgo de ajustar patrones antiguos.
- Al cambiarlo debe revisarse el campo receptivo del encoder y regenerarse cualquier
  representación preentrenada.

---

## 5. Entorno y señal de recompensa

| Parámetro | Valor | Qué controla | Razón de la elección |
|---|---:|---|---|
| `initial_balance` | 10 000 | Escala inicial del portfolio | Valor interpretable; con log-retornos no domina la escala de la recompensa. |
| `transaction_cost` | 0.001 | Coste del 0,1% sobre cada operación | Evita estrategias de rotación gratuita y aproxima fricción de mercado. |
| `reward_mode` | `log_return` | Señal paso a paso sobre el cambio relativo de equity | Es aditiva en el tiempo y comparable entre niveles de capital. |
| `random_start_train` | `True` | Diversidad de segmentos vistos durante entrenamiento | Reduce memorización de una única trayectoria de inicio. |
| `max_steps_per_episode` | 500 | Longitud máxima del segmento de entrenamiento | Equilibra contexto temporal y número de episodios distintos. |
| `reward_clip` | `None` | Recorte de recompensas extremas | Se preservan colas y shocks; se evita introducir un umbral aún no calibrado. |
| `trade_penalty` | 0.0 | Penalización adicional por operación | Se mantiene desactivada para que la fricción provenga del coste transaccional y no confunda la comparación de representaciones. |

Decisiones de ajuste:

- Elevar `transaction_cost` prueba robustez a escenarios de ejecución más exigentes; no
  debe elegirse después de observar qué valor favorece al agente preferido.
- Activar `reward_clip` puede estabilizar gradientes cuando unas pocas sesiones dominan
  el aprendizaje, pero también elimina información sobre eventos extremos.
- Añadir `trade_penalty` solo está justificado si se desea modelar fricciones no
  capturadas por el coste proporcional. Debe analizarse junto con el número de trades.
- Acortar `max_steps_per_episode` aumenta variedad de segmentos, aunque reduce el
  horizonte continuo que experimenta cada política.

---

## 6. Presupuesto y dinámica de aprendizaje DQN

Los parámetros siguientes son comunes a A y a las cabezas Q latentes.

| Parámetro | Valor | Función experimental | Efecto esperado al aumentarlo |
|---|---:|---|---|
| Semillas | 5 (`0–4`) | Estimar sensibilidad a inicialización y exploración | Mejora la estimación de dispersión; aumenta el coste linealmente. |
| Episodios de entrenamiento | 30 | Presupuesto de interacción por agente y semilla | Puede mejorar convergencia, pero también sobreajustar IS. |
| Episodios de evaluación | 3 | Repeticiones de evaluación por split | Solo aportan información adicional si existe variación en la evaluación. |
| `learning_rate` | 5e-4 | Tamaño de las actualizaciones | Aprende más rápido, con mayor riesgo de inestabilidad. |
| `gamma` | 0.99 | Peso del beneficio futuro | Amplía el horizonte efectivo, pero aumenta varianza del objetivo. |
| `hidden_dim` | 64 | Capacidad de la cabeza Q | Modela relaciones más complejas, con mayor riesgo de varianza y sobreajuste. |
| `batch_size` | 64 | Muestras por actualización | Gradientes más suaves, menor frecuencia efectiva de actualización. |
| `buffer_capacity` | 5 000 | Diversidad de transiciones recordadas | Más diversidad, pero conserva experiencias posiblemente obsoletas. |
| `target_update` | 100 | Frecuencia de sincronización de la red objetivo | Objetivo más estable pero más retrasado. |
| `epsilon_start` | 1.0 | Exploración inicial | Favorece cobertura del espacio de acciones. |
| `epsilon_end` | 0.1 | Exploración residual mínima | Evita una política totalmente determinista durante entrenamiento. |
| `epsilon_decay` | 0.998 | Velocidad de transición de explorar a explotar | Más próximo a 1 mantiene exploración durante más tiempo. |
| `weight_decay` | 1e-4 | Regularización L2 | Reduce pesos extremos; demasiado valor puede infraajustar. |
| `grad_clip_norm` | 1.0 | Límite de la norma del gradiente | Reduce actualizaciones explosivas; un umbral bajo frena el aprendizaje. |
| `dropout` | 0.0 | Ruido regularizador en la cabeza Q | Está desactivado para no añadir otra fuente estocástica en la comparación principal. |

### Por qué se eligió `hidden_dim=64`

El sweep previo A-only comparó 64 y 128 unidades. La configuración de 128 obtuvo mejor
Sharpe OOS medio en promedio, mientras que 64 mostró menor variabilidad entre semillas.
Se eligió 64 para la campaña A/B/C/D robusta priorizando estabilidad y un control de
capacidad más conservador. Esta decisión no significa que 64 domine a 128; de hecho, la
comparación debe reportarse como un compromiso entre nivel medio y dispersión.

### Orden recomendado para ajustar

1. Confirmar estabilidad numérica con `learning_rate` y `grad_clip_norm`.
2. Revisar cobertura de exploración con la trayectoria de epsilon y la diversidad de
   acciones.
3. Ajustar capacidad (`hidden_dim`) y regularización usando validación interna.
4. Aumentar episodios solo si la curva de validación aún mejora; más entrenamiento no es
   automáticamente mejor.

Cambiar varios elementos a la vez impide atribuir la mejora. Las decisiones deben
tomarse con el mismo conjunto de semillas y sin consultar OOS.

---

## 7. Selección de checkpoint con validación interna

El 20% final del IS se reserva cronológicamente para validación. Cada 5 episodios se
calcula:

```text
score_validación = Sharpe − 0.25 · |MDD| − 0.05 · (trades / steps)
```

| Parámetro | Valor | Interpretación |
|---|---:|---|
| `internal_val_ratio` | 0.20 | Cuánta historia IS se sacrifica para seleccionar el modelo. |
| `validation_eval_freq` | 5 | Compromiso entre resolución de selección y coste. |
| `validation_patience` | `None` | Se consume el presupuesto completo y al final se recupera el mejor checkpoint. |
| Peso de MDD | 0.25 | Penaliza políticas con caídas profundas. |
| Peso de trading | 0.05 | Penaliza actividad excesiva de forma secundaria. |

Los pesos expresan una preferencia de selección, no una verdad estadística. Como Sharpe,
MDD y tasa de trading tienen escalas distintas, conviene comprobar que ningún término
domina sistemáticamente el score. La tabla `validation_metrics.csv` permite revisar en
qué episodio se seleccionó cada agente y si la elección es estable entre semillas.

OOS permanece reservado para la comparación final. Elegir pesos, arquitectura o número
de episodios después de observar OOS convertiría ese tramo en validación implícita.

---

## 8. Métricas de evaluación

### Prioridad de lectura

1. **Sharpe OOS**: métrica principal de calidad riesgo-retorno fuera de muestra.
2. **Max drawdown OOS**: severidad de la peor caída; evita premiar Sharpe a costa de un
   riesgo difícil de tolerar.
3. **Retorno OOS**: magnitud económica, interpretada junto con riesgo y costes.
4. **Dispersión entre semillas**: robustez frente a inicialización y exploración.
5. **Gap IS/OOS**: señal operativa de degradación o sobreajuste.
6. **Número de trades**: ayuda a detectar políticas triviales o sobreoperativas.

Cinco semillas permiten describir media y dispersión, pero son pocas para afirmaciones
inferenciales fuertes. No debe declararse superioridad por una diferencia pequeña de
media sin revisar la distribución por semilla y ticker.

### Índice de Ventaja Latente (IVL)

El IVL resume cuatro diferencias entre un brazo latente y A:

```text
IVL = 0.35·ΔSharpe − 0.25·ΔMDD − 0.20·ΔSeedStd − 0.20·ΔGap
```

| Componente | Definición | Lectura favorable al latente |
|---|---|---|
| `ΔSharpe` | Sharpe OOS latente − directo | Positivo |
| `ΔMDD` | `abs(MDD OOS latente) − abs(MDD OOS directo)` | Negativo |
| `ΔSeedStd` | Dispersión del Sharpe OOS latente − directo | Negativo |
| `ΔGap` | Gap IS/OOS latente − directo | Negativo |

Los deltas se normalizan por su escala dentro de la campaña antes de ponderarse. Esto
evita que una magnitud domine solo por sus unidades, pero hace que el valor absoluto del
IVL dependa del conjunto de tickers y comparaciones incluido. Por ello:

- se debe inspeccionar cada componente además del índice agregado;
- los IVL de campañas con universos distintos no son directamente comparables;
- los pesos deben fijarse antes de mirar los resultados finales;
- un IVL positivo describe ventaja bajo esta función de preferencia, no significación
  estadística ni rentabilidad garantizada.

---

## 9. Walk-forward como análisis de sensibilidad

La campaña robusta usa un único split 70/30. El walk-forward queda reservado como prueba
de estabilidad temporal con ventanas expansivas:

| Parámetro | Valor de referencia | Qué controla |
|---|---:|---|
| `wf_n_windows` | 5 | Número de periodos OOS consecutivos. |
| `wf_is_ratio` | 0.60 | Tamaño del ancla inicial de entrenamiento. |
| `wf_mode` | `expanding` | Cada ventana incorpora toda la historia disponible. |

Más ventanas dan una visión más granular de los regímenes, pero acortan cada tramo OOS y
aumentan la varianza de sus métricas. Un ancla mayor mejora el entrenamiento inicial a
costa de reducir el número de observaciones futuras evaluables.

En cada ventana, cualquier encoder específico del activo debe entrenarse solo con la
información disponible hasta esa frontera. El encoder offline requiere una interpretación
separada de su periodo de preentrenamiento, descrita en `OFFLINE_PRETRAINING.md`.

---

## 10. Reglas para interpretar y extender el protocolo

- Fijar hipótesis, métrica principal y rangos de parámetros antes de consultar OOS.
- Comparar una decisión cada vez y mantener tickers, semillas y presupuesto constantes.
- Reportar nivel medio, dispersión y comportamiento por activo; el promedio puede ocultar
  contraejemplos.
- Tratar Buy & Hold como referencia económica y Random como control de cordura, no como
  sustitutos del brazo A.
- Conservar `experiment_config.json` junto a cada run para distinguir parámetros
  planificados de parámetros realmente ejecutados.
- Si se modifica ventana, orden de variables o arquitectura del encoder, invalidar y
  regenerar los artefactos preentrenados correspondientes.
- No presentar el mejor resultado de un sweep como estimación imparcial del rendimiento;
  necesita una evaluación final no utilizada durante la selección.

Documentos complementarios:

- `ENCODER_UPGRADE.md`: elección de arquitectura y capacidad de la representación.
- `OFFLINE_PRETRAINING.md`: decisiones del corpus y objetivo del encoder del brazo D.
- `RUN_ABCD_ROBUST_H64.md`: ejecución y trazabilidad operativa de la campaña concreta.
