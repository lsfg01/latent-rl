# Elección y evaluación del encoder latente

## 1. Decisión experimental

El encoder transforma una ventana financiera de forma `(30 sesiones, 8 variables)` en
un vector latente de 32 dimensiones. Su función no es decidir la acción, sino determinar
qué información del estado queda disponible para que la política la aprenda.

La pregunta relevante no es “qué arquitectura es más sofisticada”, sino:

> ¿Qué sesgo inductivo produce una representación más transferible, estable y útil para
> el aprendizaje RL fuera de muestra bajo un presupuesto comparable?

La campaña robusta utiliza una TCN como arquitectura principal. MLP y GRU permanecen como
baselines y ablaciones necesarias para saber si una posible mejora procede del modelado
temporal o simplemente de cambiar capacidad y número de parámetros.

---

## 2. Alternativas arquitectónicas

| Arquitectura | Supuesto principal | Ventajas | Riesgos | Papel experimental |
|---|---|---|---|---|
| **MLP** | Cada posición de la ventana puede tratarse como una coordenada del vector | Simple, rápida y fácil de controlar | No representa explícitamente localidad ni orden temporal | Baseline de compresión sin sesgo temporal fuerte |
| **TCN causal** | Los patrones locales a distintas escalas son informativos | Paralelizable, campo receptivo explícito, causalidad clara | Depende de kernel/dilataciones; puede sobredimensionarse | Arquitectura principal de la campaña robusta |
| **GRU** | Un estado recurrente puede resumir dependencias secuenciales | Longitud flexible y memoria temporal aprendida | Entrenamiento secuencial, mayor sensibilidad y coste | Ablación recurrente frente a la TCN |

### MLP

El MLP aplana las 30 × 8 entradas. Puede aprender interacciones entre cualquier par de
posiciones, pero no incorpora que dos días contiguos están más relacionados que dos
posiciones alejadas. Si un MLP iguala a TCN/GRU bajo capacidad comparable, el beneficio
del encoder podría provenir principalmente del cuello de botella y no del modelado
temporal.

### TCN causal

La TCN aplica convoluciones que solo usan el presente y el pasado. Las dilataciones
permiten combinar patrones de pocos días con señales distribuidas por toda la ventana.
Su campo receptivo es verificable antes de entrenar:

```text
RF = 1 + (kernel_size − 1) · Σ dilataciones
```

Con kernel 3 y dilataciones `[1,2,4,8]`, `RF=31`, suficiente para cubrir una ventana de
30 sesiones. Una TCN con `RF<lookback` no observa conjuntamente toda la historia y cambia
la hipótesis experimental, aunque el modelo siga ejecutándose.

### GRU

La GRU actualiza un estado al recorrer la ventana. Puede aprender qué información
recordar u olvidar, sin fijar manualmente escalas temporales. A cambio, su optimización es
menos paralela y su estado final puede dar un peso excesivo a las observaciones más
recientes. Debe compararse con varias semillas y un presupuesto equivalente.

---

## 3. Configuración de referencia y razones

| Parámetro | Valor | Qué determina | Justificación actual |
|---|---:|---|---|
| `encoder_type` | `tcn` | Sesgo arquitectónico | Causalidad y cobertura temporal explícita. |
| `lookback_window` | 30 | Historia disponible | Aproximadamente un mes y medio de sesiones. |
| Número de variables | 8 | Anchura informativa por sesión | Combina retorno, volatilidad, momentum, volumen y régimen. |
| `latent_dim` | 32 | Capacidad del cuello de botella | Reduce 240 entradas a 32 valores sin imponer una compresión extrema. |
| `tcn_kernel_size` | 3 | Extensión local de cada filtro | Captura patrones cortos con coste moderado. |
| `tcn_dilations` | `[1,2,4,8]` | Escalas y campo receptivo | Cubre la ventana completa con crecimiento exponencial. |
| `tcn_channels` | 64 | Capacidad interna | Da margen al preentrenamiento offline sobre un corpus amplio. |
| Activación | ReLU | Forma de la no linealidad | Estable, sencilla y fácil de reproducir. |
| Dropout del encoder | 0.0 | Regularización estocástica | Se mantiene fuera de la campaña principal al no estar calibrado. |

La compresión es aproximadamente `240 / 32 = 7,5`. Una dimensión de 32 no es un óptimo
teórico: pretende ser lo bastante pequeña para forzar síntesis y lo bastante grande para
retener señales heterogéneas.

La TCN de 64 canales se utiliza tanto en C como en D en la campaña robusta. La etiqueta
“offline/heavy” de D describe sobre todo un preentrenamiento más amplio —39 activos y un
presupuesto mayor—, no una arquitectura distinta. Esto hace que C frente a D compare con
mayor limpieza especialización IS y transferencia offline.

---

## 4. Qué hace cada parámetro y cómo decidir su ajuste

### Ventana temporal: `lookback_window`

- **Más corta**: reacciona a cambios recientes y reduce complejidad, pero puede perder
  régimen y persistencia.
- **Más larga**: aporta contexto, aunque mezcla patrones de distintas épocas y exige más
  capacidad.
- **Diagnóstico**: comparar 10/20/30/60 sesiones manteniendo el campo receptivo completo y
  reportar rendimiento por régimen, no solo promedio.

### Dimensión latente: `latent_dim`

- **Más pequeña**: regulariza y facilita la cabeza Q; puede eliminar señales útiles.
- **Más grande**: conserva más información; puede convertirse en una transformación casi
  identitaria y aumentar la varianza RL.
- **Diagnóstico**: evaluar reconstrucción, forecasting y resultados OOS para 16/32/64. Una
  menor pérdida de reconstrucción no basta para preferir una dimensión.

### Canales TCN: `tcn_channels`

- Controlan cuántos patrones puede representar cada escala temporal.
- Aumentarlos eleva capacidad y coste; también puede favorecer a TCN frente a MLP/GRU por
  puro número de parámetros.
- Una comparación arquitectónica rigurosa debería informar parámetros entrenables y, si
  es posible, incluir una variante aproximadamente igualada en tamaño.

### Kernel: `tcn_kernel_size`

- Un kernel mayor mezcla más sesiones consecutivas en cada bloque.
- Aumenta el campo receptivo, pero puede suavizar patrones cortos y elevar coste.
- Debe elegirse junto con las dilataciones; cambiar uno sin recalcular `RF` puede generar
  una cobertura accidentalmente distinta.

### Dilataciones: `tcn_dilations`

- Determinan las escalas temporales observadas.
- Una progresión exponencial cubre historia larga con pocos bloques.
- Añadir niveles más allá de la ventana no aporta nueva historia y sí añade capacidad.
- El criterio mínimo es `RF ≥ lookback`; después debe validarse si la capacidad adicional
  mejora OOS y no solo el objetivo auxiliar.

### Dropout del encoder

- Puede reducir coadaptación en corpus pequeños.
- En encoders congelados, el ruido durante preentrenamiento cambia los pesos finales pero
  no actúa durante RL.
- Debe ajustarse en el protocolo de preentrenamiento, no a partir del OOS del agente.

### Parámetros GRU

| Parámetro | Función | Riesgo al aumentarlo |
|---|---|---|
| `gru_hidden_dim` | Capacidad del estado recurrente | Más coste y posible memorización del corpus. |
| `gru_num_layers` | Profundidad temporal | Optimización más difícil y mayor sensibilidad a regularización. |

### Capas ocultas MLP

La anchura y profundidad del MLP determinan si actúa como baseline simple o como modelo
con muchos más parámetros que la TCN. Deben declararse junto con el número total de
parámetros; “MLP frente a TCN” no es una comparación interpretable si una arquitectura
tiene mucha más capacidad.

---

## 5. Entrenar conjuntamente o congelar

El estado del encoder durante RL es una decisión experimental independiente de su
arquitectura.

| Estrategia | Ventaja potencial | Riesgo | Brazo de referencia |
|---|---|---|---|
| **Finetuning conjunto** | La representación se adapta a la recompensa | Señal RL ruidosa, representación inestable y mayor varianza | B |
| **Preentrenado y congelado en IS** | Separa aprendizaje de representación y política | Puede conservar objetivos auxiliares irrelevantes para trading | C |
| **Preentrenado offline y congelado** | Facilita transferencia y reutilización | Desajuste de dominio entre corpus y activo objetivo | D |

Congelar reduce grados de libertad durante RL, pero no garantiza robustez. Un encoder fijo
puede imponer una representación inadecuada que la política no puede corregir. Del mismo
modo, que B aprenda encoder y política conjuntamente no implica que su latente sea
interpretable o estable.

Para aislar el efecto de congelar, sería necesaria una ablación adicional con el mismo
encoder preentrenado evaluado una vez congelado y otra con finetuning. Los brazos actuales
no aíslan completamente esa variable.

---

## 6. Diseño de una comparación arquitectónica justa

Al comparar MLP, TCN y GRU deben mantenerse constantes:

- activos, fechas, split temporal y normalización;
- variables y orden de entrada;
- ventana y dimensión latente;
- corpus y objetivo de preentrenamiento;
- semillas offline y semillas RL;
- cabeza Q, exploración, presupuesto y selección de checkpoint;
- estado congelado o finetuneado del encoder.

También se deben reportar:

- número de parámetros y tiempo de entrenamiento;
- pérdida de validación de reconstrucción y forecasting;
- Sharpe, drawdown, retorno, dispersión y gap IS/OOS downstream;
- resultado por ticker, no solo media agregada.

Una arquitectura puede ser preferible por estabilidad o coste aunque su media de Sharpe
sea ligeramente inferior. La decisión debe reflejar el objetivo del experimento y no una
única métrica.

### Matriz mínima recomendada

| Factor | Valores |
|---|---|
| Arquitectura | MLP, TCN, GRU |
| Dimensión latente | 16, 32, 64 |
| Estado durante RL | Congelado, finetuning |
| Objetivo auxiliar | Reconstrucción; reconstrucción + forecasting |
| Semillas offline | Al menos 3 |
| Semillas RL | Las mismas 5 por condición |

No es necesario ejecutar el producto cartesiano completo desde el principio. Un diseño
por etapas puede fijar `latent_dim=32`, comparar arquitecturas, y después estudiar
capacidad solo en las alternativas prometedoras. Las etapas y criterios de descarte deben
definirse antes de observar OOS.

---

## 7. Artefacto como contrato experimental

Un encoder preentrenado no queda identificado únicamente por sus pesos. Para reproducir e
interpretar una comparación deben conservarse:

- arquitectura y capacidad;
- ventana, número de variables y orden exacto de features;
- estadísticas de normalización;
- corpus, exclusiones y fechas;
- objetivo auxiliar y parámetros de entrenamiento;
- semilla y criterio de selección del checkpoint.

Antes de reutilizar un encoder debe comprobarse que ventana, variables y orden coinciden,
y debe declararse la política de normalización. Una coincidencia de dimensiones no es
suficiente: dos conjuntos distintos de ocho variables producirían tensores con la misma
forma y significado incompatible. La campaña actual normaliza cada activo objetivo con
su `IS_train`; aplicar las estadísticas del corpus offline sería una condición
experimental distinta, no un detalle intercambiable.

Cambiar la cabeza Q no obliga a regenerar el encoder. Cambiar entrada, arquitectura,
corpus, objetivo o normalización sí define una representación experimental nueva y debe
producir un artefacto nuevo con trazabilidad independiente.

---

## 8. Criterios de decisión y límites

La TCN se mantiene como elección principal cuando:

- cubre toda la ventana de forma causal;
- mejora o mantiene métricas OOS frente a MLP/GRU;
- no aumenta de forma material la dispersión entre semillas;
- su coste adicional es aceptable;
- el resultado no depende de un único ticker.

Debe reconsiderarse cuando una alternativa más simple iguala su rendimiento, cuando la
ventaja desaparece al igualar capacidad o cuando el resultado se concentra en el objetivo
de reconstrucción pero no llega a la política RL.

Limitaciones actuales:

- la elección TCN no procede de un sweep arquitectónico exhaustivo;
- el encoder offline de referencia usa una sola semilla de preentrenamiento;
- `latent_dim=32` y `channels=64` son decisiones razonadas, no óptimos demostrados;
- C y D difieren en corpus y escala de preentrenamiento, por lo que su comparación no
  identifica un único efecto;
- una representación transferible entre activos con fechas solapadas no equivale a una
  validación temporal de despliegue.

Para las decisiones de corpus y objetivo multitarea, consultar
`OFFLINE_PRETRAINING.md`. Para el protocolo de comparación y las métricas, consultar
`TRAINING_EVAL_PROTOCOL.md`.
