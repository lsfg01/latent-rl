# Decisiones experimentales del preentrenamiento offline

## 1. Papel del encoder offline en el experimento

El brazo D utiliza una representación entrenada antes del aprendizaje por refuerzo y
congelada durante RL. La hipótesis es que un encoder aprendido sobre varios activos puede
capturar patrones transferibles de retorno, volatilidad, momentum y régimen, reduciendo
la dependencia del agente respecto al histórico específico del activo evaluado.

Esta fase no optimiza directamente Sharpe ni retorno del agente. Su objetivo es construir
una representación general del estado de mercado. La utilidad real del encoder debe
evaluarse después, mediante el rendimiento OOS y la estabilidad del brazo D frente a A,
B y C.

La comparación más informativa es **C frente a D**:

- C aprende una representación específica del activo usando únicamente `IS_train`.
- D aprende una representación común usando activos externos y la mantiene fija.

Si D supera a C de forma consistente, existe evidencia favorable a la transferencia. Si
C supera a D, la especialización por activo puede ser más importante que la escala del
corpus. Un resultado mixto por ticker indica que la ventaja de transferencia depende del
mercado y no debe resumirse como universal.

---

## 2. Elección del corpus

### Configuración de la campaña

| Decisión | Valor | Justificación |
|---|---:|---|
| Universo solicitado | 39 activos | Aumentar diversidad sin convertir el preentrenamiento en una búsqueda masiva. |
| Periodo | 2010-01-01 a 2023-12-31 | Incluir varios regímenes de mercado y suficiente historia por activo. |
| Frecuencia | Diaria | Coincidir con la escala temporal de la evaluación RL. |
| Longitud mínima | 250 observaciones | Evitar activos con muy pocas ventanas útiles. |
| Activos objetivo excluidos | SPY, TSLA, BTC-USD | Impedir que el encoder vea directamente las series evaluadas. |
| Proxies excluidos | IVV, VOO, BTC-EUR, ETH-USD | Reducir transferencia casi directa desde instrumentos estrechamente relacionados. |

El universo combina tecnología, finanzas, salud, consumo, industria, energía,
comunicación y algunos ETFs. Esta variedad busca que el encoder no dependa de un único
sector. Sin embargo, el corpus está dominado por renta variable estadounidense; no
representa de forma equilibrada divisas, materias primas, renta fija y criptoactivos.
Que D generalice a BTC-USD es, por tanto, una prueba de transferencia fuera del dominio
más exigente que su aplicación a SPY o TSLA.

### Qué garantiza realmente la exclusión

La separación por símbolos evita que las ventanas de SPY, TSLA y BTC-USD entren en el
preentrenamiento. Excluir IVV/VOO y pares cripto relacionados reduce proxies obvios. No
elimina por completo la dependencia entre mercados:

- los activos externos comparten factores macroeconómicos y regímenes con los objetivos;
- la lista de “parientes” es una decisión manual y puede ser incompleta;
- el corpus offline llega hasta 2023 y, por tanto, coincide temporalmente con buena parte
  del periodo después evaluado sobre otros activos.

Por eso la campaña demuestra **transferencia entre activos con solapamiento temporal**,
no una simulación estricta de despliegue histórico. Para una prueba causal más fuerte,
el encoder debería entrenarse con un corte temporal anterior al inicio de cada OOS o
reentrenarse de forma walk-forward. Ambas lecturas son válidas, pero responden preguntas
distintas y deben nombrarse explícitamente.

---

## 3. Variables de entrada

El encoder recibe ventanas de 30 sesiones y ocho variables normalizadas.

| Variable | Papel en la representación | Motivo para incluirla |
|---|---|---|
| `log_return` | Movimiento relativo de cierre | Es comparable entre precios y sirve como objetivo de forecasting. |
| `high_low_range` | Volatilidad intradía | Captura amplitud de la sesión sin depender del nivel de precio. |
| `close_open_pct` | Dirección de la vela | Añade información intradía no contenida por completo en el retorno entre cierres. |
| `volume_ratio` | Actividad relativa a 20 días | Detecta sesiones anómalas de participación. |
| `rsi_14` | Momentum de corto plazo | Resume persistencia de subidas y bajadas. |
| `atr_pct` | Volatilidad relativa de 14 días | Facilita comparación de riesgo entre activos. |
| `market_regime` | Régimen discreto MA50/MA200 | Introduce una señal de tendencia lenta e interpretable. |
| `ma_ratio` | Intensidad continua del régimen | Evita que toda la información de tendencia quede reducida a tres estados. |

La selección privilegia variables adimensionales para que una misma representación pueda
transferirse entre activos con escalas de precio y volumen diferentes. `market_regime` y
`ma_ratio`, o `high_low_range` y `atr_pct`, contienen información relacionada. Esa
redundancia es aceptable si el encoder la comprime, pero debería comprobarse mediante una
ablación por familias de variables.

La normalización se ajusta únicamente con la parte de entrenamiento del corpus offline.
Las estadísticas resultantes forman parte del experimento: usar otras medias o
desviaciones en el brazo D cambiaría la distribución de entrada aunque los pesos fueran
idénticos.

En la campaña RL actual, las variables del activo objetivo se normalizan con estadísticas
de su propio `IS_train`, no con las estadísticas guardadas del corpus offline. Esto reduce
el cambio de escala entre activos sin consultar OOS, pero introduce una adaptación de
distribución en el dominio objetivo. Por tanto, D transfiere pesos sin finetuning, aunque
no transfiere literalmente la normalización del corpus. Comparar ambas políticas de
normalización es una ablación metodológica relevante.

---

## 4. Objetivo multitarea

El encoder se entrena con dos objetivos:

```text
L_total = L_reconstrucción + λ · L_forecasting
```

### Reconstrucción

Obliga al espacio latente a conservar información suficiente para aproximar la ventana
original. Favorece una representación general, pero por sí sola puede dedicar capacidad
a detalles fáciles de reconstruir y poco relevantes para decisiones financieras.

### Forecasting

Predice los próximos `k` log-retornos. Introduce presión para que el latente conserve
información temporal potencialmente predictiva. No debe interpretarse como un predictor
de trading autónomo: una pérdida MSE baja no garantiza utilidad económica ni una señal
explotable después de costes.

| Parámetro | Valor | Interpretación | Efecto de aumentarlo |
|---|---:|---|---|
| `k` | 5 sesiones | Horizonte aproximado de una semana bursátil | Exige contexto más persistente, pero hace el objetivo más incierto. |
| `lambda_forecast` | 0.5 | Forecasting secundario frente a reconstrucción | Orienta más el latente a predicción y menos a fidelidad general. |

`λ=0.5` se eligió como compromiso: el forecasting guía la representación sin desplazar
por completo el objetivo de reconstrucción. No es un óptimo demostrado. La ablación
mínima recomendable es `λ ∈ {0, 0.5, 1}` y `k ∈ {1, 5, 10}`, manteniendo arquitectura,
corpus y semilla constantes.

---

## 5. Elección de arquitectura y capacidad

La configuración de referencia es una TCN causal:

| Parámetro | Valor | Qué controla | Razón experimental |
|---|---:|---|---|
| `lookback` | 30 | Historia disponible por muestra | Aproxima un mes y medio sin hacer excesivamente largo el estado. |
| `latent_dim` | 32 | Capacidad del cuello de botella | Comprime 240 entradas a 32 dimensiones, aproximadamente 7,5 veces. |
| `kernel_size` | 3 | Patrón temporal local por convolución | Captura interacciones cortas con coste moderado. |
| `dilations` | `[1,2,4,8]` | Escalas temporales cubiertas | Produce campo receptivo 31, suficiente para toda la ventana de 30. |
| `channels` | 64 | Capacidad interna de la TCN | Mantiene la misma arquitectura que C para que la diferencia principal sea el régimen de preentrenamiento. |
| Activación | ReLU | No linealidad | Opción estable y sencilla para la comparación principal. |
| Dropout | 0.0 | Regularización estocástica | Se evita añadir ruido sin una calibración previa específica. |

La TCN se eligió por causalidad, cobertura explícita de la ventana y entrenamiento
paralelo. En la campaña robusta, C y D comparten la misma arquitectura TCN de 64 canales;
“offline/heavy” describe principalmente el corpus y presupuesto de preentrenamiento de D,
no una red downstream más grande. Esta elección es una hipótesis de diseño, no evidencia
de que TCN sea siempre mejor que MLP o GRU. `ENCODER_UPGRADE.md` detalla las alternativas
y el diseño de una comparación justa.

---

## 6. Parámetros de optimización

| Parámetro | Valor | Qué hace | Señal para revisarlo |
|---|---:|---|---|
| Épocas máximas | 100 | Límite superior del presupuesto | La validación sigue mejorando al alcanzar el límite. |
| `batch_size` | 256 | Ventanas usadas por gradiente | Gradientes ruidosos o uso ineficiente de memoria. |
| `learning_rate` | 5e-4 | Magnitud de actualización | Pérdida oscilante, divergente o descenso excesivamente lento. |
| `val_ratio` | 0.15 | Fracción temporal reservada para selección | Validación demasiado corta o entrenamiento insuficiente. |
| Paciencia | 10 épocas | Tolerancia sin mejora antes de detener | Paradas prematuras o muchas épocas sin progreso. |
| Semilla | 42 | Inicialización y orden reproducibles | Debe ampliarse a varias semillas para medir estabilidad del preentrenamiento. |

El early stopping limita sobreajuste al corpus y recupera el punto con mejor pérdida de
validación. Aun así, una sola semilla no caracteriza la incertidumbre del encoder. Antes
de atribuir a la arquitectura una diferencia downstream, conviene preentrenar varios
encoders y separar la variabilidad de representación de la variabilidad del agente RL.

El `batch_size` grande estabiliza el objetivo multitarea sobre un corpus de más de cien
mil ventanas. Reducirlo introduce más ruido, que puede regularizar, pero también cambia la
dinámica de optimización; no es una modificación neutral.

---

## 7. Cómo evaluar si el preentrenamiento es útil

`best_val_loss` solo mide ajuste a reconstrucción y forecasting. La evaluación debe
separar tres niveles:

1. **Calidad del preentrenamiento**: curvas train/val, ausencia de divergencia y brecha de
   generalización razonable.
2. **Calidad de transferencia**: rendimiento de D por ticker frente a C y A, usando la
   misma cabeza Q y presupuesto RL.
3. **Robustez**: dispersión entre semillas, gap IS/OOS, drawdown y sensibilidad a corpus,
   horizonte y dimensión latente.

Un encoder puede lograr menor pérdida de validación y empeorar el trading si aprende
detalles irrelevantes para la política. La selección final no debería basarse solo en
`best_val_loss`, ni tampoco elegir retrospectivamente el artefacto con mejor OOS del
mismo periodo.

### Ablaciones prioritarias

| Pregunta | Comparación mínima |
|---|---|
| ¿Aporta algo el forecasting? | `λ=0` frente a `λ=0.5`. |
| ¿La mejora proviene de mayor capacidad? | TCN ligera frente a TCN de 64 canales con igual corpus. |
| ¿Importa la diversidad? | Universo completo frente a subconjuntos sectoriales balanceados. |
| ¿La transferencia es temporalmente honesta? | Corpus completo frente a corpus con cutoff anterior a OOS. |
| ¿El cuello de botella es adecuado? | `latent_dim` 16, 32 y 64. |
| ¿El resultado depende del preentrenamiento? | Varias semillas offline con semillas RL fijas. |

---

## 8. Trazabilidad del artefacto de referencia

El artefacto actual se entrenó con:

- 39 activos solicitados y efectivos;
- 134 679 ventanas;
- periodo 2010-01-01 a 2023-12-31;
- semilla 42;
- TCN con ventana 30, 8 variables, 32 dimensiones latentes y 64 canales;
- `best_val_loss ≈ 0.3245`.

El artefacto debe conservar, como mínimo, arquitectura, orden de variables, estadísticas
de normalización, universo, exclusiones, fechas, semilla y pérdida de validación. Estos
datos permiten distinguir un resultado reproducible de un conjunto de pesos sin contexto.

Debe regenerarse cuando cambie cualquiera de estos elementos:

- ventana o variables de entrada, incluido su orden;
- tipo, dimensión o capacidad del encoder;
- objetivo multitarea (`k` o `λ`);
- corpus, exclusiones o corte temporal;
- estrategia de normalización;
- protocolo de entrenamiento si la modificación altera los pesos seleccionados.

No es necesario regenerarlo para cambiar únicamente la cabeza Q o parámetros RL, siempre
que el contrato de entrada permanezca idéntico.

---

## 9. Ejecución reproducible

Comprobación corta del protocolo:

```powershell
python -m scripts.experiments.pretrain_encoder --smoke
```

Preentrenamiento de referencia:

```powershell
python -m scripts.experiments.pretrain_encoder
```

El modo smoke valida que la campaña puede ejecutarse, pero sus pocas épocas y su corpus
reducido no sirven para extraer conclusiones experimentales.
