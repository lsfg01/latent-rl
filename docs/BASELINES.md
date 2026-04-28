# Baselines para Aprendizaje por Refuerzo Financiero

## Qué es un Baseline en el Contexto del TFM

En el contexto de este Trabajo de Fin de Máster (TFM) sobre aprendizaje por refuerzo financiero con representaciones latentes, un **baseline** es un punto de referencia mínimo que sirve para:

1. **Validar el entorno**: Demostrar que FinancialEnv funciona correctamente
2. **Establecer expectativas**: Definir qué nivel de rendimiento es aceptable
3. **Comparar agentes**: Proporcionar un estándar contra el cual medir agentes RL entrenados
4. **Debugging**: Identificar problemas en la implementación de agentes más complejos
5. **Análisis de coste-beneficio**: Evaluar si la complejidad adicional de RL vale la pena

Los baselines implementados (RandomAgent y BuyAndHoldAgent) representan estrategias simples que no requieren entrenamiento, permitiendo comparar el rendimiento de futuros agentes RL (como DQN, PPO, etc.) contra estrategias triviales.

## RandomAgent

### Qué Hace

RandomAgent es el baseline más simple posible. Su comportamiento es:

- **Selección aleatoria de acciones**: Elige uniformemente entre hold (0), buy (1) y sell (2)
- **Sin memoria**: No mantiene estado interno entre pasos
- **Sin aprendizaje**: No se entrena ni adapta al entorno
- **Reproducible**: Puede usar una semilla para generar secuencias deterministas

**Implementación**:
```python
def select_action(self, observation):
    return self.action_space.sample()  # Acción aleatoria
```

### Por qué es Útil

RandomAgent es útil por varias razones:

1. **Piso de rendimiento**: Establece el mínimo rendimiento esperado sin estrategia
2. **Validación de entorno**: Si el entorno funciona, RandomAgent debe ejecutarse sin errores
3. **Control de aleatoriedad**: Permite separar el efecto de la aleatoriedad del efecto de la estrategia
4. **Benchmark de complejidad**: Cualquier agente que aprenda debe superar significativamente a RandomAgent
5. **Debugging**: Ayuda a identificar si problemas provienen del agente o del entorno

### Limitaciones

RandomAgent tiene limitaciones obvias:

1. **Sin estrategia financiera**: No considera precios, tendencias o patrones
2. **Alto coste de transacción**: Genera muchos trades innecesarios
3. **Comportamiento errático**: Puede comprar y vender consecutivamente sin lógica
4. **Sin adaptación**: No responde a cambios en las condiciones del mercado
5. **Ineficiente**: Típicamente pierde dinero debido a costes de transacción

**Resultado esperado**: En mercados con costes de transacción, RandomAgent suele mostrar retornos negativos debido al exceso de trades.

## BuyAndHoldAgent

### Qué Hace

BuyAndHoldAgent implementa la estrategia clásica "buy and hold":

- **Compra inicial**: Ejecuta acción buy (1) en el primer paso
- **Mantiene posición**: Ejecuta hold (0) en todos los pasos siguientes
- **No vende**: Nunca ejecuta sell (2) a menos que se resetee
- **Determinista**: Siempre produce la misma secuencia de acciones

**Implementación**:
```python
def select_action(self, observation):
    if not self.has_bought:
        self.has_bought = True
        return 1  # Buy
    else:
        return 0  # Hold
```

### Por qué es Útil en Mercados Financieros

BuyAndHoldAgent es particularmente relevante en finanzas por:

1. **Estrategia real**: Representa una estrategia de inversión común y ampliamente utilizada
2. **Bajo coste**: Solo genera un trade (compra inicial), minimizando costes de transacción
3. **Captura tendencia**: Se beneficia de movimientos alcistas del mercado
4. **Simplicidad**: No requiere análisis técnico ni fundamental
5. **Referencia académica**: Usado frecuentemente en literatura financiera como benchmark

**Razón financiera**: En mercados eficientes, muchas estrategias activas no superan al buy-and-hold después de costes, por lo que es un baseline exigente.

### Limitaciones

BuyAndHoldAgent tiene limitaciones importantes:

1. **Sin gestión de riesgo**: No protege contra caídas del mercado
2. **Sin adaptación**: No responde a cambios en las condiciones del mercado
3. **Sin timing**: Compra independientemente del momento del mercado
4. **Sin diversificación**: Invierte todo en un solo activo
5. **Vulnerabilidad a drawdowns**: Puede sufrir pérdidas significativas en mercados bajistas

**Resultado esperado**: En mercados alcistas, BuyAndHoldAgent suele mostrar retornos positivos. En mercados bajistas, puede mostrar pérdidas significativas.

## Métricas Principales Usadas para Comparar

### Final Equity

**Definición**: Valor total de la cuenta al final del episodio.

```
final_equity = cash + position_units × current_price
```

**Interpretación**:
- Representa el valor total del portafolio
- Incluye tanto efectivo como valor de posiciones abiertas
- Es la métrica más directa de rendimiento financiero

**Ejemplo**: Si `final_equity = $11,750`, el agente ha ganado $1,750 sobre el capital inicial de $10,000.

### Total Return

**Definición**: Retorno porcentual del capital inicial.

```
total_return = (final_equity - initial_balance) / initial_balance
```

**Interpretación**:
- Normaliza el rendimiento por el capital inicial
- Permite comparar entre diferentes niveles de capital
- Es la métrica estándar en finanzas

**Ejemplo**: Si `total_return = 17.50%`, el agente ha ganado un 17.5% sobre el capital inicial.

### Reward Acumulado

**Definición**: Suma de rewards obtenidos durante el episodio.

```
total_reward = Σ reward_t
```

**Interpretación**:
- Refleja el cambio acumulado de equity normalizado
- Debe estar alineado con `total_return`
- Útil para análisis de convergencia durante entrenamiento

**Relación con equity**: `total_reward ≈ total_return` (diferencias pequeñas por normalización).

### Número de Trades

**Definición**: Cantidad de trades ejecutados durante el episodio.

```
n_trades = len(trade_history)
```

**Interpretación**:
- Indica la actividad de trading del agente
- Más trades = más costes de transacción
- Estrategias eficientes minimizan trades innecesarios

**Ejemplo**: RandomAgent suele generar muchos trades (50-100), mientras BuyAndHoldAgent genera solo 1 trade.

### Realized Profit

**Definición**: Beneficio/pérdida realizado tras cerrar posiciones.

```
realized_profit = cash - initial_balance  # Después de sell
```

**Interpretación**:
- Solo incluye profit/pérdida de trades cerrados
- No incluye P&L no realizado de posiciones abiertas
- Es una métrica complementaria, no principal

**Importante**: `realized_profit` puede ser engañoso para agentes que mantienen posiciones abiertas.

## Explicación de Realized Profit

### Qué Incluye

Realized profit incluye:

- **Profit/pérdida de trades cerrados**: Diferencia entre precio de compra y venta
- **Costes de transacción**: Restados del profit bruto
- **Cash final**: Efectivo disponible después de cerrar todas las posiciones

### Qué NO Incluye

Realized profit NO incluye:

- **P&L no realizado**: Ganancias/pérdidas de posiciones abiertas
- **Valor de posición**: No refleja el valor actual de posiciones mantenidas
- **Equity total**: No representa el valor total del portafolio

### Ejemplo Ilustrativo

Considera un agente que compra a $100 y el precio sube a $150:

- **Con posición abierta**:
  - `realized_profit = $0` (no se ha cerrado la posición)
  - `equity = $1,500` (valor de la posición)

- **Después de vender**:
  - `realized_profit = $500` (profit realizado)
  - `equity = $1,500` (cash final)

**Conclusión**: `realized_profit` es útil para analizar trades individuales, pero `equity` y `total_return` son métricas más completas para evaluar rendimiento global.

## Por qué Equity y Total Return son Métricas Principales

### Equity como Métrica Principal

**Equity** es la métrica principal porque:

1. **Completeness**: Representa el valor total del portafolio
2. **Realismo**: Refleja lo que el inversor realmente tiene
3. **Inclusión**: Incluye tanto efectivo como valor de posiciones
4. **Consistencia**: Alineado con el objetivo financiero real
5. **Comparabilidad**: Permite comparar diferentes estrategias

**Ventajas sobre otras métricas**:
- Más completo que `realized_profit` (incluye P&L no realizado)
- Más interpretable que `reward` (escala financiera directa)
- Más robusto que métricas de trades individuales

### Total Return como Métrica Principal

**Total return** es la métrica principal porque:

1. **Normalización**: Permite comparar independientemente del capital inicial
2. **Estándar financiero**: Es la métrica universal en finanzas
3. **Interpretación**: Fácil de entender (porcentaje de ganancia/pérdida)
4. **Escalabilidad**: Aplicable a diferentes niveles de capital
5. **Benchmarking**: Facilita comparación con índices y otros activos

**Relación con equity**: `total_return` es simplemente `equity` normalizado por el capital inicial.

### Complementariedad

Ambas métricas son complementarias:

- **Equity**: Valor absoluto ($11,750)
- **Total return**: Valor relativo (17.50%)
- **Juntas**: Proporcionan una imagen completa del rendimiento

## Cómo Ejecutar la Comparación de Baselines

### Comando Básico

```bash
python examples/baseline_comparison.py
```

### Qué Hace el Script

El script `baseline_comparison.py`:

1. **Genera datos sintéticos**: Crea datos OHLCV con tendencia alcista
2. **Crea el entorno**: Inicializa FinancialEnv con los datos
3. **Crea los agentes**: Inicializa RandomAgent y BuyAndHoldAgent
4. **Ejecuta episodio único**: Compara ambos agentes en el mismo entorno
5. **Ejecuta análisis estadístico**: Repite 10 episodios para análisis
6. **Muestra resultados**: Presenta métricas comparativas

### Salida Esperada

El script produce:

```
=== Comparación de Baselines ===
RandomAgent vs BuyAndHoldAgent

1. Creando datos sintéticos...
   - Datos creados: 200 filas
   - Precio inicial: 100.99
   - Precio final: 117.71
   - Retorno del activo: 16.56%

2. Creando entorno financiero...
   - Balance inicial: $10,000.00
   - Coste de transacción: 0.1%

3. Creando agentes...
   - RandomAgent creado
   - BuyAndHoldAgent creado

4. Ejecutando episodio único...

   --- RandomAgent ---
   - Pasos: 189
   - Recompensa total: -0.0039
   - Cash final: $0.00
   - Equity final: $9,770.13
   - Retorno total: -2.30%
   - Profit realizado: $-43.14
   - Número de trades: 67

   --- BuyAndHoldAgent ---
   - Pasos: 189
   - Recompensa total: 0.1980
   - Cash final: $0.00
   - Equity final: $11,750.45
   - Retorno total: 17.50%
   - Profit realizado: $0.00
   - Número de trades: 1

5. Comparación de resultados...
6. Análisis estadístico (10 episodios)...
7. Conclusión...
```

## Cómo Interpretar el Resultado

### Diferencia en Puntos Porcentuales

La conclusión expresa la diferencia en **puntos porcentuales**:

```
BuyAndHoldAgent supera a RandomAgent por 19.80 puntos porcentuales de retorno total.
```

**Interpretación**:
- BuyAndHoldAgent: 17.50% de retorno
- RandomAgent: -2.30% de retorno
- Diferencia: 17.50% - (-2.30%) = 19.80 puntos porcentuales

**Ventajas de esta expresión**:
- Clara y académicamente correcta
- Funciona con retornos positivos y negativos
- Evita confusiones con porcentajes relativos

### Comparación RandomAgent vs BuyAndHoldAgent

#### RandomAgent

**Características típicas**:
- **Retorno**: Generalmente negativo debido a costes de transacción
- **Trades**: Muchos trades (50-100)
- **Comportamiento**: Errático, sin lógica financiera
- **Profit realizado**: Puede ser positivo o negativo (último trade cerrado)

**Cuándo es útil**:
- Como piso de rendimiento mínimo
- Para validar que el entorno funciona
- Para controlar efectos de aleatoriedad

#### BuyAndHoldAgent

**Características típicas**:
- **Retorno**: Positivo en mercados alcistas, negativo en mercados bajistas
- **Trades**: Solo 1 trade (compra inicial)
- **Comportamiento**: Determinista, simple
- **Profit realizado**: 0 (posición abierta)

**Cuándo es útil**:
- Como benchmark de estrategia pasiva
- Para evaluar si RL vale la pena en mercados tendenciales
- Como referencia académica estándar

### Interpretación de Resultados

**Escenario 1: BuyAndHold > RandomAgent**
- **Interpretación**: La estrategia pasiva supera a la aleatoria
- **Conclusión**: El mercado tiene tendencia positiva
- **Expectativa**: Agentes RL deben superar a BuyAndHold

**Escenario 2: RandomAgent > BuyAndHold**
- **Interpretación**: La aleatoriedad supera a la estrategia pasiva
- **Conclusión**: El mercado es lateral o bajista
- **Expectativa**: Agentes RL deben superar a RandomAgent

**Escenario 3: Agentes RL ≈ Baselines**
- **Interpretación**: RL no añade valor significativo
- **Conclusión**: Posible problema con el agente o el entorno
- **Acción**: Revisar arquitectura, hiperparámetros, datos

## Qué Debería Demostrar un Futuro Agente RL

Para considerarse mejor que los baselines, un agente RL entrenado debería:

### Superar a RandomAgent

**Mínimo requerido**:
- Retorno total > RandomAgent
- Menos costes de transacción
- Comportamiento más coherente

**Nivel básico**:
- Retorno total > RandomAgent + margen de seguridad
- Número de trades significativamente menor
- Estrategia más inteligente que aleatoria

### Superar a BuyAndHoldAgent

**Nivel intermedio**:
- Retorno total > BuyAndHoldAgent
- Mejor gestión de riesgo (menor drawdown)
- Adaptación a diferentes condiciones de mercado

**Nivel avanzado**:
- Retorno total > BuyAndHoldAgent + margen significativo
- Protección contra drawdowns en mercados bajistas
- Generalización a diferentes activos/periodos

### Demostrar Valor Adicional

**Más allá de los baselines**:
- **Consistencia**: Rendimiento estable a través de diferentes episodios
- **Robustez**: Funciona bien en diferentes condiciones de mercado
- **Interpretabilidad**: Estrategia comprensible y justificable
- **Eficiencia**: Buen rendimiento con costes computacionales razonables

### Criterios de Éxito

**Criterios cuantitativos**:
- `total_return_RL > total_return_BuyAndHold + 5%`
- `max_drawdown_RL < max_drawdown_BuyAndHold`
- `sharpe_ratio_RL > sharpe_ratio_BuyAndHold`

**Criterios cualitativos**:
- Estrategia coherente y interpretable
- Comportamiento robusto en diferentes condiciones
- Ventaja clara sobre baselines en múltiples escenarios

## Limitaciones de Estas Baselines

### Limitaciones de RandomAgent

1. **Sin representatividad**: No representa ninguna estrategia financiera real
2. **Alta varianza**: Resultados muy variables entre ejecuciones
3. **Sin contexto**: No considera información del mercado
4. **Ineficiencia extrema**: Genera costes de transacción innecesarios
5. **Piso muy bajo**: Es un estándar mínimo muy fácil de superar

### Limitaciones de BuyAndHoldAgent

1. **Sin gestión de riesgo**: No protege contra pérdidas
2. **Dependencia de tendencia**: Solo funciona bien en mercados alcistas
3. **Sin adaptación**: No responde a cambios del mercado
4. **Sin timing**: Compra independientemente de las condiciones
5. **Vulnerabilidad a drawdowns**: Puede sufrir pérdidas severas en crisis

### Limitaciones Compartidas

1. **Sin aprendizaje**: No se adaptan ni mejoran con el tiempo
2. **Sin generalización**: No transfieren conocimiento entre activos
3. **Sin complejidad**: No capturan patrones complejos del mercado
4. **Sin optimización**: No ajustan parámetros según condiciones
5. **Sin representaciones**: No usan representaciones latentes del mercado

### Limitaciones del Contexto del MVP

1. **Datos sintéticos**: Los ejemplos usan datos generados, no reales
2. **Un solo activo**: No se evalúa en carteras diversificadas
3. **Periodo corto**: No se evalúa a largo plazo
4. **Sin validación cruzada**: No se prueba en diferentes periodos
5. **Sin comparación externa**: No se compara con índices o estrategias profesionales

## Conclusión

Los baselines implementados (RandomAgent y BuyAndHoldAgent) proporcionan puntos de referencia claros y significativos para evaluar agentes de aprendizaje por refuerzo en el contexto financiero del TFM.

**RandomAgent** establece un piso mínimo de rendimiento, mientras que **BuyAndHoldAgent** representa un estándar exigente basado en una estrategia financiera real y ampliamente utilizada.

Las métricas principales (`final_equity` y `total_return`) proporcionan una evaluación completa y coherente del rendimiento, mientras que las métricas complementarias (`reward acumulado`, `n_trades`, `realized_profit`) ofrecen información adicional sobre el comportamiento del agente.

Para que un agente RL sea considerado exitoso, debe superar significativamente a estos baselines, demostrando que el aprendizaje por refuerzo añade valor real más allá de estrategias simples y no entrenadas.