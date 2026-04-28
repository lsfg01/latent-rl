# FinancialEnv - Entorno Financiero para Reinforcement Learning

## Objetivo

FinancialEnv es un entorno de aprendizaje por refuerzo compatible con Gymnasium diseñado para el trading algorítmico. Su objetivo principal es proporcionar un entorno simplificado y controlado para entrenar y evaluar agentes de RL en tareas de toma de decisiones financieras, adecuado para un MVP experimental.

El entorno está diseñado específicamente para el TFM sobre aprendizaje por refuerzo financiero con representaciones latentes, sirviendo como base para experimentar con diferentes arquitecturas de agentes y estrategias de trading.

## Acciones Disponibles

El espacio de acciones es `Discrete(3)`, con las siguientes acciones:

- **Acción 0 = Hold**: Mantener la posición actual sin cambios
- **Acción 1 = Buy**: Abrir una posición larga (comprar)
- **Acción 2 = Sell**: Cerrar una posición larga (vender)

## Modelo Long-Only

El entorno implementa un modelo **long-only**, lo que significa:

- **No se implementan posiciones short**: El entorno no permite vender en descubierto
- **Sell cierra posición long**: La acción de sell solo cierra una posición larga existente, no abre posiciones short
- **position = -1 no se usa**: El valor -1 para position está reservado para futuras versiones que implementen short
- **Sin apalancamiento**: No se permite operar con más capital del disponible
- **Posición única**: Solo se puede mantener una posición a la vez (flat o long)

Esta simplificación es adecuada para un MVP y reduce la complejidad del espacio de acciones.

## Estado Interno

El entorno mantiene el siguiente estado interno:

### Variables de Cartera

- **`cash`**: Efectivo disponible en la cuenta (no invertido)
- **`position`**: Estado de la posición actual
  - `0 = flat`: Sin posición abierta
  - `1 = long`: Posición larga abierta
  - `-1 = short`: No implementado en este MVP
- **`position_units`**: Número de unidades del activo en posición larga
- **`equity`**: Valor total de la cuenta, calculado como `equity = cash + position_units * current_price`
- **`portfolio_value`**: Alias de equity (valor total del portafolio)

### Variables de Seguimiento

- **`realized_profit`**: Beneficio/pérdida realizado tras cerrar una posición
  - Solo se actualiza al cerrar una posición
  - No incluye P&L no realizado de posiciones abiertas
  - `realized_profit = cash - initial_balance` después de sell
- **`previous_equity`**: Equity del paso anterior (usado para calcular reward)
- **`trade_history`**: Lista de trades ejecutados
  - Cada trade incluye: tipo, precio, unidades, paso

### Variables de Configuración

- **`initial_balance`**: Balance inicial de la cuenta (default: $10,000)
- **`transaction_cost`**: Coste de transacción como porcentaje (default: 0.1%)
- **`lookback_window`**: Ventana de observación histórica (default: 10)
- **`current_step`**: Paso actual en el episodio

## Funcionamiento de las Acciones

### Buy (Acción 1)

Cuando el agente ejecuta la acción de buy:

1. **Validación**: Solo se ejecuta si `position == 0` (sin posición abierta)
2. **Cálculo del coste**: `cost = cash * transaction_cost`
3. **Cálculo de unidades**: `position_units = (cash - cost) / current_price`
4. **Actualización de estado**:
   - `cash = 0` (todo el efectivo se invierte)
   - `position = 1` (posición larga)
   - `entry_price = current_price` (precio de entrada)
5. **Registro**: Se añade un trade a `trade_history`

**Ejemplo**: Con $10,000, precio $100 y coste 0.1%:
- Coste = $10,000 × 0.001 = $10
- Cash disponible = $9,990
- Unidades compradas = $9,990 / $100 = 99.9 unidades

### Hold (Acción 0)

Cuando el agente ejecuta la acción de hold:

1. **Mantener posición**: No se modifica el estado de la cartera
2. **Actualización de equity**: `equity = cash + position_units × current_price`
3. **Cálculo de reward**: Basado en el cambio de equity desde el paso anterior

Hold es la acción por defecto cuando ya se tiene una posición abierta y se quiere mantenerla.

### Sell (Acción 2)

Cuando el agente ejecuta la acción de sell:

1. **Validación**: Solo se ejecuta si `position == 1` (posición larga abierta)
2. **Cálculo del proceeds**: `proceeds = position_units × current_price`
3. **Cálculo del coste**: `cost = proceeds × transaction_cost`
4. **Actualización de cash**: `cash = proceeds - cost`
5. **Cálculo de profit realizado**: `realized_profit = cash - initial_balance`
6. **Cierre de posición**:
   - `position_units = 0`
   - `position = 0`
   - `entry_price = 0`
7. **Registro**: Se añade un trade a `trade_history`

**Ejemplo**: Con 99.9 unidades, precio $118 y coste 0.1%:
- Proceeds = 99.9 × $118 = $11,788.20
- Coste = $11,788.20 × 0.001 = $11.79
- Cash final = $11,788.20 - $11.79 = $11,776.41
- Realized profit = $11,776.41 - $10,000 = $1,776.41

## Supuestos del MVP

El entorno actual se basa en los siguientes supuestos simplificados:

1. **Un único activo**: Solo se opera con un activo financiero a la vez
2. **Long-only**: No se permiten posiciones short; position = -1 está reservado para futuras versiones
3. **Inversión total del cash**: Al comprar, se invierte el 100% del cash disponible (limitación por no existir position sizing)
4. **Ejecución al current_price**: Las órdenes se ejecutan al precio actual del activo, normalmente el precio de cierre
5. **Costes proporcionales fijos**: Los costes de transacción son un porcentaje fijo del valor de la operación
6. **Sin slippage ni spread**: No se modela el deslizamiento de precio ni el bid/ask spread
7. **Reward basado en cambio de equity**: El reward se calcula como el cambio de equity normalizado por initial_balance
8. **Evaluación basada en equity y total_return**: La evaluación principal se basa en final_equity y total_return

Estos supuestos son intencionales para mantener el entorno controlado y adecuado para un MVP experimental.

## Cálculo de Reward

El reward se calcula de forma consistente para todas las acciones:

```
reward = (new_equity - previous_equity) / initial_balance
```

### Propiedades del Reward

1. **Consistencia**: Se aplica el mismo cálculo para hold, buy y sell
2. **Normalización**: Dividido por `initial_balance` para escala consistente
3. **Alineación con equity**: El reward acumulado refleja el cambio total de equity
4. **Signo**:
   - Reward positivo → equity aumentó
   - Reward negativo → equity disminuyó
   - Reward cero → equity sin cambios

### Nota sobre Reward Acumulado

La suma de rewards a lo largo de un episodio aproxima el cambio acumulado de equity normalizado:

```
sum(rewards) ≈ (final_equity - initial_equity) / initial_balance
```

Por lo tanto, el reward acumulado debe interpretarse junto con `final_equity` y `total_return` para una evaluación completa del rendimiento.

### Ejemplo de Cálculo

Si `initial_balance = $10,000`:
- Equity paso anterior: $10,000
- Equity actual: $10,500
- Reward = ($10,500 - $10,000) / $10,000 = 0.05 (5%)

## Información Devuelta (Info)

En cada paso, el entorno devuelve un diccionario `info` con:

```python
{
    "balance": cash,              # Alias de cash (compatibilidad)
    "cash": cash,                 # Efectivo disponible
    "equity": equity,             # Valor total de la cuenta
    "portfolio_value": equity,    # Alias de equity
    "position": position,         # Estado de posición (0/1)
    "position_units": position_units,  # Unidades en posición
    "current_price": current_price,     # Precio actual del activo
    "entry_price": entry_price,        # Precio de entrada (si hay posición)
    "realized_profit": realized_profit,    # Profit/pérdida realizada
    "total_profit": realized_profit,     # Alias temporal (compatibilidad)
    "current_step": current_step,        # Paso actual
    "n_trades": len(trade_history)       # Número de trades ejecutados
}
```

## Limitaciones Actuales

El entorno actual tiene las siguientes limitaciones, intencionales para un MVP:

### Limitaciones de Modelo

1. **Long-only**: No se implementan posiciones short
2. **Sin slippage**: Se asume ejecución perfecta al precio de mercado
3. **Sin tamaño variable**: No se permite fraccionar la posición (todo o nada)
4. **Sin apalancamiento**: No se permite operar con capital prestado
5. **Sin múltiples activos**: Solo se opera con un activo a la vez

### Limitaciones de Costes

6. **Costes de transacción simples**: Solo un porcentaje fijo por trade
7. **Sin costes de financiación**: No se cobran intereses por posiciones overnight
8. **Sin costes de custody**: No se incluyen costes de mantenimiento
9. **Sin impactos de mercado**: No se modela el impacto de grandes órdenes

### Limitaciones de Datos

10. **Valoración simplificada**: La observación contiene datos OHLCV, pero la valoración y ejecución se simplifican usando `current_price`, normalmente el precio de cierre
11. **Sin microestructura**: No se considera bid/ask spread, depth, etc.
12. **Sin eventos corporativos**: No se modelan dividendos, splits, etc.

### Limitaciones de Realismo

13. **Sin restricciones de riesgo**: No hay límites de pérdida máxima
14. **Sin gestión de capital**: No se implementa sizing dinámico
15. **Sin modelado de calendario de mercado**: No se modelan horarios de mercado, gaps intradía ni restricciones de sesión; se asume ejecución continua sin considerar horarios de apertura/cierre

## Mejoras Futuras Posibles

Para versiones posteriores del entorno, se podrían implementar:

### Mejoras de Modelo

1. **Posiciones short**: Permitir vender en descubierto
2. **Sizing variable**: Permitir fraccionar el capital en múltiples trades
3. **Múltiples activos**: Operar con una cartera de activos
4. **Apalancamiento**: Permitir operar con capital prestado
5. **Opciones**: Implementar estrategias con derivados

### Mejoras de Realismo

6. **Slippage**: Modelar el impacto de ejecución real
7. **Bid/ask spread**: Usar precios de compra/venta reales
8. **Costes de financiación**: Cobrar intereses por posiciones overnight
9. **Impacto de mercado**: Modelar cómo grandes órdenes afectan precios
10. **Microestructura**: Incluir order book depth, flow, etc.

### Mejoras de Gestión de Riesgo

11. **Stop-loss**: Implementar límites de pérdida automática
12. **Take-profit**: Implementar límites de ganancia automática
13. **Position sizing**: Gestión dinámica del tamaño de posición
14. **Risk limits**: Límites de exposición máxima

### Mejoras de Datos

15. **Eventos corporativos**: Modelar dividendos, splits, etc.
16. **Datos alternativos**: Incluir sentimiento, noticias, etc.
17. **Datos de alta frecuencia**: Soporte para intraday
18. **Datos fundamentales**: Incluir métricas contables, económicas

## Por qué este Entorno es Suficiente para el MVP del TFM

A pesar de las limitaciones, FinancialEnv es adecuado para el MVP del TFM por las siguientes razones:

### 1. Enfoque en Representaciones Latentes

El objetivo principal del TFM es experimentar con representaciones latentes en RL financiero, no crear un entorno de trading de producción. El entorno actual proporciona suficiente complejidad para:

- Probar diferentes arquitecturas de encoders latentes
- Evaluar la calidad de las representaciones aprendidas
- Comparar agentes RL con baselines simples

### 2. Simplicidad Controlada

Las limitaciones intencionales permiten:

- **Reproducibilidad**: Resultados consistentes entre experimentos
- **Debugging**: Fácil identificación de problemas
- **Interpretación**: Comportamiento del agente más transparente
- **Comparación**: Baselines claros y significativos

### 3. Complejidad Adecuada

El entorno actual incluye los elementos esenciales para RL financiero:

- **Dinámica de precios**: Movimientos de precios basados en datos históricos
- **Costes de transacción**: Impacto de los trades en el capital
- **Gestión de posición**: Lógica de compra/venta/hold
- **Reward coherente**: Alineado con objetivos financieros

### 4. Extensibilidad

La arquitectura del entorno permite:

- **Añadir nuevas features**: Fácil incorporación de datos adicionales
- **Modificar la lógica**: Cambios en el modelo de trading
- **Integrar con otros módulos**: Compatibilidad con representaciones latentes
- **Escalar a versiones complejas**: Base para mejoras futuras

### 5. Validación de Concepto

El entorno actual permite validar:

- **Viabilidad de RL financiero**: Si los agentes pueden aprender políticas útiles
- **Calidad de representaciones**: Si las representaciones latentes capturan patrones financieros
- **Comparación con baselines**: Si los agentes RL superan estrategias simples
- **Generalización**: Si los agentes aprenden patrones transferibles

## Conclusión

FinancialEnv proporciona un entorno simplificado pero funcional para experimentar con RL financiero en el contexto del TFM. Sus limitaciones son intencionales y adecuadas para un MVP, mientras que su arquitectura permite extensiones futuras según las necesidades del proyecto.

El enfoque en la claridad del modelo, la coherencia de las métricas y la simplicidad controlada hace que este entorno sea ideal para investigar representaciones latentes en RL financiero sin la complejidad de un entorno de trading de producción.