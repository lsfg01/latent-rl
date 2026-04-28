# DQNAgent: Primer Agente Entrenable

## Objetivo

DQNAgent es el primer agente entrenable implementado en el proyecto `tfm-latent-rl`. Su objetivo es proporcionar una base funcional para el aprendizaje por refuerzo en el entorno financiero, permitiendo validar el pipeline completo de RL (entorno → agente → entrenamiento → evaluación) antes de introducir representaciones latentes más complejas.

## Por qué DQN como primer agente entrenable

Deep Q-Network (DQN) se selecciona como primer agente entrenable por varias razones:

1. **Simplicidad conceptual**: DQN extiende Q-learning con aproximación de funciones, siendo fácil de entender e implementar.
2. **Base estable**: Sirve como baseline para comparaciones futuras con variantes más avanzadas (Double DQN, Dueling DQN, etc.).
3. **Validación de pipeline**: Permite verificar que el entorno financiero, el replay buffer y el bucle de entrenamiento funcionan correctamente.
4. **Literatura establecida**: DQN es un algoritmo bien documentado con patrones de implementación claros.

## Componentes Principales

### QNetwork

Red neuronal MLP que aproxima la función Q(s, a):

- **Arquitectura**: Flatten → Linear(128) → ReLU → Linear(128) → ReLU → Linear(n_actions)
- **Entrada**: Observación aplanada (batch, *observation_shape)
- **Salida**: Q-values para cada acción (batch, n_actions)
- **Propósito**: Estimar el valor esperado de cada acción dado un estado

### DQNAgent

Clase principal que implementa el algoritmo DQN:

- **Espacio de acciones**: Discrete(3) → {0: hold, 1: buy, 2: sell}
- **Exploración**: Política epsilon-greedy con decaimiento
- **Redes**: q_network (entrenable) y target_network (estable)
- **Optimizador**: Adam con learning_rate configurable
- **Loss**: SmoothL1Loss (Huber loss) para robustez frente a outliers

### ReplayBuffer

Buffer circular que almacena transiciones de experiencia:

- **Capacidad**: Configurable (default: 10000)
- **Almacenamiento**: (state, action, reward, next_state, done)
- **Muestreo**: Batch aleatorio sin reemplazo
- **Importancia**: Rompe correlación temporal, permite reutilizar experiencia

### Target Network

Copia de la red Q que se actualiza periódicamente:

- **Frecuencia**: target_update_freq (default: 100)
- **Propósito**: Estabilizar el target de aprendizaje, evitando oscilaciones
- **Actualización**: Copia completa de pesos desde q_network

### Política Epsilon-Greedy

Estrategia de exploración-explotación:

- **Epsilon inicial**: 1.0 (exploración total)
- **Epsilon final**: 0.1 (explotación con algo de exploración)
- **Decaimiento**: epsilon = max(epsilon_end, epsilon * epsilon_decay)
- **Selección**: Aleatoria con probabilidad epsilon, greedy (argmax Q) con probabilidad 1-epsilon

## Flujo de Entrenamiento

El bucle de entrenamiento sigue el patrón estándar de RL:

1. **Reset del entorno**: `obs, info = env.reset()`
2. **Selección de acción**: `action = agent.select_action(obs, training=True)`
3. **Ejecución**: `next_obs, reward, terminated, truncated, info = env.step(action)`
4. **Almacenamiento**: `agent.store_transition(obs, action, reward, next_obs, done)`
5. **Actualización**: `metrics = agent.update()` (si buffer ≥ batch_size)
6. **Actualización de target network**: Cada `target_update_freq` pasos

## Cálculo de la Pérdida

DQN minimiza el error entre Q-values actuales y target Q-values:

### Q(s, a)

```python
q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
```

Q-value de la acción ejecutada en el estado actual.

### Target Q-value

```python
next_q_values = target_network(next_states).max(1)[0]
target_q_values = rewards + gamma * next_q_values * (1 - dones)
```

Target usando Bellman optimista: reward + gamma * max Q_target(s', a').

### Loss

```python
loss = SmoothL1Loss(q_values, target_q_values)
```

SmoothL1Loss (Huber loss) es menos sensible a outliers que MSE.

## ReplayBuffer

### Funcionalidad

- **Almacenamiento**: `buffer.add(state, action, reward, next_state, done)`
- **Muestreo**: `states, actions, rewards, next_states, dones = buffer.sample(batch_size)`
- **Capacidad**: Circular buffer que descarta transiciones antiguas al llenarse

### Importancia

1. **Rompe correlación temporal**: Los datos consecutivos en RL están altamente correlacionados, lo que puede causar inestabilidad en el entrenamiento.
2. **Reutilización de experiencia**: Cada transición se puede usar múltiples veces, mejorando eficiencia de datos.
3. **Estabilidad**: Permite entrenar con batches i.i.d., condición ideal para SGD.

## Epsilon y Exploración

### Significado de Epsilon

- **Epsilon = 1.0**: Exploración total (acciones aleatorias)
- **Epsilon = 0.0**: Explotación total (siempre acción greedy)
- **Epsilon intermedio**: Balance entre exploración y explotación

### Uso en Entrenamiento

```python
if training and np.random.random() < epsilon:
    return action_space.sample()  # Acción aleatoria
else:
    return q_network(obs).argmax()  # Acción greedy
```

Durante el entrenamiento, epsilon decae gradualmente desde epsilon_start hasta epsilon_end, permitiendo exploración inicial y explotación gradual.

## Guardado y Carga de Modelos

### save(path)

Guarda un checkpoint con:

- `q_network_state_dict`: Pesos de la red Q principal
- `target_network_state_dict`: Pesos de la red target
- `optimizer_state_dict`: Estado del optimizador Adam
- `epsilon`: Valor actual de epsilon
- `update_step`: Contador de pasos de actualización

### load(path)

Restaura el estado completo del agente, permitiendo continuar entrenamiento o evaluar políticas aprendidas.

## Ejecución del Ejemplo

```bash
python examples/dqn_training_example.py
```

El ejemplo entrena un agente DQN durante 10 episodios con datos sintéticos.

## Interpretación de la Salida

### Entrenamiento

```
Episodio 1/10: Reward=-0.1188, Epsilon=0.995, Steps=<n>
Episodio 2/10: Reward=-0.0523, Epsilon=0.990, Steps=<n>
...
```

- **Reward**: Recompensa acumulada del episodio (cambio de equity normalizado)
- **Epsilon**: Valor actual de epsilon (decae cada paso)
- **Steps**: Número de pasos del episodio

### Evaluación Greedy

```
Test 1: Reward=0.0433, Steps=<n>
Test 2: Reward=0.0211, Steps=<n>
...
```

Evaluación con epsilon=0.0 (política greedy pura) para medir rendimiento sin exploración.

### Modelo Guardado

```
Modelo guardado en: models/dqn_agent.pth
```

El checkpoint se guarda en `models/dqn_agent.pth` para uso futuro.

## Limitaciones Actuales

### Arquitectura

- **MLP simple**: Sin capas convolucionales o recurrentes
- **Observaciones aplanadas**: Pierde estructura temporal de la ventana lookback
- **Sin normalización**: Observaciones sin normalización pueden causar inestabilidad

### Algoritmo

- **Sin Double DQN**: Sobreestimación de Q-values
- **Sin Dueling DQN**: No separa valor de estado y ventaja de acción
- **Sin Prioritized Experience Replay**: Muestreo uniforme no óptimo

### Entrenamiento

- **Entrenamiento corto**: 10 episodios insuficientes para convergencia
- **Hiperparámetros no optimizados**: Valores por defecto pueden no ser óptimos
- **No se afirma superioridad**: DQN actual no se afirma que supere a BuyAndHoldAgent

### Sensibilidad

- **Sensibilidad a hiperparámetros**: learning_rate, gamma, epsilon_decay afectan fuertemente el rendimiento
- **Sensibilidad a datos**: Rendimiento varía con diferentes activos y periodos

## Mejoras Futuras

### Algoritmo

1. **Double DQN**: Reducir sobreestimación de Q-values
2. **Dueling DQN**: Mejorar estimación de valor de estado
3. **Prioritized Experience Replay**: Muestreo inteligente de transiciones importantes
4. **Rainbow DQN**: Combinar múltiples mejoras de DQN

### Preprocesamiento

1. **Normalización de observaciones**: Escalar observaciones a [0, 1] o estandarizar
2. **Diferenciación**: Usar retornos en lugar de precios
3. **Features adicionales**: Indicadores técnicos, volumen, sentimiento

### Evaluación

1. **Comparación formal**: Tests estadísticos contra RandomAgent y BuyAndHoldAgent
2. **Evaluación walk-forward**: Validación en periodos fuera de muestra
3. **Cross-validation temporal**: Validación robusta en series temporales
4. **Métricas financieras**: Sharpe ratio, drawdown máximo, etc.

### Integración

1. **Representaciones latentes**: Usar MLPLatentEncoder para comprimir observaciones
2. **Logging de experimentos**: MLflow, Weights & Biases, o TensorBoard
3. **Hiperparámetro tuning**: Optuna, Ray Tune, o búsqueda manual

## Relación con el TFM

DQNAgent juega un papel fundamental en el Trabajo de Fin de Máster:

1. **Primer agente entrenable**: Valida que el pipeline RL funciona correctamente
2. **Baseline para comparaciones**: Servirá como referencia para evaluar agentes con representaciones latentes
3. **Validación de entorno**: Confirma que FinancialEnv proporciona señales de aprendizaje útiles
4. **Base para extensiones**: Arquitectura modular permite añadir Double DQN, Dueling DQN, etc.

El objetivo final del TFM es comparar agentes entrenados con representaciones latentes contra baselines como DQNAgent, RandomAgent y BuyAndHoldAgent, demostrando si las representaciones latentes mejoran el rendimiento en trading financiero.
