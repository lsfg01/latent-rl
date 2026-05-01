# Latent RL

Plataforma experimental para Reinforcement Learning financiero con representaciones latentes.

## Descripción

Este proyecto es una librería/paquete Python de RL financiero para un TFM que incluye:

- **FinancialEnv**: Entorno financiero compatible con Gymnasium
- **Agentes**: RandomAgent, BuyAndHoldAgent, DQNAgent, LatentDQNAgent
- **Representaciones latentes**: MLPLatentEncoder
- **Preentrenamiento**: AutoencoderTrainer
- **Evaluación**: LatentAdvantageIndex / IVL (Índice de Ventaja Latente)
- **Scripts de experimentación**: Ejemplos en `examples/`
- **Dashboard interactivo**: Visualización en `dashboard/app.py`
- **Resultados**: CSVs generados en `results/`
- **Modelos**: Modelos entrenados en `models/`

## Requisitos

- Python 3.9 o superior
- Git
- VS Code (opcional pero recomendado)

## Instalación desde cero

### 1. Clonar el repositorio

```powershell
git clone <url-del-repositorio>
cd tfm-latent-rl
```

### 2. Crear entorno virtual

```powershell
python -m venv .venv
```

### 3. Activar entorno virtual

```powershell
.\.venv\Scripts\activate
```

### 4. Instalar dependencias

```powershell
pip install -e .
```

### 5. Crear archivo .env

Si existe `.env.example`:

```powershell
cp .env.example .env
```

Si no existe `.env.example`, crea manualmente un archivo `.env` en la raíz del proyecto con las siguientes variables:

```env
PROJECT_ENV=local
PYTHONUTF8=1
RESULTS_DIR=results
MODELS_DIR=models
```

**Nota**: El archivo `.env` puede dejarse vacío inicialmente, pero se reserva para configuración local. No se requieren claves de API para yfinance.

## Verificar instalación

```powershell
python -m pytest
```

Si obtienes el error `No module named pytest`, instala las dependencias de desarrollo:

```powershell
pip install pytest pytest-cov
```

## Ejecutar ejemplos principales

```powershell
python examples\latent_index_example.py
python examples\latent_advantage_example.py
python examples\compare_agents_multiseed_experiment.py
```

## Generar datos del dashboard

El dashboard necesita los siguientes archivos CSV en `results/`:

- `results/latent_index.csv`
- `results/ivl_results.csv`
- `results/agent_summary.csv`
- `results/agent_seed_metrics.csv`

Estos archivos se generan ejecutando los scripts de ejemplo:

```powershell
python examples\latent_index_example.py
python examples\latent_advantage_example.py
python examples\compare_agents_multiseed_experiment.py
```

**Alternativa**: El dashboard tiene un botón "▶ Generar datos del dashboard" que ejecuta estos scripts automáticamente.

## Abrir dashboard

```powershell
python dashboard\app.py
```

El dashboard se abrirá en: **http://127.0.0.1:8050**

Para detener el servidor, presiona **Ctrl+C** en la terminal.

## Uso básico como librería

### BuyAndHoldAgent (Línea Base Financiera)

```python
from latent_rl.envs import FinancialEnv
from latent_rl.agents import BuyAndHoldAgent
import pandas as pd
import numpy as np

# Crear datos de prueba
data = pd.DataFrame({
    "open": np.random.randn(100) + 100,
    "high": np.random.randn(100) + 105,
    "low": np.random.randn(100) + 95,
    "close": np.random.randn(100) + 100,
    "volume": np.random.randint(1000, 10000, 100)
})

# Crear entorno
env = FinancialEnv(data, lookback_window=10)

# Crear agente buy-and-hold
agent = BuyAndHoldAgent(env.action_space)

# Ejecutar un episodio
obs, info = env.reset(seed=42)
done = False
total_reward = 0.0

while not done:
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Recompensa total: {total_reward:.2f}")
env.close()
```

### Comparación de Baselines

```python
from latent_rl.envs import FinancialEnv
from latent_rl.agents import RandomAgent, BuyAndHoldAgent
import pandas as pd
import numpy as np

# Crear datos
data = pd.DataFrame({
    "open": np.random.randn(100) + 100,
    "high": np.random.randn(100) + 105,
    "low": np.random.randn(100) + 95,
    "close": np.random.randn(100) + 100,
    "volume": np.random.randint(1000, 10000, 100)
})

# Crear entorno
env = FinancialEnv(data, lookback_window=10)

# Comparar agentes
random_agent = RandomAgent(env.action_space, seed=42)
buy_and_hold_agent = BuyAndHoldAgent(env.action_space)

# Ejecutar episodios y comparar métricas
# (ver examples/baseline_comparison.py para ejemplo completo)
```

### Carga de datos desde CSV

```python
from latent_rl.data import CSVDataLoader, DataPreprocessor
from latent_rl.envs import FinancialEnv
from latent_rl.representations import MLPLatentEncoder
from latent_rl.evaluation import FinancialMetrics

# Cargar datos
loader = CSVDataLoader("data/ohlcv.csv")
data = loader.load()

# Preprocesar
preprocessor = DataPreprocessor()
processed_data = preprocessor.normalize(data)

# Crear entorno
env = FinancialEnv(processed_data)

# Crear encoder
encoder = MLPLatentEncoder(input_dim=10, latent_dim=5)

# Evaluar
metrics = FinancialMetrics()
sharpe = metrics.sharpe_ratio(returns)
```

### Descarga de datos desde Yahoo Finance

```python
from latent_rl.data import YahooFinanceLoader

# Descargar datos desde Yahoo Finance
loader = YahooFinanceLoader()

df = loader.download(
    symbol="AAPL",
    start="2020-01-01",
    end="2024-12-31",
    interval="1d"
)

# Guardar como CSV compatible con CSVDataLoader
loader.save_csv(df, "data/raw/AAPL.csv")

# O descargar y guardar en un solo paso
df = loader.download_and_save(
    symbol="MSFT",
    start="2020-01-01",
    end="2024-12-31",
    file_path="data/raw/MSFT.csv"
)
```

## Estructura del proyecto

```
tfm-latent-rl/
├── src/
│   └── latent_rl/
│       ├── agents/          # Agentes de RL
│       ├── envs/            # Entornos financieros
│       ├── data/            # Carga y preprocesado de datos
│       ├── evaluation/      # Métricas financieras y IVL
│       ├── pretraining/     # Preentrenamiento de encoders
│       ├── analysis/        # Análisis de representaciones latentes
│       └── representations/  # Encoders latentes
├── examples/               # Scripts de experimentación
├── dashboard/              # Dashboard interactivo
├── tests/                  # Tests unitarios
├── results/                # CSVs generados (no subir a Git)
├── models/                 # Modelos entrenados (no subir a Git)
├── reports/                # Reportes de análisis (no subir a Git)
├── data/                   # Datos financieros (no subir a Git)
└── pyproject.toml          # Configuración del proyecto
```

## Archivos generados

Las siguientes carpetas se generan durante la ejecución y **no deben subirse a Git**:

- `results/`: CSVs generados por scripts de ejemplo y dashboard
- `reports/`: Reportes de análisis y visualizaciones
- `models/`: Modelos entrenados (encoders, agentes)
- `data/`: Datos financieros descargados o procesados
- `.venv/`: Entorno virtual

Estas carpetas están excluidas en `.gitignore`.

## Problemas frecuentes

### 1. No module named pytest

**Error**: `ModuleNotFoundError: No module named 'pytest'`

**Solución**:
```powershell
pip install pytest pytest-cov
```

### 2. collected 0 items

**Error**: `pytest` no encuentra tests

**Solución**: Asegúrate de estar en la raíz del proyecto donde están `tests/` y `pyproject.toml`.

```powershell
cd C:\lab\tfm-latent-rl
python -m pytest
```

### 3. No module named latent_rl.data

**Error**: `ModuleNotFoundError: No module named 'latent_rl.data'`

**Solución**:
1. Haz `git pull` para obtener los últimos cambios
2. Comprueba que existe `src/latent_rl/data/`
3. Verifica que `.gitignore` usa `/data/` y no `data/` para no ignorar la carpeta del módulo

### 4. Dashboard vacío

**Problema**: El dashboard se abre pero no muestra datos

**Solución**: Genera los CSVs necesarios:

```powershell
python examples\latent_index_example.py
python examples\latent_advantage_example.py
python examples\compare_agents_multiseed_experiment.py
```

O usa el botón "▶ Generar datos del dashboard" en la interfaz.

### 5. UnicodeEncodeError en Windows

**Error**: `UnicodeEncodeError: 'charmap' codec can't encode character`

**Solución**: Ejecuta con UTF-8:

```powershell
$env:PYTHONUTF8=1
python examples\latent_advantage_example.py
```

O añade esto a tu archivo `.env`:
```env
PYTHONUTF8=1
```

### 6. VS Code usa Python incorrecto

**Problema**: VS Code no usa el entorno virtual

**Solución**:
1. Presiona `Ctrl+Shift+P`
2. Selecciona "Python: Select Interpreter"
3. Elige `.venv\Scripts\python.exe`

## Flujo recomendado de desarrollo

```powershell
# Actualizar código
git pull

# Activar entorno virtual
.\.venv\Scripts\activate

# Instalar/actualizar dependencias
pip install -e .

# Ejecutar tests
python -m pytest

# Generar datos del dashboard (opcional)
python examples\latent_index_example.py
python examples\latent_advantage_example.py
python examples\compare_agents_multiseed_experiment.py

# Abrir dashboard
python dashboard\app.py
```

## Notas metodológicas

- **IVL (Índice de Ventaja Latente)**: Es la métrica principal del proyecto, que combina múltiples dimensiones de rendimiento (Sharpe ratio, drawdown, estabilidad entre semillas, gap IS/OOS).
- **Proyecciones latentes**: Las proyecciones PCA/norm/first_component son análisis auxiliares del espacio latente, no corresponden al IVL principal.
- **Experimentos sintéticos**: Los ejemplos en `examples/` usan datos sintéticos y son demostrativos, no optimizados para producción.
- **Conclusiones formales**: Para obtener conclusiones válidas se requieren:
  - Datos financieros reales (no sintéticos)
  - Múltiples semillas aleatorias
  - Splits temporales In-Sample / Out-of-Sample apropiados
  - Validación cruzada temporal
  - Tests de significancia estadística

## Licencia

MIT
