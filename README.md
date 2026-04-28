# Latent RL

Plataforma experimental para Reinforcement Learning financiero con representaciones latentes.

## Instalación

```bash
pip install -e .
```

## Uso Básico

### Agentes

#### BuyAndHoldAgent (Línea Base Financiera)

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

#### Comparación de Baselines

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

## Estructura

- `data/`: Carga y preprocesado de datos financieros
- `envs/`: Entornos compatibles con Gymnasium
- `agents/`: Agentes de aprendizaje por refuerzo
- `representations/`: Encoders latentes en PyTorch
- `evaluation/`: Métricas financieras
- `utils/`: Utilidades de configuración

## Desarrollo

```bash
# Ejecutar tests
pytest

# Ejemplo básico
python examples/basic_example.py

# Ejemplo de RandomAgent
python examples/random_agent_example.py

# Comparación de baselines
python examples/baseline_comparison.py
```

## Licencia

MIT