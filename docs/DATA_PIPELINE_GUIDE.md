# Guia del Pipeline de Datos — latent-rl

Flujo completo de datos desde la fuente hasta el entorno de trading.

---

## Flujo general

```
Yahoo Finance / CSV local
         |
         v
   [ DataCache ]              <- evita re-descargas (CSV.gz en disco)
         |
         v
  [ FeatureEngineer ]         <- anade indicadores tecnicos a OHLCV
         |
         v
  [ FeatureNormalizer ]       <- z-score fit en IS, transform en OOS/IS
    (fit solo en IS)
         |
   +-----+------+
   |             |
   v             v
 data_is       data_oos       <- division temporal estricta (train_ratio)
   |             |
   v             v
FinancialEnv  FinancialEnv    <- entorno Gymnasium (lookback_window x n_features)
   |             |
   v             v
Entrenamiento  Evaluacion     <- agentes DQN / LatentDQN
```

Con Walk-Forward activo, el flujo aplica el bloque de normalizacion
+ entrenamiento + evaluacion N veces sobre ventanas temporales distintas.

---

## Modulos del pipeline de datos

### 1. DataCache  (`src/latent_rl/data/cache.py`)

Cache local de datos OHLCV en formato CSV.gz.

```python
from latent_rl.data.cache import DataCache

cache = DataCache(".data_cache")
df = cache.get_or_download("SPY", "2020-01-01", "2024-01-01", interval="1d")
```

- Primera llamada: descarga de Yahoo Finance, guarda en `.data_cache/SPY_2020-01-01_2024-01-01_1d.csv.gz`
- Llamadas siguientes: lee del disco (sin red, en milisegundos)
- `force_refresh=True`: fuerza re-descarga aunque exista cache
- `cache.clear("SPY")`: borra archivos de ese ticker

**Columnas de salida:** `[open, high, low, close, volume]` (minusculas, indice numerico)

---

### 2. FeatureEngineer  (`src/latent_rl/data/features.py`)

Calcula indicadores tecnicos y los anade como columnas extra al DataFrame OHLCV.

```python
from latent_rl.data.features import FeatureEngineer, AVAILABLE_FEATURES

fe = FeatureEngineer()
df_features = fe.transform(df_ohlcv, ["log_return", "rsi_14", "atr_pct"])
# Columnas: [open, high, low, close, volume, log_return, rsi_14, atr_pct]
```

**8 features disponibles:**

| Feature        | Descripcion                          | Rango tipico |
|----------------|--------------------------------------|--------------|
| log_return     | log(close_t / close_{t-1})           | [-0.1, 0.1]  |
| high_low_range | (high - low) / close                 | [0, 0.1]     |
| close_open_pct | (close - open) / open                | [-0.05, 0.05]|
| volume_ratio   | volume / rolling_mean(vol, 20)       | [0, 5]       |
| rsi_14         | RSI de 14 periodos                   | [0, 100]     |
| atr_pct        | ATR(14) / close                      | [0, 0.05]    |
| market_regime  | -1 bajista / 0 transicion / +1 alcista | {-1, 0, 1} |
| ma_ratio       | (MA50 - MA200) / close               | [-0.05, 0.05]|

**Invariante critica:** Las columnas OHLCV siempre ocupan las 5 primeras posiciones.
`close` siempre es la columna 3. FinancialEnv depende de esto.

---

### 3. FeatureNormalizer  (`src/latent_rl/data/normalizer.py`)

Z-score con parametros ajustados SOLO en datos IS. Previene data leakage.

```python
from latent_rl.data.normalizer import FeatureNormalizer

norm = FeatureNormalizer()
data_is_norm  = norm.fit_transform(data_is)   # fit aqui
data_oos_norm = norm.transform(data_oos)       # aplica params IS

# Inspeccion
print(norm.mean_)  # media de cada feature en IS
print(norm.std_)   # std de cada feature en IS
```

**Por que no normalizar en OOS:**
Si calculamos la media/std en OOS, el agente "veria el futuro" durante el entrenamiento.
Usamos siempre los parametros de IS para ambos splits.

**OHLCV no se normaliza:** FinancialEnv necesita precios reales para calcular equity.

Funcion de utilidad en el pipeline:
```python
from latent_rl.experiments.utils import normalize_is_oos

data_is_n, data_oos_n, norm = normalize_is_oos(data_is, data_oos)
```

---

### 4. Walk-Forward Analysis  (`src/latent_rl/experiments/utils.py`)

Divide los datos en N ventanas temporales consecutivas. Cada ventana tiene su propio
IS y OOS. El normalizador se recalcula por ventana.

```python
from latent_rl.experiments.utils import walk_forward_splits

splits = walk_forward_splits(data, n_windows=5, is_ratio=0.6)
for wf_is, wf_oos in splits:
    wf_is_n, wf_oos_n, _ = normalize_is_oos(wf_is, wf_oos)
    # ... entrenar y evaluar
```

**Estructura de ventanas (n_windows=5, datos=1000 filas):**

```
|-- Ventana 1 --|-- Ventana 2 --|-- Ventana 3 --|-- Ventana 4 --|-- Ventana 5 --|
|  IS  | OOS   |  IS  | OOS   |  IS  | OOS   |  IS  | OOS   |  IS  | OOS   |
|  120 |  80   |  120 |  80   |  120 |  80   |  120 |  80   |  120 |  80+r |
```

Ventaja: detecta si el agente es robusto en multiples regimenes (alcista, lateral, bajista).

---

### 5. Context Tickers  (`src/latent_rl/experiments/utils.py`)

Activos correlacionados como features adicionales.

```python
from latent_rl.experiments.config import ExperimentConfig, TickerConfig

cfg = ExperimentConfig(
    tickers=["AAPL"],
    ticker_configs=[
        TickerConfig("AAPL", context_tickers=["SPY", "QQQ"])
    ],
)
# load_ticker_with_config("AAPL", cfg) devuelve:
# [open, high, low, close, volume, log_return, ..., spy_log_return, qqq_log_return]
```

Columnas anadidas: `{ticker_normalizado}_log_return` por cada activo de contexto.
Alineacion posicional: para activos del mismo mercado los dias de trading coinciden.

---

## ExperimentConfig — parametros del pipeline de datos

```python
from latent_rl.experiments.config import ExperimentConfig, TickerConfig

cfg = ExperimentConfig(
    # Fuente de datos
    tickers         = ["SPY", "AAPL"],
    start_date      = "2020-01-01",
    end_date        = "2024-01-01",
    train_ratio     = 0.7,           # 70% IS, 30% OOS
    interval        = "1d",          # "1d", "1wk", "1h"
    cache_dir       = ".data_cache", # directorio de cache
    n_obs           = None,          # None = todo el rango

    # Features tecnicos
    features = [
        "log_return", "rsi_14", "atr_pct",
        "market_regime", "ma_ratio"
    ],

    # Normalizacion IS/OOS
    normalize_features = True,       # z-score fit en IS

    # Walk-Forward
    wf_enabled   = False,            # True activa walk-forward
    wf_n_windows = 5,                # numero de ventanas
    wf_is_ratio  = 0.6,              # fraccion IS dentro de c/u

    # Configuracion por ticker (sobreescribe globales)
    ticker_configs = [
        TickerConfig(
            "SPY",
            start_date     = "2015-01-01",   # mas historia que global
            context_tickers = ["QQQ", "VIX"],
        ),
        TickerConfig(
            "AAPL",
            start_date = "2018-01-01",
            n_obs      = 1000,
            interval   = "1d",
        ),
    ],
)
```

---

## Configuraciones tipicas

### Minima (solo OHLCV, sin normalizacion)
```python
cfg = ExperimentConfig(tickers=["SPY"], features=[])
```

### Standard (features + normalizacion)
```python
cfg = ExperimentConfig(
    tickers=["SPY"],
    features=["log_return", "high_low_range", "close_open_pct", "volume_ratio"],
    normalize_features=True,
)
```

### Avanzada (features + normalizacion + walk-forward)
```python
cfg = ExperimentConfig(
    tickers=["SPY"],
    features=["log_return", "rsi_14", "atr_pct", "market_regime"],
    normalize_features=True,
    wf_enabled=True,
    wf_n_windows=5,
    wf_is_ratio=0.6,
)
```

### Con activos correlacionados
```python
cfg = ExperimentConfig(
    tickers=["BTC-USD"],
    ticker_configs=[
        TickerConfig("BTC-USD", context_tickers=["SPY", "GLD"])
    ],
    features=["log_return", "rsi_14"],
    normalize_features=True,
)
```

---

## Archivos de datos

```
latent-rl/
├── .data_cache/                    <- DataCache (auto-generado)
│   ├── SPY_2020-01-01_2024-01-01_1d.csv.gz
│   └── BTC-USD_2018-01-01_2024-01-01_1d.csv.gz
│
├── results/                        <- Resultados de experimentos
│   ├── SPY/
│   │   ├── agent_summary.csv       <- metricas IS/OOS por agente (5 agentes)
│   │   ├── agent_seed_metrics.csv  <- metricas por (agente, semilla, split)
│   │   └── ivl_results.csv         <- IVL por par (DQN vs LatentDQN)
│   └── ticker_comparison.csv       <- IVL cross-ticker (si hay >1 ticker)
```

---

## Scripts de casos de uso

| Script                              | Que demuestra                          |
|-------------------------------------|----------------------------------------|
| examples/uc1_load_data_with_features.py | DataCache + todos los features     |
| examples/uc2_is_oos_normalization.py    | FeatureNormalizer sin leakage      |
| examples/uc3_walk_forward.py            | Walk-Forward en 5 ventanas         |
| examples/uc4_context_tickers.py         | Features de activos correlacionados|
| examples/uc5_full_data_pipeline.py      | Pipeline completo end-to-end       |

Ejecutar todos:
```
python examples/uc1_load_data_with_features.py
python examples/uc2_is_oos_normalization.py
python examples/uc3_walk_forward.py
python examples/uc4_context_tickers.py
python examples/uc5_full_data_pipeline.py
```

Para el dashboard con datos reales:
```
python examples/compare_agents_multiseed_experiment.py
python -m scripts.utilities.compute_ivl
python dashboard/app.py
```

---

## n_features — como se calcula

`FinancialEnv` y el encoder reciben observaciones de forma `(lookback_window, n_features)`.

```
n_features = 5 (OHLCV)
           + len(features)          # features tecnicos
           + len(context_tickers)   # log_return por activo de contexto
```

Ejemplos:
- Solo OHLCV:                        n_features = 5
- + 4 features basicos:              n_features = 9
- + 4 features + 2 context tickers:  n_features = 11

El runner calcula `n_features` automaticamente desde `env.observation_space.shape[1]`.
