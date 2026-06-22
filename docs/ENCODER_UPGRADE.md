# Upgrade del Encoder Latente (familia de arquitecturas + artefactos)

> Contexto de la sesión: commit `encoder robusto` (`e9f6a22`). Este documento resume
> el rediseño del subsistema `representations/`: de un único `MLPLatentEncoder` plano
> a una **familia de encoders temporales** (MLP / TCN / GRU) construidos por factory,
> con un **sistema de artefactos** autocontenido para encoders preentrenados.

## Objetivo

Antes del upgrade, la única representación latente era un MLP que aplanaba la ventana
`(L, F)` a un vector `L·F`, perdiendo toda la estructura temporal. El objetivo del
upgrade es:

1. Tratar la observación como **serie temporal** `(N, L, F)` y dejar que el encoder
   modele la dinámica (no solo correlaciones estáticas entre posiciones aplanadas).
2. Poder intercambiar la arquitectura del encoder (`mlp` / `tcn` / `gru`) sin tocar
   el agente ni el pipeline, solo cambiando `encoder_type` en la configuración.
3. Persistir un encoder preentrenado como **artefacto reutilizable** que lleva su
   propia arquitectura, estadísticas de normalización y procedencia, de modo que
   cargarlo en un experimento distinto sea seguro y falle ruidosamente si es
   incompatible.

## Arquitectura del módulo

```
representations/
├── base.py          # LatentEncoder (ABC): contrato común
├── mlp_encoder.py   # MLPLatentEncoder  (compatibilidad + decoder (L,F))
├── tcn_encoder.py   # TCNLatentEncoder  (convoluciones causales dilatadas)
├── gru_encoder.py   # GRULatentEncoder  (recurrente)
├── factory.py       # build_encoder(encoder_type, **kwargs)
└── artifact.py      # save_encoder_artifact / load_encoder_artifact
```

## Contrato común: `LatentEncoder` (base.py)

Clase base abstracta (`nn.Module` + `ABC`). Define la interfaz que todo encoder debe
cumplir, lo que permite que agente, pretrainer y sistema de artefactos sean agnósticos
a la arquitectura concreta:

- `forward(x) -> (N, latent_dim)`: codifica `(N, L, F)` o `(N, input_dim)` al latente.
- `decode(z)`: reconstruye la entrada (necesario para la pérdida de reconstrucción).
- `get_arch_config() -> Dict`: devuelve `encoder_type` + kwargs de arquitectura.
  Es la pieza clave que permite **reconstruir el encoder desde un artefacto**.
- `encode` (alias de forward), `reconstruction_loss` (MSE recon, aplana si hace falta).
- `freeze()` / `unfreeze()`: congelan/descongelan todos los parámetros y fijan
  `is_frozen`. Usado por los brazos C y D (encoder congelado).

## Las tres arquitecturas

### MLPLatentEncoder (mlp_encoder.py)

Encoder MLP retrocompatible. Acepta entrada plana `(N, input_dim)` **o** secuencial
`(N, L, F)` (aplana internamente). Novedad respecto a la versión previa: si se construye
con `input_len` y `n_features`, el **decoder devuelve `(N, L, F)`** en lugar de un vector
plano, homogeneizando la forma de salida con TCN/GRU. Mantiene métodos finos de
congelado (`freeze_encoder`/`freeze_decoder`, `get_encoder_parameters`, …) para
compatibilidad con el `AutoencoderTrainer` legado. Activaciones soportadas: relu, tanh,
sigmoid, leaky_relu, gelu.

### TCNLatentEncoder (tcn_encoder.py)

Encoder temporal con **convoluciones 1D causales y dilatadas** (Temporal Convolutional
Network). Es el encoder por defecto de los experimentos robustos.

- Bloque base `_CausalResBlock`: `Conv1d` dilatada con **padding causal izquierdo**
  `(kernel_size-1)·dilation` (no ve el futuro), `LayerNorm` sobre canal, activación,
  dropout y **conexión residual** (skip 1×1 si cambian los canales).
- Entrada `(N, L, F)` → transpone a `(N, F, L)` para `Conv1d` → pila de bloques con
  dilataciones crecientes → proyección lineal del **último paso temporal** (causal) a
  `latent_dim`.
- Decoder MVP: lineal `latent_dim → L·F`, reshape a `(N, L, F)`.
- **Campo receptivo** `RF = 1 + (K-1)·Σ dilations`. Si `RF < L` emite un `warning`
  (la ventana no se cubre por completo). En la run robusta se eligió
  `dilations=[1,2,4,8]`, `K=3` → `RF = 1 + 2·15 = 31 ≥ L=30` (cobertura completa).

### GRULatentEncoder (gru_encoder.py)

Encoder recurrente. `nn.GRU(batch_first=True)` sobre `(N, L, F)`; toma el **último
estado oculto** de la última capa y lo proyecta a `latent_dim`. Decoder lineal análogo
al de la TCN. Pensado como alternativa/ablación frente a la TCN.

## Factory: `build_encoder` (factory.py)

Punto único de construcción. `build_encoder(encoder_type, **kwargs)` despacha a la clase
concreta según `"mlp" | "tcn" | "gru"` y normaliza nombres de argumentos
(`L`/`F` ↔ `input_len`/`n_features`; para MLP deriva `input_dim = L·F` si no se da).
Lanza `ValueError` con tipos desconocidos. `encoder_kwargs_from_config(arch_config)`
extrae los kwargs de un `arch_config` de artefacto descartando claves no constructoras
(`encoder_type`, `feature_names`).

Esto es lo que permite que el agente y la config solo manejen `encoder_type` + kwargs,
sin importar la clase concreta.

## Sistema de artefactos (artifact.py)

Un **artefacto** es un `.pt` autocontenido con cuatro piezas:

| Clave | Contenido |
|---|---|
| `encoder_state_dict` | Pesos del encoder **sin** cabezas auxiliares (p. ej. sin la cabeza de forecasting del pretrainer). |
| `arch_config` | `encoder_type` + kwargs de arquitectura + `L`, `F`, `feature_names`. Permite reconstruir el encoder con `build_encoder`. |
| `norm_stats` | `mean`/`std` por feature del corpus de entrenamiento + `feature_names`. |
| `provenance` | Metadatos: universo, eval_tickers, excluidos, fechas, semilla, nº ventanas, `best_val_loss`, `trained_at`, `git_commit`. |

- `save_encoder_artifact(encoder, norm_stats, provenance, path)`: serializa las cuatro
  piezas.
- `load_encoder_artifact(path, cfg=None)`: reconstruye el encoder desde `arch_config`
  vía `build_encoder`, carga los pesos y devuelve `(encoder, norm_stats, provenance)`.
  Si se pasa `cfg`, **valida compatibilidad** (`_validate_artifact_compat`): que `L`,
  `F` y `feature_names` del artefacto coincidan con `cfg.lookback_window`,
  `len(cfg.features)` y `cfg.features`. Si no coinciden, lanza `ValueError` detallado
  (fallo ruidoso, no silencioso).

Esta validación es la red de seguridad que evita usar por error un encoder entrenado con
otra ventana/feature-set en un experimento incompatible.

## Integración aguas abajo

- **`LatentDQNAgent`** construye su encoder por factory a partir de `encoder_type` y
  kwargs, y expone `load_artifact_encoder(path, cfg)` para inyectar un artefacto v2 ya
  entrenado (brazo D). También admite `precomputed_latents=True` (sin encoder interno).
- **`EncoderPretrainer` / `pretrain_offline`** entrenan cualquiera de las tres
  arquitecturas y exportan solo el encoder (sin la cabeza de forecasting) como artefacto.
- **`ExperimentConfig.encoder_kwargs()`** traduce la config del experimento a los kwargs
  que espera la factory según `encoder_type`.

## Notas de diseño

- La forma de salida del decoder se homogeneizó a `(N, L, F)` en las tres arquitecturas
  (el MLP la respeta cuando recibe `input_len`/`n_features`), de modo que las pérdidas de
  reconstrucción aplanan de forma consistente.
- La cabeza de forecasting **nunca** se guarda en el artefacto: vive solo en el
  `EncoderPretrainer`. El artefacto contiene exclusivamente el encoder reutilizable.
- `arch_config` es la fuente de verdad de la arquitectura: cambiar la arquitectura sin
  regenerar el artefacto se detecta al cargar (mismatch de pesos o de `L`/`F`).
