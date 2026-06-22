# Preentrenamiento Offline del Encoder Robusto (Fase 1 — corpus del brazo D)

> Contexto de la sesión: commit `encoder robusto` (`e9f6a22`). Este documento resume el
> pipeline de **preentrenamiento offline** que produce el artefacto `tcn_heavy.pt`
> usado por el brazo D, su objetivo multitarea (reconstrucción + forecasting) y, sobre
> todo, sus **garantías anti-fuga de datos**.

## Objetivo

El brazo D del experimento usa un encoder "gordo" entrenado **una sola vez, fuera del
bucle de RL**, sobre un universo amplio de activos. La idea es obtener una representación
de mercado **transferible y robusta**, entrenada con datos que el agente nunca verá en
evaluación, y reutilizarla congelada en SPY / TSLA / BTC-USD.

El pipeline es la "Fase 1" del experimento; la "Fase 2" (entrenamiento RL A/B/C/D) lo
consume vía artefacto. Ambas fases comparten arquitectura a través de
`scripts/experiments/config.py` (fuente única de verdad).

## Módulos

```
pretraining/
├── config.py               # PretrainConfig (incluye validación anti-fuga)
├── encoder_pretrainer.py   # EncoderPretrainer: recon + forecast multitarea
└── offline.py              # pretrain_offline(cfg): pipeline completo end-to-end
scripts/experiments/pretrain_encoder.py  # Runner de Fase 1 (define el UNIVERSE)
scripts/experiments/config.py            # L, F, arquitectura, k, λ compartidos
```

## Anti-fuga de datos (lo más importante)

`PretrainConfig.__post_init__` **aborta** si el universo de entrenamiento solapa con los
activos de evaluación o con sus "parientes":

- `eval_tickers`: SPY, TSLA, BTC-USD (los que se evalúan en Fase 2).
- `relatives`: símbolos correlacionados a excluir, p. ej.
  `SPY → [IVV, VOO]`, `BTC-USD → [BTC-EUR, ETH-USD]`.
- El conjunto excluido es `eval_tickers ∪ relatives`. Si **cualquiera** de ellos aparece
  en `universe`, lanza `ValueError` (violación anti-fuga).

De este modo se garantiza por construcción que el encoder gordo **no ha visto** el activo
(ni un proxy cercano) sobre el que luego se evalúa. El runner define un universo variado
por sector (tecnología, financiero, salud, consumo, industrial/energía, comunicación,
ETFs no-familia-SPY) — 39 activos en la run real.

Validaciones adicionales de la config: `val_ratio ∈ (0,1)`, `k ≥ 1`, `λ_forecast ≥ 0`,
features ⊆ `AVAILABLE_FEATURES`, y **`log_return` obligatorio** (es el target de
forecasting).

## Objetivo multitarea: `EncoderPretrainer`

Pérdida total:

```
L = L_recon + λ · L_forecast
```

- **L_recon**: MSE entre `decode(z)` y la ventana de entrada `(L, F)`.
- **L_forecast**: MSE entre una **cabeza lineal** `latent_dim → k` y los `k` log-returns
  futuros. Fuerza al latente a contener información **predictiva**, no solo
  comprimir/reconstruir.

Detalles:

- `make_windows(df, features, lookback)`: ventana deslizante. `X[i] = feat[i:i+L]`,
  `Y[i] = log_return[i+L : i+L+k]`. No cruza activos (se llama por activo).
- `train(...)`: split temporal train/val **sin barajar**, estandariza los targets de
  forecasting, bucle Adam con **early stopping por `val_loss`** y restauración del mejor
  checkpoint. Mini-batches barajados con semilla.
- La cabeza de forecasting vive **solo** en el trainer; `save_encoder` exporta únicamente
  el encoder (sin cabeza) como artefacto v2.

## Pipeline end-to-end: `pretrain_offline(cfg)` (offline.py)

1. **Carga** cada activo del universo con caché (`DataCache.get_or_download`) y calcula
   features (`FeatureEngineer`). Descarta los que no alcanzan
   `min_asset_length + lookback + k` filas (también tras `dropna`).
2. **Normalización sin fuga**: ajusta `FeatureNormalizer` **solo sobre el tramo train**
   de cada activo (la fracción `1 - val_ratio` inicial), concatenado entre activos.
   `norm_stats` (mean/std por feature) se guarda en el artefacto.
3. **Construye** el encoder por factory con la arquitectura de `cfg`.
4. **Genera ventanas `(X, Y)` por activo** sobre los features normalizados (el
   `log_return` para `Y` se toma sin normalizar; el trainer lo estandariza). No se mezclan
   ventanas de activos distintos dentro de una misma ventana.
5. **Entrena** con recon + forecast (early stopping).
6. **Guarda el artefacto** en `cfg.output_path` con `arch_config` (incluye `feature_names`,
   `L`, `F`), `norm_stats` y `provenance` (universo efectivo, excluidos, fechas, semilla,
   nº ventanas totales, `best_val_loss`, `trained_at`, `git_commit`).

## Configuración compartida (`scripts/experiments/config.py`)

Garantiza que Fase 1 y Fase 2 usen **exactamente** la misma arquitectura (si difieren,
el artefacto falla al cargar). Valores de la run robusta:

| Parámetro | Valor |
|---|---|
| `LOOKBACK` (L) | 30 |
| `FEATURES` (F=8) | log_return, high_low_range, close_open_pct, volume_ratio, rsi_14, atr_pct, market_regime, ma_ratio |
| `ENCODER_TYPE` | tcn |
| `LATENT_DIM` | 32 |
| `TCN_KERNEL` | 3 |
| `TCN_DILATIONS` | [1, 2, 4, 8] → RF = 31 ≥ 30 |
| `TCN_CHANNELS` | 64 |
| `K_FORECAST` | 5 |
| `LAMBDA_FORECAST` | 0.5 |
| `EVAL_TICKERS` | SPY, TSLA, BTC-USD |
| `ENCODER_ARTIFACT` | models/encoders/tcn_heavy.pt |

## Artefacto producido (`models/encoders/tcn_heavy.pt`)

Metadatos reales de procedencia del artefacto entrenado en esta sesión:

- **Arquitectura**: TCN, L=30, F=8, latent_dim=32, kernel=3, dilations=[1,2,4,8],
  channels=64.
- **Universo**: 39 activos (39 solicitados / 39 efectivos), periodo 2010-01-01 → 2023-12-31,
  intervalo diario.
- **Excluidos** (anti-fuga): BTC-EUR, BTC-USD, ETH-USD, IVV, SPY, TSLA, VOO.
- **Ventanas totales**: 134 679.
- **best_val_loss**: ≈ 0.3245.
- **seed**: 42 · **git_commit**: 8bff144.

## Uso

```bash
# Smoke (5 activos, 3 épocas) para validar el flujo
python -m scripts.experiments.pretrain_encoder --smoke

# Entrenamiento real (~15-30 min) → genera models/encoders/tcn_heavy.pt
python -m scripts.experiments.pretrain_encoder
```

El artefacto es **reutilizable**: solo hay que regenerarlo si cambia la arquitectura
(L, F, latent_dim, dilations…) o el corpus.
