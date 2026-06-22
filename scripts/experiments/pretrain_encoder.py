"""
Fase 1 — Preentrenamiento offline del encoder TCN robusto.

Genera el artefacto en ENCODER_ARTIFACT (models/encoders/tcn_heavy.pt).
El artefacto es reutilizable: solo hay que regenerarlo si cambia la
arquitectura (L, F, latent_dim, dilations…) o el corpus.

Uso:
    # Smoke-run rápido para validar el flujo de punta a punta
    python -m scripts.experiments.pretrain_encoder --smoke

    # Entrenamiento real (tarda ~15-30 min según hardware)
    python -m scripts.experiments.pretrain_encoder
"""

import argparse
import logging

from latent_rl.pretraining import PretrainConfig, pretrain_offline
from scripts.experiments.config import (
    LOOKBACK, FEATURES, ENCODER_TYPE, LATENT_DIM,
    TCN_KERNEL, TCN_DILATIONS, TCN_CHANNELS,
    K_FORECAST, LAMBDA_FORECAST,
    EVAL_TICKERS, ENCODER_ARTIFACT, SMOKE_ENCODER_ARTIFACT,
)

# ---------------------------------------------------------------------------
# Universo de preentrenamiento
# ---------------------------------------------------------------------------
# Variado por sector. Excluye los eval_tickers (SPY, TSLA, BTC-USD)
# y sus parientes (IVV, VOO, BTC-EUR, ETH-USD) por anti-fuga de datos.
UNIVERSE = [
    # Tecnología
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "ADBE", "CRM", "ORCL", "INTC",
    # Financiero
    "JPM", "BAC", "GS", "MS", "V",
    # Salud
    "JNJ", "PFE", "UNH", "MRK", "ABBV",
    # Consumo / retail
    "KO", "PG", "WMT", "MCD", "COST", "NKE",
    # Industrial / energía
    "CAT", "BA", "HON", "XOM", "CVX",
    # Comunicación / entretenimiento
    "DIS", "NFLX", "T",
    # ETFs (no familia SPY)
    "QQQ", "IWM", "DIA", "TLT", "GLD",
]

# ---------------------------------------------------------------------------
# Construcción de la configuración
# ---------------------------------------------------------------------------

def build_config(
    smoke: bool = False,
    output_path: str | None = None,
) -> PretrainConfig:
    """Devuelve PretrainConfig para entrenamiento real o smoke-run."""
    if smoke:
        return PretrainConfig(
            universe=UNIVERSE[:5],  # solo 5 activos
            eval_tickers=EVAL_TICKERS,
            start_date="2020-01-01",
            end_date="2023-12-31",
            interval="1d",
            cache_dir=".data_cache",
            features=FEATURES,
            lookback=LOOKBACK,
            encoder_type=ENCODER_TYPE,
            latent_dim=LATENT_DIM,
            kernel_size=TCN_KERNEL,
            dilations=TCN_DILATIONS,
            channels=TCN_CHANNELS,
            k=K_FORECAST,
            lambda_forecast=LAMBDA_FORECAST,
            n_epochs=3,
            batch_size=64,
            learning_rate=5e-4,
            val_ratio=0.15,
            early_stopping_patience=3,
            min_asset_length=100,
            seed=42,
            output_path=output_path or SMOKE_ENCODER_ARTIFACT,
        )

    return PretrainConfig(
        universe=UNIVERSE,
        eval_tickers=EVAL_TICKERS,
        relatives={
            "SPY":     ["IVV", "VOO"],
            "BTC-USD": ["BTC-EUR", "ETH-USD"],
            "TSLA":    [],
        },
        start_date="2010-01-01",
        end_date="2023-12-31",
        interval="1d",
        cache_dir=".data_cache",
        features=FEATURES,
        lookback=LOOKBACK,
        encoder_type=ENCODER_TYPE,
        latent_dim=LATENT_DIM,
        kernel_size=TCN_KERNEL,
        dilations=TCN_DILATIONS,
        channels=TCN_CHANNELS,
        k=K_FORECAST,
        lambda_forecast=LAMBDA_FORECAST,
        n_epochs=100,
        batch_size=256,
        learning_rate=5e-4,
        val_ratio=0.15,
        early_stopping_patience=10,
        min_asset_length=250,
        seed=42,
        output_path=output_path or ENCODER_ARTIFACT,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1: preentrenar encoder TCN")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-run con parámetros mínimos para validar el flujo end-to-end",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Ruta del artefacto (por defecto separa modelos smoke y reales)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = build_config(smoke=args.smoke, output_path=args.output_path)

    print("=" * 70)
    print(f"{'SMOKE RUN' if args.smoke else 'ENTRENAMIENTO REAL'}")
    print("=" * 70)
    print(f"  Universo         : {len(cfg.universe)} activos")
    print(f"  Eval (excluidos) : {cfg.eval_tickers}")
    print(f"  Excluidos total  : {cfg.excluded_symbols()}")
    print(f"  Periodo          : {cfg.start_date} -> {cfg.end_date}")
    print(f"  Features ({len(cfg.features)})   : {cfg.features}")
    print(f"  Lookback (L)     : {cfg.lookback}")
    print(f"  Encoder          : {cfg.encoder_type}  latent_dim={cfg.latent_dim}")
    print(f"  TCN kernel={cfg.kernel_size} dilations={cfg.dilations} channels={cfg.channels}")
    print(f"  RF               : {1 + (cfg.kernel_size - 1) * sum(cfg.dilations)} (>= L={cfg.lookback}?)")
    print(f"  Épocas           : {cfg.n_epochs}  batch={cfg.batch_size}  lr={cfg.learning_rate}")
    print(f"  Salida           : {cfg.output_path}")
    print("=" * 70)

    output_path = pretrain_offline(cfg)

    # Verificar el artefacto
    import torch
    ckpt = torch.load(output_path, map_location="cpu", weights_only=False)
    arch = ckpt["arch_config"]
    prov = ckpt["provenance"]

    print("\n=== Verificación del artefacto ===")
    print(f"  encoder_type : {arch.get('encoder_type')}")
    print(f"  L            : {arch.get('L')}")
    print(f"  F            : {arch.get('F')}")
    print(f"  latent_dim   : {arch.get('latent_dim')}")
    print(f"  feature_names: {arch.get('feature_names')}")
    print(f"  n_assets_eff : {prov.get('n_assets_effective')}")
    print(f"  n_windows    : {prov.get('n_total_windows')}")
    print(f"  best_val_loss: {prov.get('best_val_loss', 'N/A'):.6f}")
    print(f"  trained_at   : {prov.get('trained_at')}")
    print(f"\nArtefacto guardado en: {output_path}")


if __name__ == "__main__":
    main()
