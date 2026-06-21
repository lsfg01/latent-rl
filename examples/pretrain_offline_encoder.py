"""
Script de preentrenamiento offline del encoder robusto (brazo D).

Uso:
    python examples/pretrain_offline_encoder.py

Produce un artefacto autocontenido en ``models/encoders/tcn_heavy.pt`` que puede
cargarse en el pipeline de experimentos vía ``cfg.heavy_encoder_path``.

El universo de activos es diverso (ETFs de renta variable, commodities, divisas)
y está diseñado para excluir los eval_tickers SPY, TSLA y BTC-USD y sus parientes,
garantizando que el encoder no ha visto esos activos durante el preentrenamiento.
"""

from latent_rl.pretraining import PretrainConfig, pretrain_offline


def main() -> None:
    cfg = PretrainConfig(
        # Universo amplio (excluye SPY, IVV, VOO, TSLA, BTC-USD, BTC-EUR, ETH-USD)
        universe=[
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "BAC",
            "XOM", "CVX", "JNJ", "PFE", "GE", "CAT", "BA",
            "GLD", "SLV", "USO",        # commodities
            "EFA", "EEM",               # ETFs internacionales
            "QQQ", "DIA",               # índices US (distintos de SPY)
        ],
        eval_tickers=["SPY", "TSLA", "BTC-USD"],
        relatives={
            "SPY": ["IVV", "VOO"],
            "BTC-USD": ["BTC-EUR", "ETH-USD"],
            "TSLA": [],
        },
        start_date="2015-01-01",
        end_date="2023-12-31",
        interval="1d",
        cache_dir=".data_cache",
        features=[
            "log_return", "high_low_range", "close_open_pct",
            "volume_ratio", "rsi_14", "atr_pct", "market_regime", "ma_ratio",
        ],
        lookback=20,
        encoder_type="tcn",
        latent_dim=32,
        kernel_size=3,
        dilations=[1, 2, 4, 8],
        channels=64,
        k=5,
        lambda_forecast=0.5,
        n_epochs=100,
        batch_size=256,
        learning_rate=5e-4,
        val_ratio=0.15,
        early_stopping_patience=10,
        min_asset_length=200,
        seed=42,
        output_path="models/encoders/tcn_heavy.pt",
    )

    print("=== Preentrenamiento offline del encoder TCN ===")
    print(f"Universo: {len(cfg.universe)} activos")
    print(f"Eval tickers (excluidos): {cfg.eval_tickers}")
    print(f"Features ({len(cfg.features)}): {cfg.features}")
    print(f"Lookback={cfg.lookback}  latent_dim={cfg.latent_dim}")
    print(f"Épocas={cfg.n_epochs}  batch={cfg.batch_size}  lr={cfg.learning_rate}")

    output = pretrain_offline(cfg)
    print(f"\nArtefacto guardado en: {output}")
    print("\nPara usar en el experimento:")
    print("  cfg = ExperimentConfig(")
    print(f"      heavy_encoder_path='{output}',")
    print("      run_arms=['A', 'B', 'C', 'D'],")
    print("  )")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
