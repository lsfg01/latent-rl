"""Ejemplo de descarga de datos desde Yahoo Finance.

Este ejemplo demuestra cómo descargar datos OHLCV desde Yahoo Finance
y guardarlos como CSV para usarlos con CSVDataLoader.
"""

from pathlib import Path
from latent_rl.data import YahooFinanceLoader, CSVDataLoader


def main():
    """Función principal del ejemplo."""
    print("=== Ejemplo de Descarga desde Yahoo Finance ===\n")

    # 1. Crear directorio de datos si no existe
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"1. Directorio de datos: {data_dir}\n")

    # 2. Crear loader de Yahoo Finance
    print("2. Creando loader de Yahoo Finance...")
    loader = YahooFinanceLoader()
    print("   Loader creado\n")

    # 3. Descargar datos de ejemplo
    print("3. Descargando datos de ejemplo...")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2023-01-01"
    end_date = "2024-12-31"

    for symbol in symbols:
        print(f"   Descargando {symbol}...")

        try:
            # Descargar datos
            df = loader.download(
                symbol=symbol,
                start=start_date,
                end=end_date,
                interval="1d"
            )

            print(f"   - Filas descargadas: {len(df)}")
            print(f"   - Columnas: {list(df.columns)}")
            print(f"   - Rango de fechas: {df['date'].min()} a {df['date'].max()}")

            # Guardar como CSV
            csv_path = data_dir / f"{symbol}.csv"
            loader.save_csv(df, csv_path)
            print(f"   - Guardado en: {csv_path}\n")

        except Exception as e:
            print(f"   - Error al descargar {symbol}: {e}\n")

    # 4. Verificar compatibilidad con CSVDataLoader
    print("4. Verificando compatibilidad con CSVDataLoader...")
    for symbol in symbols:
        csv_path = data_dir / f"{symbol}.csv"

        if csv_path.exists():
            try:
                csv_loader = CSVDataLoader(csv_path)
                loaded_df = csv_loader.load(date_column="date")

                print(f"   {symbol}:")
                print(f"   - Filas cargadas: {len(loaded_df)}")
                print(f"   - Columnas: {list(loaded_df.columns)}")

            except Exception as e:
                print(f"   {symbol}: Error al cargar - {e}")

    print("\n=== Ejemplo completado ===")
    print(f"\nLos datos CSV están disponibles en: {data_dir}")
    print("Puedes usarlos con CSVDataLoader para entrenar modelos.")


if __name__ == "__main__":
    main()