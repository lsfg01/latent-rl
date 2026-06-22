"""CLI para exportar figuras internas desde una run ya ejecutada."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from latent_rl.reporting.memory_figures import export_memory_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera figuras reproducibles sin recalcular resultados."
    )
    parser.add_argument("--results-dir", required=True, help="Directorio de la run")
    parser.add_argument("--out-dir", required=True, help="Directorio de artefactos")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "svg"],
        choices=["png", "svg", "html"],
        help="Formatos de salida (default: png svg)",
    )
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Incluye F5/F6 y benchmarks donde resulten informativos.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = export_memory_figures(
        args.results_dir,
        args.out_dir,
        formats=args.formats,
        width=args.width,
        height=args.height,
        scale=args.scale,
        include_optional=args.include_optional,
    )
    print(f"Generadas {len(manifest['figures'])} figuras en {manifest['out_dir']}")
    for item in manifest["figures"]:
        print(f"  {item['id']}: {', '.join(item['files'])}")
    print("  manifest: figure_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
