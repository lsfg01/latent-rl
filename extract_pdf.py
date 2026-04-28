from pathlib import Path
from pypdf import PdfReader

pdf_path = Path("docs/tfm.pdf")
out_path = Path("docs/tfm_extracted.txt")

print(f"Buscando PDF en: {pdf_path.resolve()}")

if not pdf_path.exists():
    raise FileNotFoundError(f"No existe el archivo: {pdf_path}")

reader = PdfReader(str(pdf_path))
parts = []

print(f"Páginas encontradas: {len(reader.pages)}")

for i, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    parts.append(f"\n\n===== PÁGINA {i} =====\n\n{text}")

out_path.write_text("".join(parts), encoding="utf-8")

print(f"Texto extraído en: {out_path.resolve()}")
print(f"Tamaño del texto extraído: {out_path.stat().st_size} bytes")
