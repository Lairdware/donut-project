"""
Konvertiert alle PDFs in dataset/pdfs/ zu PNG-Bildern in dataset/images/.
Nutzt pypdfium2 für hohe Qualität und Geschwindigkeit.

Aufruf:
    python tools/pdf_to_images.py                    # alle PDFs
    python tools/pdf_to_images.py --dpi 150          # niedrigere Auflösung
    python tools/pdf_to_images.py --file meine.pdf   # einzelne Datei

Ausgabe:
    dataset/images/<dateiname>_p0.png   ← Seite 1
    dataset/images/<dateiname>_p1.png   ← Seite 2  (falls mehrseitig)
    ...

Nach der Konvertierung:
    - Für synthetische PDFs: Labels bereits in dataset/labels.jsonl vorhanden.
    - Für eigene PDFs: dataset/labels.jsonl manuell ergänzen (siehe Vorlage unten).
"""

import argparse
import sys
from pathlib import Path

import pypdfium2 as pdfium


def convert_pdf(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list[str]:
    """Konvertiert Seite 1 einer PDF-Datei in ein PNG-Bild."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    created = []

    page = pdf[0]
    bitmap = page.render(scale=scale, rotation=0)
    pil_img = bitmap.to_pil()

    img_name = f"{pdf_path.stem}_p0.png"
    img_path = output_dir / img_name
    pil_img.save(img_path, "PNG")
    created.append(img_name)

    pdf.close()
    return created


def main():
    parser = argparse.ArgumentParser(
        description="PDF → PNG Konverter (pypdfium2)"
    )
    parser.add_argument("--pdf-dir", default="dataset/pdfs",
                        help="Verzeichnis mit PDFs (Standard: dataset/pdfs)")
    parser.add_argument("--img-dir", default="dataset/images",
                        help="Ausgabe-Verzeichnis für Bilder (Standard: dataset/images)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Auflösung in DPI (Standard: 200)")
    parser.add_argument("--file", type=str, default=None,
                        help="Nur diese eine PDF-Datei konvertieren")
    parser.add_argument("--force", action="store_true",
                        help="Bereits vorhandene Bilder überschreiben")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    img_dir = Path(args.img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        pdf_files = [Path(args.file)]
    else:
        pdf_files = sorted(pdf_dir.rglob("*.pdf"))

    if not pdf_files:
        print(f"Keine PDFs in '{pdf_dir}' gefunden.")
        sys.exit(0)

    print(f"{len(pdf_files)} PDFs gefunden → konvertiere mit {args.dpi} DPI ...\n")

    total_images = 0
    skipped = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        # Prüfen ob Bilder bereits existieren
        first_page_img = img_dir / f"{pdf_path.stem}_p0.png"
        if first_page_img.exists() and not args.force:
            skipped += 1
            continue

        try:
            created = convert_pdf(pdf_path, img_dir, dpi=args.dpi)
            total_images += len(created)
            pages_str = ", ".join(created)
            print(f"  [{i:>4}/{len(pdf_files)}] {pdf_path.name:<35} → {pages_str}")
        except Exception as e:
            print(f"  [FEHLER] {pdf_path.name}: {e}")

    print(f"\nFertig! {total_images} Bilder erzeugt", end="")
    if skipped:
        print(f" ({skipped} bereits vorhanden, übersprungen)", end="")
    print(f"\nBilder: {img_dir}/")

    # Hinweis für eigene PDFs
    unlabeled = []
    labels_file = Path("dataset/labels.jsonl")
    if labels_file.exists():
        labeled = set()
        with open(labels_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = __import__("json").loads(line)
                    labeled.add(entry.get("image", ""))
        unlabeled = [
            p.name for p in img_dir.glob("*.png") if p.name not in labeled
        ]

    if unlabeled:
        print(f"\n⚠  {len(unlabeled)} Bilder ohne Label in dataset/labels.jsonl:")
        for name in unlabeled[:5]:
            print(f'   {{"image": "{name}", "order_number": "", "total": ""}}')
        if len(unlabeled) > 5:
            print(f"   ... und {len(unlabeled) - 5} weitere")
        print(f"\n   → Bitte dataset/labels.jsonl ergänzen, dann:")
        print(f"     python tools/prepare_dataset.py")


if __name__ == "__main__":
    main()
