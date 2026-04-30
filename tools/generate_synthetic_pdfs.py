"""
Erzeugt synthetische Bestellungs-PDFs mit reportlab.
Ca. 15% der Dokumente haben absichtlich ein fehlendes Feld —
sowohl visuell im PDF als auch im Label.

Aufruf:
    python tools/generate_synthetic_pdfs.py
    python tools/generate_synthetic_pdfs.py --count 500
"""

import argparse
import json
import random
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

random.seed(42)

COMPANIES = [
    "TechSupply GmbH", "Bürobedarf AG", "Digital Solutions KG",
    "Handels & Co.", "LogiPro GmbH", "Systemhaus Müller",
    "IT-Vertriebs GmbH", "Bürotechnik Schulz", "Netzwerk AG",
    "Hardware Plus KG",
]
PRODUCTS = [
    ("Laptop Dell XPS 15", 1299.00),
    ("Monitor 27 Zoll 4K", 449.99),
    ("Tastatur Logitech MX", 89.99),
    ("Maus ergonomisch", 49.99),
    ("USB-Hub 7-Port", 34.99),
    ("Headset Bluetooth", 129.99),
    ("Drucker HP LaserJet", 289.00),
    ("Webcam Full HD", 79.99),
    ("Netzwerkkabel Cat6 10m", 12.99),
    ("SSD 1TB Samsung", 109.99),
    ("RAM 16GB DDR4", 64.99),
    ("Router WLAN 6", 199.99),
    ("Switch 24-Port", 159.00),
    ("Netzteil 650W", 89.99),
    ("Grafikkarte RTX 3060", 399.99),
]
CITIES = [
    ("10115", "Berlin"), ("20095", "Hamburg"), ("80331", "München"),
    ("50667", "Köln"), ("60311", "Frankfurt"), ("70173", "Stuttgart"),
    ("40213", "Düsseldorf"), ("30159", "Hannover"), ("04109", "Leipzig"),
    ("01067", "Dresden"),
]
STREETS = [
    "Hauptstraße", "Bahnhofstraße", "Schillerstraße", "Goethestraße",
    "Industriestraße", "Am Marktplatz", "Ringstraße", "Gartenweg",
]

BRAND_COLOR = colors.HexColor("#003C7E")
LIGHT_BG    = colors.HexColor("#E8F0FE")


def generate_order(i: int) -> dict:
    plz, city = random.choice(CITIES)
    items = []
    total = 0.0
    for _ in range(random.randint(1, 4)):
        product, price = random.choice(PRODUCTS)
        qty = random.randint(1, 5)
        line = round(price * qty, 2)
        total += line
        items.append({"product": product, "qty": qty, "price": price, "total": line})
    return {
        "order_number": f"ORD-2024-{i:05d}",
        "company":      random.choice(COMPANIES),
        "address":      f"{random.choice(STREETS)} {random.randint(1,200)}, {plz} {city}",
        "date":         f"{random.randint(1,28):02d}.{random.randint(1,12):02d}.2024",
        "items":        items,
        "total":        round(total, 2),
    }


def build_pdf(order: dict, path: str, show_order_number: bool = True, show_total: bool = True):
    """
    Rendert eine Bestellung als PDF.
    show_order_number / show_total steuern ob das jeweilige Feld
    im Dokument sichtbar ist — muss mit dem Label übereinstimmen.
    """
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    h1     = ParagraphStyle("h1", fontSize=16, textColor=BRAND_COLOR,
                             fontName="Helvetica-Bold", spaceAfter=4)
    normal = ParagraphStyle("n",  fontSize=10, fontName="Helvetica", spaceAfter=3)
    small  = ParagraphStyle("s",  fontSize=8,  fontName="Helvetica", textColor=colors.grey)
    bold   = ParagraphStyle("b",  fontSize=10, fontName="Helvetica-Bold")

    story = []

    # Kopfzeile
    header = Table(
        [[Paragraph("MusterFirma GmbH", h1), Paragraph(f"Datum: {order['date']}", normal)]],
        colWidths=["70%", "30%"],
    )
    header.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ALIGN",  (1,0), (1,0),  "RIGHT"),
    ]))
    story.append(header)
    story.append(Paragraph("Musterstraße 1 · 12345 Musterstadt", small))
    story.append(Spacer(1, 0.4*cm))

    # Empfänger
    story.append(Paragraph(order["company"], bold))
    story.append(Paragraph(order["address"], normal))
    story.append(Spacer(1, 0.5*cm))

    # Bestellnummer-Box — nur wenn show_order_number
    if show_order_number:
        t = Table(
            [[Paragraph("Bestellnummer:", bold),
              Paragraph(f"<b>{order['order_number']}</b>", bold)]],
            colWidths=["35%", "65%"],
        )
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), LIGHT_BG),
            ("BOX",           (0,0), (-1,-1), 1.5, BRAND_COLOR),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5*cm))
    else:
        # Platzhalter damit Layout ähnlich bleibt
        story.append(Paragraph("Interne Referenz: wird nachgereicht", small))
        story.append(Spacer(1, 0.5*cm))

    # Positionen-Tabelle
    table_data = [["Produkt", "Menge", "Einzelpreis", "Gesamt"]]
    for item in order["items"]:
        table_data.append([
            item["product"],
            str(item["qty"]),
            f"{item['price']:.2f} EUR",
            f"{item['total']:.2f} EUR",
        ])

    pos_table = Table(table_data, colWidths=["50%", "10%", "20%", "20%"])
    pos_table.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0), BRAND_COLOR),
        ("TEXTCOLOR",      (0,0), (-1,0), colors.white),
        ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F4F7FF")]),
        ("ALIGN",          (1,0), (-1,-1), "RIGHT"),
        ("GRID",           (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",     (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
    ]))
    story.append(pos_table)
    story.append(Spacer(1, 0.3*cm))

    # Gesamtbetrag — nur wenn show_total, mit farbiger Box wie Bestellnummer
    if show_total:
        total_table = Table(
            [[Paragraph("Gesamtbetrag:", bold),
              Paragraph(f"<b>{order['total']:.2f} EUR</b>", bold)]],
            colWidths=["35%", "65%"],
        )
        total_table.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), LIGHT_BG),
            ("BOX",           (0,0), (-1,-1), 1.5, BRAND_COLOR),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        story.append(total_table)
    else:
        story.append(Paragraph("* Betrag wird separat in Rechnung gestellt.", small))

    # Footer
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        "Bitte bei Rückfragen die Bestellnummer angeben. "
        "MusterFirma GmbH · USt-IdNr: DE123456789",
        small,
    ))

    doc.build(story)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()

    pdf_dir     = Path("dataset/pdfs")
    labels_file = Path("dataset/labels.jsonl")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Bestehende Labels laden (um Duplikate zu vermeiden)
    existing = set()
    if labels_file.exists():
        with open(labels_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(json.loads(line).get("image", ""))

    new_labels = []
    stats = {"both": 0, "no_total": 0, "no_order_number": 0}
    print(f"Generiere {args.count} Bestellungs-PDFs ...")

    for i in range(1, args.count + 1):
        order = generate_order(i)
        pdf_name = f"order_{i:05d}.pdf"
        img_name = f"order_{i:05d}_p0.png"

        # Fehlende-Feld-Wahrscheinlichkeit: 8% kein Betrag, 7% keine Bestellnr.
        rnd = random.random()
        show_order_number = rnd >= 0.07
        show_total        = rnd < 0.07 or rnd >= 0.15

        build_pdf(order, str(pdf_dir / pdf_name),
                  show_order_number=show_order_number,
                  show_total=show_total)

        if img_name not in existing:
            label = {
                "image":        img_name,
                "order_number": order["order_number"] if show_order_number else "",
                "total":        f"{order['total']:.2f} EUR" if show_total else "",
            }
            new_labels.append(label)

            if show_order_number and show_total:
                stats["both"] += 1
            elif not show_total:
                stats["no_total"] += 1
            else:
                stats["no_order_number"] += 1

        if i % 50 == 0:
            print(f"  {i}/{args.count} ...")

    with open(labels_file, "a", encoding="utf-8") as f:
        for entry in new_labels:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nFertig! {len(new_labels)} neue Labels geschrieben.")
    print(f"  Beide Felder    : {stats['both']}")
    print(f"  Kein Betrag     : {stats['no_total']}")
    print(f"  Keine Bestell-Nr: {stats['no_order_number']}")
    print(f"\nNächster Schritt: python tools/pdf_to_images.py")


if __name__ == "__main__":
    main()
