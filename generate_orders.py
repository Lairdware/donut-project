"""
Generiert 100 synthetische Bestellungs-Dokumente als PNG-Bilder
und erstellt die zugehörigen JSON-Annotationen für Donut-Training.
"""

import json
import random
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap

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
    "Hauptstraße", "Bahnhofstraße", "Schillerstraße",
    "Goethestraße", "Industriestraße", "Am Marktplatz",
    "Ringstraße", "Gartenweg", "Kirchgasse", "Bergstraße",
]


def generate_order_number(i: int) -> str:
    year = 2024
    return f"ORD-{year}-{i:05d}"


def generate_order(i: int) -> dict:
    order_number = generate_order_number(i)
    company = random.choice(COMPANIES)
    street = random.choice(STREETS)
    house_nr = random.randint(1, 200)
    plz, city = random.choice(CITIES)
    date_day = random.randint(1, 28)
    date_month = random.randint(1, 12)
    date = f"{date_day:02d}.{date_month:02d}.2024"

    num_items = random.randint(1, 4)
    items = []
    total = 0.0
    for _ in range(num_items):
        product, price = random.choice(PRODUCTS)
        qty = random.randint(1, 5)
        line_total = price * qty
        total += line_total
        items.append({"product": product, "qty": qty, "price": price, "total": line_total})

    return {
        "order_number": order_number,
        "company": company,
        "address": f"{street} {house_nr}, {plz} {city}",
        "date": date,
        "items": items,
        "total": round(total, 2),
    }


def render_order_image(order: dict, output_path: str, width=480, height=640):
    """Rendert eine DIN-A4-ähnliche Bestellung als PNG."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font_header = ImageFont.truetype("arial.ttf", 22)
        font_title = ImageFont.truetype("arial.ttf", 18)
        font_bold = ImageFont.truetype("arialbd.ttf", 14)
        font_normal = ImageFont.truetype("arial.ttf", 13)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except IOError:
        font_header = ImageFont.load_default()
        font_title = font_header
        font_bold = font_header
        font_normal = font_header
        font_small = font_header

    margin = 50
    y = margin

    # Firmenkopf (fiktiv)
    draw.text((margin, y), "MusterFirma GmbH", fill=(0, 60, 120), font=font_header)
    draw.text((width - margin - 200, y), f"Datum: {order['date']}", fill=(80, 80, 80), font=font_normal)
    y += 30
    draw.text((margin, y), "Musterstraße 1 · 12345 Musterstadt", fill=(100, 100, 100), font=font_small)
    y += 20
    draw.line([(margin, y), (width - margin, y)], fill=(0, 60, 120), width=2)
    y += 20

    # Empfänger
    draw.text((margin, y), order["company"], fill=(0, 0, 0), font=font_bold)
    y += 18
    draw.text((margin, y), order["address"], fill=(60, 60, 60), font=font_normal)
    y += 40

    # Bestellnummer — das wichtigste Feld
    draw.rectangle([(margin, y), (width - margin, y + 36)], fill=(230, 240, 255), outline=(0, 60, 120), width=2)
    draw.text((margin + 10, y + 8), "Bestellnummer:", fill=(0, 60, 120), font=font_bold)
    draw.text((margin + 160, y + 8), order["order_number"], fill=(0, 0, 0), font=font_bold)
    y += 55

    # Tabellenheader
    draw.rectangle([(margin, y), (width - margin, y + 24)], fill=(0, 60, 120))
    draw.text((margin + 5, y + 5), "Produkt", fill=(255, 255, 255), font=font_bold)
    draw.text((width - 320, y + 5), "Menge", fill=(255, 255, 255), font=font_bold)
    draw.text((width - 230, y + 5), "Einzelpreis", fill=(255, 255, 255), font=font_bold)
    draw.text((width - 110, y + 5), "Gesamt", fill=(255, 255, 255), font=font_bold)
    y += 28

    # Positionen
    row_colors = [(248, 248, 255), (255, 255, 255)]
    for idx, item in enumerate(order["items"]):
        color = row_colors[idx % 2]
        draw.rectangle([(margin, y), (width - margin, y + 22)], fill=color)
        product_text = item["product"]
        if len(product_text) > 35:
            product_text = product_text[:32] + "..."
        draw.text((margin + 5, y + 4), product_text, fill=(0, 0, 0), font=font_normal)
        draw.text((width - 315, y + 4), str(item["qty"]), fill=(0, 0, 0), font=font_normal)
        draw.text((width - 225, y + 4), f"{item['price']:.2f} EUR", fill=(0, 0, 0), font=font_normal)
        draw.text((width - 110, y + 4), f"{item['total']:.2f} EUR", fill=(0, 0, 0), font=font_normal)
        draw.line([(margin, y + 22), (width - margin, y + 22)], fill=(200, 200, 200), width=1)
        y += 24

    y += 15
    draw.line([(width - 300, y), (width - margin, y)], fill=(0, 60, 120), width=2)
    y += 5
    draw.text((width - 295, y), "Gesamtbetrag:", fill=(0, 60, 120), font=font_bold)
    draw.text((width - 130, y), f"{order['total']:.2f} EUR", fill=(0, 0, 0), font=font_bold)

    # Footer
    y_footer = height - 60
    draw.line([(margin, y_footer), (width - margin, y_footer)], fill=(200, 200, 200), width=1)
    draw.text((margin, y_footer + 8), "Bitte bei Rückfragen die Bestellnummer angeben.", fill=(120, 120, 120), font=font_small)
    draw.text((margin, y_footer + 22), "MusterFirma GmbH · USt-IdNr: DE123456789 · IBAN: DE89 3704 0044 0532 0130 00", fill=(150, 150, 150), font=font_small)

    img.save(output_path, "PNG", dpi=(96, 96))


def main():
    out_dir = Path("data/train")
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    print("Generiere 500 Bestellungs-Dokumente ...")
    for i in range(1, 501):
        order = generate_order(i)
        img_name = f"order_{i:03d}.png"
        img_path = out_dir / img_name
        render_order_image(order, str(img_path))

        gt_parse = {
            "order_number": order["order_number"],
            "total": f"{order['total']:.2f} EUR",
        }
        metadata.append({
            "file_name": img_name,
            "ground_truth": json.dumps({"gt_parse": gt_parse}, ensure_ascii=False),
        })

        if i % 10 == 0:
            print(f"  {i}/100 erstellt ...")

    # metadata.jsonl speichern
    meta_path = out_dir / "metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nFertig! Bilder: {out_dir}/  |  Metadaten: {meta_path}")
    print(f"Beispiel Ground-Truth: {metadata[0]['ground_truth']}")


if __name__ == "__main__":
    main()
