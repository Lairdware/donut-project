"""
Erzeugt/ergänzt dataset/labels.jsonl mit einem Platzhalter-Eintrag pro Bild
in dataset/images/. Bereits gelabelte Bilder werden nicht verändert.

Aufruf:
    python tools/generate_labels_skeleton.py
    python tools/generate_labels_skeleton.py --img-dir dataset/images --labels dataset/labels.jsonl
    python tools/generate_labels_skeleton.py --num-positions 2   # mehrere Positionsslots vorbefüllen

Danach: dataset/labels.jsonl öffnen und die "" Werte ausfüllen.
"""

import argparse
import json
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

# Muss mit POSITION_FIELD_NAMES in tools/prepare_dataset.py übereinstimmen.
POSITION_FIELD_NAMES = [
    "position_number",
    "delivery_date",
    "material",
    "customer_material",
    "quantity",
    "base_unit",
    "price_per_base",
    "currency",
    "price_base",
    "net_revenue",
    "drawing_number",
    "drawing_number_index",
    # ▼ NEUES FELD: Feldname hier eintragen
    # "mein_feld",
]


def empty_position() -> dict:
    return {field: "" for field in POSITION_FIELD_NAMES}


def main():
    parser = argparse.ArgumentParser(
        description="Erzeugt Platzhalter-Einträge in labels.jsonl für ungelabelte Bilder"
    )
    parser.add_argument("--img-dir", default="dataset/images")
    parser.add_argument("--labels",  default="dataset/labels.jsonl")
    parser.add_argument("--num-positions", type=int, default=1,
                        help="Wie viele leere Positions-Slots pro Bild vorbefüllt werden (Standard: 1)")
    args = parser.parse_args()

    img_dir     = Path(args.img_dir)
    labels_file = Path(args.labels)

    if not img_dir.is_dir():
        print(f"Fehler: '{img_dir}' nicht gefunden.")
        return

    images = sorted(p.name for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"Keine Bilder in '{img_dir}' gefunden.")
        return

    # Bereits vorhandene Labels einlesen (Reihenfolge erhalten)
    existing_lines = []
    labeled_images = set()
    if labels_file.exists():
        with open(labels_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                existing_lines.append(line)
                try:
                    entry = json.loads(line)
                    labeled_images.add(entry.get("image", ""))
                except json.JSONDecodeError:
                    pass

    new_lines = []
    for img_name in images:
        if img_name in labeled_images:
            continue
        entry = {
            "image": img_name,
            "positions": [empty_position() for _ in range(args.num_positions)],
        }
        new_lines.append(json.dumps(entry, ensure_ascii=False))

    if not new_lines:
        print("Alle Bilder sind bereits in labels.jsonl vorhanden. Nichts zu tun.")
        return

    labels_file.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_file, "w", encoding="utf-8") as f:
        for line in existing_lines + new_lines:
            f.write(line + "\n")

    print(f"{len(new_lines)} neue Platzhalter-Einträge hinzugefügt.")
    print(f"{len(existing_lines)} bereits vorhandene Einträge unverändert übernommen.")
    print(f"\n→ Jetzt {labels_file} öffnen und die \"\"-Werte ausfüllen.")


if __name__ == "__main__":
    main()
