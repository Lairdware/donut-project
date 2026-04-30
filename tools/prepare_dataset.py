"""
Liest dataset/labels.jsonl + dataset/images/ und baut daraus
die Trainings- und Validierungs-Splits für Donut.

Ausgabe:
    data/train/   ← 90% der gelabelten Bilder + metadata.jsonl
    data/val/     ← 10% der gelabelten Bilder + metadata.jsonl

Aufruf:
    python tools/prepare_dataset.py
    python tools/prepare_dataset.py --val-split 0.15   # 15% Validierung
    python tools/prepare_dataset.py --no-copy          # nur metadata, keine Bilder kopieren
"""

import argparse
import json
import random
import shutil
from pathlib import Path

SEED = 42


def load_labels(labels_file: Path) -> list[dict]:
    labels = []
    with open(labels_file, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARNUNG] Zeile {lineno} ungültig: {e}")
                continue

            # Pflichtfelder prüfen
            if "image" not in entry:
                print(f"  [WARNUNG] Zeile {lineno}: 'image' fehlt, übersprungen.")
                continue
            if not entry.get("order_number") and not entry.get("total"):
                print(f"  [WARNUNG] {entry['image']}: alle Felder leer, übersprungen.")
                continue

            labels.append(entry)
    return labels


def to_ground_truth(entry: dict) -> str:
    """Baut die Donut-Ground-Truth-Sequenz aus einem Label-Eintrag."""
    gt_parse = {}
    if entry.get("order_number"):
        gt_parse["order_number"] = entry["order_number"]
    if entry.get("total"):
        gt_parse["total"] = entry["total"]
    return json.dumps({"gt_parse": gt_parse}, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Erstellt Train/Val-Split aus labels.jsonl"
    )
    parser.add_argument("--labels",    default="dataset/labels.jsonl")
    parser.add_argument("--img-dir",   default="dataset/images")
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--val-dir",   default="data/val")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Anteil Validierungsdaten (Standard: 0.1 = 10%%)")
    parser.add_argument("--no-copy",   action="store_true",
                        help="Bilder nicht kopieren (nur metadata.jsonl schreiben)")
    args = parser.parse_args()

    labels_file = Path(args.labels)
    img_dir     = Path(args.img_dir)
    train_dir   = Path(args.train_dir)
    val_dir     = Path(args.val_dir)

    if not labels_file.exists():
        print(f"Fehler: '{labels_file}' nicht gefunden.")
        return

    print(f"Lade Labels aus {labels_file} ...")
    labels = load_labels(labels_file)
    print(f"  {len(labels)} gültige Einträge geladen.")

    # Nur Einträge mit vorhandenem Bild
    valid = []
    missing = []
    for entry in labels:
        img_path = img_dir / entry["image"]
        if img_path.exists():
            valid.append(entry)
        else:
            missing.append(entry["image"])

    if missing:
        print(f"  [WARNUNG] {len(missing)} Bilder nicht gefunden (ignoriert):")
        for name in missing[:3]:
            print(f"    - {name}")
        if len(missing) > 3:
            print(f"    ... und {len(missing)-3} weitere")

    if not valid:
        print("Keine gültigen Einträge. Abbruch.")
        return

    # Split
    random.seed(SEED)
    random.shuffle(valid)
    n_val   = max(1, int(len(valid) * args.val_split))
    n_train = len(valid) - n_val
    train_entries = valid[:n_train]
    val_entries   = valid[n_train:]

    print(f"\nSplit: {n_train} Train | {n_val} Val")

    # Verzeichnisse anlegen
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    def write_split(entries: list[dict], target_dir: Path, split_name: str):
        metadata = []
        for entry in entries:
            src = img_dir / entry["image"]
            dst = target_dir / entry["image"]

            if not args.no_copy:
                shutil.copy2(src, dst)

            metadata.append({
                "file_name":    entry["image"],
                "ground_truth": to_ground_truth(entry),
            })

        meta_path = target_dir / "metadata.jsonl"
        with open(meta_path, "w", encoding="utf-8") as f:
            for m in metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        print(f"  {split_name}: {len(metadata)} Einträge → {meta_path}")

    write_split(train_entries, train_dir, "Train")
    write_split(val_entries,   val_dir,   "Val  ")

    print(f"\nFertig! Nächster Schritt: python train_donut.py")


if __name__ == "__main__":
    main()
