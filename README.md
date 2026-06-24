# Donut Dokument-Extraktion

Trainiert ein [Donut](https://github.com/clovaai/donut)-Modell, das **Positionsdaten** (Tabellenzeilen) direkt aus Dokument-Bildern oder PDFs extrahiert — ohne OCR-Vorverarbeitung. Ein Dokument kann mehrere Positionen enthalten (z.B. Bestellpositionen).

> Branch-Hinweis: Diese Version (`multipage`) extrahiert Positionsdaten mit Mehrseiten-Support. Die ursprüngliche Version mit Adressfeldern (sold_to/ship_to/invoice_to_party) und Single-Page liegt auf `main`.

---

## Verzeichnisstruktur

```
donut_try/
├── tools/
│   ├── pdf_to_images.py          # PDFs → PNG-Bilder (alle Seiten)
│   ├── prepare_dataset.py        # labels.jsonl → Train/Val-Split
│   └── generate_synthetic_pdfs.py
├── train_donut.py                # Modell trainieren
├── inference_donut.py            # Positionen aus Bildern/PDFs extrahieren
├── donut_extract.py              # Standalone-Skript für externe Projekte
├── donut_client.py               # Wrapper für externe Projekte (DataFrame)
├── data/
│   ├── train/                    # Trainingsbilder + metadata.jsonl
│   └── val/                      # Validierungsbilder + metadata.jsonl
├── dataset/
│   ├── pdfs/                     # Eingabe-PDFs
│   ├── images/                   # Konvertierte PNG-Bilder
│   └── labels.jsonl              # Ground-Truth-Labels
└── output/
    └── donut_orders/
        ├── best_model/           # Bestes Checkpoint (niedrigster Val-Loss)
        └── final_model/          # Letzter Epoch
```

---

## Positionsfelder

Jede Position innerhalb eines Dokuments hat diese Felder:

```
position_number, delivery_date, material, customer_material,
quantity, base_unit, price_per_base, currency, price_base,
net_revenue, drawing_number, drawing_number_index
```

Felder dürfen pro Position leer sein — das Modell lernt dann, das Tag wegzulassen.

---

## Installation

**1. PyTorch installieren** (vor requirements.txt):

```bash
# CUDA 12.4 / 12.8
pip install "torch==2.6.0" torchvision --index-url https://download.pytorch.org/whl/cu124
```

**2. Abhängigkeiten installieren:**

```bash
pip install -r requirements.txt
```

---

## Pipeline

### Schritt 1 — PDFs zu Bildern konvertieren

```bash
python tools/pdf_to_images.py               # alle PDFs in dataset/pdfs/
python tools/pdf_to_images.py --dpi 150     # niedrigere Auflösung
python tools/pdf_to_images.py --file x.pdf  # einzelne Datei
```

Ausgabe: `dataset/images/<name>_page1.png`, `_page2.png`, ... — **eine Datei pro PDF-Seite**, jede Seite wird einzeln gelabelt.

---

### Schritt 2 — Labels erstellen

`dataset/labels.jsonl` manuell befüllen oder ergänzen. Ein Eintrag pro **Bild/Seite**, mit einer Liste von Positionen:

```jsonl
{"image": "bestellung_001_page1.png", "positions": [{"position_number": "10", "delivery_date": "15.03.2026", "material": "ABC-123", "customer_material": "K-998", "quantity": "50", "base_unit": "ST", "price_per_base": "12.50", "currency": "EUR", "price_base": "1", "net_revenue": "625.00", "drawing_number": "DRW-001", "drawing_number_index": "A"}, {"position_number": "20", "material": "XYZ-456", "quantity": "10", "base_unit": "ST", "currency": "EUR"}]}
{"image": "bestellung_002_page1.png", "positions": []}
```

- Eine Seite kann **mehrere Positionen** haben (Liste von Objekten)
- Einzelne Felder innerhalb einer Position dürfen fehlen/leer sein
- Eine Seite ohne Positionen → leere Liste `[]`
- `image` muss mit dem Dateinamen in `dataset/images/` übereinstimmen
- **Reihenfolge der Positionen in der Liste muss der Reihenfolge im Dokument entsprechen** (von oben nach unten) — Eval vergleicht positionsweise nach Index

---

### Schritt 3 — Train/Val-Split erstellen

```bash
python tools/prepare_dataset.py
python tools/prepare_dataset.py --val-split 0.15   # 15% Validierung
python tools/prepare_dataset.py --no-copy          # nur metadata, Bilder nicht kopieren
```

Erstellt `data/train/metadata.jsonl` und `data/val/metadata.jsonl`.

> **Wichtig:** Bei neuen Labels oder geänderten Feldern immer neu ausführen, bevor trainiert wird.

---

### Schritt 4 — Trainieren

```bash
python train_donut.py
```

Das Modell wird nach jedem Epoch evaluiert. Das beste Modell (niedrigster Val-Loss) wird in `output/donut_orders/best_model/` gespeichert.

> **MAX_LENGTH prüfen:** Jede Position braucht ~25-40 Tokens. Bei vielen Positionen pro Dokument (z.B. 10+) muss `MAX_LENGTH` in `train_donut.py` **und** `inference_donut.py` (identisch!) entsprechend hoch sein, sonst werden Positionen am Ende abgeschnitten. Aktueller Standard: `768`.

> **Neu trainieren:** Bei geänderten Feldern zuerst `output/donut_orders/` löschen, damit kein altes Modell geladen wird.

---

### Schritt 5 — Inferenz

```bash
# Einzelnes Bild
python inference_donut.py --image dataset/images/bestellung_001_page1.png

# PDF (Seite 1)
python inference_donut.py --pdf dataset/pdfs/bestellung_001.pdf

# Ganzes Verzeichnis
python inference_donut.py --dir dataset/images/

# Ergebnisse als JSON speichern
python inference_donut.py --dir dataset/images/ --output ergebnisse.json

# Evaluation gegen labels.jsonl
python inference_donut.py --eval dataset/labels.jsonl --img-dir dataset/images/

# Ehrlicher Test auf dem Val-Split (Modell hat diese Bilder nie gesehen)
python inference_donut.py --eval data/val/metadata.jsonl --img-dir data/val/

# Anderes Modell verwenden
python inference_donut.py --image x.png --model output/donut_orders/checkpoint-200
```

Ausgabe pro Bild zeigt alle gefundenen Positionen einzeln mit Konfidenz (`avg` = Durchschnitt, `min` = unsicherster Einzeltoken — fällt deutlich ab bei Dokumenten die das Modell nie gesehen hat).

---

## Verwendung in einem anderen Projekt

`donut_client.py` in das andere Projekt kopieren (braucht dort nur `pandas`):

```python
from donut_client import extract_from_pdf

df = extract_from_pdf(r"C:\Bestellungen\bestellung.pdf")
print(df)
```

Gibt einen DataFrame zurück mit **einer Zeile pro Position** (Spalten: `page`, `position_index`, `position_number`, `material`, ... `confidence`, `confidence_min`). Läuft über die venv aus dem `donut_try`-Ordner via Subprocess — im anderen Projekt wird kein Torch/Transformers benötigt.

---

## Neues Feld hinzufügen

Felder werden zentral in `POSITION_FIELDS` (train/inference) bzw. `POSITION_FIELD_NAMES` (prepare_dataset) gepflegt — an 5 Stellen eintragen:

---

### 1. `train_donut.py` — Token-Konstanten + `POSITION_FIELDS`

```python
MEIN_FELD_TOKEN = "<s_mein_feld>"
MEIN_FELD_END   = "</s_mein_feld>"

POSITION_FIELDS = [
    ...
    ("mein_feld", MEIN_FELD_TOKEN, MEIN_FELD_END),
]
```

Token-Registrierung und Sequenzbau übernehmen das Feld automatisch, da beide über `POSITION_FIELDS` iterieren.

---

### 2. `inference_donut.py` — Token-Konstanten + `POSITION_FIELDS`

```python
MEIN_FELD_TOKEN = "<s_mein_feld>"
MEIN_FELD_END   = "</s_mein_feld>"

POSITION_FIELDS = [
    ...
    ("mein_feld", MEIN_FELD_TOKEN, MEIN_FELD_END),
]
```

`structural_ids`, `field_start_map`, Ausgabe und Konfidenz übernehmen das Feld automatisch.

---

### 3. `donut_extract.py` — Token-Konstanten + `POSITION_FIELDS`

```python
MEIN_FELD_TOKEN = "<s_mein_feld>"
MEIN_FELD_END   = "</s_mein_feld>"

POSITION_FIELDS = [
    ...
    ("mein_feld", MEIN_FELD_TOKEN, MEIN_FELD_END),
]
```

---

### 4. `donut_client.py` — Spalte im DataFrame

```python
COLUMNS = [
    ...
    "mein_feld",
]
```

---

### 5. `tools/prepare_dataset.py` — Feld in `POSITION_FIELD_NAMES`

```python
POSITION_FIELD_NAMES = [
    ...
    "mein_feld",
]
```

---

### Danach

1. `dataset/labels.jsonl` — Feld in den Positions-Objekten ergänzen
2. `python tools/prepare_dataset.py` ausführen
3. `output/donut_orders/` löschen
4. `python train_donut.py` neu trainieren

---

## Bekannte Probleme

| Problem | Ursache | Fix |
|---|---|---|
| Feld nie gefunden | Altes Modell geladen | `output/donut_orders/` löschen, neu trainieren |
| Feld immer leer | JSONL hat leere Werte | `prepare_dataset.py` prüfen |
| Punktuation falsch (`s.r.o,`) | `repetition_penalty` zu hoch | `repetition_penalty=1.0` in `inference_donut.py` |
| Falsche Ausgabe trotz 100% Val-Acc | `no_repeat_ngram_size` blockiert legitime Wiederholungen (z.B. gleiches Material in zwei Positionen) | Parameter nicht verwenden, nur `repetition_penalty=1.0` |
| 100% Val-Acc, 0 Treffer | Feld in Trainings-JSONL leer | Labels prüfen, `prepare_dataset.py` neu ausführen |
| Positionen am Ende abgeschnitten | `MAX_LENGTH` zu klein für die Anzahl Positionen | Wert in `train_donut.py` **und** `inference_donut.py` erhöhen, neu trainieren |
| Eval zeigt falsche Zuordnung | Reihenfolge der Positionen in `labels.jsonl` stimmt nicht mit Dokument überein | Positionen in der Liste in Dokument-Reihenfolge (oben → unten) eintragen |
| Konfidenz immer hoch trotz falscher Ausgabe | Geometrischer Mittelwert verschleiert einzelne unsichere Tokens | `confidence_min` (Minimum) zusätzlich beachten |
