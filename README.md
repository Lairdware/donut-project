# Donut Dokument-Extraktion

Trainiert ein [Donut](https://github.com/clovaai/donut)-Modell, das strukturierte Felder (z.B. Bestellnummer, Firmenname) direkt aus Dokument-Bildern oder PDFs extrahiert — ohne OCR-Vorverarbeitung.

---

## Verzeichnisstruktur

```
donut_try/
├── tools/
│   ├── pdf_to_images.py          # PDFs → PNG-Bilder
│   ├── prepare_dataset.py        # labels.jsonl → Train/Val-Split
│   └── generate_synthetic_pdfs.py
├── train_donut.py                # Modell trainieren
├── inference_donut.py            # Felder aus Bildern/PDFs extrahieren
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

Ausgabe: `dataset/images/<name>_p0.png` (Seite 1), `_p1.png` (Seite 2), ...

---

### Schritt 2 — Labels erstellen

`dataset/labels.jsonl` manuell befüllen oder ergänzen. Ein Eintrag pro Zeile:

```jsonl
{"image": "rechnung_001_p0.png", "sold_to_party_name": "Siemens AG", "sold_to_party_street": "Wittelsbacherplatz", "sold_to_party_street_number": "2", "sold_to_party_zip": "80333", "sold_to_party_city": "München", "sold_to_party_country": "Deutschland"}
{"image": "rechnung_002_p0.png", "sold_to_party_name": "KYOCERA AVX Components s.r.o.", "sold_to_party_street": "Masarykova", "sold_to_party_street_number": "1", "sold_to_party_zip": "678 01", "sold_to_party_city": "Blansko", "sold_to_party_country": "Czech Republic"}
```

- Felder dürfen leer sein (`""`) — das Modell lernt dann, sie wegzulassen
- `image` muss mit dem Dateinamen in `dataset/images/` übereinstimmen

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

> **Neu trainieren:** Bei geänderten Feldern zuerst `output/donut_orders/` löschen, damit kein altes Modell geladen wird.

---

### Schritt 5 — Inferenz

```bash
# Einzelnes Bild
python inference_donut.py --image dataset/images/rechnung_001_p0.png

# PDF (Seite 1)
python inference_donut.py --pdf dataset/pdfs/rechnung_001.pdf

# Ganzes Verzeichnis
python inference_donut.py --dir dataset/images/

# Ergebnisse als JSON speichern
python inference_donut.py --dir dataset/images/ --output ergebnisse.json

# Evaluation gegen labels.jsonl
python inference_donut.py --eval dataset/labels.jsonl --img-dir dataset/images/

# Anderes Modell verwenden
python inference_donut.py --image x.png --model output/donut_orders/checkpoint-200
```

---

## Neues Feld hinzufügen

An allen mit `# ▼ NEUES FELD` markierten Stellen eintragen. Reihenfolge:

### 1. `train_donut.py` — Token-Konstanten

```python
MEIN_FELD_TOKEN = "<s_mein_feld>"
MEIN_FELD_END   = "</s_mein_feld>"
```

### 2. `train_donut.py` — Token registrieren

```python
processor.tokenizer.add_special_tokens({"additional_special_tokens": [
    ...
    MEIN_FELD_TOKEN, MEIN_FELD_END,
]})
```

### 3. `train_donut.py` — Wert lesen + Sequenz bauen

```python
mein_feld = gt["gt_parse"].get("mein_feld", "")
...
if mein_feld:
    parts += f"{MEIN_FELD_TOKEN}{mein_feld}{MEIN_FELD_END}"
```

### 4. `inference_donut.py` — Token-Konstanten (identisch zu train)

```python
MEIN_FELD_TOKEN = "<s_mein_feld>"
MEIN_FELD_END   = "</s_mein_feld>"
```

### 5. `inference_donut.py` — `structural_ids`, `field_start_map`, `field_end_ids`

```python
structural_ids = set(tok.convert_tokens_to_ids([..., MEIN_FELD_TOKEN, MEIN_FELD_END]))

field_start_map = {
    ...
    tok.convert_tokens_to_ids(MEIN_FELD_TOKEN): "mein_feld",
}

field_end_ids = {
    ...
    tok.convert_tokens_to_ids(MEIN_FELD_END),
}
```

### 6. `inference_donut.py` — Ausgabe

```python
mein_feld = parsed.get("mein_feld", MISSING_LABEL)
# im results-Dict:
"mein_feld": mein_feld,
"confidence_mein_feld": result["confidence"]["fields"].get("mein_feld", 0.0),
```

### 7. `tools/prepare_dataset.py` — Labels übernehmen

```python
# in to_ground_truth():
if entry.get("mein_feld"):
    gt_parse["mein_feld"] = entry["mein_feld"]
```

Außerdem `dataset/labels.jsonl` um das neue Feld ergänzen und `prepare_dataset.py` neu ausführen, dann neu trainieren.

---

## Bekannte Probleme

| Problem | Ursache | Fix |
|---|---|---|
| Feld nie gefunden | Altes Modell geladen | `output/donut_orders/` löschen, neu trainieren |
| Feld immer leer | JSONL hat leere Werte | `prepare_dataset.py` prüfen |
| Punktuation falsch (`s.r.o,`) | `repetition_penalty` zu hoch | `repetition_penalty=1.0` in `inference_donut.py` |
| Name doppelt ausgegeben | Modell wiederholt Sequenz | `no_repeat_ngram_size=4` in `inference_donut.py` |
| 100% Val-Acc, 0 Treffer | Feld in Trainings-JSONL leer | Labels prüfen, `prepare_dataset.py` neu ausführen |
| Sequenz abgeschnitten | `MAX_LENGTH` zu klein | Wert in `train_donut.py` + `inference_donut.py` erhöhen |
