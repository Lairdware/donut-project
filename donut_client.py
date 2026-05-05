"""
Verwendung in einem anderen Projekt:

    from donut_client import extract_from_pdf

    df = extract_from_pdf(r"C:\Rechnungen\rechnung.pdf")
    print(df)

Gibt einen pandas DataFrame zurück mit einer Zeile pro PDF-Seite.
"""

import json
import subprocess
from pathlib import Path

import pandas as pd

# Absolute Pfade — einmalig anpassen wenn das Projekt verschoben wird
_VENV_PYTHON    = r"C:\Users\Hyperhaven\Dev\donut_try\.venv\Scripts\python.exe"
_EXTRACT_SCRIPT = r"C:\Users\Hyperhaven\Dev\donut_try\donut_extract.py"
_DEFAULT_MODEL  = r"C:\Users\Hyperhaven\Dev\donut_try\output\donut_orders\best_model"

COLUMNS = [
    "page",
    "sold_to_party_name",
    "sold_to_party_street",
    "sold_to_party_street_number",
    "sold_to_party_zip",
    "sold_to_party_city",
    "sold_to_party_country",
    "confidence",
    "raw_output",
]


def extract_from_pdf(pdf_path: str, model_path: str = _DEFAULT_MODEL) -> pd.DataFrame:
    """
    Extrahiert Adressfelder aus einer PDF mit dem trainierten Donut-Modell.

    Args:
        pdf_path:   Pfad zur PDF-Datei
        model_path: Pfad zum trainierten Modell (Standard: best_model)

    Returns:
        DataFrame mit einer Zeile pro Seite und den extrahierten Feldern.
        Leere Spalten wenn ein Feld nicht gefunden wurde.
    """
    pdf_path = str(Path(pdf_path).resolve())

    result = subprocess.run(
        [_VENV_PYTHON, _EXTRACT_SCRIPT, pdf_path, "--model", model_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"Extraktion fehlgeschlagen (exit {result.returncode}):\n"
            f"STDOUT: {result.stdout!r}\n"
            f"STDERR: {result.stderr}"
        )

    data = json.loads(result.stdout)

    if isinstance(data, dict) and "error" in data:
        raise FileNotFoundError(data["error"])

    df = pd.DataFrame(data, columns=COLUMNS)
    return df
