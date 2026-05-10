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
    "ship_to_party_name",
    "ship_to_party_street",
    "ship_to_party_street_number",
    "ship_to_party_zip",
    "ship_to_party_city",
    "ship_to_party_country",
    "invoice_to_party_name",
    "invoice_to_party_street",
    "invoice_to_party_street_number",
    "invoice_to_party_zip",
    "invoice_to_party_city",
    "invoice_to_party_country",
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

    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [_VENV_PYTHON, _EXTRACT_SCRIPT, pdf_path, "--model", model_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")

    if result.returncode != 0 or not stdout.strip():
        raise RuntimeError(
            f"Extraktion fehlgeschlagen (exit {result.returncode}):\n"
            f"STDOUT: {stdout!r}\n"
            f"STDERR: {stderr}"
        )

    data = json.loads(stdout)

    if isinstance(data, dict) and "error" in data:
        raise FileNotFoundError(data["error"])

    df = pd.DataFrame(data, columns=COLUMNS)
    return df
