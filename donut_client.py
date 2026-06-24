"""
Verwendung in einem anderen Projekt:

    from donut_client import extract_from_pdf

    df = extract_from_pdf(r"C:\Rechnungen\rechnung.pdf")
    print(df)

Gibt einen pandas DataFrame zurück mit einer Zeile pro Position
(eine PDF-Seite kann mehrere Positionen/Zeilen enthalten).
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
    "position_index",
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
    "confidence",
    "confidence_min",
    "raw_output",
]


def extract_from_pdf(pdf_path: str, model_path: str = _DEFAULT_MODEL) -> pd.DataFrame:
    """
    Extrahiert Positionsdaten aus einer PDF mit dem trainierten Donut-Modell.

    Args:
        pdf_path:   Pfad zur PDF-Datei
        model_path: Pfad zum trainierten Modell (Standard: best_model)

    Returns:
        DataFrame mit einer Zeile pro Position. Wenn eine Seite keine
        Position enthält, gibt es trotzdem eine Zeile mit leeren Feldern
        (position_index=0), damit keine Seite "verschwindet".
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
