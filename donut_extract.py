"""
Extrahiert Positionsdaten aus einer PDF und gibt das Ergebnis als JSON
auf stdout aus. Wird von donut_client.py via subprocess aufgerufen.
Ein Dokument kann mehrere Positionen pro Seite enthalten.

Aufruf:
    python donut_extract.py rechnung.pdf
    python donut_extract.py rechnung.pdf --model output/donut_orders/best_model
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    DonutProcessor, VisionEncoderDecoderModel,
    StoppingCriteria, StoppingCriteriaList,
)

DEFAULT_MODEL      = str(Path(__file__).parent / "output/donut_orders/best_model")
TASK_TOKEN         = "<s_order>"
TASK_END_TOKEN     = "</s_order>"
POSITION_TOKEN     = "<s_position>"
POSITION_END       = "</s_position>"

POS_NUM_TOKEN         = "<s_position_number>"
POS_NUM_END           = "</s_position_number>"
DELIVERY_DATE_TOKEN   = "<s_delivery_date>"
DELIVERY_DATE_END     = "</s_delivery_date>"
MATERIAL_TOKEN        = "<s_material>"
MATERIAL_END          = "</s_material>"
CUST_MATERIAL_TOKEN   = "<s_customer_material>"
CUST_MATERIAL_END     = "</s_customer_material>"
QUANTITY_TOKEN        = "<s_quantity>"
QUANTITY_END          = "</s_quantity>"
BASE_UNIT_TOKEN       = "<s_base_unit>"
BASE_UNIT_END         = "</s_base_unit>"
PRICE_PER_BASE_TOKEN  = "<s_price_per_base>"
PRICE_PER_BASE_END    = "</s_price_per_base>"
CURRENCY_TOKEN        = "<s_currency>"
CURRENCY_END          = "</s_currency>"
PRICE_BASE_TOKEN      = "<s_price_base>"
PRICE_BASE_END        = "</s_price_base>"
NET_REVENUE_TOKEN     = "<s_net_revenue>"
NET_REVENUE_END       = "</s_net_revenue>"
DRAWING_NUMBER_TOKEN  = "<s_drawing_number>"
DRAWING_NUMBER_END    = "</s_drawing_number>"
DRAWING_NUM_IDX_TOKEN = "<s_drawing_number_index>"
DRAWING_NUM_IDX_END   = "</s_drawing_number_index>"

POSITION_FIELDS = [
    ("position_number",      POS_NUM_TOKEN,         POS_NUM_END),
    ("delivery_date",        DELIVERY_DATE_TOKEN,   DELIVERY_DATE_END),
    ("material",             MATERIAL_TOKEN,        MATERIAL_END),
    ("customer_material",    CUST_MATERIAL_TOKEN,   CUST_MATERIAL_END),
    ("quantity",              QUANTITY_TOKEN,        QUANTITY_END),
    ("base_unit",             BASE_UNIT_TOKEN,       BASE_UNIT_END),
    ("price_per_base",        PRICE_PER_BASE_TOKEN,  PRICE_PER_BASE_END),
    ("currency",               CURRENCY_TOKEN,        CURRENCY_END),
    ("price_base",             PRICE_BASE_TOKEN,      PRICE_BASE_END),
    ("net_revenue",            NET_REVENUE_TOKEN,     NET_REVENUE_END),
    ("drawing_number",         DRAWING_NUMBER_TOKEN,  DRAWING_NUMBER_END),
    ("drawing_number_index",   DRAWING_NUM_IDX_TOKEN, DRAWING_NUM_IDX_END),
]

MAX_LENGTH = 768


class StopOnTaskEnd(StoppingCriteria):
    def __init__(self, task_end_token_id: int):
        self.task_end_token_id = task_end_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0, -1].item() == self.task_end_token_id


def _geo_mean(probs: list[float]) -> float:
    if not probs:
        return 0.0
    result = math.exp(sum(math.log(max(p, 1e-10)) for p in probs) / len(probs))
    return result if math.isfinite(result) else 0.0


def parse_output(token_sequence: str) -> list[dict]:
    outer = re.search(r"<s_order>(.*?)</s_order>", token_sequence, re.DOTALL)
    inner = outer.group(1) if outer else token_sequence

    positions = []
    for pos_match in re.finditer(r"<s_position>(.*?)</s_position>", inner, re.DOTALL):
        pos_content = pos_match.group(1)
        pos_dict = {}
        for match in re.finditer(r"<s_(\w+)>(.*?)</s_\1>", pos_content, re.DOTALL):
            pos_dict[match.group(1)] = " ".join(match.group(2).split())
        if pos_dict:
            positions.append(pos_dict)
    return positions


def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = DonutProcessor.from_pretrained(model_path)
    model     = VisionEncoderDecoderModel.from_pretrained(model_path)
    decoder_start_id = processor.tokenizer.convert_tokens_to_ids([TASK_TOKEN])[0]
    model.config.decoder_start_token_id            = decoder_start_id
    model.generation_config.decoder_start_token_id = decoder_start_id
    model.generation_config.pad_token_id           = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id           = processor.tokenizer.eos_token_id
    model.to(device)
    model.eval()
    return model, processor, device


def predict(image: Image.Image, model, processor, device) -> dict:
    pixel_values  = processor(image, return_tensors="pt").pixel_values.to(device)
    task_end_id   = processor.tokenizer.convert_tokens_to_ids(TASK_END_TOKEN)
    stop_criteria = StoppingCriteriaList([StopOnTaskEnd(task_end_id)])

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=torch.full(
                (1, 1), model.config.decoder_start_token_id, device=device
            ),
            max_length=MAX_LENGTH,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            stopping_criteria=stop_criteria,
            num_beams=1,
            repetition_penalty=1.0,
            output_scores=True,
            return_dict_in_generate=True,
        )

    seq_str = processor.tokenizer.batch_decode(
        outputs.sequences, skip_special_tokens=False
    )[0]
    seq_str = seq_str.replace(processor.tokenizer.eos_token, "")
    seq_str = seq_str.replace(processor.tokenizer.pad_token, "")
    seq_str = re.sub(r"(<s_\w+>)\s+", r"\1", seq_str)

    positions = parse_output(seq_str)

    # Dokument-Konfidenz
    tok = processor.tokenizer
    structural_ids = {tok.convert_tokens_to_ids(t) for t in
                      [TASK_TOKEN, TASK_END_TOKEN, POSITION_TOKEN, POSITION_END]}
    for _, start_tok, end_tok in POSITION_FIELDS:
        structural_ids.add(tok.convert_tokens_to_ids(start_tok))
        structural_ids.add(tok.convert_tokens_to_ids(end_tok))

    generated_ids = outputs.sequences[0][1:].tolist()
    per_token_probs = []
    for step, step_scores in enumerate(outputs.scores):
        if step >= len(generated_ids):
            break
        tok_id = generated_ids[step]
        prob   = F.softmax(step_scores[0], dim=-1)[tok_id].item()
        per_token_probs.append((tok_id, prob))

    skip_ids = structural_ids | {tok.eos_token_id, tok.pad_token_id}
    content_probs = [p for tid, p in per_token_probs if tid not in skip_ids]
    doc_conf     = round(_geo_mean(content_probs), 4)
    doc_min_conf = round(min(content_probs), 4) if content_probs else 0.0

    return {
        "positions":    positions,
        "confidence":     doc_conf,
        "confidence_min": doc_min_conf,
        "raw_output":     seq_str.strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Pfad zur PDF-Datei")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(json.dumps({"error": f"Datei nicht gefunden: {args.pdf}"}))
        sys.exit(1)

    model, processor, device = load_model(args.model)

    doc     = pdfium.PdfDocument(args.pdf)
    results = []
    last_position_row = None  # letzte angehängte Zeile, für Seitenumbruch-Merge

    for page_idx in range(len(doc)):
        page   = doc[page_idx]
        bitmap = page.render(scale=300 / 72.0, rotation=0)
        image  = bitmap.to_pil().convert("RGB")
        result = predict(image, model, processor, device)
        positions = result["positions"]

        for pos_idx, pos in enumerate(positions, 1):
            # Erste Position einer Seite ohne position_number → vermutlich
            # Fortsetzung der letzten Position der Vorseite (Seitenumbruch
            # mitten in der Tabelle). Fehlende Felder dort ergänzen statt
            # eine neue Zeile zu erzeugen.
            is_continuation = (
                pos_idx == 1
                and not pos.get("position_number")
                and last_position_row is not None
            )
            if is_continuation:
                for name, _, _ in POSITION_FIELDS:
                    if not last_position_row.get(name) and pos.get(name):
                        last_position_row[name] = pos[name]
                last_position_row.setdefault("continued_on_pages", [])
                last_position_row["continued_on_pages"].append(page_idx)
                continue

            row = {name: pos.get(name, "") for name, _, _ in POSITION_FIELDS}
            row["page"]                = page_idx
            row["position_index"]      = pos_idx
            row["continued_on_pages"]  = []
            row["confidence"]          = result["confidence"]
            row["confidence_min"]      = result["confidence_min"]
            row["raw_output"]          = result["raw_output"]
            results.append(row)
            last_position_row = row

        if not positions:
            row = {name: "" for name, _, _ in POSITION_FIELDS}
            row["page"]               = page_idx
            row["position_index"]     = 0
            row["continued_on_pages"] = []
            row["confidence"]         = result["confidence"]
            row["confidence_min"]     = result["confidence_min"]
            row["raw_output"]         = result["raw_output"]
            results.append(row)
    doc.close()

    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
