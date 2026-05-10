"""
Extrahiert Felder aus einer PDF und gibt das Ergebnis als JSON auf stdout aus.
Wird von donut_client.py via subprocess aufgerufen.

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
NAME_TOKEN         = "<s_sold_to_party_name>"
NAME_END           = "</s_sold_to_party_name>"
STREET_TOKEN       = "<s_sold_to_party_street>"
STREET_END         = "</s_sold_to_party_street>"
STREET_NUM_TOKEN   = "<s_sold_to_party_street_number>"
STREET_NUM_END     = "</s_sold_to_party_street_number>"
ZIP_TOKEN          = "<s_sold_to_party_zip>"
ZIP_END            = "</s_sold_to_party_zip>"
CITY_TOKEN         = "<s_sold_to_party_city>"
CITY_END           = "</s_sold_to_party_city>"
COUNTRY_TOKEN          = "<s_sold_to_party_country>"
COUNTRY_END            = "</s_sold_to_party_country>"
SHIP_NAME_TOKEN        = "<s_ship_to_party_name>"
SHIP_NAME_END          = "</s_ship_to_party_name>"
SHIP_STREET_TOKEN      = "<s_ship_to_party_street>"
SHIP_STREET_END        = "</s_ship_to_party_street>"
SHIP_STREET_NUM_TOKEN  = "<s_ship_to_party_street_number>"
SHIP_STREET_NUM_END    = "</s_ship_to_party_street_number>"
SHIP_ZIP_TOKEN         = "<s_ship_to_party_zip>"
SHIP_ZIP_END           = "</s_ship_to_party_zip>"
SHIP_CITY_TOKEN        = "<s_ship_to_party_city>"
SHIP_CITY_END          = "</s_ship_to_party_city>"
SHIP_COUNTRY_TOKEN     = "<s_ship_to_party_country>"
SHIP_COUNTRY_END       = "</s_ship_to_party_country>"
INV_NAME_TOKEN         = "<s_invoice_to_party_name>"
INV_NAME_END           = "</s_invoice_to_party_name>"
INV_STREET_TOKEN       = "<s_invoice_to_party_street>"
INV_STREET_END         = "</s_invoice_to_party_street>"
INV_STREET_NUM_TOKEN   = "<s_invoice_to_party_street_number>"
INV_STREET_NUM_END     = "</s_invoice_to_party_street_number>"
INV_ZIP_TOKEN          = "<s_invoice_to_party_zip>"
INV_ZIP_END            = "</s_invoice_to_party_zip>"
INV_CITY_TOKEN         = "<s_invoice_to_party_city>"
INV_CITY_END           = "</s_invoice_to_party_city>"
INV_COUNTRY_TOKEN      = "<s_invoice_to_party_country>"
INV_COUNTRY_END        = "</s_invoice_to_party_country>"
MAX_LENGTH             = 192


class StopOnTaskEnd(StoppingCriteria):
    def __init__(self, task_end_token_id: int):
        self.task_end_token_id = task_end_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0, -1].item() == self.task_end_token_id


def _geo_mean(probs: list[float]) -> float:
    if not probs:
        return 0.0
    return math.exp(sum(math.log(max(p, 1e-10)) for p in probs) / len(probs))


def parse_output(token_sequence: str) -> dict:
    result = {}
    outer = re.search(r"<s_order>(.*?)</s_order>", token_sequence, re.DOTALL)
    inner = outer.group(1) if outer else token_sequence
    for match in re.finditer(r"<s_(\w+)>(.*?)</s_\1>", inner, re.DOTALL):
        result[match.group(1)] = " ".join(match.group(2).split())
    return result


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

    parsed = parse_output(seq_str)

    # Konfidenz
    tok = processor.tokenizer
    structural_ids = set(tok.convert_tokens_to_ids([
        TASK_TOKEN, TASK_END_TOKEN,
        NAME_TOKEN, NAME_END,
        STREET_TOKEN, STREET_END,
        STREET_NUM_TOKEN, STREET_NUM_END,
        ZIP_TOKEN, ZIP_END,
        CITY_TOKEN, CITY_END,
        COUNTRY_TOKEN, COUNTRY_END,
        SHIP_NAME_TOKEN, SHIP_NAME_END,
        SHIP_STREET_TOKEN, SHIP_STREET_END,
        SHIP_STREET_NUM_TOKEN, SHIP_STREET_NUM_END,
        SHIP_ZIP_TOKEN, SHIP_ZIP_END,
        SHIP_CITY_TOKEN, SHIP_CITY_END,
        SHIP_COUNTRY_TOKEN, SHIP_COUNTRY_END,
        INV_NAME_TOKEN, INV_NAME_END,
        INV_STREET_TOKEN, INV_STREET_END,
        INV_STREET_NUM_TOKEN, INV_STREET_NUM_END,
        INV_ZIP_TOKEN, INV_ZIP_END,
        INV_CITY_TOKEN, INV_CITY_END,
        INV_COUNTRY_TOKEN, INV_COUNTRY_END,
    ]))
    generated_ids = outputs.sequences[0][1:].tolist()
    per_token_probs = []
    for step, step_scores in enumerate(outputs.scores):
        if step >= len(generated_ids):
            break
        tok_id = generated_ids[step]
        prob   = F.softmax(step_scores[0], dim=-1)[tok_id].item()
        per_token_probs.append((tok_id, prob))

    skip_ids = structural_ids | {tok.eos_token_id, tok.pad_token_id}
    doc_conf = round(_geo_mean([p for tid, p in per_token_probs if tid not in skip_ids]), 4)

    return {
        "sold_to_party_name":              parsed.get("sold_to_party_name", ""),
        "sold_to_party_street":            parsed.get("sold_to_party_street", ""),
        "sold_to_party_street_number":     parsed.get("sold_to_party_street_number", ""),
        "sold_to_party_zip":               parsed.get("sold_to_party_zip", ""),
        "sold_to_party_city":              parsed.get("sold_to_party_city", ""),
        "sold_to_party_country":           parsed.get("sold_to_party_country", ""),
        "ship_to_party_name":              parsed.get("ship_to_party_name", ""),
        "ship_to_party_street":            parsed.get("ship_to_party_street", ""),
        "ship_to_party_street_number":     parsed.get("ship_to_party_street_number", ""),
        "ship_to_party_zip":               parsed.get("ship_to_party_zip", ""),
        "ship_to_party_city":              parsed.get("ship_to_party_city", ""),
        "ship_to_party_country":           parsed.get("ship_to_party_country", ""),
        "invoice_to_party_name":           parsed.get("invoice_to_party_name", ""),
        "invoice_to_party_street":         parsed.get("invoice_to_party_street", ""),
        "invoice_to_party_street_number":  parsed.get("invoice_to_party_street_number", ""),
        "invoice_to_party_zip":            parsed.get("invoice_to_party_zip", ""),
        "invoice_to_party_city":           parsed.get("invoice_to_party_city", ""),
        "invoice_to_party_country":        parsed.get("invoice_to_party_country", ""),
        "confidence":                      doc_conf,
        "raw_output":                      seq_str.strip(),
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
    for page_idx in range(len(doc)):
        page   = doc[page_idx]
        bitmap = page.render(scale=300 / 72.0, rotation=0)
        image  = bitmap.to_pil().convert("RGB")
        result = predict(image, model, processor, device)
        result["page"] = page_idx
        results.append(result)
    doc.close()

    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
