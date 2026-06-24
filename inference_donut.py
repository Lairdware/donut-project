"""
Donut Inferenz — extrahiert Positionsdaten (Tabellenzeilen) aus
Dokument-Bildern inkl. Konfidenz. Ein Dokument kann mehrere Positionen
enthalten.

Aufruf:
    python inference_donut.py --image pfad/zum/bild.png
    python inference_donut.py --pdf   pfad/zur/datei.pdf      ← immer Seite 1
    python inference_donut.py --dir   dataset/images/
    python inference_donut.py --dir   dataset/images/ --output ergebnisse.json
    python inference_donut.py --eval  dataset/labels.jsonl

Voraussetzungen:
    pip install torch torchvision transformers sentencepiece Pillow pypdfium2
"""

import argparse
import json
import math
import re
import sys
import time
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

# ---------------------------------------------------------------------------
# Konfiguration (muss mit train_donut.py übereinstimmen)
# ---------------------------------------------------------------------------
DEFAULT_MODEL      = "output/donut_orders/best_model"
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
# ▼ NEUES FELD: Tokens hier definieren (muss mit train_donut.py übereinstimmen)
# MEIN_FELD_TOKEN = "<s_mein_feld>"
# MEIN_FELD_END   = "</s_mein_feld>"

# Reihenfolge + Tokens der Positionsfelder — an einer Stelle gepflegt.
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
    # ▼ NEUES FELD: ("mein_feld", MEIN_FELD_TOKEN, MEIN_FELD_END),
]

MAX_LENGTH         = 768
CONFIDENCE_HIGH    = 0.85
CONFIDENCE_LOW     = 0.50
IMAGE_EXTENSIONS   = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
MISSING_LABEL      = "—"


# ---------------------------------------------------------------------------
# Konfidenz
# ---------------------------------------------------------------------------
def _geo_mean(probs: list[float]) -> float:
    if not probs:
        return 0.0
    result = math.exp(sum(math.log(max(p, 1e-10)) for p in probs) / len(probs))
    return result if math.isfinite(result) else 0.0


def compute_confidences(sequences: torch.Tensor, scores: tuple,
                        processor: DonutProcessor) -> dict:
    tok = processor.tokenizer

    field_start_map = {tok.convert_tokens_to_ids(start_tok): name
                       for name, start_tok, _ in POSITION_FIELDS}
    field_end_ids    = {tok.convert_tokens_to_ids(end_tok)
                       for _, _, end_tok in POSITION_FIELDS}
    position_start_id = tok.convert_tokens_to_ids(POSITION_TOKEN)
    position_end_id    = tok.convert_tokens_to_ids(POSITION_END)

    structural_ids = set(field_start_map.keys()) | field_end_ids | {
        tok.convert_tokens_to_ids(TASK_TOKEN),
        tok.convert_tokens_to_ids(TASK_END_TOKEN),
        position_start_id,
        position_end_id,
    }
    structural_ids.discard(tok.unk_token_id)

    generated_ids = sequences[0][1:].tolist()

    per_token: list[dict] = []
    for step, step_scores in enumerate(scores):
        if step >= len(generated_ids):
            break
        tok_id = generated_ids[step]
        prob   = F.softmax(step_scores[0], dim=-1)[tok_id].item()
        per_token.append({
            "token": tok.convert_ids_to_tokens([tok_id])[0],
            "id":    tok_id,
            "prob":  prob,
        })

    # Konfidenz pro Position (eine Liste von {feld: konfidenz}-Dicts,
    # parallel zu parse_output's Rückgabe-Liste)
    positions_confidence: list[dict] = []
    current_position_probs: Optional[dict] = None
    current_field: Optional[str] = None

    for item in per_token:
        tid = item["id"]
        if tid == position_start_id:
            current_position_probs = {}
            current_field = None
        elif tid == position_end_id:
            if current_position_probs is not None:
                positions_confidence.append({
                    f: round(_geo_mean(probs), 4)
                    for f, probs in current_position_probs.items()
                })
            current_position_probs = None
            current_field = None
        elif tid in field_start_map:
            current_field = field_start_map[tid]
            if current_position_probs is not None:
                current_position_probs.setdefault(current_field, [])
        elif tid in field_end_ids:
            current_field = None
        elif (current_field is not None and current_position_probs is not None
              and tid not in structural_ids):
            current_position_probs[current_field].append(item["prob"])

    skip_ids = structural_ids | {tok.eos_token_id, tok.pad_token_id}
    content_probs = [item["prob"] for item in per_token if item["id"] not in skip_ids]
    doc_conf     = round(_geo_mean(content_probs), 4)
    doc_min_conf = round(min(content_probs), 4) if content_probs else 0.0

    return {
        "document":     doc_conf,
        "document_min": doc_min_conf,
        "positions":    positions_confidence,
        "per_token":    [{"token": t["token"], "prob": round(t["prob"], 4)}
                         for t in per_token],
    }


def confidence_label(score: float) -> str:
    if score >= CONFIDENCE_HIGH:
        return "HIGH"
    if score >= CONFIDENCE_LOW:
        return "MED"
    return "LOW"


# ---------------------------------------------------------------------------
# Stopping Criteria — stoppt sobald </s_order> generiert wurde.
# ---------------------------------------------------------------------------
class StopOnTaskEnd(StoppingCriteria):
    def __init__(self, task_end_token_id: int):
        self.task_end_token_id = task_end_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0, -1].item() == self.task_end_token_id


# ---------------------------------------------------------------------------
# PDF → PIL (erste Seite)
# ---------------------------------------------------------------------------
def pdf_first_page_to_image(pdf_path: str, dpi: int = 300) -> Image.Image:
    doc  = pdfium.PdfDocument(pdf_path)
    page = doc[0]
    bitmap = page.render(scale=dpi / 72.0, rotation=0)
    img = bitmap.to_pil().convert("RGB")
    doc.close()
    return img


# ---------------------------------------------------------------------------
# Modell
# ---------------------------------------------------------------------------
def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Lade Modell von '{model_path}' auf {device} ...")
    processor = DonutProcessor.from_pretrained(model_path)
    model     = VisionEncoderDecoderModel.from_pretrained(model_path)

    decoder_start_id = processor.tokenizer.convert_tokens_to_ids([TASK_TOKEN])[0]
    model.config.decoder_start_token_id           = decoder_start_id
    model.generation_config.decoder_start_token_id = decoder_start_id
    model.generation_config.pad_token_id           = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id           = processor.tokenizer.eos_token_id

    model.to(device)
    model.eval()
    return model, processor, device


# ---------------------------------------------------------------------------
# Parser — gibt eine Liste von Positionen zurück (ein Dokument = mehrere
# Positionen möglich)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Inferenz
# ---------------------------------------------------------------------------
def predict_single(image_path: str, model, processor, device,
                   image: Image.Image = None) -> dict:
    if image is None:
        image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

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

    positions  = parse_output(seq_str)
    confidence = compute_confidences(outputs.sequences, outputs.scores, processor)

    return {
        "raw_output": seq_str.strip(),
        "positions":  positions,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------
def process_directory(dir_path: str, model, processor, device) -> list:
    images = sorted(
        p for p in Path(dir_path).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print(f"Keine Bilder in '{dir_path}'.")
        return []

    print(f"{len(images)} Bilder. Starte Inferenz ...\n")

    results = []
    for i, img_path in enumerate(images, 1):
        t0      = time.time()
        result  = predict_single(str(img_path), model, processor, device)
        elapsed = time.time() - t0

        positions    = result["positions"]
        doc_conf     = result["confidence"]["document"]
        doc_min_conf = result["confidence"]["document_min"]
        status       = confidence_label(doc_conf)

        print(f"[{i}/{len(images)}] {img_path.name}  "
              f"{len(positions)} Position(en)  "
              f"avg={doc_conf:>5.1%} min={doc_min_conf:>5.1%}  [{status}]  ({elapsed:.2f}s)")
        for p_idx, pos in enumerate(positions, 1):
            vals = "  ".join(f"{name}={pos.get(name, MISSING_LABEL)}"
                             for name, _, _ in POSITION_FIELDS if pos.get(name))
            print(f"    Pos {p_idx}: {vals}")
        print(f"    RAW: {result['raw_output']}")

        results.append({
            "file":           img_path.name,
            "positions":      positions,
            "confidence":     result["confidence"]["positions"],
            "confidence_document":     doc_conf,
            "confidence_document_min": doc_min_conf,
            "confidence_label":        status,
            "raw_output":              result["raw_output"],
            "time_s":                  round(elapsed, 3),
        })

    return results


# ---------------------------------------------------------------------------
# Evaluation gegen labels.jsonl
# ---------------------------------------------------------------------------
def _parse_label_entry(entry: dict) -> dict:
    """Normalisiert beide JSONL-Formate auf ein einheitliches Dict."""
    # Format 1 (dataset/labels.jsonl): {"image": "...", "positions": [...]}
    if "image" in entry or "ground_truth" not in entry:
        result = dict(entry)
        result.setdefault("image", entry.get("file_name", ""))
        result.setdefault("positions", entry.get("positions", []))
        return result
    # Format 2 (data/val/metadata.jsonl): {"file_name": "...", "ground_truth": "{\"gt_parse\": {...}}"}
    parsed = json.loads(entry["ground_truth"]).get("gt_parse", {})
    return {"image": entry.get("file_name", ""), "positions": parsed.get("positions", [])}


def _normalize_value(v) -> str:
    return " ".join(str(v).split())


def evaluate(labels_file: str, img_dir: str, model, processor, device):
    with open(labels_file, encoding="utf-8") as f:
        labels = [_parse_label_entry(json.loads(l)) for l in f if l.strip()]

    field_names = [name for name, _, _ in POSITION_FIELDS]
    correct = {f: [0, 0] for f in field_names}   # [hits, total]
    n_docs        = 0
    n_pos_correct = 0
    n_pos_total   = 0

    print(f"{len(labels)} Labels. Starte Evaluation ...\n")

    for entry in labels:
        img_name = entry["image"]
        img_path = Path(img_dir) / img_name
        if not img_path.exists():
            print(f"  [NICHT GEFUNDEN] {img_path}")
            continue

        result   = predict_single(str(img_path), model, processor, device)
        pred_pos = result["positions"]
        gt_pos   = [
            {k: _normalize_value(v) for k, v in p.items() if v}
            for p in entry["positions"]
        ]

        n_docs += 1
        print(f"  {img_name}  ({len(gt_pos)} GT-Positionen, {len(pred_pos)} gefunden)")

        # Naiver 1:1-Vergleich nach Index — setzt voraus dass Reihenfolge
        # der Positionen in Bild und Label übereinstimmt (von oben nach unten).
        for idx in range(max(len(gt_pos), len(pred_pos))):
            gt  = gt_pos[idx]  if idx < len(gt_pos)  else {}
            pred = pred_pos[idx] if idx < len(pred_pos) else {}

            labeled = [f for f in field_names if gt.get(f)]
            ok = {f: pred.get(f, "") == gt[f] for f in labeled}

            for f in labeled:
                correct[f][1] += 1
                if ok[f]:
                    correct[f][0] += 1

            n_pos_total += 1
            if labeled and all(ok.values()):
                n_pos_correct += 1

            status = "✓" if (labeled and all(ok.values())) else ("~" if any(ok.values()) else "✗")
            print(f"    {status} Pos {idx+1}: "
                  + "  ".join(f"{f}: pred={pred.get(f, MISSING_LABEL)} gt={gt.get(f, MISSING_LABEL)}"
                              for f in field_names if gt.get(f) or pred.get(f)))
        print(f"    RAW: {result['raw_output']}\n")

    if n_docs:
        print(f"{'='*55}")
        for f in field_names:
            hits, total = correct[f]
            if total:
                print(f"  {f:<25}: {hits}/{total}  ({hits/total:.1%})")
            else:
                print(f"  {f:<25}: nicht gelabelt")
        if n_pos_total:
            print(f"  {'Position komplett korrekt':<25}: {n_pos_correct}/{n_pos_total}  ({n_pos_correct/n_pos_total:.1%})")
        print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Donut Inferenz: Positionsdaten + Konfidenz aus Dokument-Bildern"
    )
    parser.add_argument("--image",  help="Einzelnes Bild")
    parser.add_argument("--pdf",    help="PDF-Datei (verarbeitet Seite 1)")
    parser.add_argument("--dir",    help="Verzeichnis mit Bildern")
    parser.add_argument("--eval",   help="labels.jsonl für Evaluation")
    parser.add_argument("--img-dir", default="dataset/images",
                        help="Bild-Verzeichnis für --eval (Standard: dataset/images)")
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--output", default=None, help="JSON-Ausgabedatei")
    parser.add_argument("--show-tokens", action="store_true")
    args = parser.parse_args()

    if not args.image and not args.pdf and not args.dir and not args.eval:
        parser.print_help()
        sys.exit(1)

    model, processor, device = load_model(args.model)

    if args.image or args.pdf:
        src = args.pdf or args.image
        if not Path(src).exists():
            print(f"Fehler: '{src}' nicht gefunden.")
            sys.exit(1)

        print(f"\nVerarbeite{'  (Seite 1)' if args.pdf else ''}: {src}")
        t0 = time.time()
        if args.pdf:
            img    = pdf_first_page_to_image(args.pdf)
            result = predict_single(src, model, processor, device, image=img)
        else:
            result = predict_single(src, model, processor, device)
        elapsed = time.time() - t0

        positions = result["positions"]
        conf      = result["confidence"]
        doc_lbl   = confidence_label(conf["document"])

        print("\n" + "=" * 55)
        print(f"  {len(positions)} Position(en) gefunden")
        for idx, pos in enumerate(positions, 1):
            print(f"\n  [Position {idx}]")
            for name, _, _ in POSITION_FIELDS:
                print(f"    {name:<22}: {pos.get(name, MISSING_LABEL)}")
            if idx - 1 < len(conf["positions"]):
                for field, fc in conf["positions"][idx - 1].items():
                    print(f"      Konf '{field}': {fc:.1%}  [{confidence_label(fc)}]")
        # ▼ NEUES FELD: erscheint automatisch da POSITION_FIELDS durchlaufen wird
        print(f"\n  Rohausgabe         : {result['raw_output']}")
        print(f"  Dok-Konfidenz (avg): {conf['document']:.1%}  [{doc_lbl}]")
        print(f"  Dok-Konfidenz (min): {conf['document_min']:.1%}")
        print(f"  Dauer              : {elapsed:.3f}s")

        if args.show_tokens:
            print("\n  Per-Token-Konfidenz:")
            for t in conf["per_token"]:
                bar = "█" * int(t["prob"] * 20)
                print(f"    {t['token']:<30} {t['prob']:.3f}  {bar}")

        print("=" * 55)

    elif args.dir:
        if not Path(args.dir).is_dir():
            print(f"Fehler: '{args.dir}' nicht gefunden.")
            sys.exit(1)

        results = process_directory(args.dir, model, processor, device)

        if results:
            found    = sum(1 for r in results if r["positions"])
            high     = sum(1 for r in results if r["confidence_label"] == "HIGH")
            med      = sum(1 for r in results if r["confidence_label"] == "MED")
            low      = sum(1 for r in results if r["confidence_label"] == "LOW")
            avg_time = sum(r["time_s"] for r in results) / len(results)
            total_pos = sum(len(r["positions"]) for r in results)
            print(f"\n{'='*55}")
            print(f"  Dokumente mit Positionen: {found}/{len(results)} | Positionen gesamt: {total_pos}")
            print(f"  HIGH={high} MED={med} LOW={low}")
            print(f"  Ø Zeit  : {avg_time:.2f}s/Bild")
            print(f"{'='*55}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nErgebnisse: {args.output}")

    elif args.eval:
        evaluate(args.eval, args.img_dir, model, processor, device)


if __name__ == "__main__":
    main()
