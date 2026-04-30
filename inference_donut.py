"""
Donut Inferenz — extrahiert Felder aus Dokument-Bildern inkl. Konfidenz.

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
from transformers import DonutProcessor, VisionEncoderDecoderModel

# ---------------------------------------------------------------------------
# Konfiguration (muss mit train_donut.py übereinstimmen)
# ---------------------------------------------------------------------------
DEFAULT_MODEL      = "output/donut_orders/best_model"
TASK_TOKEN         = "<s_order>"
TASK_END_TOKEN     = "</s_order>"
ORDER_NUM_TOKEN    = "<s_order_number>"
ORDER_NUM_END      = "</s_order_number>"
TOTAL_TOKEN        = "<s_total>"
TOTAL_END          = "</s_total>"
MAX_LENGTH         = 56   # etwas mehr Puffer für lange Beträge
CONFIDENCE_HIGH    = 0.85
CONFIDENCE_LOW     = 0.50
IMAGE_EXTENSIONS   = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
MISSING_LABEL      = "—"          # Anzeige wenn Feld im Dokument fehlt


# ---------------------------------------------------------------------------
# Konfidenz
# ---------------------------------------------------------------------------
def _geo_mean(probs: list[float]) -> float:
    if not probs:
        return 0.0
    return math.exp(sum(math.log(max(p, 1e-10)) for p in probs) / len(probs))


def compute_confidences(sequences: torch.Tensor, scores: tuple,
                        processor: DonutProcessor) -> dict:
    tok = processor.tokenizer

    structural_ids = set(tok.convert_tokens_to_ids([
        TASK_TOKEN, TASK_END_TOKEN,
        ORDER_NUM_TOKEN, ORDER_NUM_END,
        TOTAL_TOKEN, TOTAL_END,
    ]))
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

    # Feld-Konfidenzen
    field_start_map = {
        tok.convert_tokens_to_ids(ORDER_NUM_TOKEN): "order_number",
        tok.convert_tokens_to_ids(TOTAL_TOKEN):     "total",
    }
    field_end_ids = {
        tok.convert_tokens_to_ids(ORDER_NUM_END),
        tok.convert_tokens_to_ids(TOTAL_END),
        tok.convert_tokens_to_ids(TASK_END_TOKEN),
    }

    current_field: Optional[str] = None
    field_probs: dict[str, list[float]] = {}

    for item in per_token:
        tid = item["id"]
        if tid in field_start_map:
            current_field = field_start_map[tid]
            field_probs.setdefault(current_field, [])
        elif tid in field_end_ids:
            current_field = None
        elif current_field is not None and tid not in structural_ids:
            field_probs[current_field].append(item["prob"])

    field_confidences = {
        field: round(_geo_mean(probs), 4)
        for field, probs in field_probs.items()
    }

    # Dokument-Konfidenz
    skip_ids = structural_ids | {tok.eos_token_id, tok.pad_token_id}
    doc_conf = round(_geo_mean([
        item["prob"] for item in per_token if item["id"] not in skip_ids
    ]), 4)

    return {
        "document": doc_conf,
        "fields":   field_confidences,
        "per_token": [{"token": t["token"], "prob": round(t["prob"], 4)}
                      for t in per_token],
    }


def confidence_label(score: float) -> str:
    if score >= CONFIDENCE_HIGH:
        return "HIGH"
    if score >= CONFIDENCE_LOW:
        return "MED"
    return "LOW"


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
    model.to(device)
    model.eval()
    return model, processor, device


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def parse_output(token_sequence: str) -> dict:
    result = {}

    # Inhalt innerhalb <s_order>…</s_order> extrahieren, dann Feld-Tags suchen.
    outer = re.search(r"<s_order>(.*?)</s_order>", token_sequence, re.DOTALL)
    inner = outer.group(1) if outer else token_sequence

    for match in re.finditer(r"<s_(\w+)>(.*?)</s_\1>", inner, re.DOTALL):
        result[match.group(1)] = match.group(2).strip()

    # Fallback Bestellnummer
    if "order_number" not in result:
        m = re.search(r"ORD-\d{4}-\d{5}", token_sequence)
        if m:
            result["order_number"] = m.group(0)
            result["_fallback"] = True

    return result


# ---------------------------------------------------------------------------
# Inferenz
# ---------------------------------------------------------------------------
def predict_single(image_path: str, model, processor, device,
                   image: Image.Image = None) -> dict:
    if image is None:
        image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    task_end_id = processor.tokenizer.convert_tokens_to_ids(TASK_END_TOKEN)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=torch.full(
                (1, 1), model.config.decoder_start_token_id, device=device
            ),
            max_length=MAX_LENGTH,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=[processor.tokenizer.eos_token_id, task_end_id],
            num_beams=1,
            repetition_penalty=1.8,
            output_scores=True,
            return_dict_in_generate=True,
        )

    seq_str = processor.tokenizer.batch_decode(
        outputs.sequences, skip_special_tokens=False
    )[0]
    seq_str = seq_str.replace(processor.tokenizer.eos_token, "")
    seq_str = seq_str.replace(processor.tokenizer.pad_token, "")
    seq_str = re.sub(r"(<s_\w+>)\s+", r"\1", seq_str)  # BPE-Leerzeichen

    parsed     = parse_output(seq_str)
    confidence = compute_confidences(outputs.sequences, outputs.scores, processor)

    return {
        "raw_output":   seq_str.strip(),
        "parsed":       parsed,
        "confidence":   confidence,
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
    hdr = f"{'#':<5} {'Datei':<28} {'Bestellnummer':<22} {'Betrag':<16} {'Konf':>6}  Status"
    print(hdr)
    print("-" * len(hdr))

    results = []
    for i, img_path in enumerate(images, 1):
        t0     = time.time()
        result = predict_single(str(img_path), model, processor, device)
        elapsed = time.time() - t0

        parsed   = result["parsed"]
        doc_conf = result["confidence"]["document"]
        status   = confidence_label(doc_conf)

        order_number = parsed.get("order_number", MISSING_LABEL)
        total        = parsed.get("total",        MISSING_LABEL)

        print(
            f"{i:<5} {img_path.name:<28} {order_number:<22} {total:<16} "
            f"{doc_conf:>5.1%}  [{status}]  ({elapsed:.2f}s)"
        )

        results.append({
            "file":                    img_path.name,
            "order_number":            order_number,
            "total":                   total,
            "confidence_document":     doc_conf,
            "confidence_order_number": result["confidence"]["fields"].get("order_number", 0.0),
            "confidence_total":        result["confidence"]["fields"].get("total", 0.0),
            "confidence_label":        status,
            "raw_output":              result["raw_output"],
            "time_s":                  round(elapsed, 3),
        })

    return results


# ---------------------------------------------------------------------------
# Evaluation gegen labels.jsonl
# ---------------------------------------------------------------------------
def evaluate(labels_file: str, img_dir: str, model, processor, device):
    with open(labels_file, encoding="utf-8") as f:
        labels = [json.loads(l) for l in f if l.strip()]

    correct_order = correct_total = correct_both = n = 0

    print(f"{len(labels)} Labels. Starte Evaluation ...\n")
    hdr = f"{'Datei':<28} {'Pred-Nr':<22} {'GT-Nr':<22} {'Pred-€':<16} {'GT-€':<16}  OK?"
    print(hdr)
    print("-" * len(hdr))

    for entry in labels:
        img_path = Path(img_dir) / entry["image"]
        if not img_path.exists():
            continue

        result = predict_single(str(img_path), model, processor, device)
        parsed = result["parsed"]

        pred_nr    = parsed.get("order_number", "")
        pred_total = parsed.get("total", "")
        gt_nr      = entry.get("order_number", "")
        gt_total   = entry.get("total", "")

        ok_nr    = pred_nr    == gt_nr
        ok_total = pred_total == gt_total
        ok_both  = ok_nr and ok_total

        if ok_nr:    correct_order += 1
        if ok_total: correct_total += 1
        if ok_both:  correct_both  += 1
        n += 1

        status = "✓" if ok_both else ("~" if (ok_nr or ok_total) else "✗")
        print(
            f"{entry['image']:<28} {pred_nr or MISSING_LABEL:<22} {gt_nr or MISSING_LABEL:<22} "
            f"{pred_total or MISSING_LABEL:<16} {gt_total or MISSING_LABEL:<16}  {status}"
        )

    if n:
        print(f"\n{'='*60}")
        print(f"  Bestellnummer korrekt : {correct_order}/{n}  ({correct_order/n:.1%})")
        print(f"  Gesamtbetrag korrekt  : {correct_total}/{n}  ({correct_total/n:.1%})")
        print(f"  Beide korrekt         : {correct_both}/{n}  ({correct_both/n:.1%})")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Donut Inferenz: Felder + Konfidenz aus Dokument-Bildern"
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

        parsed   = result["parsed"]
        conf     = result["confidence"]
        doc_lbl  = confidence_label(conf["document"])

        print("\n" + "=" * 55)
        print(f"  Bestellnummer      : {parsed.get('order_number', MISSING_LABEL)}")
        print(f"  Gesamtbetrag       : {parsed.get('total',        MISSING_LABEL)}")
        print(f"  Rohausgabe         : {result['raw_output']}")
        print(f"  Dok-Konfidenz      : {conf['document']:.1%}  [{doc_lbl}]")
        for field, fc in conf["fields"].items():
            name = "Bestellnummer" if field == "order_number" else "Gesamtbetrag"
            print(f"  Feld '{name}'  : {fc:.1%}  [{confidence_label(fc)}]")
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
            found    = sum(1 for r in results if r["order_number"] != MISSING_LABEL)
            high     = sum(1 for r in results if r["confidence_label"] == "HIGH")
            med      = sum(1 for r in results if r["confidence_label"] == "MED")
            low      = sum(1 for r in results if r["confidence_label"] == "LOW")
            avg_time = sum(r["time_s"] for r in results) / len(results)
            print(f"\n{'='*55}")
            print(f"  Erkannt : {found}/{len(results)} | HIGH={high} MED={med} LOW={low}")
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
