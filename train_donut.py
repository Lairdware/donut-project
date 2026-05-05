"""
Donut Trainings-Script — extrahiert Bestellnummer + Gesamtbetrag
aus Bestellungs-Dokumenten. Unterstützt Dokumente mit fehlenden Feldern.

Voraussetzungen:
    pip install "torch>=2.6.0" torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install transformers datasets sentencepiece Pillow

Aufruf:
    python train_donut.py
"""

import json
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DonutProcessor, VisionEncoderDecoderModel, get_scheduler,
    StoppingCriteria, StoppingCriteriaList,
)
from PIL import Image

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
PRETRAINED_MODEL  = "naver-clova-ix/donut-base"
TRAIN_DIR         = Path("data/train")
VAL_DIR           = Path("data/val")
OUTPUT_DIR        = Path("output/donut_orders")

TASK_TOKEN        = "<s_order>"
TASK_END_TOKEN    = "</s_order>"
ORDER_NUM_TOKEN   = "<s_order_number>"
ORDER_NUM_END     = "</s_order_number>"
SOLD_TO_TOKEN     = "<s_sold_to_party_name>"
SOLD_TO_END       = "</s_sold_to_party_name>"
# ▼ NEUES FELD: Tokens hier definieren (Schema: "<s_feldname>" / "</s_feldname>")
# MEIN_FELD_TOKEN = "<s_mein_feld>"
# MEIN_FELD_END   = "</s_mein_feld>"

# Zielformat (Labels, <s_order> wird automatisch vorne eingefügt):
#   beide Felder : <s_order_number>ORD-…</s_order_number><s_sold_to_party_name>…</s_sold_to_party_name></s_order>
#   nur Nummer   : <s_order_number>ORD-…</s_order_number></s_order>
#   nur Name     : <s_sold_to_party_name>…</s_sold_to_party_name></s_order>

IMAGE_SIZE        = (1280, 960)
MAX_LENGTH        = 56
BATCH_SIZE        = 4   # 1280×960 braucht mehr VRAM
NUM_EPOCHS        = 50
LEARNING_RATE     = 3e-5
GRAD_ACCUMULATION = 2
SAVE_STEPS        = 50
EARLY_STOP_PATIENCE = 8   # Epochen ohne Val-Loss-Verbesserung → Stop
SEED              = 42
USE_AMP           = True

torch.manual_seed(SEED)
random.seed(SEED)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class OrderDataset(Dataset):
    def __init__(self, data_dir: Path, processor: DonutProcessor):
        self.data_dir  = data_dir
        self.processor = processor

        meta_path = data_dir / "metadata.jsonl"
        with open(meta_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        img_path = self.data_dir / sample["file_name"]
        image    = Image.open(img_path).convert("RGB")

        gt           = json.loads(sample["ground_truth"])
        order_number = gt["gt_parse"].get("order_number", "")
        sold_to      = gt["gt_parse"].get("sold_to_party_name", "")
        # ▼ NEUES FELD: Wert aus gt_parse lesen (Key = Feldname in der JSONL)
        # mein_feld = gt["gt_parse"].get("mein_feld", "")

        # Fehlende Felder → Tag weglassen.
        # Modell lernt: kein Tag im Output = Feld nicht im Dokument.
        parts = ""
        if order_number:
            parts += f"{ORDER_NUM_TOKEN}{order_number}{ORDER_NUM_END}"
        if sold_to:
            parts += f"{SOLD_TO_TOKEN}{sold_to}{SOLD_TO_END}"
        # ▼ NEUES FELD: Sequenz-Block anhängen
        # if mein_feld:
        #     parts += f"{MEIN_FELD_TOKEN}{mein_feld}{MEIN_FELD_END}"
        target_sequence = parts + TASK_END_TOKEN

        pixel_values = self.processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# Modell-Setup
# ---------------------------------------------------------------------------
def setup_model_and_processor():
    processor = DonutProcessor.from_pretrained(PRETRAINED_MODEL)

    processor.tokenizer.add_special_tokens({"additional_special_tokens": [
        TASK_TOKEN, TASK_END_TOKEN,
        ORDER_NUM_TOKEN, ORDER_NUM_END,
        SOLD_TO_TOKEN, SOLD_TO_END,
        # ▼ NEUES FELD: beide Tokens hier eintragen
        # MEIN_FELD_TOKEN, MEIN_FELD_END,
    ]})

    processor.image_processor.size = {"height": IMAGE_SIZE[0], "width": IMAGE_SIZE[1]}
    processor.image_processor.do_align_long_axis = False

    model = VisionEncoderDecoderModel.from_pretrained(PRETRAINED_MODEL)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    decoder_start_id = processor.tokenizer.convert_tokens_to_ids([TASK_TOKEN])[0]

    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = decoder_start_id

    # In transformers 5.x wird generation_config statt config verwendet —
    # beide setzen damit es auf 4.x und 5.x identisch funktioniert.
    model.generation_config.decoder_start_token_id = decoder_start_id
    model.generation_config.pad_token_id           = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id           = processor.tokenizer.eos_token_id

    return model, processor


# ---------------------------------------------------------------------------
# Stopping Criteria (version-robust)
# ---------------------------------------------------------------------------
class StopOnTaskEnd(StoppingCriteria):
    def __init__(self, task_end_token_id: int):
        self.task_end_token_id = task_end_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0, -1].item() == self.task_end_token_id


# ---------------------------------------------------------------------------
# Normalisierung für Acc-Berechnung (außerhalb der Schleife definiert)
# ---------------------------------------------------------------------------
def make_normalize(processor):
    pad = processor.tokenizer.pad_token
    eos = processor.tokenizer.eos_token

    def normalize(s: str) -> str:
        s = s.replace(pad, "").replace(eos, "")
        s = s.replace(TASK_TOKEN, "").replace(TASK_END_TOKEN, "")
        return " ".join(s.split())

    return normalize


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (USE_AMP and device.type == "cuda") else None
    print(f"Device: {device}  |  AMP: {'bfloat16' if amp_dtype else 'aus'}")

    model, processor = setup_model_and_processor()
    model.to(device)

    train_dataset = OrderDataset(TRAIN_DIR, processor)
    val_dataset   = OrderDataset(VAL_DIR,   processor)
    print(f"Train: {len(train_dataset)} Samples | Val: {len(val_dataset)} Samples")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps  = (len(train_loader) // GRAD_ACCUMULATION) * NUM_EPOCHS
    scheduler    = get_scheduler("cosine", optimizer,
                                 num_warmup_steps=total_steps // 10,
                                 num_training_steps=total_steps)
    scaler       = GradScaler(device="cuda") if (USE_AMP and amp_dtype == torch.float16) else None
    normalize    = make_normalize(processor)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss  = float("inf")
    global_step    = 0
    no_improve     = 0   # Early-Stop-Zähler

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            pv  = batch["pixel_values"].to(device)
            lbl = batch["labels"].to(device)

            ctx = autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
            with ctx:
                loss = model(pixel_values=pv, labels=lbl).loss / GRAD_ACCUMULATION

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            train_loss += loss.item() * GRAD_ACCUMULATION

            if (step + 1) % GRAD_ACCUMULATION == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % SAVE_STEPS == 0:
                    ckpt = OUTPUT_DIR / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt)
                    processor.save_pretrained(ckpt)
                    print(f"  Checkpoint gespeichert: {ckpt}")

        avg_train_loss = train_loss / len(train_loader)

        # --- Val ---
        model.eval()
        val_loss = 0.0
        correct  = 0
        n_total  = 0

        task_end_id   = processor.tokenizer.convert_tokens_to_ids(TASK_END_TOKEN)
        stop_criteria = StoppingCriteriaList([StopOnTaskEnd(task_end_id)])

        with torch.no_grad():
            for batch in val_loader:
                pv  = batch["pixel_values"].to(device)
                lbl = batch["labels"].to(device)

                ctx = autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
                with ctx:
                    val_loss += model(pixel_values=pv, labels=lbl).loss.item()

                generated = model.generate(
                    pv,
                    decoder_input_ids=torch.full(
                        (pv.size(0), 1),
                        model.config.decoder_start_token_id,
                        device=device,
                    ),
                    max_length=MAX_LENGTH,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    stopping_criteria=stop_criteria,
                )

                for gen_ids, lbl_ids in zip(generated, lbl):
                    # Generierte Sequenz bei </s_order> abschneiden —
                    # robuster als eos_token_id-Liste (Versionsunabhängig)
                    gen_list = gen_ids.tolist()
                    if task_end_id in gen_list:
                        gen_list = gen_list[:gen_list.index(task_end_id)]
                    pred = processor.tokenizer.decode(gen_list, skip_special_tokens=False)

                    lbl_clean = lbl_ids.clone()
                    lbl_clean[lbl_clean == -100] = processor.tokenizer.pad_token_id
                    gt = processor.tokenizer.decode(lbl_clean, skip_special_tokens=False)

                    if normalize(pred) == normalize(gt):
                        correct += 1
                    n_total += 1

        avg_val_loss = val_loss / len(val_loader)
        accuracy     = correct / n_total if n_total > 0 else 0

        print(
            f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {accuracy:.2%}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve    = 0
            best_path = OUTPUT_DIR / "best_model"
            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)
            print(f"  -> Bestes Modell gespeichert (Val Loss: {best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly Stop: {EARLY_STOP_PATIENCE} Epochen ohne Verbesserung.")
                break

    final_path = OUTPUT_DIR / "final_model"
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\nTraining abgeschlossen. Finales Modell: {final_path}")


if __name__ == "__main__":
    train()
