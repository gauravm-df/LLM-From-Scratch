import os
import sys
import time
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_fine_tuning():
    # Detect training jsonl from project root, 01_data, or parent
    possible_train = [
        PROJECT_ROOT / "01_data" / "train_whatsapp_700k.jsonl",
        PROJECT_ROOT / "train_whatsapp_700k.jsonl",
        PROJECT_ROOT.parent / "train_whatsapp_700k.jsonl"
    ]
    possible_val = [
        PROJECT_ROOT / "01_data" / "val_whatsapp_20k.jsonl",
        PROJECT_ROOT / "val_whatsapp_20k.jsonl",
        PROJECT_ROOT.parent / "val_whatsapp_20k.jsonl"
    ]

    train_jsonl = None
    for p in possible_train:
        if p.exists():
            train_jsonl = p
            break

    val_jsonl = None
    for p in possible_val:
        if p.exists():
            val_jsonl = p
            break

    pretrained_base = PROJECT_ROOT / "04_pretrain" / "base_model"
    chat_output_dir = PROJECT_ROOT / "05_finetune" / "chat_model"
    chat_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("STAGE 5: SUPERVISED FINE-TUNING (SFT) WITH WHATSAPP CONVERSATIONS")
    print(f"Pre-Trained Base Model: {pretrained_base}")
    print(f"Train Dataset: {train_jsonl}")
    print(f"Output Chat Model: {chat_output_dir}")
    print("=" * 65)

    if not pretrained_base.exists():
        print(f"Error: Base model not found in {pretrained_base}.")
        print("Please ensure Stage 4 Pre-training has run first.")
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware compute device: {device}")

    # 1. Load Tokenizer & Pre-trained Base Model
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(pretrained_base))
    model = AutoModelForCausalLM.from_pretrained(str(pretrained_base))

    # 2. Load Chat Datasets
    dataset = load_dataset("json", data_files={
        "train": str(train_jsonl),
        "validation": str(val_jsonl)
    })

    # 3. Tokenize ChatML conversations
    def tokenize_chat(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")

    print("Tokenizing ChatML conversations for SFT...")
    tokenized_train = dataset["train"].map(tokenize_chat, batched=True, remove_columns=["text", "messages", "metadata"])
    tokenized_val = dataset["validation"].map(tokenize_chat, batched=True, remove_columns=["text", "messages", "metadata"])

    # 4. Data Collator for Causal Language Modeling
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. Training Arguments (300 fast pilot steps: ~2 minutes on Mac)
    args = TrainingArguments(
        output_dir=str(chat_output_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=300,
        logging_steps=20,
        save_strategy="steps",
        save_steps=150,
        eval_strategy="steps",
        eval_steps=150,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val
    )

    print("Starting SFT Conversational Fine-Tuning...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    # 7. Save Chat Model
    model.save_pretrained(str(chat_output_dir))
    tokenizer.save_pretrained(str(chat_output_dir))
    print(f"\nFine-Tuning Completed in {elapsed/60:.2f} minutes!")
    print(f"Final Conversational Model saved to: {chat_output_dir}")

if __name__ == "__main__":
    run_fine_tuning()
