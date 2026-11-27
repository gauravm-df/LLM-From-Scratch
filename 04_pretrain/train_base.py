import os
import sys
import math
import time
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "03_model"))
from configuration import create_blank_model

def run_pretraining():
    corpus_file = PROJECT_ROOT / "01_data" / "raw_pretrain_corpus.txt"
    tokenizer_dir = PROJECT_ROOT / "02_tokenizer" / "model"
    output_dir = PROJECT_ROOT / "04_pretrain" / "base_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("STAGE 4: PRE-TRAINING LANGUAGE MODEL FROM ABSOLUTE ZERO")
    print(f"Corpus: {corpus_file}")
    print(f"Output Model: {output_dir}")
    print("=" * 65)

    # 1. Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Hardware compute device: {device}")

    # 2. Load Tokenizer
    tok_json = tokenizer_dir / "tokenizer.json"
    if not tok_json.exists():
        from tokenizers import ByteLevelBPETokenizer
        tok = ByteLevelBPETokenizer(
            str(tokenizer_dir / "vocab.json"),
            str(tokenizer_dir / "merges.txt")
        )
        tok.save(str(tok_json))

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_json),
        bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>"
    )

    # 3. Create Blank Model (Zero pre-existing weights)
    model = create_blank_model(vocab_size=len(tokenizer))
    print(f"Initialized blank Transformer with {model.num_parameters():,} random weights.")

    # 4. Load Raw Text Dataset
    dataset = load_dataset("text", data_files={"train": str(corpus_file)})

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 5. Data Collator for Causal LM (Labels = Input IDs shifted by 1)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. Training Arguments
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,       # Effective batch size = 16
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        max_steps=500,                       # Fast Pilot Run: ~3-4 minutes on Mac
        logging_steps=20,
        save_strategy="steps",
        save_steps=250,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    # 7. Start Training
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=tokenized_dataset["train"]
    )

    start_time = time.time()
    result = trainer.train()
    elapsed = time.time() - start_time

    # 8. Evaluation (Perplexity)
    loss = result.training_loss
    ppl = math.exp(loss) if loss < 20 else float("inf")

    print(f"\nPre-Training Completed in {elapsed/60:.2f} minutes!")
    print(f"Final Pre-Training Loss: {loss:.4f}")
    print(f"Final Perplexity: {ppl:.2f}")

    # 9. Save Base Model
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Base foundation model saved to: {output_dir}")

if __name__ == "__main__":
    run_pretraining()

