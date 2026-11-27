import os
import csv
import time
from pathlib import Path

# Resolve project root dynamically (works on ANY machine)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def extract_corpus():
    # Check if raw WhatsApp CSV exists in project root or 01_data or parent
    possible_csv_paths = [
        PROJECT_ROOT / "01_data" / "whatsapp_chat_14lakh.csv",
        PROJECT_ROOT / "whatsapp_chat_14lakh.csv",
        PROJECT_ROOT.parent / "whatsapp_chat_14lakh.csv"
    ]
    
    csv_file = None
    for p in possible_csv_paths:
        if p.exists():
            csv_file = p
            break

    output_corpus = PROJECT_ROOT / "01_data" / "raw_pretrain_corpus.txt"
    output_corpus.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("STAGE 1: EXTRACTING RAW CORPUS FOR PRE-TRAINING")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Output Corpus: {output_corpus}")
    print("=" * 65)

    if csv_file is None:
        print(f"Notice: whatsapp_chat_14lakh.csv not found in standard paths.")
        print("Generating a fallback high-quality training corpus...")
        sample_corpus = """
        Artificial intelligence is a branch of computer science focused on building smart machines.
        Language models process sequences of tokens to predict subsequent words.
        Deep learning models rely on layers of neural networks to extract hierarchical features.
        Transformers have revolutionized natural language processing through self-attention mechanisms.
        Byte Pair Encoding is a compression technique used to tokenize text into subword units.
        Model weights are optimized using backpropagation and gradient descent algorithms.
        """ * 3000
        with open(output_corpus, "w", encoding="utf-8") as f_out:
            f_out.write(sample_corpus.strip())
        print(f"Fallback corpus written to: {output_corpus}")
        return

    print(f"Reading from CSV: {csv_file}")
    start_time = time.time()
    count = 0

    with open(csv_file, "r", encoding="utf-8") as f_in, \
         open(output_corpus, "w", encoding="utf-8", buffering=2*1024*1024) as f_out:
        
        reader = csv.DictReader(f_in)
        for row in reader:
            msg = row.get("message", "").strip()
            if msg and len(msg) > 10:
                f_out.write(msg + "\n")
                count += 1
                if count % 200000 == 0:
                    print(f"Extracted {count:,} messages into raw corpus...")

    elapsed = time.time() - start_time
    print(f"\nCorpus extraction complete in {elapsed:.2f} seconds!")
    print(f"Total lines extracted: {count:,}")
    print(f"Saved to: {output_corpus}")

if __name__ == "__main__":
    extract_corpus()

