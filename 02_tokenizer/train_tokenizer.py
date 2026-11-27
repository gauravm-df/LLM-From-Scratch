import os
import time
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def train_custom_bpe():
    corpus_file = PROJECT_ROOT / "01_data" / "raw_pretrain_corpus.txt"
    model_dir = PROJECT_ROOT / "02_tokenizer" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("STAGE 2: TRAINING CUSTOM BPE TOKENIZER FROM SCRATCH")
    print(f"Corpus: {corpus_file}")
    print(f"Target Output Directory: {model_dir}")
    print("Vocabulary Size: 16,384 tokens (Zero Unknown Tokens)")
    print("=" * 65)

    if not corpus_file.exists():
        print(f"Notice: {corpus_file} not found. Running extractor first...")
        from importlib.machinery import SourceFileLoader
        extractor = SourceFileLoader("extractor", str(PROJECT_ROOT / "01_data" / "prepare_pretrain_corpus.py")).load_module()
        extractor.extract_corpus()

    start_time = time.time()

    # Initialize ByteLevelBPETokenizer (Guarantees zero <unk> out-of-vocabulary tokens)
    tokenizer = ByteLevelBPETokenizer()

    # Train directly on raw corpus
    tokenizer.train(
        files=[str(corpus_file)],
        vocab_size=16384,
        min_frequency=2,
        special_tokens=[
            "<s>",                            # BOS: Beginning of Sequence
            "<pad>",                          # PAD: Padding token
            "</s>",                           # EOS: End of Sequence
            "<unk>",                          # UNK: Unknown character
            "<|im_start|>",                   # ChatML block start
            "<|im_end|>"                      # ChatML block end
        ]
    )

    # Save vocab.json, merges.txt AND tokenizer.json (Hugging Face standard)
    tokenizer.save_model(str(model_dir))
    tokenizer.save(str(model_dir / "tokenizer.json"))
    elapsed = time.time() - start_time

    print(f"\nTokenizer successfully trained from scratch in {elapsed:.2f} seconds!")
    print(f"Vocabulary and tokenizer.json saved to: {model_dir}")

if __name__ == "__main__":
    train_custom_bpe()

