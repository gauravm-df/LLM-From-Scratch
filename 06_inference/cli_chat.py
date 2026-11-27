import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    chat_dir = PROJECT_ROOT / "05_finetune" / "chat_model"
    base_dir = PROJECT_ROOT / "04_pretrain" / "base_model"

    if chat_dir.exists():
        model_dir = chat_dir
    elif base_dir.exists():
        model_dir = base_dir
    else:
        print("Error: No trained model found in 05_finetune/chat_model or 04_pretrain/base_model.")
        print("Please run train_base.py and train_sft.py first.")
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print(f"CUSTOM LLM INTERACTIVE TERMINAL (Running 100% Offline on {device})")
    print(f"Loading weights from: {model_dir}")
    print("Type 'exit' or 'quit' to end chat.")
    print("=" * 65)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir)).to(device)
    model.eval()

    while True:
        try:
            prompt = input("\nYou: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            formatted = (
                "<|im_start|>system\n"
                "You are a friendly, natural English WhatsApp conversational companion.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=128,
                    temperature=0.2,
                    top_p=0.85,
                    top_k=40,
                    repetition_penalty=1.15,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )

            reply = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            clean_reply = reply.split("<|im_end|>")[0].strip()
            print(f"AI: {clean_reply}")
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break

if __name__ == "__main__":
    main()

