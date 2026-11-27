import os
import sys
from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(
    title="Custom Scratch LLM REST API",
    description="100% Offline Local LLM Built and Trained From Scratch with Zero External Models",
    version="1.0.0"
)

chat_dir = PROJECT_ROOT / "05_finetune" / "chat_model"
base_dir = PROJECT_ROOT / "04_pretrain" / "base_model"
model_dir = chat_dir if chat_dir.exists() else base_dir

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global tokenizer, model
    if not model_dir.exists():
        print(f"Warning: Model directory {model_dir} does not exist yet. Please run training first.")
        return
    print(f"Loading custom LLM weights from {model_dir} on {device}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir)).to(device)
    model.eval()
    print("Model loaded successfully into RAM/VRAM!")

class ChatQuery(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.2
    top_p: float = 0.85

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_directory": str(model_dir),
        "device": str(device),
        "model_loaded": (model is not None)
    }

@app.post("/v1/chat")
def chat_endpoint(query: ChatQuery):
    global tokenizer, model
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not initialized or trained yet.")

    try:
        formatted = f"<|im_start|>user\n{query.prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=query.max_tokens,
                temperature=query.temperature,
                top_p=query.top_p,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        reply = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        clean = reply.split("<|im_end|>")[0].strip()

        return {
            "status": "success",
            "prompt": query.prompt,
            "response": clean,
            "tokens_generated": len(outputs[0]) - len(inputs.input_ids[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn 06_inference.server_fastapi:app --host 127.0.0.1 --port 8000

